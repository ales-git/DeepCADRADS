import cv2
import numpy as np
from numpy import asarray
import os
from pathlib import Path
from config import Config_00_Preprocessing as config

"""
This script contains support functions useful to remove artefacts
and enhance the contrast of straightened MPR coronary images.
"""


def minMaxNormalise(img):
    """
    This function does min-max normalisation on
    the given image.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to normalise.
    Returns
    -------
    norm_img: {numpy.ndarray}
        The min-max normalised image.
    """
    norm_img = (img - img.min()) / (img.max() - img.min())

    return norm_img


def globalBinarise(img, thresh, maxval):
    """"
    This function takes as input a numpy array image and
    returns a corresponding mask that is a global binarisation
    on of the original image based on a given threshold and maxval.
    Any elements in the array that is greater than or equals to the given threshold
    will be assigned maxval, else zero.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to perform binarisation on.
    thresh : {int or float}
        The global threshold for binarisation.
    maxval : {np.uint8}
        The value assigned to an element that is greater
        than or equals to `thresh`.
    Returns
    -------
    binarised_img : {numpy.ndarray, dtype=np.uint8}
        A binarised image of {0, 1}.
        """
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval

    return binarised_img


def editMask(mask, ksize=(23, 23), operation="open"):
    """
    This function edits a given mask (binary image) by performing
    closing then opening morphological operations.
    Parameters
    ----------
    mask : {numpy.ndarray}
        The mask to edit.
    ksize : {tuple}
        Size of the structuring element.
    operation : {str}
        Either "open" or "close", each representing open and close
        morphological operations respectively.
    Returns
    -------
    edited_mask : {numpy.ndarray}
        The mask after performing close and open morphological
        operations.
    """
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask


def sortContoursByArea(contours, reverse=True):
    """
    This function takes as inpu a list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.
    Parameters
    ----------
    contours : {list}
        The list of contours to sort.
    Returns
    -------
    sorted_contours : {list}
        The list of contours sorted by contour area in descending
        order.
    bounding_boxes : {list}
        The list of bounding boxes ordered corresponding to the
        contours in `sorted_contours`.
    """
    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes


def xLargestBlobs(mask, top_x=None, reverse=True):
    """
    This function finds contours in the given image and
    keeps only the top X largest ones.
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to get the top X largest blobs.
    top_x : {int}
        The top X contours to keep based on contour area
        ranked in descending order.
    Returns
    -------
    n_contours : {int}
        The number of contours found in the given `mask`.
    X_largest_blobs : {numpy.ndarray}
        The corresponding mask of the image containing only
        the top X largest contours in white.
    """
    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    n_contours = len(contours)

    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:

        # Make sure that the number of contours to keep is at most equal
        # to the number of contours present in the mask.
        if n_contours < top_x or top_x == None:
            top_x = n_contours

        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = sortContoursByArea(
            contours=contours, reverse=reverse
        )

        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_x]

        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,  # Draw the contours on `to_draw_on`.
            contours=X_largest_contours,  # List of contours to draw.
            contourIdx=-1,  # Draw all contours in `contours`.
            color=1,  # Draw the contours in white.
            thickness=-1,  # Thickness of the contour lines.
        )

    return n_contours, X_largest_blobs


def applyMask(img, mask):
    """
    This function applies a mask to a given image. White
    areas of the mask are kept, while black areas are
    removed.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to mask.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to apply.
    Returns
    -------
    masked_img: {numpy.ndarray}
        The masked image.
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img


def clahe(img, clip=2.0, tile=(8, 8)):
    """
    This function applies the Contrast-Limited Adaptive
    Histogram Equalisation filter to a given image.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to edit.
    clip : {int or float}
        Threshold for contrast limiting.
    tile : {tuple (int, int)}
        Size of grid for histogram equalization. Input
        image will be divided into equally sized
        rectangular tiles. `tile` defines the number of
        tiles in row and column.
    Returns
    -------
    clahe_img : {numpy.ndarray, np.uint8}
        The CLAHE edited image, with values ranging from [0, 255]
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img)

    return clahe_img


def removeColor(img):
    """
    Function to remove any colored elements from the image.
    HSV color table (Hue, Saturation, Value):
    # mask of green (36,25,25) to (70, 255,255)
    # mask of yellow (20, 25, 25) to (36, 255, 255)
    """
    # convert image to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 25, 25), (160, 255, 255))
    # Remove all colored elements
    imask = mask > 0
    color = np.zeros_like(img, np.uint8)
    color[imask] = img[imask]
    img[imask] = (0, 0, 0)
    return img


def autoCrop(img, threshold=0):
    """
    Crops any edges below or equal to threshold and returns cropped image.
    The optimal threshold should be defined based on your dataset characteristics.
    """
    if len(img.shape) == 3:
        flatImage = np.max(img, 2)
    else:
        flatImage = img
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        img = img[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        'no non zero pixels found'
    return img


def fullImgPreprocess(img, thresh, maxval, ksize, operation, reverse, top_x, clip, tile):
    """
    This function chains and executes all the preprocessing
    steps for a Img, in the following order:
    Step 1 - Remove colors
    Step 2 - Remove artefacts
    Step 3 - CLAHE enhancement
    Step 4 - Crop

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to preprocess.
    Returns
    -------
    img_pre : {numpy.ndarray}
        The preprocessed image.
    """

    # Step 1: Remove colors
    img = removeColor(img=img)
    array_img = asarray(img)

    # Step 2: Remove artefacts
    bin_img = globalBinarise(img=array_img, thresh=thresh, maxval=maxval)
    edited_mask = editMask(mask=bin_img, ksize=(ksize, ksize), operation=operation)
    n_countours, X_largest_mask = xLargestBlobs(mask=edited_mask, top_x=top_x, reverse=reverse)
    masked_image = applyMask(img=array_img, mask=X_largest_mask)

    # Step 3: CLAHE enhancement
    clahe_img = clahe(img=masked_image, clip=clip, tile=(tile, tile))

    # Step 4: Crop
    img_pre = autoCrop(img=clahe_img, threshold=1)

    return img_pre


if __name__ == "__main__":
    ''' You may need to adjust the parameters and the preprocessing operations needed given your input dataset characteristics'''

    path_orig = config.orig_path  # Path to the original dataset
    for f in sorted(os.listdir(path_orig)):  # loop on images
        file_path = os.path.join(path_orig, f)
        path = Path(file_path)
        img = cv2.imread(file_path)
        img_pre = fullImgPreprocess(
            img=img,
            thresh=config.thresh,
            maxval=config.maxval,
            ksize=config.ksize,
            operation=config.operation,
            reverse=config.reverse,
            top_x=config.top_x,
            clip=config.clip,
            tile=config.tile
        )

        save_path = os.path.join(config.out_path, f)
        cv2.imwrite(save_path, img_pre)
