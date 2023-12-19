import os
import numpy as np
import pandas as pd
from PIL import Image
from config import Config_02_SyntheticData as config

def filelist(directory, keywords, vessel_name):
    list_files = []
    for filename in os.listdir(directory):
        for keyword in keywords:
            if keyword in filename and vessel_name in filename:
                list_files.append(filename)
    return list_files

def avg_vessel(list_vessels, seg, num_set=None):
    c = 0
    for vessel in list_vessels:
        if c == 0:
            name = 'LCX_se00'+str(seg)
        elif c == 1:
            name = 'LAD_se00'+str(seg)
        elif c == 2:
            name = 'RCA_se00'+str(seg)

        # get dimensions of first image
        w, h = Image.open(os.path.join(config.dir, vessel[0])).size
        N = len(vessel)

        # Create a numpy array of floats to store the average
        arr = np.zeros((h, w), float)

        # Build up average pixel intensities, casting each image as an array of floats
        for im in vessel:
            im = Image.open(os.path.join(config.dir, vessel[0]))
            im = im.resize((w, h))
            imarr = np.array(im, dtype=float)
            arr = arr + imarr / N

        # Round values in array and cast as 8-bit integer
        arr = np.array(np.round(arr), dtype=np.uint8)
        c += 1

        # Generate, save and preview final image
        out = Image.fromarray(arr, mode="L")

        if num_set is not None:
            directory = os.path.join(config.main_out_path, 'train', num_set)
        else:
            directory = os.path.join(config.main_out_path, 'test')

        if not os.path.exists(directory):
            os.makedirs(directory)

        out_path = os.path.join(directory, f"avg_{name}.png")
        out.save(out_path)

def create_avg_images(num_set):
    if num_set is not None:
        num_set = f"set{num_set}"
        csv_path = os.path.join(config.main_csv_path,'train', num_set, "csv/X_train.csv")
    else:
        csv_path = os.path.join(config.main_csv_path, 'test', "csv/X_train.csv")

    df = pd.read_csv(csv_path)
    CR0_pat = df[df['Class'] == 'CR0']['uniq_ID']

    # create lists of vessel files for each segment
    vessel_names = ['LCX', 'LAD', 'RCA']
    all_files = os.listdir(config.dir)
    vessels_segs = []
    for seg in range(8):
        vessels = []
        for vessel_name in vessel_names:
            keywords = list(CR0_pat.values)
            vessel_files = filelist(config.dir, keywords, f"{vessel_name}_se{seg:03d}")
            vessels.append(vessel_files)
        vessels_segs.append(vessels)

    # create avg images for each vessel in each segment
    for seg, vessels in enumerate(vessels_segs):
        avg_vessel(vessels, seg, num_set)

if __name__ == "__main__":

    '''
    Run this script to create avg images of
    LCX, LAD and RCA vessels considering CR0 patients.
    '''

    ''' avg images are generated using using a different training set for each fold of the cross val. Saved in train folder.'''
    for num_set in range(1,config.n_fold+1):
        create_avg_images(num_set)

    ''' avg images using the whole training set (train+val) before final test. Saved in test folder.'''
    create_avg_images(num_set=None)