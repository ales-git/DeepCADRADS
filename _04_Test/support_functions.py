import os.path
import numpy as np
import torch
from torchmetrics import Accuracy, ConfusionMatrix, AUROC
from torchvision import transforms
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import shap
import matplotlib.pyplot as plt
from config import Config_04_Test as config
import sys


def metrics(outputs, targets, probs, exp_name, w, type_metrics):
    if exp_name == 'BinaryClass':
        # Accuracy
        compute_Accuracy = Accuracy()
        if type_metrics == 'per_img':
            Accuracy_tot = compute_Accuracy(outputs.squeeze(1), targets)
        else:
            Accuracy_tot = compute_Accuracy(outputs, targets)

        # ConfusionMatrix
        compute_CM = ConfusionMatrix(num_classes=2)
        ConfusionMatrix_tot = compute_CM(outputs, targets)
        TN = ConfusionMatrix_tot[0][0]
        FP = ConfusionMatrix_tot[0][1]
        FN = ConfusionMatrix_tot[1][0]
        TP = ConfusionMatrix_tot[1][1]

        # Precision
        precision = TP / (TP + FP)

        # Recall
        recall = TP / (TP + FN)

        # Specificity
        specificity = TN / (TN + FP)

        # F1
        if TP.item() == 0:
            F1 = torch.tensor(0)
        else:
            F1 = (2 * precision * recall) / (precision + recall)

        # AUROC
        compute_AUROC = AUROC(num_classes=2, pos_label=1)
        AUROC_tot = compute_AUROC(probs.view(-1), targets.view(-1))

        # ROC
        y_true = np.array(targets)
        y_prob = np.array(probs)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        # print(optimal_threshold)
        plt.figure(0)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(fpr, tpr)
        plt.show()

        # File with test results
        f = open(os.path.join('../results', exp_name + '_' + type_metrics + ".txt"), "w")
        f.write('** Metrics computed on testset **\n\n')
        f.write('Experiment: ' + exp_name + '\n')

        results = ['\n\nAccuracy: ' + str(Accuracy_tot.item()),

                   '\n\n ConfusionMatrix: ',
                   '\n' + str([TN.item(), FP.item()]),
                   '\n' + str([FN.item(), TP.item()]),

                   '\n\n AUROC: ' + str(AUROC_tot.item()),
                   '\n\n Precision: ' + str(precision.item()),
                   '\n\n Recall: ' + str(recall.item()),
                   '\n\n Specificity: ' + str(specificity.item()),
                   '\n\n F1: ' + str(F1.item())
                   ]

        f.writelines(results)
        f.close()

    else:  # multiclass

        # Accuracy
        compute_Accuracy = Accuracy()
        Accuracy_tot = compute_Accuracy(outputs, targets)

        # ConfusionMatrix
        compute_CM = ConfusionMatrix(num_classes=3)
        ConfusionMatrix_tot = compute_CM(outputs, targets)
        print(ConfusionMatrix_tot)

        # Report
        report = classification_report(targets, outputs, digits=3)
        print(report)

        # AUROC
        labels = targets.view(-1)
        # Convert the labels to binary format
        n_classes = 3
        binary_labels = np.zeros((len(labels), n_classes))
        for i in range(n_classes):
            binary_labels[:, i] = (labels == i)

        # Compute the AUROC for each positive class
        aurocs = []
        fprlist = []
        tprlist = []
        for i in range(n_classes):
            auroc = roc_auc_score(binary_labels[:, i], probs[:, i])
            y_true = np.array(binary_labels[:, i])
            y_prob = np.array(probs[:, i])
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            fprlist.append(fpr)
            tprlist.append(tpr)
            aurocs.append(auroc)

        # Compute the overall AUROC as the average of the individual AUROCs

        if type_metrics == 'per_img':
            auroc = np.average(aurocs, weights=(w[0], w[1], w[2]))  # weights per img (define in config.py)
        else:
            auroc = np.average(aurocs, weights=(w[0], w[1], w[2]))  # weights per pat (define in config.py)

        ### average metrics
        # Compute the precision, recall, F1 score, and support (number of true instances) for each class
        precision, recall, f1, support = precision_recall_fscore_support(targets, outputs, average=None)

        # Compute the weighted average precision, recall, F1 score, and accuracy
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)

        # File with test results
        f = open(os.path.join('../results', exp_name + '_' + type_metrics + ".txt"), "w")
        f.write('** Metrics computed on testset **\n\n')
        f.write('Experiment: ' + exp_name + '\n')

        results = ['\n\nAccuracy: ' + str(Accuracy_tot.item()),
                   '\n\n ConfusionMatrix: ',
                   '\n' + str(ConfusionMatrix_tot),
                   '\n\n Report: ',
                   '\n' + str(report),
                   '\n\n AUROC: ' + str(auroc),
                   '\n\n Weighted Precision: ' + str(weighted_precision),
                   '\n\n Weighted Recall: ' + str(weighted_recall),
                   '\n\n Weighted F1: ' + str(weighted_f1)
                   ]

        f.writelines(results)
        f.close()


def image_plot(ch, shap_values, pixel_values, labels=None, preds=None, names=None, width=20, aspect=0.2, hspace=0.2,
               labelpad=None, multi_output=False, show=True, fig_size=None):
    """ Plots SHAP values for image inputs.
    It is a slightly modified version of the original shap.image_plot function from
    the shap package that allows to plot single channel images.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being explained.
    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
        shape as each array in the shap_values list of arrays.
    labels : list
        List of names for each of the model outputs that are being explained. This list should be the same length
        as the shap_values list.
    width : float
        The width of the produced matplotlib plot.
    labelpad : float
        How much padding to use around the model output labels.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    # make sure labels
    if labels is not None:
        assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
        if multi_output:
            assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
        else:
            assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    x = pixel_values
    if fig_size is None:
        fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
        if fig_size[0] > width:
            fig_size *= width / fig_size[0]
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                    0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
        else:
            x_curr_gray = x_curr

        axes[row, 0].imshow(x_curr, cmap=plt.get_cmap('gray'))
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        # preds = preds.T

        for i in range(len(shap_values)):
            # if labels is not None:
            # axes[row,i+1].set_title(f'Label: {int(labels[row])} Pred: {int(preds[row,i])}', **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.2,
                                    extent=(-1, sv.shape[1], sv.shape[0], -1))
            im = axes[row, i + 1].imshow(sv, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis('off')

    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect, anchor=(0.5, 2))
    cb.outline.set_visible(False)
    '''
    plt.savefig(f'shap_ch{ch}.jpg',format='jpg',dpi=1200)
    if show:
        plt.show()
    '''
    ''' 
    # plot single images
    plt.figure()
    sv = shap_values[0][row] if len(shap_values[0][row].shape) == 2 else shap_values[0][row].sum(-1)
    plt.imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.3, extent=(-1, sv.shape[1], sv.shape[0], -1))
    plt.imshow(sv, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
    plt.axis('off')
    plt.savefig(f'shap_ch{ch}.jpg', format='jpg', dpi=1200)

    plt.figure()
    plt.imshow(x_curr, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.savefig(f'fig_shap_ch{ch}.jpg', format='jpg', dpi=1200)
    '''
    if show:
        plt.show()


def shapValues(testloader, checkpoints_path):
    path_checkp = checkpoints_path + config.checkpoints_name + '.pt'
    checkpoint = torch.load(path_checkp, map_location=torch.device("cpu"))
    net = checkpoint['model']
    net.load_state_dict(checkpoint['model_state_dict'])

    images, labels, names = next(iter(testloader))

    background = images[:config.n_img_background]

    e = shap.DeepExplainer(net, background)

    ''' 
    If you want to explain several images at the same time you can modify the code and specify 
    in config.py just the starting index and the number of images to explain.
    '''

    test_images = images[config.index_img_shap]
    test_images = test_images.unsqueeze(0)
    test_labels = np.array(labels[config.index_img_shap])

    with open(os.devnull, 'w') as null_device:
        sys.stdout = null_device
        shap_values = e.shap_values(test_images)
        sys.stdout = sys.__stdout__

    outputs = net(test_images)

    if outputs.size(1) > 2:  # multiclass
        outputs = torch.nn.Softmax()(outputs.data)
        probs = outputs.clone()
        predicted = probs.argmax(1)
        # print(predicted)
    else:
        outputs = torch.nn.Sigmoid()(outputs.data)
        probs = outputs.clone()
        predicted = torch.round(outputs)
        # print(predicted)

    # plot
    resize_img = transforms.Resize(config.img_dim_plot)
    img_resized = resize_img(test_images)
    if outputs.size(1) > 2:  # multiclass
        shap_tensor = torch.FloatTensor(np.concatenate(shap_values))
        shap_resized = resize_img(shap_tensor.squeeze())
    else:
        shap_tensor = torch.from_numpy(np.concatenate(shap_values))
        shap_resized = resize_img(shap_tensor)

    shap_resized_numpy = shap_resized.numpy()
    if outputs.size(1)<2:
        shap_resized_numpy = shap_resized_numpy[np.newaxis, :]

    for ch in [0, 1, 2]:
        if outputs.size(1) > 2:  # multiclass
            test_images = img_resized[:, ch, :, :]  # check test_images dimensions
            shap_values = [shap_resized_numpy[:, ch, :, :]]
            labels = torch.tensor([0, 1, 2]).unsqueeze(0)
            shap_values = shap_values[0]
            shap_values = [shap_values[np.newaxis, 0, :, :], shap_values[np.newaxis, 1, :, :],
                           shap_values[np.newaxis, 2, :, :]]

            shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
            test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

            image_plot(ch, shap_numpy, test_numpy, labels, predicted, multi_output=True, show=True)
        else:
            test_images = img_resized[:, ch, :, :]
            shap_values = [shap_resized_numpy[:,ch, :, :]]

            shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
            test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

            image_plot(ch, shap_numpy, test_numpy, np.atleast_1d(test_labels), predicted, show=True)
