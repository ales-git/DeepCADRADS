import os.path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_03_Train')))
import _03_Train.CAD_Dataset as dat
import torch
import numpy as np
import pandas as pd
from support_functions import metrics, shapValues
from config import Config_04_Test as config


def test(exp_name, dim, batch_size, test_path, test_txt_path, checkpoints_path, checkpoint_name, type_metrics):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # data loader
    testdata = dat.CAD_Dataset(test_path, test_txt_path, dim, extract_names=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
    testloader_shap = torch.utils.data.DataLoader(testdata, batch_size=len(testdata), shuffle=False)

    # model
    path_checkp = os.path.join(checkpoints_path, checkpoint_name + '.pt')
    checkpoint = torch.load(path_checkp, map_location=torch.device(device))
    net = checkpoint['model']
    # print(net)
    net.load_state_dict(checkpoint['model_state_dict'])

    list_pred = []
    list_probs = []
    list_names = []
    list_labels = []
    dict_probs = {}

    net.eval()
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():

            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print(i)
            # print(names)

            outputs = net(inputs)

            if exp_name == 'BinaryClass':
                outputs = torch.nn.Sigmoid()(outputs.data)
                probs = outputs.clone()
                predicted = torch.round(outputs)

                if names not in list(dict_probs.keys()):
                    dict_probs[names] = {1: []}

                dict_probs[names][1].append(outputs[0].item())

            else:
                outputs = torch.nn.Softmax(dim=-1)(outputs.data)
                probs = outputs.clone()
                predicted = probs.argmax(1)

                if names not in list(dict_probs.keys()):
                    dict_probs[names] = {0: [], 1: [], 2: []}

                for i in range(outputs.shape[1]):
                    dict_probs[names][i].append(outputs[0, i].item())

            list_pred.append(predicted)
            list_probs.append(probs)
            list_names.append(names[0])
            list_labels.append(labels.type(torch.int))

    if type_metrics == 'per_img':
        outputs = torch.cat(list_pred).detach().cpu()
        probs = torch.cat(list_probs).detach().cpu()
        names = list_names
        labels = torch.cat(list_labels).detach().cpu()

        # SHAP
        shapValues(testloader_shap, checkpoints_path)

        return outputs, probs, names, labels

    else:
        list_preds_pat, list_probs_pat = [], []
        for p, dict_class in dict_probs.items():
            if config.exp_name == 'BinaryClass':
                list_preds_pat.append(np.round([np.mean(v) for k, v in dict_class.items()])[0])
                list_probs_pat.append([np.mean(v) for k, v in dict_class.items()])
            else:
                list_preds_pat.append(np.argmax([np.mean(v) for k, v in dict_class.items()]))
                list_probs_pat.append([np.mean(v) for k, v in dict_class.items()])

        if config.exp_name == 'BinaryClass':
            df_pat = pd.read_csv(config.csv_path)
            list_labels_pat = list(df_pat['bin_Class'])
            list_names_pat = list(df_pat['uniq_ID'])
        else:
            df_pat = pd.read_csv(config.csv_path)
            list_labels_pat = list(df_pat['multi_Class_3'])
            list_names_pat = list(df_pat['uniq_ID'])

        return torch.tensor(list_preds_pat), torch.tensor(list_probs_pat), list_names_pat, torch.tensor(list_labels_pat)


if __name__ == "__main__":
    '''
    You can choose if you want the test to be performed per image or per patient. 
    Each image represent a set of 3 coronary arteries for a specific patient in a specific view.
    You can test the ability of the model to predict the class of the patient based on the 3 images or aggregate 
    the results considering all the different views available for each patient. 
    In the clinical practice this gives you the possibility to analyse just the most useful views or to consider 
    all the views available depending on the difficulty of the case. 
    '''

    ''' Test per image '''
    outputs, probs, names, labels = test(config.exp_name, config.dim, config.batch_size, config.data_path,
                                         config.txt_path, config.checkpoints_path, config.checkpoints_name,
                                         type_metrics='per_img')
    metrics(outputs, labels, probs, config.exp_name, config.class_weights, type_metrics='per_img')

    ''' Test per patient '''
    outputs, probs, names, labels = test(config.exp_name, config.dim, config.batch_size, config.data_path,
                                         config.txt_path, config.checkpoints_path, config.checkpoints_name,
                                         type_metrics='per_pat')
    metrics(outputs, labels, probs, config.exp_name, config.class_weights, type_metrics='per_pat')

    print('Done!')
