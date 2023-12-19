import torch
import wandb
import itertools
import train as t
import random
import timm
import numpy as np
import datetime
from config import Config_03_Train as config

if __name__ == "__main__":
    random.seed(config.seed)

    project_name = config.project_name

    # Training params
    lists = [config.dropout_list, config.l2_list, config.lr_list, config.milestones_list, config.lsmooth_list]

    param_comb = list(itertools.product(*lists))
    list_config_results = []
    list_config_std = []
    for ncomb, comb in enumerate(param_comb):
        exit_flag = False
        list_best_val_acc = []
        print(datetime.datetime.now())
        print(f'--- Cross Val CONFIG {ncomb + 1} ---')
        for fold in range(1, 11):
            if exit_flag == False:
                nset = 'set' + str(fold)

                data_path = config.main_data_path + nset
                train_txt_path = config.main_txt_path + '/' + nset + '/txt/train.txt'
                val_txt_path = config.main_txt_path + '/' + nset + '/txt/val.txt'

                # Check for GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("Using {} device".format(device))

                ######################## PARAMS ###############################

                dropout = comb[0]
                l2 = comb[1]
                lr = comb[2]
                milestones = comb[3]
                lsmooth = comb[4]

                ######################## MODEL ###############################
                if project_name == 'BinaryClass':
                    num_classes = 1
                elif project_name == 'MultiClass':
                    num_classes = 3

                pretrain = True  # True
                model_name = config.model_name
                model = timm.create_model(model_name,
                                          pretrained=pretrain,
                                          num_classes=num_classes,
                                          drop_rate=dropout)

                ######################### TRAINING #########################
                comb_name = 'CONFIG' + str(ncomb + 1)
                fold_name = '_FOLD' + str(fold)
                param_name = '_dropout=' + str(dropout) + '_l2=' + str(l2) + '_lr=' + str(lr) + '_milestones=' + str(
                    milestones) + '_lsmooth_' + str(lsmooth)
                exp_name = comb_name + fold_name + param_name
                best_val_acc = t.training(device, project_name, exp_name, model_name, model, data_path, train_txt_path,
                                          val_txt_path, config.epochs, config.batch_size, lr, dropout, l2, milestones,
                                          lsmooth,
                                          config.pos_weight, config.dim, pretrain)
                list_best_val_acc.append(best_val_acc)
                wandb.finish()
                if round(best_val_acc, 2) < config.min_acc_fold:
                    exit_flag = True
                    list_best_val_acc = [0.0]

        print('########################### RESULTS CV #######################################')
        print(list_best_val_acc)
        print(f'avg_val_acc: {np.mean(list_best_val_acc)}')
        print(f'std_val_acc: {np.std(list_best_val_acc)}')
        print('###############################################################################')
        list_config_results.append(np.mean(list_best_val_acc))
        list_config_std.append(np.std(list_best_val_acc))
        print(list_config_results)

    print('\n###################### RESULTS GRIDSEARCH #########################################')
    print(list_config_results)
    print(f'max_avg_val_acc: {np.max(list_config_results)}')
    print(f'config number: {np.argmax(list_config_results) + 1}')
    print('####################################################################################')
    print(datetime.datetime.now())
