import random
import timm
import torch
import train as t
import wandb
from config import Config_03_Train as config

if __name__ == "__main__":
    random.seed(config.seed)

    project_name = config.project_name

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    ######################## PARAMS ###############################

    dropout = config.dropout
    l2 = config.l2
    lr = config.lr
    milestones = config.milestones
    lsmooth = config.lsmooth

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
    param_name = '_dropout=' + str(config.dropout) + '_l2=' + str(config.l2) + '_lr=' + str(
        config.lr) + '_milestones=' + str(config.milestones) + '_lsmooth_' + str(config.lsmooth)
    exp_name = config.project_name + param_name
    best_val_acc = t.training(device, project_name, exp_name, model_name, model, config.data_path,
                              config.txt_path + 'train.txt', config.txt_path + 'test.txt',
                              config.epochs, config.batch_size, lr, dropout, l2, milestones, lsmooth,
                              config.pos_weight, config.dim, pretrain)
    wandb.finish()
