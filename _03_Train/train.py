import wandb
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import LoadData
import numpy as np
import random
from torch.utils.data.sampler import WeightedRandomSampler
from config import Config_03_Train as config


# train loop
def training(device, project_name, exp_name, model_name, net, data_path, train_txt_path, val_txt_path, epochs,
             batch_size, lr, dropout, l2, milestones, lsmooth, pos_weight, dim, pretrain):
    # set wandb options
    torch.manual_seed(config.seed)

    wandb.init(project=project_name,
               config={"exp_name": exp_name, "model": model_name,
                       "pretrain": pretrain, "learning_rate": lr,
                       "dropout": dropout, "l2": l2,
                       "batch_size": batch_size, "dim": dim,
                       "epochs": epochs, "milestones": milestones,
                       "lsmooth": lsmooth, "pos_weight": pos_weight})
    wandb.run.name = exp_name

    net = net.to(device)

    # data loaders
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    # Create dataset and data loader for training and validation
    traindata, valdata = LoadData.load_data(data_path, train_txt_path, val_txt_path, dim=dim)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, worker_init_fn=seed_worker,
                                              generator=g, shuffle=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, worker_init_fn=seed_worker,
                                            generator=g, shuffle=False)

    # training params
    if project_name == 'BinaryClass':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif project_name == 'MultiClass':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=lsmooth,
                                              weight=torch.tensor([1 - trainloader.dataset.label.count('0') / len(
                                                  trainloader.dataset.label),
                                                                   1 - trainloader.dataset.label.count('1') / len(
                                                                       trainloader.dataset.label),
                                                                   1 - trainloader.dataset.label.count('2') / len(
                                                                       trainloader.dataset.label)]).to(device))

    optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999), weight_decay=l2, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ### Train Loop
    best_val_acc = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch + 1}\n-------------------------------")
        loss_list = []
        total_train = 0
        correct_train = 0

        net.train()  # set the network in training mode

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if project_name == 'MultiClass':
                labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            if project_name == 'BinaryClass':
                smooth_labels = (labels.unsqueeze(1).float()) * (1 - lsmooth) + (
                            lsmooth / 2)  # smooth_labels = [orig labels] * (1-lsmooth) + (lsmooth/n_classes)
                loss = criterion(outputs, smooth_labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            loss_list.append(loss.item())

            # output predictions on training set
            if project_name == 'BinaryClass':
                predicted = torch.round(torch.nn.Sigmoid()(outputs.data))
                correct_train += (predicted.squeeze(1) == labels).sum().item()
            else:
                predicted = torch.nn.Softmax(dim=-1)(outputs.data)
                correct_train += (predicted.argmax(1) == labels).type(torch.float).sum().item()

            total_train += labels.size(0)

        if scheduler is not False:
            scheduler.step()

        # Validation loss
        val_loss_list = []
        total = 0
        correct = 0

        net.eval()  # set the network in evaluation mode

        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                if project_name == 'MultiClass':
                    labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                if project_name == 'BinaryClass':
                    predicted = torch.round(torch.nn.Sigmoid()(outputs.data))
                    correct += (predicted.squeeze(1) == labels).sum().item()
                else:
                    predicted = torch.nn.Softmax(dim=-1)(outputs.data)
                    correct += (predicted.argmax(1) == labels).type(torch.float).sum().item()

                total += labels.size(0)

                if project_name == 'BinaryClass':
                    smooth_labels = (labels.unsqueeze(1).float()) * (1 - lsmooth) + (
                                lsmooth / 2)  # smooth_labels = [orig labels] * (1-lsmooth) + (lsmooth/n_classes)
                    val_loss = criterion(outputs, smooth_labels)
                else:
                    val_loss = criterion(outputs, labels)

                val_loss_list.append(val_loss.item())

        wandb.log({'epoch': epoch + 1,
                   'acc_train': correct_train / total_train,
                   'loss_train': sum(loss_list) / len(loss_list),
                   'acc_val': correct / total,
                   'loss_val': sum(val_loss_list) / len(val_loss_list)})

        print(f"val loss: ", sum(val_loss_list) / len(val_loss_list))
        print(f"val accuracy: ", correct / total)
        print(f"train loss: ", sum(loss_list) / len(loss_list))
        print(f"train accuracy: ", correct_train / total_train)

        val_acc = correct / total
        if val_acc > best_val_acc:
            torch.save({
                'epoch': epoch + 1,
                'model': net,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
                'val_loss': val_loss_list,
            }, config.checkpoints_path + '/' + exp_name + '.pt') 

            print('finished epoch: better model')
            best_val_acc = val_acc

    print("Finished Training")
    print('Best_val_acc = ', best_val_acc)

    return best_val_acc
