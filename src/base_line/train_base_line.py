import torch
import albumentations as A
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import time
import copy
import glob

import os
import sys
path = os.path.dirname(__file__)
root_folder = os.path.join(
    os.path.abspath(path).split("domain_calibration")[0],
    "domain_calibration"
)
sys.path.insert(0, root_folder)

from src.base_line.data_loader import Dataset
from src.base_line.restnet34 import initialize_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import pkbar

def train_model(
    model, dataloaders, criterion, 
    optimizer, num_epochs=25, 
    folder_save="./experiments/", 
    k_fold=0,
    is_inception=False):
    
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Training epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            kbar = pkbar.Kbar(
                target=len(dataloaders[phase]), 
                epoch=epoch, num_epochs=num_epochs, 
                width=80, always_stateful=False
            )
            
            running_loss = 0.0
            running_corrects = 0
            
            index = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                kbar.update(index, values=[(f"{phase}_loss", loss.item())])
                index += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                path_save = folder_save + f"/resnet_34_kfold_{k_fold}.pth"
                torch.save(model.state_dict(), path_save)
                print("\n Saving best model ... ")

            if not os.path.isfile(folder_save + f"/resnet_34_kfold_{k_fold}_train_history.txt"):
                f = open(folder_save + f"/resnet_34_kfold_{k_fold}_train_history.txt", "a")
                f.write("epoch_loss, epoch_acc")
                f.close()
            
            if not os.path.isfile(folder_save + f"/resnet_34_kfold_{k_fold}_val_history.txt"):
                f = open(folder_save + f"/resnet_34_kfold_{k_fold}_val_history.txt", "a")
                f.write("epoch_loss, epoch_acc")
                f.close()

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                f = open(folder_save + f"/resnet_34_kfold_{k_fold}_train_history.txt", "a")
                f.write("\n" + str(epoch_loss) + ", " + str(epoch_acc.item()))
                f.close()
            
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                f = open(folder_save + f"/resnet_34_kfold_{k_fold}_val_history.txt", "a")
                f.write("\n" + str(epoch_loss) + ", " + str(epoch_acc.item()))
                f.close()

        print()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('\nBest val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history


if __name__ == "__main__":
    config_params = {
        "dataset_name": ["Office", "Office-Home", "OfficeHomeDataset_10072016"],
        "input_size": 256,
        "batch_size": 132,
        "num_workers": 4,
        "learning_rate": 0.001,
        "beta": (0.9,0.999),
        "number_epochs": 50,
    }
    
    for dataset_name in config_params["dataset_name"]:
        if dataset_name == "Office-Home":
            path_information =\
                f"/vinbrain/anhng/domain_adaptation/datasets/{dataset_name}/OfficeHomeDataset_10072016/information/"
        else:
            path_information =\
                f"/vinbrain/anhng/domain_adaptation/datasets/{dataset_name}/information/"

        for path_csv in glob.glob(path_information + "/*.csv"):
            domain_name = path_csv.split("/")[-1].replace("_kfold.csv", "")

            print(f"Running on {dataset_name} with domain: {domain_name} .... ")
            
            path_weight_save =\
                f"/workspace/domain_calibration/experiments/{dataset_name}/{domain_name}/"
            
            if dataset_name ==  "Office-Home":
                path_root = f"/vinbrain/anhng/domain_adaptation/datasets/{dataset_name}/{domain_name}/images/"
            else:
                path_root = f"/vinbrain/anhng/domain_adaptation/datasets/{dataset_name}/{domain_name}/"

            if not os.path.isdir(path_weight_save):
                os.makedirs(path_weight_save)

            data_transforms = {
                'train': A.Compose([
                    A.RandomResizedCrop(width=224, height=224),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)
                    )
                ]),
                'val': A.Compose([
                    A.Resize(width=224, height=224),
                    A.CenterCrop(width=224, height=224),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)
                    )
                ])
            }

            print("Initializing Datasets and Dataloaders...")
            df_info = pd.read_csv(path_csv)
            
            number_classes = len(set(df_info["classes"]))

            k_fold = 3
            for k in range(3):
                df_info_k_fold = df_info.copy()
                df_info_k_fold["phase"] = "train"
                df_info_k_fold.loc[df_info_k_fold.kfold==k, ['phase']] = 'val'
                image_datasets = {
                    "train": Dataset(
                        df_info=df_info_k_fold[df_info_k_fold.phase=="train"],
                        folder_data=path_root,
                        image_size=config_params["input_size"], 
                        transforms=data_transforms["train"]),
                    "val": Dataset(
                        df_info=df_info_k_fold[df_info_k_fold.phase=="val"], 
                        folder_data=path_root,
                        image_size=config_params["input_size"], 
                        transforms=data_transforms["val"]),

                }

                dataloaders_dict = {
                    x: torch.utils.data.DataLoader(
                        image_datasets[x], 
                        batch_size=config_params["batch_size"], 
                        shuffle=True, 
                        num_workers=config_params["num_workers"]) for x in ['train', 'val']}

                model_ft = initialize_model(
                    num_classes=number_classes,
                    feature_extract=True, 
                    use_pretrained=True
                )
                model_ft = model_ft.to(device)

                params_to_update = model_ft.parameters()
                print("Params to learn:")
                params_to_update = []
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t",name)

                optimizer_ft = optim.Adam(
                    params_to_update,
                    lr=config_params["learning_rate"],
                    betas=config_params["beta"],
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False
                )

                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                model_ft, hist = train_model(
                    model_ft, dataloaders_dict, 
                    criterion, optimizer_ft, 
                    num_epochs=config_params["number_epochs"],
                    folder_save=path_weight_save,
                    k_fold=k,
                    is_inception=False
                )
    