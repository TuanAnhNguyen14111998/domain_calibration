from distutils.command import config
from matplotlib import transforms
import torch
import albumentations as A
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import pkbar
import glob
import yaml

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
from src.base_line.early_stopping import EarlyStopping


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_from_yaml(fname):
    with open(fname, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    return base_config

def get_agument(augment_name):
    list_augment = []
    for augment in augment_name:
        if augment == "CenterCrop":
            list_augment.append(
                A.CenterCrop(width=224, height=224),
            )
        if augment == "Flip":
            list_augment.apend(
                A.Flip(always_apply=False, p=1.0)
            )
        if augment == "Blur":
            list_augment.append(
                A.Blur(always_apply=False, p=0.5, blur_limit=(3, 7))
            )
        if augment == "GaussNoise":
            list_augment.append(
                A.GaussNoise(always_apply=False, p=0.5, var_limit=(10.0, 50.0))
            )
        if augment == "RandomBrightness":
            list_augment.append(
                A.RandomBrightness(
                    always_apply=False, p=0.5, 
                    limit=(-0.20000000298023224, 0.20000000298023224))
            )
        if augment == "Normalize":
            list_augment.append(
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            )

    return A.Compose(list_augment) 


def train_model(
    model, dataloaders, criterion, 
    optimizer, num_epochs=25, 
    folder_save="./experiments/", 
    k_fold=0,
    patience=10):
    
    path_save = folder_save + f"/resnet_34_kfold_{k_fold}.pth"
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_save)

    train_acc_history = []
    val_acc_history = []

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print('Training epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        
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
            
            if phase == 'val':
                if not os.path.isfile(folder_save + f"/resnet_34_kfold_{k_fold}_val_history.txt"):
                    f = open(folder_save + f"/resnet_34_kfold_{k_fold}_val_history.txt", "a")
                    f.write("epoch_loss, epoch_acc")
                    f.write("\n" + str(epoch_loss) + ", " + str(epoch_acc.item()))
                    f.close()
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
                    f = open(folder_save + f"/resnet_34_kfold_{k_fold}_val_history.txt", "a")
                    f.write("\n" + str(epoch_loss) + ", " + str(epoch_acc.item()))
                    f.close()

                early_stopping(epoch_acc, model)
                if early_stopping.early_stop:
                    print("Early stopping ... ")
                    break

            if phase == "train":
                if not os.path.isfile(folder_save + f"/resnet_34_kfold_{k_fold}_train_history.txt"):
                    f = open(folder_save + f"/resnet_34_kfold_{k_fold}_train_history.txt", "a")
                    f.write("epoch_loss, epoch_acc")
                    f.write("\n" + str(epoch_loss) + ", " + str(epoch_acc.item()))
                    f.close()
                else:
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                    f = open(folder_save + f"/resnet_34_kfold_{k_fold}_train_history.txt", "a")
                    f.write("\n" + str(epoch_loss) + ", " + str(epoch_acc.item()))
                    f.close()

        print()


if __name__ == "__main__":
    config_params = load_from_yaml("./configs/exp.yaml")
    for dataset_name in config_params["dataset_name"]:
        if dataset_name == "Office-Home":
            path_information =\
                f"{config_params['path_data']}/{dataset_name}/OfficeHomeDataset_10072016/information/"
        elif dataset_name == "Bing-Caltech":
            path_information =\
                f"{config_params['path_data']}/{dataset_name}/information/"
        else:
            path_information =\
                f"{config_params['path_data']}/{dataset_name}/information/"

        for path_csv in glob.glob(path_information + "/*.csv"):
            domain_name = path_csv.split("/")[-1].replace("_kfold.csv", "")
            if domain_name in config_params["domain_name"]:
                print(f"Running on {dataset_name} with domain: {domain_name} .... ")

                path_weight_save =\
                    f"/vinbrain/anhng/domain_adaptation/experiments/{dataset_name}/{domain_name}/"
                
                if dataset_name ==  "Office-Home":
                    path_root = f"{config_params['path_data']}/{dataset_name}/OfficeHomeDataset_10072016/{domain_name}/"
                elif dataset_name == "Bing-Caltech":
                    path_root = f"{config_params['path_data']}/{dataset_name}/BingLarge_C256_deduped/"
                elif dataset_name == "Domain-net":
                    path_root = f"{config_params['path_data']}/{dataset_name}/{domain_name}/"
                else:
                    path_root = f"{config_params['path_data']}/{dataset_name}/{domain_name}/images/"

                if not os.path.isdir(path_weight_save):
                    os.makedirs(path_weight_save)
                
                data_transforms = {
                    'train': get_agument(config_params["train_augment"]),
                    'val': get_agument(config_params["val_augment"])
                }

                print("Initializing Datasets and Dataloaders...")
                df_info = pd.read_csv(path_csv)

                number_classes = len(set(df_info["classes"]))

                k_fold = 3
                for k in range(3):
                    if k in config_params["kfold_exp"]:
                        df_info_k_fold = df_info.copy()
                        df_info_k_fold["phase"] = "train"
                        df_info_k_fold.loc[df_info_k_fold.kfold==k, ['phase']] = 'val'
                        image_datasets = {
                            "train": Dataset(
                                df_info=df_info_k_fold[df_info_k_fold.phase=="train"],
                                folder_data=path_root,
                                image_size_ratio=config_params["image_size_ratio"], 
                                input_size=config_params["input_size"], 
                                transforms=data_transforms["train"]),
                            "val": Dataset(
                                df_info=df_info_k_fold[df_info_k_fold.phase=="val"], 
                                folder_data=path_root,
                                image_size_ratio=config_params["image_size_ratio"], 
                                input_size=config_params["input_size"],
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
                            betas=(0.9,0.999),
                            eps=1e-08,
                            weight_decay=0,
                            amsgrad=False
                        )

                        criterion = nn.CrossEntropyLoss()
                        train_model(
                            model_ft, dataloaders_dict, 
                            criterion, optimizer_ft, 
                            num_epochs=config_params["number_epochs"],
                            folder_save=path_weight_save,
                            k_fold=k,
                            patience=config_params["patience"]
                        )
    