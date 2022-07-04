from threading import main_thread
import torch
import albumentations as A
import pandas as pd
from tqdm import tqdm
import time
import copy
import glob
import numpy as np
import yaml

import os
import sys
path = os.path.dirname(__file__)
root_folder = os.path.join(
    os.path.abspath(path).split("domain_calibration")[0],
    "domain_calibration"
)
sys.path.insert(0, root_folder)

import warnings
warnings.filterwarnings("ignore")

from src.base_line.data_loader_export_logits import Dataset
from src.base_line.restnet34 import initialize_model

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
            list_augment.append(
                A.Flip(always_apply=False, p=0.5)
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

def default_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    images = torch.stack([torch.tensor(item[0]) for item in batch])
    labels = torch.stack([torch.tensor(item[1]) for item in batch])
    records = [item[2] for item in batch]
    
    return images, labels, records


def export_logits(
    model, dataloaders, 
    export_main_domain=True):
    
    df_finish_pred = pd.DataFrame()
    if export_main_domain:
        phases = ['train', 'val']
    else:
        phases = ['val']
    
    for phase in phases:
        model.eval()
        for inputs, labels, records in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for index, pred in enumerate(preds):
                records[index]["logit"] = list(outputs[index].cpu().detach().numpy())
                records[index]["predict"] = int(pred.cpu().detach().numpy())
                df_finish_pred = df_finish_pred.append(records[index])

    return df_finish_pred


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

        all_csv_domain = glob.glob(path_information + "/*.csv")
        for path_csv in all_csv_domain:
            main_domain = path_csv.split("/")[-1].replace("_kfold.csv", "")

            print(f"Running on {dataset_name} with domain: {main_domain} .... ")
            path_weight_save =\
                f"/vinbrain/anhng/domain_adaptation/experiments/{dataset_name}/{main_domain}/"

            if not os.path.isdir(path_weight_save):
                os.makedirs(path_weight_save)

            data_transforms = {
                'train': get_agument(config_params["train_augment"]),
                'val': get_agument(config_params["val_augment"])
            }

            print("Initializing Datasets and Dataloaders...")
            informations = [
                {
                    "dataframe": pd.read_csv(path_csv),
                    "domain_name": path_csv.split("/")[-1].replace("_kfold.csv", "")
                }
                for path_csv in all_csv_domain
            ]

            number_classes = len(set(informations[0]["dataframe"]["classes"]))

            k_fold = 3
            with pd.ExcelWriter(f'{path_weight_save}/resnet_34_kfold_val_logits.xlsx') as writer:
                for k in range(3):
                    if k in config_params["kfold_exp"]:
                        model_ft = initialize_model(
                            num_classes=number_classes,
                            feature_extract=True, 
                            use_pretrained=True
                        )

                        path_weight = path_weight_save + f"resnet_34_kfold_{str(k)}.pth"
                        
                        model_ft.load_state_dict(
                            torch.load(
                                path_weight,
                                map_location=torch.device(device=device)
                            )
                        )
                        model_ft.eval()
                        model_ft = model_ft.to(device)

                        dataframe_logits = []
                        for information in informations:
                            domain_name = information["domain_name"]
                            if dataset_name ==  "Office-Home":
                                path_root = f"{config_params['path_data']}/{dataset_name}/OfficeHomeDataset_10072016/{domain_name}/"
                            elif dataset_name == "Bing-Caltech":
                                path_root = f"{config_params['path_data']}/{dataset_name}/BingLarge_C256_deduped/"
                            elif dataset_name == "Domain-net":
                                path_root = f"{config_params['path_data']}/{dataset_name}/{domain_name}/"
                            else:
                                path_root = f"{config_params['path_data']}/{dataset_name}/{domain_name}/images/"
                            
                            df_info_k_fold = information["dataframe"].copy()
                            df_info_k_fold["domain_name"] = domain_name
                            df_info_k_fold["logit"] = np.nan
                            df_info_k_fold["predict"] = np.nan
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
                                    collate_fn=default_collate,
                                    batch_size=config_params["batch_size"], 
                                    shuffle=True, 
                                    num_workers=config_params["num_workers"]) for x in ['train', 'val']}
                            
                            if main_domain == domain_name:
                                export_main_domain = True
                            else:
                                export_main_domain = False
                            
                            df_finish_pred = export_logits(
                                model_ft, dataloaders_dict, 
                                export_main_domain=export_main_domain,
                            )

                            dataframe_logits.append(df_finish_pred)
                        
                        dataframe_logits = pd.concat(dataframe_logits)
                        dataframe_logits.to_excel(writer, sheet_name=f'kfold_{k}')
