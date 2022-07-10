from threading import main_thread
import torch
import albumentations as A
import pandas as pd
from tqdm import tqdm
import glob
import numpy as np
import yaml
import json

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

from src.style_transfer.data_loader_styler_transfer import Dataset
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
    
    list_df = []
    if export_main_domain:
        phases = ['train', 'val']
    else:
        phases = ['train', 'val']
    
    for phase in phases:
        model.eval()
        for inputs, labels, records in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            df = pd.DataFrame(records)
            df["logit"] = outputs.cpu().detach().numpy().tolist()
            df["predict"] = preds.cpu().detach().numpy().tolist()
            list_df.append(df)
    
    return pd.concat(list_df)


def get_image_id_source(path_style_transfer, dataset_name, main_domain, kfold=0):
    df_mean_feature = pd.read_excel(
        f"{path_style_transfer}/{dataset_name}/{main_domain}/resnet_34_export_feature_vector.xlsx",
        sheet_name=f"kfold_{kfold}"
    )

    dictionary_dist = {}
    pdist = torch.nn.PairwiseDistance(p=2)

    for index, row in df_mean_feature.iterrows():
        mean_feature_vector = torch.tensor(json.loads(df_mean_feature.iloc[0]["mean_feature_vector"]))
        feature_vector = torch.tensor(json.loads(row["feature_vector"]))
        pdist = torch.nn.PairwiseDistance(p=2)
        dist = pdist(feature_vector.view(1, -1), mean_feature_vector.cpu().view(1, -1))
        dictionary_dist[row["imageid"]] = dist.cpu().numpy()[0]
    
    sorted_dictionary_dists = {k: str(v) for k, v in sorted(dictionary_dist.items(), key=lambda item: item[1])}
    image_id_closest = list(sorted_dictionary_dists.keys())[0]

    return image_id_closest


if __name__ == "__main__":
    config_params = load_from_yaml("./configs/config_style_transfer/exp_office.yaml")

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
                f"{config_params['path_weight']}/{dataset_name}/{main_domain}/"

            path_style_transfer =\
                f"{config_params['path_style_transfer']}/{dataset_name}/{main_domain}/"

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
            with pd.ExcelWriter(f'{path_style_transfer}/resnet_34_kfold_val_logits.xlsx') as writer:
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

                        image_id_source = get_image_id_source(
                            path_style_transfer, dataset_name, 
                            main_domain, kfold=k
                        )

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

                            if main_domain == domain_name:
                                export_main_domain = True
                            else:
                                export_main_domain = False
                            
                            image_datasets = {
                                "train": Dataset(
                                    df_info=df_info_k_fold[df_info_k_fold.phase=="train"],
                                    folder_data=path_root,
                                    image_size_ratio=config_params["image_size_ratio"], 
                                    input_size=config_params["input_size"], 
                                    transforms=data_transforms["train"],
                                    image_id_source=image_id_source,
                                    L=config_params["L"],
                                    export_main_domain=export_main_domain),
                                "val": Dataset(
                                    df_info=df_info_k_fold[df_info_k_fold.phase=="val"], 
                                    folder_data=path_root,
                                    image_size_ratio=config_params["image_size_ratio"], 
                                    input_size=config_params["input_size"],
                                    transforms=data_transforms["val"],
                                    image_id_source=image_id_source,
                                    L=config_params["L"],
                                    export_main_domain=export_main_domain),
                            }

                            dataloaders_dict = {
                                x: torch.utils.data.DataLoader(
                                    image_datasets[x],
                                    collate_fn=default_collate,
                                    batch_size=config_params["batch_size"], 
                                    shuffle=True, 
                                    num_workers=config_params["num_workers"]) for x in ['train', 'val']}
                            
                            df_finish_pred = export_logits(
                                model_ft, dataloaders_dict, 
                                export_main_domain=export_main_domain,
                            )

                            dataframe_logits.append(df_finish_pred)
                        
                        dataframe_logits = pd.concat(dataframe_logits)
                        dataframe_logits.to_excel(writer, sheet_name=f'kfold_{k}', index=False)
