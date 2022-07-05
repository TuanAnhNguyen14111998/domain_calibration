from hashlib import sha1
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import yaml
import glob
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_from_yaml(fname):
    with open(fname, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    return base_config

class TempScaleParams(nn.Module):
    def __init__(self):
        super(TempScaleParams, self).__init__()
        self.temperature = nn.Parameter(torch.tensor([1.5]))
    
    def forward(self, logit):
        return logit / self.temperature


class _ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class CaliberatedLoss(nn.Module):
    def __init__(self, type_loss = 'entropy'):
        super(CaliberatedLoss, self).__init__()
        self.prob = nn.CrossEntropyLoss()
        self.ece = _ECELoss()
        self.type = type_loss

    def forward(self, logits, labels):
        if self.type == 'entropy':
            return self.prob(logits, labels)
        elif self.type == 'both':
            return self.prob(logits, labels) + self.ece(logits, labels)
        else:
            return self.ece(logits, labels)


def compute_calibration_origin(true_labels, pred_labels, confidences, num_bins=10):
    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    return ece


def compute_calibration_adaptive(true_labels, pred_labels, confidences, num_bins=10):
    bin_size = 1.0 / num_bins
    _, bins = pd.qcut(confidences, q = 10, retbins = True)
    indices = np.digitize(confidences, bins, right=True)
    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    return ece


def train_temp_scaling(df, type_loss = 'entropy',
                        type_ece = "origin",
                        path_save = './experiments/',
                        k_fold=0):
    
    full_train_logit = np.vstack([json.loads(x) for x in df[df.phase == 'train']['logit'].values])
    full_train_label = df[df.phase == 'train']['classes'].tolist()
    full_valid_logit = np.vstack([json.loads(x) for x in df[df.phase == 'val']['logit'].values])
    full_valid_label = df[df.phase == 'val']['classes'].tolist()
    logit_train = torch.from_numpy(full_train_logit).to(device)
    label_train = torch.tensor(full_train_label).squeeze().to(device)
    logit_valid = torch.from_numpy(full_valid_logit).to(device)
    label_valid = torch.tensor(full_valid_label).squeeze().to(device)
    losses = []
    eces = []
    Ts = []
    Net_ = TempScaleParams().to(device)
    criterion = CaliberatedLoss(type_loss)
    optimizer = optim.Adam(Net_.parameters(), lr=3e-3)
    best_ece = 3000
    best_T = 0
    for epoch in range(200):
        Ts.append(Net_.temperature.item())
        Net_.train()
        caliberated_logit = Net_(logit_train)
        Loss = criterion(caliberated_logit.to(device), label_train.to(device))
        Loss.backward()
        optimizer.step()
        losses.append(Loss)
        Net_.eval()
        confidences, pred = torch.max(torch.softmax(logit_valid / Net_.temperature , 1), 1)
        if type_ece == "origin":
            ece = compute_calibration_origin(
                label_valid.detach().cpu().numpy(), 
                pred.detach().cpu().numpy(), 
                confidences.detach().cpu().numpy()
            )
        else:
            ece = compute_calibration_adaptive(
                label_valid.detach().cpu().numpy(), 
                pred.detach().cpu().numpy(), 
                confidences.detach().cpu().numpy()
            )
        eces.append(ece)
        if ece < best_ece:
            best_ece = ece
            best_T = Net_.temperature.item()
    
    # write Best T
    f = open(f'{path_save}' + f"/{type_loss}_{type_ece}_kfold_{k_fold}_temperature_checkpoint.txt", "w")
    f.write(str(best_T))
    f = open(f'{path_save}' + f"/{type_loss}_{type_ece}_kfold_{k_fold}_valid.txt", "w")
    f.write(str(best_ece))
    f.close()

    figure = plt.gcf() 
    figure.set_size_inches(8, 6)
    fig, ax = plt.subplots(1,3, figsize = (8, 4))
    ax[0].plot(losses)
    ax[0].set_title('train loss')
    ax[1].plot(eces)
    ax[1].set_title('valid ece')
    ax[2].plot(Ts)
    ax[2].set_title('T update')
    plt.savefig(f'{path_save}' + f"/monitor_{type_loss}_kfold_{k_fold}.png")


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
            path_save =\
                f"{config_params['path_save']}/{dataset_name}/{main_domain}/"
            
            if not os.path.isdir(path_save):
                os.makedirs(path_save)

            k_fold = 3
            for k in range(3):
                if k in config_params["kfold_exp"]:
                    path_excel = path_save + "resnet_34_kfold_val_logits.xlsx"
                    df = pd.read_excel(path_excel, sheet_name=f"kfold_{k}")
                    df_main_domain = df[df.domain_name == main_domain]
                    train_temp_scaling(
                        df=df_main_domain,
                        type_loss="entropy",
                        type_ece="origin",
                        path_save=path_save,
                        k_fold=k
                    )
