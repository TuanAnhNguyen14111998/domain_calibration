import pandas as pd
import numpy as np
import yaml
import glob
import os
import json
import torch
import warnings
warnings.filterwarnings("ignore")


def load_from_yaml(fname):
    with open(fname, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    return base_config


def get_acc(train_text_paths, val_text_paths):
    training_loss = {
        'fold_0': [],
        'fold_1': [],
        'fold_2': []
    }
    training_acc = {
        'fold_0': [],
        'fold_1': [],
        'fold_2': []
    }

    for text in train_text_paths:
        k_fold = text.split("/")[-1].split("_")[3]
        f = open(text, "r")
        for index, x in enumerate(f):
            if index == 0:
                continue
            loss = float(x.replace("\n", "").split(", ")[0])
            acc = float(x.replace("\n", "").split(", ")[1])
            if k_fold == "0":
                training_loss["fold_0"].append(loss)
                training_acc["fold_0"].append(acc)
            elif k_fold == "1":
                training_loss["fold_1"].append(loss)
                training_acc["fold_1"].append(acc)
            else:
                training_loss["fold_2"].append(loss)
                training_acc["fold_2"].append(acc)

    val_loss = {
        'fold_0': [],
        'fold_1': [],
        'fold_2': []
    }

    val_acc = {
        'fold_0': [],
        'fold_1': [],
        'fold_2': []
    }

    for text in val_text_paths:
        k_fold = text.split("/")[-1].split("_")[3]
        f = open(text, "r")
        for index, x in enumerate(f):
            if index == 0:
                continue
            loss = float(x.replace("\n", "").split(", ")[0])
            acc = float(x.replace("\n", "").split(", ")[1])
            if k_fold == "0":
                val_loss["fold_0"].append(loss)
                val_acc["fold_0"].append(acc)
            elif k_fold == "1":
                val_loss["fold_1"].append(loss)
                val_acc["fold_1"].append(acc)
            else:
                val_loss["fold_2"].append(loss)
                val_acc["fold_2"].append(acc)
    
    
    df_result_training_loss = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in training_loss.items() ]))
    df_result_val_loss = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in val_loss.items() ]))
    
    df_result_training_acc = pd.DataFrame(pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in training_acc.items() ])))
    df_result_val_acc = pd.DataFrame(pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in val_acc.items() ])))
    
    return [df_result_training_loss, df_result_val_loss, df_result_training_acc, df_result_val_acc]


def report_result_acc(dictionary_metrics, path_save, train=True):
    path_folder = f"{path_save}/{dataset_name}"
    if not os.path.isdir(path_folder + "/result_visualize"):
        os.makedirs(path_folder + "/result_visualize")
    if train:
        index = 2
        path_save = f"{path_folder}/result_visualize/train_metrics.csv"
    else:
        index = 3
        path_save = f"{path_folder}/result_visualize/val_metrics.csv"

    dictionary = {
        "Domain Name": [],
        "Fold-1": [],
        "Fold-2": [],
        "Fold-3": [],
        "Mean Accuracy": [],
        "Standard Deviation": [],
    }
    for domain_name in domain_names:
        df = dictionary_metrics[domain_name][index]
        keys = ["fold_0", "fold_1", "fold_2"]
        optimize_values = []
        for key in keys:
            optimize_values.append(df[key].max())

        mean_value = np.mean(optimize_values)
        std_values = np.std(optimize_values)
        
        dictionary['Domain Name'].append(domain_name)
        dictionary['Mean Accuracy'].append(mean_value)
        dictionary['Standard Deviation'].append(std_values)
        dictionary['Fold-1'].append(optimize_values[0])
        dictionary['Fold-2'].append(optimize_values[1])
        dictionary['Fold-3'].append(optimize_values[2])
    
    pd.DataFrame(dictionary).to_csv(path_save, index=False)


def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
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


def compute_adaptation_calibration(true_labels, pred_labels, confidences, num_bins=10):
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


def compute_calibration_class_conditional(true_labels, confidences_all, num_bins=10):
    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece_all = []
    for i in range(confidences_all.shape[1]):
        confidences = confidences_all[:, i]
        pred_labels = np.array([i] * true_labels.shape[0])
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
        try:
            if ece > 0:
                ece_all.append(ece)
        except:
            pass
    return np.array(ece_all).sum() / len(ece_all)


def caculate_ece_not_calibrate(df, domain_names, type_calibration="ece"):
    ece_values = {}
    for index, key in enumerate(domain_names):
        condition = (df.domain_name == key) & (df.phase == "val")
        
        val_logits = np.vstack([json.loads(x) for x in df[condition]['logit'].values])
        val_ground_truth = df[condition]["classes"].values
        val_pred = df[condition]["predict"].values
        confidences_all = torch.softmax(torch.from_numpy(val_logits), 1).numpy()
        val_confidences, val_pred = torch.max(torch.softmax(torch.from_numpy(val_logits), 1), 1)
        
        n_bins = 10
        if type_calibration == "ece":
            uncalibrated_score = compute_calibration(
                np.array(val_ground_truth), 
                val_pred.detach().cpu().numpy(),
                val_confidences.detach().cpu().numpy(), 
                num_bins=n_bins
            )
        elif type_calibration == "adapt":
            uncalibrated_score = compute_adaptation_calibration(
                np.array(val_ground_truth), 
                val_pred.detach().cpu().numpy(),
                val_confidences.detach().cpu().numpy(), 
                num_bins=n_bins
            )
        else:
            uncalibrated_score = compute_calibration_class_conditional(
                val_ground_truth,
                confidences_all, 
                num_bins=n_bins
            )

        ece_values[key] = uncalibrated_score
        
    return ece_values


def caculate_ece_calibrate(df, domain_names, T, type_calibration="ece"):
    ece_values = {}
    for index, key in enumerate(domain_names):
        condition = (df.domain_name == key) & (df.phase == "val")
        
        val_logits = np.vstack([json.loads(x) for x in df[condition]['logit'].values])
        val_ground_truth = df[condition]["classes"].values
        if type(T) == list:
            confidences_all = torch.softmax(torch.from_numpy(val_logits)*T[0] + T[1], 1).numpy()
            val_confidences, val_pred = torch.max(
                torch.softmax(torch.from_numpy(val_logits)*T[0] + T[1], 1), 1
            )
        else:
            confidences_all = torch.softmax(torch.from_numpy(val_logits)/T, 1).numpy()
            val_confidences, val_pred = torch.max(torch.softmax(torch.from_numpy(val_logits)/T, 1), 1)
        
        n_bins = 10
        if type_calibration == "ece":
            calibrated_score = compute_calibration(
                np.array(val_ground_truth), 
                val_pred.detach().cpu().numpy(), 
                val_confidences.detach().cpu().numpy(), 
                num_bins=n_bins
            )
        elif type_calibration == "adapt":
            calibrated_score = compute_adaptation_calibration(
                np.array(val_ground_truth), 
                val_pred.detach().cpu().numpy(), 
                val_confidences.detach().cpu().numpy(), 
                num_bins=n_bins
            )
        else:
            calibrated_score = compute_calibration_class_conditional(
                val_ground_truth,
                confidences_all, 
                num_bins=n_bins
            )

        ece_values[key] = calibrated_score
        
    return ece_values


def report_ece_not_calibrate(type_eces, path_save, kfold, domain_names):
    for type_ece in type_eces:
        with pd.ExcelWriter(f'{path_save}/{type_ece}_not_calibrate_results_logits.xlsx') as writer:
            for i in range(3):
                if i in kfold:
                    dictionary = {
                        "Domain Name": [],
                    }

                    for domain_name in domain_names:
                        dictionary[domain_name] = []

                    for domain_name in domain_names:
                        path_excel = f"{config_params['path_save']}/{dataset_name}/{domain_name}/resnet_34_kfold_val_logits.xlsx"
                        dictionary['Domain Name'].append(domain_name)
                        df = pd.read_excel(path_excel, sheet_name=f"kfold_{i}")
                        ece_values = caculate_ece_not_calibrate(df, domain_names, type_calibration=type_ece)

                        for name in domain_names:
                            dictionary[name].append(ece_values[name])

                    pd.DataFrame(dictionary).to_excel(writer, sheet_name=f'kfold_{i}', index=False)


def report_ece_calibrate_temperature(type_eces, type_loss, path_save, kfold, domain_names):
    for type_ece in type_eces:
        with pd.ExcelWriter(f'{path_save}/{type_ece}_{type_loss}_calibrate_temperature_results_logits.xlsx') as writer:
            for i in range(3):
                if i in kfold:
                    dictionary = {
                        "Domain Name": [],
                    }

                    for domain_name in domain_names:
                        dictionary[domain_name] = []

                    for domain_name in domain_names:
                        path_excel = f"{config_params['path_save']}/{dataset_name}/{domain_name}/resnet_34_kfold_val_logits.xlsx"
                        dictionary['Domain Name'].append(domain_name)
                        df = pd.read_excel(path_excel, sheet_name=f"kfold_{i}")
                        file = open(f"{config_params['path_save']}/{dataset_name}/{domain_name}/{type_loss}_origin_kfold_{i}_temperature_checkpoint.txt")
                        T = float(file.read())
                        ece_values = caculate_ece_calibrate(df, domain_names, T, type_calibration=type_ece)

                        for name in domain_names:
                            dictionary[name].append(ece_values[name])
                    
                    pd.DataFrame(dictionary).to_excel(writer, sheet_name=f'kfold_{i}', index=False)


def report_ece_calibrate_platt(type_eces, type_loss, path_save, kfold, domain_names):
    for type_ece in type_eces:
        with pd.ExcelWriter(f'{path_save}/{type_ece}_{type_loss}_calibrate_platscaling_results_logits.xlsx') as writer:
            for i in range(3):
                if i in kfold:
                    dictionary = {
                        "Domain Name": [],
                    }

                    for domain_name in domain_names:
                        dictionary[domain_name] = []

                    for domain_name in domain_names:
                        path_excel = f"{config_params['path_save']}/{dataset_name}/{domain_name}/resnet_34_kfold_val_logits.xlsx"
                        dictionary['Domain Name'].append(domain_name)
                        df = pd.read_excel(path_excel, sheet_name=f"kfold_{i}")
                        file = open(f"{config_params['path_save']}/{dataset_name}/{domain_name}/{type_loss}_origin_kfold_{i}_platscaling_checkpoint.txt")
                        list_weight = json.loads(file.read())
                        ece_values = caculate_ece_calibrate(
                            df, domain_names, list_weight, 
                            type_calibration=type_ece
                        )

                        for name in domain_names:
                            dictionary[name].append(ece_values[name])
                    
                    pd.DataFrame(dictionary).to_excel(writer, sheet_name=f'kfold_{i}', index=False)


def report_ece(config_params):
    dataset_name = config_params["dataset_name"]
    domain_names = config_params["domain_names"]
    kfold = config_params["k_fold"]

    path_save = f"{config_params['path_save']}/{dataset_name}/result_visualize/"

    if not os.path.isdir(path_save):
        os.mkdir(path_save)
        
    type_eces = ["ece", "adapt", "cce"]
    report_ece_not_calibrate(type_eces, path_save, kfold, domain_names)
    
    type_losses = ["entropy", "both"]
    for type_loss in type_losses:
        report_ece_calibrate_temperature(type_eces, type_loss, path_save, kfold, domain_names)
    
    type_losses = ["entropy", "both"]
    for type_loss in type_losses:
        report_ece_calibrate_platt(type_eces, type_loss, path_save, kfold, domain_names)
    

if __name__ == "__main__":

    config_params = load_from_yaml("./configs/config_metrics/config_office_home.yaml")

    dataset_name = config_params["dataset_name"]
    domain_names = config_params["domain_names"]
    dictionary_metrics = {domain_name: None for domain_name in domain_names}

    for domain in dictionary_metrics:
        train_paths = f"{config_params['path_save']}/{dataset_name}/{domain}/*_train_history.txt"
        train_text_paths = glob.glob(train_paths)

        val_paths = f"{config_params['path_save']}/{dataset_name}/{domain}/*_val_history.txt"
        val_text_paths = glob.glob(val_paths)
        
        dictionary_metrics[domain] = get_acc(train_text_paths, val_text_paths)

    # report accuracy
    report_result_acc(dictionary_metrics, config_params['path_save'], train=True)
    report_result_acc(dictionary_metrics, config_params['path_save'], train=False)

    # report ece
    report_ece(config_params)