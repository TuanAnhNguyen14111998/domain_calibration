import pandas as pd
import numpy as np
import yaml
import glob
import os

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
    
    
    df_result_training_loss = pd.DataFrame(training_loss)
    df_result_val_loss = pd.DataFrame(val_loss)
    
    df_result_training_acc = pd.DataFrame(training_acc)
    df_result_val_acc = pd.DataFrame(val_acc)
    
    return [df_result_training_loss, df_result_val_loss, df_result_training_acc, df_result_val_acc]


def report_result_acc(dictionary_metrics, path_save, train=True):
    if train:
        index = 2
        path_save = f"{path_save}/{dataset_name}/result_visualize/train_metrics.csv"
    else:
        index = 3
        path_save = f"{path_save}/{dataset_name}/result_visualize/val_metrics.csv"
    
    if not os.path.isdir(path_save):
        os.makedirs(path_save)

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
        dictionary['Fold-3'].append(optimize_values[1])
    
    pd.DataFrame(dictionary).to_csv(path_save, index=False)


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


