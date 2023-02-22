# Score SWAN manually

## Read prepared data

import pandas as pd
import os, datetime
import json

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

path = get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data/data/make_dataset/")
data = pd.read_csv(path+'item_lvl_w_impairment.csv')

swan_cols = [col for col in data.columns if col.startswith("SWAN")]
diag_cols = [col for col in data.columns if col.startswith("Diag.")]

data = data[swan_cols + diag_cols]
print(data.columns)
print(data) 

inattentive_cols = swan_cols[0:9] # First 9 cols: https://www.amerihealth.com/pdfs/providers/resources/worksheets/prevhealth_swan.pdf
print("Inattentive cols: ", inattentive_cols)
data["Inattentive_score"] = data[inattentive_cols].sum(axis=1)/len(inattentive_cols)
print(data[inattentive_cols + ["Inattentive_score"]])
print(data["Inattentive_score"].mean())

# Use one (arbitraty) cut-off value for diagnosis (ADHD Inattentive type)

# Print mean among non-diagnosed
print(data[data["Diag.ADHD-Inattentive Type"] == 0]["Inattentive_score"].mean())
# Print mean among diagnosed
print(data[data["Diag.ADHD-Inattentive Type"] == 1]["Inattentive_score"].mean())

# 0.53 for non-diagnosed, 1.14 for diagnosed => let's set cut-off value to 0.84

data["Inattentive_diagnosis"] = data["Inattentive_score"] >= 0.84

def calculate_sens_spec(data, manual_diag_col, data_diag_col):

    # Calculate confusion matrix values between manual diagnosis and diagnosis from data
    TP = data[(data[data_diag_col] == 1) & (data[manual_diag_col] == 1)].shape[0]
    TN = data[(data[data_diag_col] == 0) & (data[manual_diag_col] == 0)].shape[0]
    FP = data[(data[data_diag_col] == 0) & (data[manual_diag_col] == 1)].shape[0]
    FN = data[(data[data_diag_col] == 1) & (data[manual_diag_col] == 0)].shape[0]

    # Calculate sensititvity and specificity
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)

    return sensitivity, specificity

sensitivity, specificity = calculate_sens_spec(data, "Inattentive_diagnosis", "Diag.ADHD-Inattentive Type")
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)

# => On this cut-off, sensitivity is 0.63, specificity is 0.60. Using the model (only SWAN items as input), on the cut-off where sensitivity is 0.63, specificity is 0.74.

# Calculate AUC

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(data["Diag.ADHD-Inattentive Type"], data["Inattentive_score"])
print("AUC: ", auc)

# => AUC is 0.66, which is lower than the AUC of the model (only SWAN items as input) (0.79)
