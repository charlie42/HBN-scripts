# Score SWAN manually

## Read prepared data

import pandas as pd
import os, datetime

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

path = get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data/data/make_dataset/")
all_data = pd.read_csv(path+'item_lvl_w_impairment.csv')

swan_cols = [col for col in all_data.columns if col.startswith("SWAN")]
srs_cols = [col for col in all_data.columns if col.startswith("SRS")]
diag_cols = [col for col in all_data.columns if col.startswith("Diag.")]




### Score SWAN manually for ADHD-Inattentive Type
print("Diag.ADHD-Inattentive Type")

data = all_data[swan_cols + diag_cols].copy()

inattentive_cols = swan_cols[0:9] # First 9 cols: https://www.aacap.org/App_Themes/AACAP/docs/member_resources/toolbox_for_clinical_practice_and_outcomes/symptoms/SWAN_Attention-DeficitHyperactivity_Disorder_Symptoms_Child.pdf 

data["Inattentive_score"] = data[inattentive_cols].sum(axis=1)/len(inattentive_cols)

# Print mean score 
print(data["Inattentive_score"].mean())
# Print mean among non-diagnosed
print(data[data["Diag.ADHD-Inattentive Type"] == 0]["Inattentive_score"].mean())
# Print mean among diagnosed
print(data[data["Diag.ADHD-Inattentive Type"] == 1]["Inattentive_score"].mean())

# Calculate AUC

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(data["Diag.ADHD-Inattentive Type"], data["Inattentive_score"])
print("AUC: ", auc)

# => AUC is 0.66, which is lower than the AUC of the model (only SWAN items as input) (0.79)



### Score SWAN manually for ADHD-Combined Type
print("Diag.ADHD-Combined Type")

data["Combined_score"] = data[swan_cols].sum(axis=1) # https://www.academia.edu/4479470/Across_the_continuum_of_attention_skills_A_twin_study_of_the_SWAN_ADHD_rating_scale 

# Print mean score
print(data["Combined_score"].mean())
# Print mean among non-diagnosed
print(data[data["Diag.ADHD-Combined Type"] == 0]["Combined_score"].mean())
# Print mean among diagnosed
print(data[data["Diag.ADHD-Combined Type"] == 1]["Combined_score"].mean())

# Calculate AUC

auc = roc_auc_score(data["Diag.ADHD-Combined Type"], data["Combined_score"])
print("AUC: ", auc)

# => AUC is 0.82, which is lower than the AUC of the model (only SWAN items as input) (0.85)



### Score SRS manually for ASD
print("Diag.Autism Spectrum Disorder")

data = all_data[srs_cols + diag_cols].copy()
data["SRS_score"] = data[srs_cols].sum(axis=1) # https://journals.sagepub.com/doi/abs/10.1177/0734282913517525?journalCode=jpaa 

# Print mean score
print(data["SRS_score"].mean())
# Print mean among non-diagnosed
print(data[data["Diag.Autism Spectrum Disorder"] == 0]["SRS_score"].mean())
# Print mean among diagnosed
print(data[data["Diag.Autism Spectrum Disorder"] == 1]["SRS_score"].mean())

# Calculate AUC

auc = roc_auc_score(data["Diag.Autism Spectrum Disorder"], data["SRS_score"])
print("AUC: ", auc)

# => AUC is 0.84, which is lower than the AUC of the model (only SRS items as input) (0.89)