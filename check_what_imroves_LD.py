import pandas as pd

from helpers import get_newest_non_empty_dir_in_dir

def get_pos_examples_dir(path):
    import os, datetime
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # Find lowest timestamp (make_datasets script runs before eval_orig script)
    timestamps = [d.split("___")[0] for d in dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    oldest_dir_name = dir_names[timestamps.index(min(timestamps))]
    return path + oldest_dir_name + "/"

dir = "../learning_diagnosis_predictor_data/check_what_improves_LD_data/"

# Read scores
nothing_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "Nothing/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
nih_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NIH/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
conners_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "Conners/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
newdiag_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NewDiag/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
newdiag_nih_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NewDiag+NIH/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
newdiag_conners_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NewDiag+Conners/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
newdiag_conners_nih_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NewDiag+Conners+NIH/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]

# Read # of positive examples
nothing_pos = pd.read_csv(get_pos_examples_dir(dir + "Nothing/") + "number-of-positive-examples.csv", index_col=1)
nothing_pos = nothing_pos[[nothing_pos.columns[-1]]]
nothing_ds_size = pd.read_csv(get_pos_examples_dir(dir + "Nothing/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

nih_pos = pd.read_csv(get_pos_examples_dir(dir + "NIH/") + "number-of-positive-examples.csv", index_col=1)
nih_pos = nih_pos[[nih_pos.columns[-1]]]
nih_ds_size = pd.read_csv(get_pos_examples_dir(dir + "NIH/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

conners_pos = pd.read_csv(get_pos_examples_dir(dir + "Conners/") + "number-of-positive-examples.csv", index_col=1)
conners_pos = conners_pos[[conners_pos.columns[-1]]]
conners_ds_size = pd.read_csv(get_pos_examples_dir(dir + "Conners/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

newdiag_pos = pd.read_csv(get_pos_examples_dir(dir + "NewDiag/") + "number-of-positive-examples.csv", index_col=1)
newdiag_pos = newdiag_pos[[newdiag_pos.columns[-1]]]
newdiag_ds_size = pd.read_csv(get_pos_examples_dir(dir + "NewDiag/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

newdiag_nih_pos = pd.read_csv(get_pos_examples_dir(dir + "NewDiag+NIH/") + "number-of-positive-examples.csv", index_col=1)
newdiag_nih_pos = newdiag_nih_pos[[newdiag_nih_pos.columns[-1]]]
newdiag_nih_ds_size = pd.read_csv(get_pos_examples_dir(dir + "NewDiag+NIH/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

newdiag_conners_pos = pd.read_csv(get_pos_examples_dir(dir + "NewDiag+Conners/") + "number-of-positive-examples.csv", index_col=1)
newdiag_conners_pos = newdiag_conners_pos[[newdiag_conners_pos.columns[-1]]]
newdiag_conners_ds_size = pd.read_csv(get_pos_examples_dir(dir + "NewDiag+Conners/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

newdiag_conners_nih_pos = pd.read_csv(get_pos_examples_dir(dir + "NewDiag+Conners+NIH/") + "number-of-positive-examples.csv", index_col=1)
newdiag_conners_nih_pos = newdiag_conners_nih_pos[[newdiag_conners_nih_pos.columns[-1]]]
newdiag_conners_nih_ds_size = pd.read_csv(get_pos_examples_dir(dir + "NewDiag+Conners+NIH/") + "dataset_stats.csv", index_col=0).loc["n_rows_full_ds"].values[0]

# Add # of positive examples to scores
nothing_scores["# of positive examples"] = nothing_pos
nothing_scores["Total examples"] = nothing_ds_size
nih_scores["# of positive examples"] = nih_pos
nih_scores["Total examples"] = nih_ds_size
conners_scores["# of positive examples"] = conners_pos
conners_scores["Total examples"] = conners_ds_size
newdiag_scores["# of positive examples"] = newdiag_pos
newdiag_scores["Total examples"] = newdiag_ds_size
newdiag_nih_scores["# of positive examples"] = newdiag_nih_pos
newdiag_nih_scores["Total examples"] = newdiag_nih_ds_size
newdiag_conners_scores["# of positive examples"] = newdiag_conners_pos
newdiag_conners_scores["Total examples"] = newdiag_conners_ds_size
newdiag_conners_nih_scores["# of positive examples"] = newdiag_conners_nih_pos
newdiag_conners_nih_scores["Total examples"] = newdiag_conners_nih_ds_size

# Add suffixes for merging
nothing_scores = nothing_scores.add_suffix("_nothing")
nih_scores = nih_scores.add_suffix("_nih")
conners_scores = conners_scores.add_suffix("_conners")
newdiag_scores = newdiag_scores.add_suffix("_newdiag")
newdiag_nih_scores = newdiag_nih_scores.add_suffix("_newdiag_nih")
newdiag_conners_scores = newdiag_conners_scores.add_suffix("_newdiag_conners")
newdiag_conners_nih_scores = newdiag_conners_nih_scores.add_suffix("_newdiag_conners_nih")

# Merge into one table on Diag
scores = pd.concat([nothing_scores, nih_scores, conners_scores, newdiag_scores, newdiag_nih_scores, newdiag_conners_scores, newdiag_conners_nih_scores], axis=1, join="outer")

print(scores.columns)

# Make all columns except those that start with ROC AUC into ints (if not NA) (bug in pd)
for col in scores.columns:
    if not col.startswith("ROC AUC"):
        scores[col] = scores[col].astype('Int64')

scores.to_csv("output/what_improves_LD.csv")