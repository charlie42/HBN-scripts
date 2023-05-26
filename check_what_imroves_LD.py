import pandas as pd

from helpers import get_newest_non_empty_dir_in_dir

dir = "input/check_what_improves_LD_data/"

nothing_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "Nothing/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
nih_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NIH/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
conners_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "Conners/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]
newdiag_scores = pd.read_csv(get_newest_non_empty_dir_in_dir(dir + "NewDiag/", ["use_test_set__1"]) + "performance_table_all_features.csv", index_col=0)[["ROC AUC Mean CV"]]

# Merge into one table
scores = nothing_scores.merge(nih_scores, suffixes=("_nothing", "_nih"), how="outer", on="Diag")
scores = scores.merge(conners_scores, suffixes=("_nih", "_conners"), how="outer", on="Diag")
scores = scores.merge(newdiag_scores, suffixes=("_conners", "_newdiag"), how="outer", on="Diag")

scores.to_csv("output/what_improves_LD.csv")

# Plot stacked bar with improvement for each row (diag)


