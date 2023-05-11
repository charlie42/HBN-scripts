# HBN-scripts

Scripts to format output of the `diagnosis_predictor` for the paper

- `count_transdiag_items.py` calculates for how many diagnoses each item appears in each subset. Rows: item, colsumns: subset, values: count
- `all_vs_opt_features_auc_test_set.py` prints AUCROC on test set on all features and on the optimal number of features for each diagnosis
- `manual_scoring_analysis/score_manually.py` calculates AUROC of each subscale and total score from HBN for each diagnosis, and prints it alongside AUROC from my model on the number of features equal to the number of items in the best performing subscale (e.g. if CBCL_Ext_T has the highest AUROC for ODD, and it contains 10 items, I print AUROC of my model at 10 items). I also print the number of items my model needs to reach the performance of the best subscale. `score_manually_total.py` only uses total scores.
- `misc` folder contains other useful scripts that don't generate pretty tables