from helpers import get_newest_non_empty_dir_in_dir, make_dir_if_not_exists

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from make_final_tables import read_data_eval_subsets, read_data_eval_subsets_learning

diagnosis_dict = {
        'Diag.Major Depressive Disorder': 'MDD',
        'Diag.Autism Spectrum Disorder': 'ASD',
        'Diag.Enuresis': 'Enuresis',
        'Diag.ADHD-Combined Type': 'ADHD-C',
        'Diag.Social Anxiety (Social Phobia)': 'SAD',
        'Diag.Generalized Anxiety Disorder': 'GAD',
        'Diag.Oppositional Defiant Disorder': 'ODD',
        'Diag.Any Diag': 'Any',
        'Diag.No Diagnosis Given': 'None',
        'Diag.Separation Anxiety': 'SA',
        'Diag.ADHD-Inattentive Type': 'ADHD-I',
        'Diag.Specific Learning Disorder with Impairment in Mathematics': 'SLD-Math',
        'New Diag.Specific Learning Disorder with Impairment in Mathematics': 'SLD-Math',
        'New Diag.Specific Learning Disorder with Impairment in Mathematics - Consensus': 'SLD-Math-Consensus',
        'Diag.Language Disorder': 'Language',
        'Diag.Specific Phobia': 'Phobia',
        'Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'New Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'New Diag.Specific Learning Disorder with Impairment in Reading - Consensus': 'SLD-Reading-Consensus',
        'Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'New Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'New Diag.Specific Learning Disorder with Impairment in Written Expression - Consensus': 'SLD-Writing-Consensus',
        'Diag.Other Specified Anxiety Disorder': 'Other Anxiety',
        'New Diag.Processing Speed Deficit': 'LD-PS',
        'New Diag.Borderline Intellectual Functioning': 'LD-BIF',
    }

def make_df_to_plot_thresholds(dir_eval_subsets_all, filename):

    df_opt_features_all = pd.read_csv(dir_eval_subsets_all + "sens-spec-on-subsets-test-set-optimal-nb-features/" + filename, 
                                      index_col=0)

    return df_opt_features_all

def plot_eval_orig(eval_orig_df):
    # Plot on same plot, one point for each diagnosis, connext diagnoses with lines
    fig = plt.figure(figsize=(10, 5))
    plt.title("AUROC on all features")
    plt.plot(eval_orig_df["AUC all features all assessments"], label="AUROC on test set all assessments", marker="o", linestyle="", color="lightblue")
    plt.plot(eval_orig_df["AUC CV all features all assessments"], label="AUROC CV all assessments", marker="o", linestyle="", color="blue")
    plt.plot(eval_orig_df["AUC all features free assessments"], label="AUROC on test set free assessments", marker="o", linestyle="", color="lightblue", markerfacecolor='none')
    plt.plot(eval_orig_df["AUC CV all features free assessments"], label="AUROC CV free assessments", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.plot(eval_orig_df["AUC all features healthy controls all assessments"], label="AUROC on test set healthy controls all assessments", marker="o", linestyle="", color="lightcoral")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylim([0.5, 1.0])
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # For each diag, also plot bar plots with the number of positive examples for all assessments and for free assessments
    ax2 = plt.twinx()
    ax2.set_ylabel("Number of positive cases")
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples free assessments"].values, label="# of positive cases free assessments", color="blue", width=0.5, alpha=0.2, fill=False)
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples all assessments"].values, label="# of positive cases all assessments", color="blue", width=0.5, alpha=0.2)
    tot_n_features_all_assessments = eval_orig_df["# rows full dataset all assessments"].values[0]
    tot_n_features_free_assessments = eval_orig_df["# rows full dataset free assessments"].values[0]
    ax2.legend([f"# of positive cases free assessments/{tot_n_features_free_assessments}", f"# of positive cases all assessments/{tot_n_features_all_assessments}"], loc="lower right")
    
    plt.tight_layout()

    return fig

def plot_eval_orig_for_prez(eval_orig_df):
    eval_orig_df = eval_orig_df.sort_values(by="AUC all features all assessments", ascending=False)
    # Plot on same plot, one point for each diagnosis, connext diagnoses with lines
    fig_all = plt.figure(figsize=(10, 5))
    plt.title("AUROC on all features")
    plt.plot(eval_orig_df["AUC all features all assessments"], label="AUROC on test set", marker="o", linestyle="", color="blue")
    plt.plot(eval_orig_df["AUC all features healthy controls all assessments"], label="AUROC on test set healthy controls", marker="o", linestyle="", color="lightcoral")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylim([0.5, 1.0])
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # For each diag, also plot bar plots with the number of positive examples for all assessments and for free assessments
    ax2 = plt.twinx()
    ax2.set_ylabel("Number of positive cases")
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples all assessments"].values, label="# of positive cases all assessments", color="blue", width=0.5, alpha=0.1)
    tot_n_features_all_assessments = eval_orig_df["# rows full dataset all assessments"].values[0]
    ax2.legend([f"# of positive cases all assessments/{tot_n_features_all_assessments}"], loc="lower right")
    
    plt.tight_layout()

    fig_all_vs_free = plt.figure(figsize=(10, 5))
    plt.title("AUROC on all features")
    plt.plot(eval_orig_df["AUC all features all assessments"], label="AUROC on test set all assessments", marker="o", linestyle="", color="blue")
    plt.plot(eval_orig_df["AUC all features free assessments"], label="AUROC on test set free assessments", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylim([0.5, 1.0])
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    return fig_all, fig_all_vs_free

def plot_manual_vs_ml(eval_subsets_df):
    eval_subsets_df = eval_subsets_df.sort_values(by="Best subscale score", ascending=False)
    
    fig = plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of features")
    plt.plot(eval_subsets_df["Best subscale score"], label="AUROC of best subscale (+ # of items to reach it with ML)", marker="o", linestyle="", color="red")
    plt.plot(eval_subsets_df["ML score at # of items of best subscale (all assessments)"], label="AUROC of ML model on # of items in best subscale (+ # of items in best subscale) (all assessments)", marker="o", linestyle="", color="blue")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Print number of items next to AUC scores to the right of the markers
    for i, row in eval_subsets_df.iterrows():
        #plt.text(i, row["AUC all assessments"]+0.01, str(row["Optimal # of features all assessments"]), ha="left", va="center", size=8)
        plt.text(i, row["ML score at # of items of best subscale (all assessments)"]+0.01, str(row["# of items in best subscale"]), ha="center", va="bottom", size=8, color="blue")
        plt.text(i, row["Best subscale score"]-0.01, str(row["# of items to reach best subscale (all assessments)"]), ha="center", va="top", size=8, color="blue")
    
    # Append best subscale name to the diag name on x axis
    eval_subsets_df["Best subscale"] = eval_subsets_df["Best subscale"].str.split(",").str[1] # Remove prefix before , from subscale names
    plt.xticks(range(len(eval_subsets_df.index)), eval_subsets_df.index + " (" + eval_subsets_df["Best subscale"] + ")", rotation=45, ha="right", size=8)
    plt.xlabel("Diagnosis (best subscale)")

    plt.tight_layout()
    
    return fig

def plot_manual_vs_ml_for_prez(eval_subsets_df):
    eval_subsets_df = eval_subsets_df.sort_values(by="Best subscale score", ascending=False)
    
    fig = plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of features")
    plt.plot(eval_subsets_df["Best subscale score"], label="AUROC of best subscale (+ # of items to reach it with ML)", marker="o", linestyle="", color="red")
    plt.plot(eval_subsets_df["ML score at # of items of best subscale (all assessments)"], label="AUROC of ML model on # of items in best subscale (+ # of items in best subscale)", marker="o", linestyle="", color="blue")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Print number of items next to AUC scores to the right of the markers
    for i, row in eval_subsets_df.iterrows():
        #plt.text(i, row["AUC all assessments"]+0.01, str(row["Optimal # of features all assessments"]), ha="left", va="center", size=8)
        plt.text(i, row["ML score at # of items of best subscale (all assessments)"]+0.01, str(row["# of items in best subscale"]), ha="center", va="bottom", size=8, color="blue")
        plt.text(i, row["Best subscale score"]-0.01, str(row["# of items to reach best subscale (all assessments)"]), ha="center", va="top", size=8, color="blue")
    
    # Append best subscale name to the diag name on x axis
    eval_subsets_df["Best subscale"] = eval_subsets_df["Best subscale"].str.split(",").str[1] # Remove prefix before , from subscale names
    plt.xticks(range(len(eval_subsets_df.index)), eval_subsets_df.index + " (" + eval_subsets_df["Best subscale"] + ")", rotation=45, ha="right", size=8)
    plt.xlabel("Diagnosis (best subscale)")

    plt.tight_layout()
    
    return fig

def plot_opt_num_features(opt_vs_all_df):
    opt_vs_all_df = opt_vs_all_df.sort_values(by="AUC optimal features all assessments", ascending=False)

    # Get total number of features in all assessments and free assessments
    tot_n_features_all_assessments = opt_vs_all_df["# input features all assessments"].values[0]
    tot_n_features_free_assessments = opt_vs_all_df["# input features free assessments"].values[0]
    
    # Plot AUC on all features all assessments, AUC on optimal # of features all assessments, AUC on optimal # of features free assessments, AUC on all features free assessments
    fig = plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of items vs all items")
    plt.plot(opt_vs_all_df[f"AUC all features all assessments"], label=f"AUC on all items ({tot_n_features_all_assessments}) (all assessments)", marker="o", linestyle="", color="blue")
    plt.plot(opt_vs_all_df[f"AUC all features free assessments"], label=f"AUC on all items ({tot_n_features_free_assessments}) (free assessments)", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.plot(opt_vs_all_df["AUC optimal features all assessments"], label="AUC on optimal # of items (all assessments)", marker="o", linestyle="", color="red")
    plt.plot(opt_vs_all_df["AUC optimal features free assessments"], label="AUC on optimal # of items (free assessments)", marker="o", linestyle="", color="red", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")
    plt.ylim([0.5, 1.0])

    # Print number of items next to AUC scores to the right of the markers
    opt_vs_all_df = opt_vs_all_df.reset_index()
    for i, row in opt_vs_all_df.iterrows():
        plt.text(i+0.1, row["AUC optimal features all assessments"]-0.001, str(row["Optimal # of features all assessments"]), ha="left", va="center", fontsize=6)
        plt.text(i+0.1, row["AUC optimal features free assessments"]-0.0005, str(row["Optimal # of features free assessments"]), ha="left", va="center", fontsize=6)

    plt.tight_layout()

    return fig

def plot_opt_num_features_for_prez(opt_vs_all_df):
    opt_vs_all_df = opt_vs_all_df.sort_values(by="AUC optimal features all assessments", ascending=False)

    # Get total number of features in all assessments and free assessments
    tot_n_features_all_assessments = opt_vs_all_df["# input features all assessments"].values[0]
    
    # Plot AUC on all features all assessments, AUC on optimal # of features all assessments, AUC on optimal # of features free assessments, AUC on all features free assessments
    fig_all = plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of items vs all items")
    plt.plot(opt_vs_all_df[f"AUC all features all assessments"], label=f"AUC on all items ({tot_n_features_all_assessments})", marker="o", linestyle="", color="blue")
    plt.plot(opt_vs_all_df["AUC optimal features all assessments"], label="AUC on optimal # of items", marker="o", linestyle="", color="red")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")
    plt.ylim([0.5, 1.0])

    # Print number of items next to AUC scores to the right of the markers
    opt_vs_all_df_reset_index = opt_vs_all_df.reset_index()
    for i, row in opt_vs_all_df_reset_index.iterrows():
        plt.text(i+0.1, row["AUC optimal features all assessments"]-0.001, str(row["Optimal # of features all assessments"]), ha="left", va="center", fontsize=6)

    plt.tight_layout()

    fig_all_vs_free = plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of items vs all items")
    plt.plot(opt_vs_all_df["AUC optimal features all assessments"], label="AUC on optimal # of items (all assessments)", marker="o", linestyle="", color="red")
    plt.plot(opt_vs_all_df["AUC optimal features free assessments"], label="AUC on optimal # of items (free assessments)", marker="o", linestyle="", color="red", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    return fig_all, fig_all_vs_free

def plot_thresholds(thresholds_df, diag_short):
    # Drop first row (PPV always 0.5)
    thresholds_df = thresholds_df.drop(thresholds_df.index[0])

    # Plot sensitivity, specificity, PPV, and NPV for each threshold
    fig = plt.figure(figsize=(10, 8))
    plt.title(f"Sensitivity, specificity, PPV, and NPV for each threshold for {diag_short} diagnosis, optimal # of features, all assessments")
    plt.plot(thresholds_df.index, thresholds_df["Sensitivity"], label="Sensitivity", marker="o", linestyle="-", color="blue")
    plt.plot(thresholds_df.index, thresholds_df["Specificity"], label="Specificity", marker="o", linestyle="-", color="red")
    plt.plot(thresholds_df.index, thresholds_df["PPV"], label="PPV", marker="o", linestyle="-", color="green")
    plt.plot(thresholds_df.index, thresholds_df["NPV"], label="NPV", marker="o", linestyle="-", color="orange")
    plt.xlim(thresholds_df.index[0], thresholds_df.index[-1])
    plt.xticks(thresholds_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("Sensitivity, specificity, PPV, and NPV")
    plt.xlabel("Threshold")

    plt.tight_layout()

    return fig 

def plot_thresholds_for_prez(thresholds_df, diag_short):
    # Drop first row (PPV always 0.5)
    thresholds_df = thresholds_df.drop(thresholds_df.index[0])

    # Plot sensitivity, specificity, PPV, and NPV for each threshold
    fig = plt.figure(figsize=(10, 8))
    plt.title(f"Sensitivity and specificity for each threshold for {diag_short} diagnosis, optimal # of features, all assessments")
    plt.plot(thresholds_df.index, thresholds_df["Sensitivity"], label="Sensitivity", marker="o", linestyle="-", color="blue")
    plt.plot(thresholds_df.index, thresholds_df["Specificity"], label="Specificity", marker="o", linestyle="-", color="red")
    plt.xlim(thresholds_df.index[0], thresholds_df.index[-1])
    plt.xticks(thresholds_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("Sensitivity and specificity")
    plt.xlabel("Threshold")

    plt.tight_layout()

    return fig 

def plot_what_improves_LD(check_what_improves_LD_df):

    check_what_improves_LD_df = check_what_improves_LD_df.sort_values("ROC AUC Mean CV_nothing", ascending=False)

    # First plot: one by one

    # Plot ROC AUC Mean CV, x=diag (index), y=ROC AUC Mean CV, color=_nothing or _nih or _conners or _newdiag
    fig_one_by_one = plt.figure(figsize=(10, 8))
    plt.title("AUROC Mean CV for each LD diagnosis, all features, all assessments")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_nothing"], label="AUROC CV - Nothing (+ positive cases/total cases)", marker="D", color="blue", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_nih"], label="AUROC CV - NIH (+ positive cases/total cases)", marker="o", color="red", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_conners"], label="AUROC CV - Conners (+ positive cases/total cases)", marker="o", color="green", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_newdiag"], label="AUROC CV - Test-based (+ positive cases/total cases)", marker="o", color="orange", markersize=10, linestyle="")
    plt.xlim(check_what_improves_LD_df.index[0], check_what_improves_LD_df.index[-1])
    plt.ylim(0.5, 1)
    plt.xticks(check_what_improves_LD_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Add text labels for each point with # of positive examples and Total examples
    check_what_improves_LD_df_reset_index = check_what_improves_LD_df.reset_index()
    for i, row in check_what_improves_LD_df_reset_index.iterrows():
        plt.text(i+0.1, row["ROC AUC Mean CV_nothing"]-0.0005, str(row["# of positive examples_nothing"]) + "/" + str(row["Total examples_nothing"]), ha="left", va="center", fontsize=8)
        plt.text(i+0.1, row["ROC AUC Mean CV_nih"]-0.0005, str(row["# of positive examples_nih"]) + "/" + str(row["Total examples_nih"]), ha="left", va="center", fontsize=8)
        plt.text(i+0.1, row["ROC AUC Mean CV_conners"]-0.0005, str(row["# of positive examples_conners"]) + "/" + str(row["Total examples_conners"]), ha="left", va="center", fontsize=8)
        plt.text(i+0.1, row["ROC AUC Mean CV_newdiag"]-0.0005, str(row["# of positive examples_newdiag"]) + "/" + str(row["Total examples_newdiag"]), ha="left", va="center", fontsize=8)
    plt.tight_layout()

    # Second plot: cummulative

    # Plot ROC AUC Mean CV, x=diag (index), y=ROC AUC Mean CV, color=_nothing or _nih or _conners or _newdiag
    fig_comul = plt.figure(figsize=(10, 8))
    plt.title("AUROC Mean CV for each LD diagnosis, all features, all assessments")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_newdiag"], label="AUROC CV - Test-based (+ positive cases/total cases)", marker="D", color="blue", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_newdiag_conners"], label="AUROC CV - Test-based and Conners (+ positive cases/total cases)", marker="o", color="green", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_newdiag_conners_nih"], label="AUROC CV - Test-based and and Conners and NIH (+ positive cases/total cases)", marker="o", color="red", markersize=10, linestyle="")
    plt.xlim(check_what_improves_LD_df.index[0], check_what_improves_LD_df.index[-1])
    plt.ylim(0.5, 1)
    plt.xticks(check_what_improves_LD_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")
    
    plt.tight_layout()

    return fig_one_by_one, fig_comul

def plot_what_improves_LD_for_prez(check_what_improves_LD_df):
    check_what_improves_LD_df = check_what_improves_LD_df.sort_values("ROC AUC Mean CV_nothing", ascending=False)

    # First plot: one by one

    # Plot ROC AUC Mean CV, x=diag (index), y=ROC AUC Mean CV, color=_nothing or _nih or _conners or _newdiag
    fig = plt.figure(figsize=(10, 8))
    plt.title("AUROC Mean CV for each LD diagnosis, all features")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_nothing"], label="AUROC CV - Nothing (+ positive cases/total cases)", marker="D", color="blue", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_nih"], label="AUROC CV - NIH (+ positive cases/total cases)", marker="o", color="red", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_conners"], label="AUROC CV - Conners (+ positive cases/total cases)", marker="o", color="green", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_newdiag"], label="AUROC CV - Test-based (+ positive cases/total cases)", marker="o", color="orange", markersize=10, linestyle="")
    plt.xlim(check_what_improves_LD_df.index[0], check_what_improves_LD_df.index[-1])
    plt.ylim(0.5, 1)
    plt.xticks(check_what_improves_LD_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")
    plt.tight_layout()
    
    return fig

def plot_coefficients(coefficients_df, diag):
    coefficients_df = coefficients_df.sort_values("Coefficient", ascending=True)

    # Plot coefficients as horizontal bars, x=Coefficient, y=Item, print text of the item inside each bar
    fig = plt.figure(figsize=(14, 8 ))
    plt.title(f"{diag}: coefficients for each item, optimal # of items")
    plt.barh(coefficients_df.index, coefficients_df["Coefficient"], color="blue")
    plt.xlim(-1, 1)
    plt.xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]) 
    plt.tight_layout()

    # Add vertical lines at 0.25, 0.5, 0.75
    plt.axvline(x=0.25, color="black", linestyle="--", linewidth=0.1)
    plt.axvline(x=0.5, color="black", linestyle="--", linewidth=0.1)
    plt.axvline(x=0.75, color="black", linestyle="--", linewidth=0.1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.1)
    plt.axvline(x=-0.25, color="black", linestyle="--", linewidth=0.1)
    plt.axvline(x=-0.5, color="black", linestyle="--", linewidth=0.1)
    plt.axvline(x=-0.75, color="black", linestyle="--",  linewidth=0.1)

    return fig


def main():

    output_dir = "output/viz/"
    make_dir_if_not_exists(output_dir)

    # Read performance tables
    compare_orig_subsets_df = pd.read_csv("output/compare_orig_vs_subsets.csv", index_col=0)
    compare_orig_subsets_learning_df = pd.read_csv("output/compare_orig_vs_subsets_learning.csv", index_col=0)
    what_improves_LD_df = pd.read_csv("output/what_improves_LD.csv", index_col=0)
    coefficients_all_df = pd.read_csv("output/coefs_all.csv", index_col=0)
    coefficients_free_df = pd.read_csv("output/coefs_free.csv", index_col=0)
    coefficients_learning_all_df = pd.read_csv("output/coefs_learning_all.csv", index_col=0)
    coefficients_learning_free_df = pd.read_csv("output/coefs_learning_free.csv", index_col=0)
    coefficients_CIS_all_df = pd.read_csv("output/coefs_CIS_all.csv", index_col=0)
    coefficients_CIS_free_df = pd.read_csv("output/coefs_CIS_free.csv", index_col=0)

    # Read thresholds data from diagnosis_predictor_data (for all assessments, that's why using dir_eval_subsets_all[0])
    dir_eval_subsets_all = read_data_eval_subsets()
    filename = "Diag.Autism Spectrum Disorder.csv"
    thresholds_df = make_df_to_plot_thresholds(dir_eval_subsets_all[0], filename)

    dir_eval_subsets_learning_all = read_data_eval_subsets_learning()
    filename = "New Diag.Specific Learning Disorder with Impairment in Reading.csv"
    thresholds_learning_df = make_df_to_plot_thresholds(dir_eval_subsets_learning_all[0], filename)
    
    # Rephrase diags to shorter names
    compare_orig_subsets_df = compare_orig_subsets_df.rename(index=diagnosis_dict)
    compare_orig_subsets_learning_df = compare_orig_subsets_learning_df.rename(index=diagnosis_dict)
    what_improves_LD_df = what_improves_LD_df.rename(index=diagnosis_dict)
    
    # Make all columns except those that start with ROC AUC into ints (if not NA) (bug in pd)
    for col in what_improves_LD_df.columns:
        print(col)
        if not "AUC" in col and not "score" in col and not "Best subscale" in col:
            what_improves_LD_df[col] = what_improves_LD_df[col].astype('Int64')
    for col in compare_orig_subsets_df.columns:
        print(col)
        if not "AUC" in col and not "score" in col and not "Best subscale" in col:
            compare_orig_subsets_df[col] = compare_orig_subsets_df[col].astype('Int64')


    plot = plot_eval_orig(compare_orig_subsets_df)
    plot.savefig("output/viz/ROC_AUC_all_features.png", bbox_inches="tight", dpi=600)
    plot_all, plot_all_vs_free = plot_eval_orig_for_prez(compare_orig_subsets_df)
    plot_all.savefig("output/viz/ROC_AUC_all_features_for_prez_all.png", bbox_inches="tight", dpi=600)
    plot_all_vs_free.savefig("output/viz/ROC_AUC_all_features_for_prez_all_vs_free.png", bbox_inches="tight", dpi=600)
    plot_all_learning, plot_all_vs_free_learning = plot_eval_orig_for_prez(compare_orig_subsets_learning_df)
    plot_all_learning.savefig("output/viz/ROC_AUC_all_features_for_prez_all_learning.png", bbox_inches="tight", dpi=600)
    plot_all_vs_free_learning.savefig("output/viz/ROC_AUC_all_features_for_prez_all_vs_free_learning.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    plot = plot_manual_vs_ml(compare_orig_subsets_df)
    plot.savefig("output/viz/ROC_AUC_subsets.png", bbox_inches="tight", dpi=600)
    plot = plot_manual_vs_ml_for_prez(compare_orig_subsets_df)
    plot.savefig("output/viz/ROC_AUC_subsets_for_prez.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    plot = plot_opt_num_features(compare_orig_subsets_df)
    plot.savefig("output/viz/ROC_AUC_optimal_vs_all_features.png", bbox_inches="tight", dpi=600)
    plot_all, plot_all_vs_free = plot_opt_num_features_for_prez(compare_orig_subsets_df)
    plot_all.savefig("output/viz/ROC_AUC_optimal_vs_all_features_for_prez_all.png", bbox_inches="tight", dpi=600)    
    plot_all_vs_free.savefig("output/viz/ROC_AUC_optimal_vs_all_features_for_prez_all_vs_free.png", bbox_inches="tight", dpi=600)    
    plot_all_learning, plot_all_vs_free_learning = plot_opt_num_features_for_prez(compare_orig_subsets_learning_df)
    plot_all_learning.savefig("output/viz/ROC_AUC_optimal_vs_all_features_learning_for_prez_all.png", bbox_inches="tight", dpi=600)    
    plot_all_vs_free_learning.savefig("output/viz/ROC_AUC_optimal_vs_all_features_learning_for_prez_all_vs_free.png", bbox_inches="tight", dpi=600) 
    plt.close("all")   

    plot = plot_thresholds(thresholds_df, "ASD")
    plot.savefig("output/viz/sensitivity_specificity_PPV_NPV.png", bbox_inches="tight", dpi=600)
    plot = plot_thresholds_for_prez(thresholds_learning_df, "SLD-Reading")
    plot.savefig("output/viz/sensitivity_specificity_PPV_NPV_learning.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    plot_one_by_one, plot_cumul = plot_what_improves_LD(what_improves_LD_df)
    plot_cumul.savefig("output/viz/what_improves_LD_one_by_one.png", dpi=600)
    plot_one_by_one.savefig("output/viz/what_improves_LD_cumulative.png", dpi=600)
    plot = plot_what_improves_LD_for_prez(what_improves_LD_df)
    plot.savefig("output/viz/what_improves_LD_one_by_one_for_prez.png", dpi=600)
    plot_all = plot_coefficients(coefficients_all_df, "ASD")
    plot_free = plot_coefficients(coefficients_free_df, "ASD")
    plot_learning_all = plot_coefficients(coefficients_learning_all_df, "SLD-Reading")
    plot_learning_free = plot_coefficients(coefficients_learning_free_df, "SLD-Reading")
    plot_CIS_free = plot_coefficients(coefficients_CIS_free_df, "CIS_P")
    plot_CIS_all = plot_coefficients(coefficients_CIS_all_df, "CIS_P")
    plot_all.savefig("output/viz/coefficients_all.png", dpi=600)
    plot_free.savefig("output/viz/coefficients_free.png", dpi=600)
    plot_learning_all.savefig("output/viz/coefficients_learning_all.png", dpi=600)
    plot_learning_free.savefig("output/viz/coefficients_learning_free.png", dpi=600)
    plot_CIS_free.savefig("output/viz/coefficients_CIS_free.png", dpi=600)
    plot_CIS_all.savefig("output/viz/coefficients_CIS_all.png", dpi=600)
    plt.close("all")

if __name__ == "__main__":
    main()



