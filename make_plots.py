from helpers import get_newest_non_empty_dir_in_dir, make_dir_if_not_exists

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from make_final_tables import data_paths

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
        'Diag.Specific Learning Disorder with Impairment in Mathematics (test)': 'SLD-Math',
        'Diag.Language Disorder': 'Language',
        'Diag.Specific Phobia': 'Phobia',
        'Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'Diag.Specific Learning Disorder with Impairment in Reading (test)': 'SLD-Reading',
        'Diag.Specific Learning Disorder with Impairment in Written Expression (test)': 'SLD-Writing',
        'Diag.Other Specified Anxiety Disorder': 'Other Anxiety',
        'Diag.Processing Speed Deficit (test)': 'LD-PS',
        'Diag.Borderline Intellectual Functioning (test)': 'LD-BIF',
        'Diag.Intellectual Disability-Borderline (test)': 'LD-BIF',
    }

def make_df_to_plot_thresholds(dir_eval_subsets_all, filename):

    df_opt_features_all = pd.read_csv(dir_eval_subsets_all + "sens-spec-on-subsets-test-set-optimal-nb-features/" + filename, 
                                      index_col=0)

    return df_opt_features_all

def plot_eval_orig(eval_orig_df, filename, plot_free_assessments=False, plot_cv=False):
    # Plot on same plot, one point for each diagnosis, connext diagnoses with lines
    plt.figure(figsize=(10, 5))
    plt.title("AUROC for all features")
    plt.plot(eval_orig_df["AUC all features all assessments"], label="AUROC on test set all assessments", marker="o", linestyle="", color="lightblue")
    plt.plot(eval_orig_df["AUC CV all features all assessments"], label="AUROC CV all assessments", marker="o", linestyle="", color="blue")
    if plot_free_assessments:
        plt.plot(eval_orig_df["AUC all features free assessments"], label="AUROC on test set free assessments", marker="o", linestyle="", color="lightblue", markerfacecolor='none')
        plt.plot(eval_orig_df["AUC CV all features free assessments"], label="AUROC CV free assessments", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.plot(eval_orig_df["AUC all features healthy controls all assessments"], label="AUROC on test set healthy controls all assessments", marker="o", linestyle="", color="lightcoral")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylim([0.5, 1.0])
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Draw horizontal line at 0.7, 0.8, 0.9
    plt.axhline(y=0.7, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.axhline(y=0.8, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.axhline(y=0.9, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

    # For each diag, also plot bar plots with the number of positive examples for all assessments and for free assessments
    ax2 = plt.twinx()
    ax2.set_ylabel("Number of positive cases")
    if plot_free_assessments:
        ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples free assessments"].values, label="# of positive cases free assessments", color="blue", width=0.5, alpha=0.2, fill=False)
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples all assessments"].values, label="# of positive cases all assessments", color="blue", width=0.5, alpha=0.2)
    tot_n_features_all_assessments = eval_orig_df["# rows full dataset all assessments"].values[0]
    if plot_free_assessments:
        tot_n_features_free_assessments = eval_orig_df["# rows full dataset free assessments"].values[0]
    if plot_free_assessments:
        ax2.legend([f"# of positive cases free assessments/{tot_n_features_free_assessments}", f"# of positive cases all assessments/{tot_n_features_all_assessments}"], loc="lower right")
    else:
        ax2.legend([f"# of positive cases all assessments/{tot_n_features_all_assessments}"], loc="lower right")
    
    plt.tight_layout()

    output_dir = "output/viz/"
    make_dir_if_not_exists(output_dir)
    plt.savefig(output_dir + filename, bbox_inches="tight", dpi=600)

def plot_manual_vs_ml(eval_subsets_df):
    eval_subsets_df = eval_subsets_df.sort_values(by="Best subscale score", ascending=False)
    
    plt.figure(figsize=(10, 8))
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
    
    plt.savefig("output/viz/ROC_AUC_subsets.png", bbox_inches="tight", dpi=600)

def plot_opt_num_features(opt_vs_all_df, filename, plot_free_assessments=False):
    opt_vs_all_df = opt_vs_all_df.sort_values(by="AUC optimal features all assessments", ascending=False)

    # Get total number of features in all assessments and free assessments
    tot_n_features_all_assessments = opt_vs_all_df["# input features all assessments"].values[0]
    if plot_free_assessments:
        tot_n_features_free_assessments = opt_vs_all_df["# input features free assessments"].values[0]
    
    # Plot AUC on all features all assessments, AUC on optimal # of features all assessments, AUC on optimal # of features free assessments, AUC on all features free assessments
    plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of items vs all items")
    plt.plot(opt_vs_all_df[f"AUC all features all assessments"], label=f"AUC on all items ({tot_n_features_all_assessments}) (all assessments)", marker="o", linestyle="", color="blue")
    if plot_free_assessments:
        plt.plot(opt_vs_all_df[f"AUC all features free assessments"], label=f"AUC on all items ({tot_n_features_free_assessments}) (free assessments)", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.plot(opt_vs_all_df["AUC optimal features all assessments"], label="AUC on optimal # of items (all assessments)", marker="o", linestyle="", color="red")
    if plot_free_assessments:
        plt.plot(opt_vs_all_df["AUC optimal features free assessments"], label="AUC on optimal # of items (free assessments)", marker="o", linestyle="", color="red", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Print number of items next to AUC scores to the right of the markers
    opt_vs_all_df = opt_vs_all_df.reset_index()
    for i, row in opt_vs_all_df.iterrows():
        plt.text(i+0.1, row["AUC optimal features all assessments"]-0.001, str(row["Optimal # of features all assessments"]), ha="left", va="center", fontsize=6)
        if plot_free_assessments:
            plt.text(i+0.1, row["AUC optimal features free assessments"]-0.0005, str(row["Optimal # of features free assessments"]), ha="left", va="center", fontsize=6)

    plt.tight_layout()

    plt.savefig("output/viz/"+filename, bbox_inches="tight", dpi=600)

def plot_thresholds(thresholds_df, diag):
    # Drop first row (PPV always 0.5)
    thresholds_df = thresholds_df.drop(thresholds_df.index[0])

    # Plot sensitivity, specificity, PPV, and NPV for each threshold
    plt.figure(figsize=(10, 8))
    plt.title(f"Sensitivity, specificity, PPV, and NPV for each threshold for {diag}, optimal # of features, all assessments")
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

    plt.savefig("output/viz/sensitivity_specificity_PPV_NPV.png", bbox_inches="tight", dpi=600)

def plot_what_improves_LD(check_what_improves_LD_df):

    check_what_improves_LD_df = check_what_improves_LD_df.sort_values("ROC AUC Mean CV_nothing", ascending=False)

    # First plot: one by one

    # Plot ROC AUC Mean CV, x=diag (index), y=ROC AUC Mean CV, color=_nothing or _nih or _conners or _newdiag
    plt.figure(figsize=(10, 8))
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
    
    plt.savefig("output/viz/what_improves_LD_one_by_one.png", dpi=600)

    # Second plot: cummulative

    # Plot ROC AUC Mean CV, x=diag (index), y=ROC AUC Mean CV, color=_nothing or _nih or _conners or _newdiag
    plt.figure(figsize=(10, 8))
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

    plt.savefig("output/viz/what_improves_LD_cumulative.png", dpi=600)

def main():

    # Read performance tables
    compare_orig_subsets_df = pd.read_csv("output/compare_orig_vs_subsets.csv", index_col=0)
    compare_orig_subsets_learning_df = pd.read_csv("output/compare_orig_vs_subsets_learning.csv", index_col=0)
    what_improves_LD_df = pd.read_csv("output/what_improves_LD.csv", index_col=0)

    # Read thresholds data from diagnosis_predictor_data (for all assessments, that's why using dir_eval_subsets_all[0])
    dir_eval_subsets_all_learning = data_paths["eval_subsets_learning"]
    print("dir_eval_subsets_all", dir_eval_subsets_all_learning)
    thresholds_filename = "Diag.Specific Learning Disorder with Impairment in Reading (test).csv"
    thresholds_df = make_df_to_plot_thresholds(dir_eval_subsets_all_learning, thresholds_filename)
    
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


    plot_eval_orig(compare_orig_subsets_df, filename="ROC_AUC_all_features.png", plot_free_assessments=False)
    plot_eval_orig(compare_orig_subsets_learning_df, filename="ROC_AUC_all_features_learning.png", plot_free_assessments=True)
    plot_manual_vs_ml(compare_orig_subsets_learning_df)
    plot_opt_num_features(compare_orig_subsets_df, filename="ROC_AUC_optimal_vs_all_features.png", plot_free_assessments=False)
    plot_opt_num_features(compare_orig_subsets_learning_df, filename="ROC_AUC_optimal_vs_all_features_learning.png", plot_free_assessments=True)
    plot_thresholds(thresholds_df, thresholds_filename.split(".")[0])
    plot_what_improves_LD(what_improves_LD_df)

if __name__ == "__main__":
    main()



