from helpers import get_newest_non_empty_dir_in_dir, make_dir_if_not_exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_reading import DataReader

DIAGNOSIS_DICT = {
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
        'Diag.Specific Learning Disorder with Impairment in Mathematics (test)': 'SLD-Math (test)',
        'Diag.Language Disorder': 'Language',
        'Diag.Specific Phobia': 'Phobia',
        'Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'Diag.Specific Learning Disorder with Impairment in Reading (test)': 'SLD-Reading (test)',
        'Diag.Specific Learning Disorder with Impairment in Written Expression (test)': 'SLD-Writing (test)',
        'Diag.Other Specified Anxiety Disorder': 'Other Anxiety',
        'Diag.Processing Speed Deficit (test)': 'PS (test)',
        'Diag.Borderline Intellectual Functioning (test)': 'BIF (test)',
        'Diag.Intellectual Disability-Borderline (test)': 'BIF (test)',
        'Diag.NVLD (test)': 'NVLD (test)',
        'Diag.NVLD without reading condition (test)': 'NVLD no read (test)',
    }
LDS = [
    'SLD-Math',
    'SLD-Math (test)',
    'SLD-Reading',
    'SLD-Writing',
    'SLD-Reading (test)',
    'SLD-Writing (test)',
    'PS (test)',
    'NVLD (test)',
    'NVLD no read (test)'
]
NON_LDS = [diag for diag in DIAGNOSIS_DICT.values() if diag not in LDS]

# Get only the diagnoses with (test) in the name
test_diagnosis_list = [diag for diag in DIAGNOSIS_DICT.keys() if "(test)" in diag]

def get_tick_ids_for_diagnoses_with_LD_in_name(diagnoses):
    # Get the xtick ids for diagnoses with "test" in the name
    tick_ids = []
    for i, diag in enumerate(diagnoses):
        if "LD" in diag:
            tick_ids.append(i)
    return tick_ids

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

    # Make xticks for the diagnoses with (test) in the name bold
    tick_ids = get_tick_ids_for_diagnoses_with_LD_in_name(eval_orig_df.index)
    for tick_id in tick_ids:
        plt.gca().get_xticklabels()[tick_id].set_weight("bold")

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

def plot_manual_vs_ml(eval_subsets_df, plot_free_assessments=False):
    eval_subsets_df = eval_subsets_df.sort_values(by="Best subscale score", ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.title("AUROC on test set for subsets of features")
    plt.plot(eval_subsets_df["Best subscale score"], label="AUROC of best subscale (+ # of items to reach it with ML)", marker="o", linestyle="", color="red")
    plt.plot(eval_subsets_df["ML score at # of items of best subscale (all assessments)"], label="AUROC of ML model on # of items in best subscale (all assessments)", marker="o", linestyle="", color="blue")
    if plot_free_assessments:
        plt.plot(eval_subsets_df["ML score at # of items of best subscale (free assessments)"], label="AUROC of ML model on # of items in best subscale (free assessments)", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)

    # Make xticks for the diagnoses with (test) in the name bold
    tick_ids = get_tick_ids_for_diagnoses_with_LD_in_name(eval_subsets_df.index)
    for tick_id in tick_ids:
        plt.gca().get_xticklabels()[tick_id].set_weight("bold")

    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Print number of items next to AUC scores to the right of the markers
    if not plot_free_assessments:
        for i, row in eval_subsets_df.iterrows():
            #plt.text(i, row["AUC all assessments"]+0.01, str(row["Optimal # of features all assessments"]), ha="left", va="center", size=8)
            #plt.text(i, row["ML score at # of items of best subscale (all assessments)"]+0.01, str(row["# of items in best subscale"]), ha="center", va="bottom", size=8, color="blue")
            plt.text(i, row["Best subscale score"]-0.01, str(row["# of items to reach best subscale (all assessments)"]), ha="center", va="top", size=8, color="blue")
    else:
        for i, row in eval_subsets_df.iterrows():
            plt.text(i, row["Best subscale score"]-0.01, str(row["# of items to reach best subscale (free assessments)"]), ha="center", va="top", size=8, color="blue", alpha=0.5)
    
    # Append best subscale name to the diag name on x axis
    plt.xticks(range(len(eval_subsets_df.index)), (
        eval_subsets_df.index +
        " (" + 
        eval_subsets_df['Best subscale'] + 
        ", " + 
        eval_subsets_df['# of items in best subscale'].astype(str) + 
        ")"
        ), rotation=45, ha="right", size=8)
    plt.xlabel("Diagnosis (best subscale)")

    plt.tight_layout()
    
    if plot_free_assessments:
        plt.savefig("output/viz/ROC_AUC_subsets_free.png", bbox_inches="tight", dpi=600)
    else:
        plt.savefig("output/viz/ROC_AUC_subsets.png", bbox_inches="tight", dpi=600)

def add_formatted_labels_to_bars(ax, container):
    heights = [rect.get_height() for rect in container]
    formatted_heights = ["{:.2f}".format(height).lstrip('0') for height in heights]
    
    ax.bar_label(container, labels=formatted_heights, label_type='edge', size=8, rotation=45)
    return ax

def plot_manual_vs_ml_bars(df, diags, filename, title, col_dict):
    df = df.sort_values(by=list(col_dict.keys())[0], ascending=False)

    # Filter df to only include the diagnoses in diags
    df = df[df.index.isin(diags)]

    # Rename columns for legend
    for col, col_renamed in col_dict.items():
        df[col_renamed] = df[col]

    # Keep only cols to be plotted in df
    df = df[col_dict.values()]    
    
    # Plot grouped bar chart
    ax = df.plot(kind='bar', figsize=(10, 6))
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('AUROC')
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", size=8)
    plt.ylim([0.5, 1.0])

    # Add y-values on top of the bars if less than three bars in a group or less than 5 groups in total
    #for p in ax.patches:
        #ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    if len(df.columns) < 3 or len(df.index) < 5:
        for container in ax.containers:
            #ax.bar_label(container, fmt='%.2f', label_type='edge', size=8, rotation=45)
            add_formatted_labels_to_bars(ax, container)

    # Show legend
    plt.legend(loc="upper right")

    plt.tight_layout()

    plt.savefig("output/viz/" + filename, bbox_inches="tight", dpi=600)


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

    # Make xticks for the diagnoses with (test) in the name bold
    tick_ids = get_tick_ids_for_diagnoses_with_LD_in_name(opt_vs_all_df.index)
    for tick_id in tick_ids:
        plt.gca().get_xticklabels()[tick_id].set_weight("bold")


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
    plt.plot(thresholds_df.index, thresholds_df["Sensitivity"], label="Sensitivity", marker="o", linestyle="", color="blue")
    plt.plot(thresholds_df.index, thresholds_df["Specificity"], label="Specificity", marker="o", linestyle="", color="red")
    plt.plot(thresholds_df.index, thresholds_df["PPV"], label="PPV", marker="o", linestyle="", color="green")
    plt.plot(thresholds_df.index, thresholds_df["NPV"], label="NPV", marker="o", linestyle="", color="orange")
    plt.xlim(thresholds_df.index[0], thresholds_df.index[-1])
    plt.xticks(thresholds_df.index, rotation=45, ha="right", size=8)

    plt.legend(loc="upper right")
    plt.ylabel("Sensitivity, specificity, PPV, and NPV")
    plt.xlabel("Threshold")

    plt.tight_layout()

    plt.savefig("output/viz/sensitivity_specificity_PPV_NPV.png", bbox_inches="tight", dpi=600)

def plot_sum_scores_vs_subscales(sum_scores_df, sum_scores_free_df = None):
    # Plot AUC of sum scores vs AUC of best subscales, add number of items in best subscale to the right of the marker
    plt.figure(figsize=(10, 8))
    plt.title("AUROC of new screener vs best existing subscale")
    plt.plot(sum_scores_df["Best subscale score"], label="AUROC of best subscale", marker="o", linestyle="", color="red")
    plt.plot(sum_scores_df["AUROC"], label="AUROC of new screener", marker="o", linestyle="", color="blue")
    if sum_scores_free_df is not None:
        plt.plot(sum_scores_free_df["AUROC"], label="AUROC of new screener (free assessments)", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)

    # Make xticks for the diagnoses with (test) in the name bold
    tick_ids = get_tick_ids_for_diagnoses_with_LD_in_name(sum_scores_df.index)
    for tick_id in tick_ids:
        plt.gca().get_xticklabels()[tick_id].set_weight("bold")

    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    # Append # of items to diag name on x axis
    plt.xticks(range(len(sum_scores_df.index)), (
        sum_scores_df.index +
        " (" + 
        sum_scores_df['Best subscale'] + 
        ", " + 
        sum_scores_df['# of items in best subscale'].astype(str) + 
        ")"
        ), rotation=45, ha="right", size=8)
    plt.xlabel("Diagnosis (# of items in best subscale)")
        
    plt.tight_layout()

    if sum_scores_free_df is not None:
        plt.savefig("output/viz/sum_scores_free.png", bbox_inches="tight", dpi=600)
    else:
        plt.savefig("output/viz/sum_scores.png", bbox_inches="tight", dpi=600)

def plot_learning_improvements(learning_improvement_df):

    # Plot AUC from "original", "more assessments", "NIH"
    df = learning_improvement_df
    plt.figure(figsize=(10, 8))
    plt.title("AUROC of original models, models with more assessments, and models with more assessments and NIH")
    plt.plot(df["original"], label="Original", marker="o", linestyle="", color="blue")
    plt.plot(df["more assessments"], label="More assessments", marker="o", linestyle="", color="red")
    plt.plot(df["NIH"], label="More assessments and NIH", marker="o", linestyle="", color="green")
    plt.xticks(rotation=45, ha="right", size=8)

    # Make xticks for the diagnoses with (test) in the name bold
    tick_ids = get_tick_ids_for_diagnoses_with_LD_in_name(df.index)
    for tick_id in tick_ids:
        plt.gca().get_xticklabels()[tick_id].set_weight("bold")

    plt.legend(loc="upper right")
    plt.ylabel("AUROC")
    plt.xlabel("Diagnosis")

    plt.tight_layout()

    plt.savefig("output/viz/learning_improvements.png", bbox_inches="tight", dpi=600)

def plot_learning_improvements_bars(learning_improvement_df):

    plot_manual_vs_ml_bars(
        df=learning_improvement_df, 
        diags=learning_improvement_df.index, 
        filename="learning_improvements_bars.png", 
        title="AUROC of original models, models with more assessments, and models with more assessments and NIH",
        col_dict={
            "original": "Original",
            "more assessments": "More assessments",
            "NIH": "More assessments and NIH"
        }
    )

def make_averages_for_learning_improvements(learning_improvement_df):
    df = learning_improvement_df
    print(df)

    # Make df with averages for each assessment subset
    col_dict = {
        "original": "Original",
        "more assessments": "More assessments",
        "NIH": "More assessments and NIH"
    }
    index_dict = {
        "Non LDs": NON_LDS,
        "LDs": LDS,
    }
    df_averages = pd.DataFrame(columns=col_dict.values(), index=index_dict.keys())
    print(df_averages)

    for index, diags in index_dict.items():
        diags = [diag for diag in diags if diag in df.index] # Remove diagnoses not in df
        for col in df.columns:
            df_averages.loc[index, col_dict[col]] = df.loc[diags, col].mean()
    
    print("df_averages:\n", df_averages)

    return df_averages


def plot_average_learning_improvements(learning_improvement_df):
    df = make_averages_for_learning_improvements(learning_improvement_df)

    plot_averages(
        df, 
        filename="learning_improvements_averages.png",
        indices=["Non LDs", "LDs"], 
        cols=df.columns)

def make_averages_for_assessment_subsets(
        df_ml,
        df_sum_scores, 
        filename_and_diag_sets_dict):
    
    # Make df with averages for each assessment subset
    col_dict_ml = {
        "Best subscale score": "Best subscale",
        "ML score at # of items of best subscale (all assessments)": "All",
        "ML score at # of items of best subscale (free assessments)": "Non-propriatary",
        "ML score at # of items of best subscale (only parent report)": "Only parent report",
        "ML score at # of items of best subscale (free assessments, only parent report)": "Non-proprietary, only parent report"
    }
    col_dict_sum_scores = {
        "Best subscale score": "Best subscale",
        "AUROC all assessments parent and sr": "All",
        "AUROC free assessments parent and sr": "Non-propriatary",
        "AUROC all assessments only parent report": "Only parent report",
        "AUROC free assessments only parent report": "Non-proprietary, only parent report"
    }
    col_list = col_dict_ml.values()
    df_averages_ml = pd.DataFrame(col_list)
    df_averages_sum_scores = pd.DataFrame(col_list)

    # Make df with averages for each assessment subset
    for index, diags in filename_and_diag_sets_dict.items():
        diags = [diag for diag in diags if diag in df_ml.index]
        for col_ml, col_renamed in col_dict_ml.items():
            df_averages_ml.loc[index, col_renamed] = df_ml.loc[diags, col_ml].mean()

        diags = [diag for diag in diags if diag in df_sum_scores.index]
        for col_sum_scores, col_renamed in col_dict_sum_scores.items():
            df_averages_sum_scores.loc[index, col_renamed] = df_sum_scores.loc[diags, col_sum_scores].mean()

    print("df_averages_ml:\n", df_averages_ml)
    print("df_averages_sum_scores:\n", df_averages_sum_scores)

    return df_averages_ml, df_averages_sum_scores, col_list

def plot_averages(df, filename, indices, cols):
    col_dict = {x:x for x in cols} # Don't need to rename columns
    plot_manual_vs_ml_bars(
        df, 
        diags=indices, 
        filename=filename, 
        title="AUROC of best existing subscale vs subset sum-score",
        col_dict=col_dict)
        
def plot_group_bar_plots_for_subsets(df_ml, df_sum_scores):
    filename_and_diag_sets_dict = {
        "Non LDs": NON_LDS,
        "LDs": LDS,
    }
    filename_and_col_sets_dict_ml = { # Rename columns for legend
        "all": {
            "Best subscale score": "AUROC of best subscale",
            "ML score at # of items of best subscale (all assessments)": "AUROC of ML model on # of items in best subscale",
        },
        "free": {
            "Best subscale score": "AUROC of best subscale",
            "ML score at # of items of best subscale (all assessments)": "AUROC of ML model on # of items in best subscale (all assessments)",
            "ML score at # of items of best subscale (free assessments)": "AUROC of ML model on # of items in best subscale (non-proprietary)",
        },
        "only_parent_report": {
            "Best subscale score": "AUROC of best subscale",
            "ML score at # of items of best subscale (all assessments)": "AUROC of ML model on # of items in best subscale (all assessments)",
            "ML score at # of items of best subscale (only parent report)": "AUROC of ML model on # of items in best subscale (only parent report)",
        },
        "free_and_only_parent_report": {
            "Best subscale score": "AUROC of best subscale",
            "ML score at # of items of best subscale (all assessments)": "AUROC of ML model on # of items in best subscale (all assessments)",
            "ML score at # of items of best subscale (free assessments)": "AUROC of ML model on # of items in best subscale (non-proprietary)",
            "ML score at # of items of best subscale (only parent report)": "AUROC of ML model on # of items in best subscale (only parent report)",
            "ML score at # of items of best subscale (free assessments, only parent report)": "AUROC of ML model on # of items in best subscale (free assessments, only parent report)",
        }
    }
    filename_and_col_sets_dict_sum_scores = {
        "all": {
            "Best subscale score": "AUROC of best subscale",
            "AUROC all assessments parent and sr": "AUROC of item subset sum-score",
        },
        "free": {
            "Best subscale score": "AUROC of best subscale",
            "AUROC all assessments parent and sr": "AUROC of item subset sum-score (all assessments)",
            "AUROC free assessments parent and sr": "AUROC of item subset sum-score (non-proprietary)",
        },
        "only_parent_report": {
            "Best subscale score": "AUROC of best subscale",
            "AUROC all assessments parent and sr": "AUROC of item subset sum-score (all assessments)",
            "AUROC all assessments only parent report": "AUROC of item subset sum-score (only parent report)",
        },
        "free_and_only_parent_report": {
            "Best subscale score": "AUROC of best subscale",
            "AUROC all assessments parent and sr": "AUROC of item subset sum-score (all assessments)",
            "AUROC free assessments parent and sr": "AUROC of item subset sum-score (non-proprietary)",
            "AUROC all assessments only parent report": "AUROC of item subset sum-score (only parent report)",
            "AUROC free assessments only parent report": "AUROC of item subset sum-score (non-proprietary, only parent report)",
        }
    }
    for filename, diags in filename_and_diag_sets_dict.items():
        for col_set_name, col_set in filename_and_col_sets_dict_ml.items():
            plot_manual_vs_ml_bars(
                df_ml, 
                diags=diags, 
                filename=f"ROC_AUC_subsets_bars_{filename}_{col_set_name}.png", 
                title="AUROC on test set for subsets of features",
                col_dict=col_set)
    for filename, diags in filename_and_diag_sets_dict.items():
        for col_set_name, col_set in filename_and_col_sets_dict_sum_scores.items():
            plot_manual_vs_ml_bars(
                df_sum_scores, 
                diags=diags, 
                filename=f"ROC_AUC_sum_scores_bars_{filename}_{col_set_name}.png", 
                title="AUROC of best existing subscale vs subset sum-score",
                col_dict=col_set)
            
    # Plot group bar plot with averages score between diagnoses for each assessment subset (separate for LD and non-LD)
    # One group for non-LDS, one for LDS. Each bar in group is assessment subset
    df_averages_ml, df_averages_sum_scores, col_list = make_averages_for_assessment_subsets(
        df_ml,
        df_sum_scores, 
        filename_and_diag_sets_dict)
    
    plot_averages(df_averages_ml, 
                  filename="ROC_AUC_averages_ml.png", 
                  indices=filename_and_diag_sets_dict.keys(), 
                  cols=col_list)
    plot_averages(df_averages_sum_scores, 
                  filename="ROC_AUC_averages_sum_scores.png", 
                  indices=filename_and_diag_sets_dict.keys(), 
                  cols=col_list)
    
def get_opt_n_features_per_diag(df):
    df = df[["Optimal # of features all assessments parent and sr", 
             "Optimal # of features free assessments parent and sr",
             "Optimal # of features all assessments only parent report",
             "Optimal # of features free assessments only parent report"]]
    df = df.rename(columns={
        "Optimal # of features all assessments parent and sr": "All assessments",
        "Optimal # of features free assessments parent and sr": "Free assessments",
        "Optimal # of features all assessments only parent report": "Only parent report",
        "Optimal # of features free assessments only parent report": "Free assessments only parent report"
    })
    return df

def plot_saturation_plot(df, optimal_ns, ax, diag):
    # Plot saturation plot for one diagnosis, 4 curves (one for each assessment subset): "All assessments", "Free assessments", "Only parent report", "Free assessments, only parent report"
    # x-axis: # of features, y-axis: AUROC

    ax.set_title(diag)
    ax.set_xlabel("# of features")
    ax.set_ylabel("AUROC")

    # Plot curves
    ax.plot(df.index, df["All assessments"], label="All assessments", linestyle="-", color="orange")
    ax.plot(df.index, df["Free assessments"], label="Non-proprietary assessments", linestyle="--", color="orange")
    ax.plot(df.index, df["Only parent report"], label="Only parent report", linestyle="-", color="red")
    ax.plot(df.index, df["Free assessments only parent report"], label="Non-proprietary assessments, only parent report", linestyle="--", color="red")

    # Highlight dots with optimal n of features (same column names), empty dot for free, filled dot for all, orange for parent and sr assessments, red for only parent report
    ax.plot(optimal_ns["All assessments"], df.loc[optimal_ns["All assessments"], "All assessments"], marker="o", color="orange", linestyle="", fillstyle="full", markersize=10)
    ax.plot(optimal_ns["Free assessments"], df.loc[optimal_ns["Free assessments"], "Free assessments"], marker="o", color="orange", linestyle="", fillstyle="none", markersize=10)
    ax.plot(optimal_ns["Only parent report"], df.loc[optimal_ns["Only parent report"], "Only parent report"], marker="o", color="red", linestyle="", fillstyle="full", markersize=10)
    ax.plot(optimal_ns["Free assessments only parent report"], df.loc[optimal_ns["Free assessments only parent report"], "Free assessments only parent report"], marker="o", color="red", linestyle="", fillstyle="none", markersize=10)
    
    # Add legend
    ax.legend(loc="lower right")
    
    return ax

def plot_saturation_plots(dfs, optimal_ns):
    # Plot saturation plots for each diagnosis, one plot per diagnosis, 4 curves per plot (one for each assessment subset), make as square as possible
    diags = list(dfs.keys())
    fig, axes = plt.subplots(int(np.ceil(len(diags)/3)), 3, figsize=(20, 20)) #calculate grid for nb of diags such that it is square
    # Add gap between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes = axes.flatten()

    # Set ylim
    for ax in axes:
        ax.set_ylim([0.5, 1.0])

    for i, diag in enumerate(diags):
        axes[i] = plot_saturation_plot(dfs[diag], optimal_ns.loc[diag], axes[i], diag)

    # Remove empty plots
    for i in range(len(diags), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    plt.savefig("output/viz/saturation_plots.png", bbox_inches="tight", dpi=600)

def make_averages_for_saturation_plots(dfs, optimal_ns):
    # Make df with average values for each number of features (without deprecated df.append)
    df_averages = pd.concat(list(dfs.values()), keys=list(dfs.keys()))
    df_averages = df_averages.groupby(level=1).mean()
    print(df_averages)

    # Make df with average values of optimal # of features for each assessment subset
    print(optimal_ns)
    optimal_ns = optimal_ns.mean().astype(int)
    print(optimal_ns)

    return df_averages, optimal_ns

def plot_average_saturation_plot(dfs, optimal_ns):
    # Get average value at each number of features among diags for all, free, only parent report, and free and only parent report

    # Make df with average values for each number of features
    df_averages, optimal_ns = make_averages_for_saturation_plots(dfs, optimal_ns)

    # Plot saturation plot for one diagnosis, 4 curves (one for each assessment subset): "All assessments", "Free assessments", "Only parent report", "Free assessments, only parent report"
    # x-axis: # of features, y-axis: AUROC

    plt.figure(figsize=(10, 8))
    plt.title("Average AUROC of ML model on # of items in best subscale")
    plt.xlabel("# of features")
    plt.ylabel("AUROC")

    # Set ylim
    plt.ylim([0.5, 1.0])

    # Plot curves
    plt.plot(df_averages.index, df_averages["All assessments"], label="All assessments", linestyle="-", color="orange")
    plt.plot(df_averages.index, df_averages["Free assessments"], label="Non-proprietary assessments", linestyle="--", color="orange")
    plt.plot(df_averages.index, df_averages["Only parent report"], label="Only parent report", linestyle="-", color="red")
    plt.plot(df_averages.index, df_averages["Free assessments only parent report"], label="Non-proprietary assessments, only parent report", linestyle="--", color="red")

    # Highlight dots with optimal n of features (same column names), empty dot for free, filled dot for all, orange for parent and sr assessments, red for only parent report
    plt.plot(optimal_ns["All assessments"], df_averages.loc[optimal_ns["All assessments"], "All assessments"], marker="o", color="orange", linestyle="", fillstyle="full", markersize=10)
    plt.plot(optimal_ns["Free assessments"], df_averages.loc[optimal_ns["Free assessments"], "Free assessments"], marker="o", color="orange", linestyle="", fillstyle="none", markersize=10)
    plt.plot(optimal_ns["Only parent report"], df_averages.loc[optimal_ns["Only parent report"], "Only parent report"], marker="o", color="red", linestyle="", fillstyle="full", markersize=10)
    plt.plot(optimal_ns["Free assessments only parent report"], df_averages.loc[optimal_ns["Free assessments only parent report"], "Free assessments only parent report"], marker="o", color="red", linestyle="", fillstyle="none", markersize=10)

    # Add legend
    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig("output/viz/average_saturation_plot.png", bbox_inches="tight", dpi=600)

def main():

    # Read data performance tables
    data_reader = DataReader()

    compare_orig_subsets_df = data_reader.read_data("compare_orig_vs_subsets")
    compare_orig_subsets_learning_df = data_reader.read_data("compare_orig_vs_subsets_learning")
    sum_scores_df = data_reader.read_data("sum_score_aurocs")
    learning_improvement_df = data_reader.read_data("learning_improvements")
    saturation_dfs = data_reader.read_data("saturation")

    # Read thresholds data from diagnosis_predictor_data (all assessments)
    thresholds_filename = "Diag.Specific Learning Disorder with Impairment in Reading (test).csv"
    thresholds_df = data_reader.read_data(
        data_type="thresholds",
        params=["parent_and_sr", "multiple_assessments", "all_assessments", "only_learning_diags"],
        filename=thresholds_filename)
    
    # Rephrase diags to shorter names
    compare_orig_subsets_df = compare_orig_subsets_df.rename(index=DIAGNOSIS_DICT)
    compare_orig_subsets_learning_df = compare_orig_subsets_learning_df.rename(index=DIAGNOSIS_DICT)
    sum_scores_df = sum_scores_df.rename(index=DIAGNOSIS_DICT)
    learning_improvement_df = learning_improvement_df.rename(index=DIAGNOSIS_DICT)
    saturation_dfs = {DIAGNOSIS_DICT[x]:y for x,y in saturation_dfs.items() if x in DIAGNOSIS_DICT.keys()}

    # Make all columns except those that start with ROC AUC into ints (if not NA) (bug in pd)
    for col in compare_orig_subsets_df.columns:
        if not "AUC" in col and not "score" in col and not "Best subscale" in col:
            compare_orig_subsets_df[col] = compare_orig_subsets_df[col].astype('Int64')
    for col in sum_scores_df.columns:
        if not "AUC" in col and not "AUROC" and not "score" in col and not "Best subscale" in col:
            sum_scores_df[col] = sum_scores_df[col].astype('Int64')

    #plot_eval_orig(compare_orig_subsets_df, filename="ROC_AUC_all_features.png", plot_free_assessments=False)
    #plot_eval_orig(compare_orig_subsets_learning_df, filename="ROC_AUC_all_features_learning.png", plot_free_assessments=False)

    #plot_manual_vs_ml(compare_orig_subsets_df)
    #plot_manual_vs_ml(compare_orig_subsets_df, plot_free_assessments=True)

    #plot_opt_num_features(compare_orig_subsets_df, filename="ROC_AUC_optimal_vs_all_features.png", plot_free_assessments=False)
    #plot_opt_num_features(compare_orig_subsets_learning_df, filename="ROC_AUC_optimal_vs_all_features_learning.png", plot_free_assessments=False)

    #plot_thresholds(thresholds_df, thresholds_filename.split(".")[0])

    #plot_sum_scores_vs_subscales(sum_scores_df)
    #plot_sum_scores_vs_subscales(sum_scores_df, sum_scores_free_df)

    #plot_group_bar_plots_for_subsets(compare_orig_subsets_df, sum_scores_df)

    #plot_learning_improvements(learning_improvement_df)
    plot_learning_improvements_bars(learning_improvement_df)
    plot_average_learning_improvements(learning_improvement_df)

    #opt_n_features_per_diag = get_opt_n_features_per_diag(compare_orig_subsets_df)
    #plot_saturation_plots(saturation_dfs, opt_n_features_per_diag)
    #plot_average_saturation_plot(saturation_dfs, opt_n_features_per_diag)

if __name__ == "__main__":
    main()