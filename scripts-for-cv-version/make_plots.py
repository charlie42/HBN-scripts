import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys, os, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_reading import DataReader

FIG_PATH = "output/viz/cv/"

def drop_duplicated_from_legend(ax0):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

def plot_bar_plot(df, title):
    df.plot(type="bar", title=title)

def plot_box_and_jitter_plot(df, title, cols, filename_base):
    plt.figure(figsize=(2, 5))
    #plt.title(title)

    for position, col_name in enumerate(cols):
        plt.boxplot(df[col_name], positions=[position], showfliers=False, vert=False)
        plt.scatter(df[col_name], np.ones(len(df[col_name]))*position)

    plt.yticks(range(len(cols)), cols, ha="right")
    plt.xlim([0.5, 0.9])

    plt.savefig(FIG_PATH+filename_base+".png", bbox_inches="tight", dpi=600)

def add_box_plot_bottom(ax, df, col, color, position):
    box_width = 0.8
    print(df[col])
    ax.boxplot(df[col], positions=[position], showfliers=False, vert=False, widths=box_width)
    ax.scatter(df[col], np.ones(len(df[col]))*position, color=color, alpha=0.8)

def make_subp_scatter_multiple_y_vals(df, col, color, label):
    y_vals = df.index
    y_positions = range(0, len(y_vals))
    print(col)
    for x, y in zip(df[col], y_positions):
        print(x, y, [y] * len(x))
        plt.scatter(x, [y] * len(x), label=label, edgecolors=color, marker='o', facecolors='none')

def plot_indiv_cv(df, labels, box_labels, title, filename_base):
    plt.figure(figsize=(8, 5))

    plt.scatter(df[labels[0]], df.index, label=labels[0], color='blue')
    plt.scatter(df[labels[2]], df.index, label=labels[2], color='red')
    
    make_subp_scatter_multiple_y_vals(df, labels[1], "red", labels[1])
    
    plt.xlim([0.5, 0.9])
    plt.legend()
    drop_duplicated_from_legend(plt)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    plt.savefig(FIG_PATH+filename_base+".png", bbox_inches="tight", dpi=600)

def plot_box_cv(df, labels, box_labels, title, filename_base):
    plt.figure(figsize=(8, 5))


    fig, ax = plt.subplots()

    
    c = "#1f77b4"
    ax.boxplot(df[labels[1]], vert=False,
                patch_artist=True,
                boxprops=dict(facecolor="#17becf", color=c),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color="#2ca02c"),
                zorder=0)
    ax.set_yticklabels(df.index)

    ax.scatter(df[labels[0]], range(1, len(df.index)+1), label=labels[0], color='red', zorder=1)

    
    #plt.xlim([0.5, 0.9])
    plt.legend()
    drop_duplicated_from_legend(plt)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    #plt.tight_layout()

    plt.savefig(FIG_PATH+filename_base+".png", bbox_inches="tight", dpi=600)
    plt.close()

def plot_box_delta(df, filename_base):
    plt.figure(figsize=(1, 5))
    df.boxplot(column="Delta ML", widths=0.6)
    plt.savefig(FIG_PATH+filename_base+".png", bbox_inches="tight", dpi=600)
                   

if __name__ == "__main__":
    # Plot mean cv auroc across diags vs existing assessments, boxplot
    # manual_vs_mean_ml_df = pd.read_csv("output/cv/manual_vs_mean_ml.csv", 
    #                                    index_col="Diag",
    #                                    keep_default_na=False, # To prevent interpreting "None" as NaN
    #                                    na_values=['NaN'])

    # plot_box_and_jitter_plot(
    #     manual_vs_mean_ml_df, 
    #     "Performance of existing subscales vs trained models",
    #     ["Score of best scale", "Mean ML score at N"],
    #     "manual_vs_mean_ml"
    # )

    # For each diag plot plot scatter plot with red point for existing 
    # assessment, and transparent blue points for auc of each fold 
    from ast import literal_eval
    manual_vs_cv_ml_df = pd.read_csv("output/cv/manual_vs_cv_ml.csv",
                                      index_col="Diag", 
                                      keep_default_na=False, # To prevent interpreting "None" as NaN
                                      na_values=['NaN'],
                                      converters={
                                          "CV ML scores at N": literal_eval,
                                          "CV sum scores at N": literal_eval,} # To read list as list and not str
                                      )
    
    diags = list(manual_vs_cv_ml_df.index)

    # print("DEBUG ORDER", manual_vs_mean_ml_df, manual_vs_mean_ml_df.index, manual_vs_mean_ml_df["Score of best scale"], 
    #       pd.Series(manual_vs_mean_ml_df["Score of best scale"], index=diags),
    #       pd.Series(manual_vs_mean_ml_df["Score of best scale"]),
    #       list(manual_vs_mean_ml_df["Score of best scale"]))
    
    #print(type(diags), type(next(iter(diags))), type(list(manual_vs_mean_ml_df.index)), type(list(manual_vs_mean_ml_df.index)[0]))

    
    # plot_indiv_cv(
    #     manual_vs_cv_ml_df,
    #     labels = [
    #         "Score of best scale",
    #         "CV ML scores at N",
    #         "Median ML score at N",
    #     ],
    #     box_labels = ["Score of best scale", "Mean ML score at N"],
    #     title="Performance of existing subscales vs trained models",
    #     filename_base="manual_vs_cv_ml_scatter"
    # )

    plot_box_cv(
        manual_vs_cv_ml_df,
        labels = [
            "Score of best scale",
            "CV ML scores at N",
            "Median ML score at N",
        ],
        box_labels = ["Score of best scale", "Mean ML score at N"],
        title="Performance of existing subscales vs trained models",
        filename_base="manual_vs_cv_ml_box"
    )

    plot_box_delta(
        manual_vs_cv_ml_df,
        filename_base="delta_box"
        )