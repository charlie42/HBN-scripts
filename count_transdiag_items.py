import os, datetime
import pandas as pd

# Calculate for how many diagnoses each item appears in each subset
# Rows: item, cols: subset, values: count

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    
    # Filter only those with first_dropped_assessment in name (not single assessment), only_free_assessments == 0
    non_empty_dir_names = [d for d in non_empty_dir_names if "first_dropped_assessment" in d and not "only_free_assessments__1" in d and "debug_mode__False" in d]
    
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

path = "../diagnosis_predictor_data/reports/identify_feature_subsets/"
dir = get_newest_non_empty_dir_in_dir(path)
print("Reading reports from: ", dir)

from joblib import load

# Load data
feature_subsets = load(dir + "feature-subsets.joblib")

# Get list of all items
all_items = []
for diag in feature_subsets.keys():
    for subset in feature_subsets[diag].keys():
        all_items += feature_subsets[diag][subset]
all_items = list(set(all_items))

# Get list of all diagnoses
diags = list(feature_subsets.keys())
print("diags", diags)

# Get list of all subsets
subsets = list(feature_subsets.values())[0].keys()

# Count for how many diagnoses each item appears in each subset
item_counts = {}
for subset in subsets:
    item_counts[subset] = {}
    for item in all_items:
        item_counts[subset][item] = 0
        for diag in diags:
            if item in feature_subsets[diag][subset]:
                item_counts[subset][item] += 1

# Make a table with rows = items, cols = subsets, values = count
item_counts_table = pd.DataFrame.from_dict(item_counts)

# Append item name to item ID
item_names = pd.read_csv("../diagnosis_predictor/references/item-names.csv")
item_names["ID"] = item_names["datadic"] + "," + item_names["keys"]

# Merge item names to item_counts_table
item_counts_table = item_names.merge(item_counts_table, left_on="ID", right_index=True)
item_counts_table = item_counts_table.drop(["datadic", "keys"], axis=1)

# Move ID column to be the first
cols = item_counts_table.columns.tolist()
cols.pop(cols.index("ID"))
print(type(["ID"]), type(cols))
item_counts_table = item_counts_table[["ID"] + cols]

# Sort by count in 126 subset
item_counts_table = item_counts_table.sort_values(30, ascending=False)
print(item_counts_table)

# Save to csv
item_counts_table.to_csv("output/item_counts.csv", index=False)