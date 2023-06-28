import pandas as pd

from helpers import get_newest_non_empty_dir_in_dir

# Calculate for how many diagnoses each item appears in each subset
# Rows: item, cols: subset, values: count

path = "../diagnosis_predictor_data/reports/identify_feature_subsets/"
dir = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                             "only_free_assessments__0",
                                             "learning?__0",]) 
                                                                                                    
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
item_counts_table = item_counts_table[["ID"] + cols]

# Only print 1, 2, 3, 5, 10, 20, 35 subsets: 
cols_to_keep = [col for col in item_counts_table.columns.tolist() if not str(col).isdigit() or int(col) in [1, 2, 3, 5, 10, 20, 30, 41]]
item_counts_table = item_counts_table[cols_to_keep]

# Sort by count in 35th subset
item_counts_table = item_counts_table.sort_values(list(item_counts_table.columns)[-1], ascending=False) # Sort by last column
print(item_counts_table)

# Save to csv
item_counts_table.to_csv("output/item_counts.csv", index=False)