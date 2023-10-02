# Read txt files from a specified folder with formatted jsons and merge them 
# into one json file, under a key with the name of the file.
# Need to copy the feature-subsets folder from reports/evaluate_models_on_feature_subsets
# to input/subsets/ and rename it to all/ or free/ depending on the input
# assessments

import os
import json

# Path to the folder with txt files
path_all = 'input/subsets/all/'
path_free = 'input/subsets/free/'

result = {
    "All assessments": {},
    "Only non-proprietary assessments": {},
}
new_result = {
    "All assessments": {},
    "Only non-proprietary assessments": {},
}

# Iterate through all files in the folder
for filename in os.listdir(path_all):
    if filename.endswith('.txt'):
        result["All assessments"][filename[:-4]] = {}
        with open(path_all + filename, 'r') as f:
            result["All assessments"][filename[:-4]] = json.load(f)

# Remove AUROC
for diag in result["All assessments"]:
    new_result["All assessments"][diag] = {}
    for subset in result["All assessments"][diag]:
        new_result["All assessments"][diag][f'{subset} items'] = result["All assessments"][diag][subset][1]

# Iterate through all files in the folder
for filename in os.listdir(path_free):
    if filename.endswith('.txt'):
        result["Only non-proprietary assessments"][filename[:-4]] = {}
        with open(path_free + filename, 'r') as f:
            result["Only non-proprietary assessments"][filename[:-4]] = json.load(f)

# Remove AUROC
for diag in result["Only non-proprietary assessments"]:
    new_result["Only non-proprietary assessments"][diag] = {}
    for subset in result["Only non-proprietary assessments"][diag]:
        new_result["Only non-proprietary assessments"][diag][f'{subset} items'] = result["Only non-proprietary assessments"][diag][subset][1]

output = 'output/subsets/subsets.json'
with open(output, 'w') as f:
    json.dump(new_result, f, indent=4)

# Make a dict with only the last subset from each file
last_subset = {
    "All assessments": {},
    "Only non-proprietary assessments": {},
}
for diag in new_result["All assessments"]:
    last_subset["All assessments"][diag] = new_result["All assessments"][diag][list(new_result["All assessments"][diag].keys())[-1]]
for diag in new_result["Only non-proprietary assessments"]:
    last_subset["Only non-proprietary assessments"][diag] = new_result["Only non-proprietary assessments"][diag][list(new_result["Only non-proprietary assessments"][diag].keys())[-1]]

output = 'output/subsets/last_subset.json'
with open(output, 'w') as f:
    json.dump(last_subset, f, indent=4)

print('Done!')

