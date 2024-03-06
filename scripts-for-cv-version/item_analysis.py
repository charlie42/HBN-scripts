from joblib import load
import pandas as pd 

def transform_lists_to_sets(lists):
    sets = []
    for list in lists:
        sets.append(set(list))
    return sets

if __name__ == "__main__":
    perf_dict = load("input/scores_objects_lr_debug.joblib")
    for diag in perf_dict:
        print(diag)
        lists = perf_dict[diag]["rfe_features"]
        sets = transform_lists_to_sets(lists)
        common_items = set.intersection(*sets)
        print(f"{len(common_items)} common items:", common_items)    
        for subset in perf_dict[diag]["perf_on_features"]:
            if subset == 10:
                print(subset)
                lists = perf_dict[diag]["perf_on_features"][subset]["features"]
                sets = transform_lists_to_sets(lists)
                # How many elements are common:
                common_items = set.intersection(*sets)
                print(f"{len(common_items)} common items:", common_items)    