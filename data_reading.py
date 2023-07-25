import pandas as pd
import os, datetime

class DataReader:
    def __init__(self):
        self.base_path = "../diagnosis_predictor_data/"
        
    def read_data(self, data_type, params=[], file_filter=""):
        self.data_path = self._generate_data_path(data_type, params, file_filter)

        if data_type == "item_lvl":
            return self._read_item_lvl()
        elif data_type == "eval_orig":
            return self._read_eval_orig()
        elif data_type == "eval_subsets":
            return self._read_eval_subsets()
        elif data_type == "make_ds":
            return self._read_make_ds()
        else:
            raise ValueError("data_type not recognized: ", data_type)
        
    def _read_item_lvl(self):
        return pd.read_csv(self.data_path + "item_lvl.csv")
    
    def _read_eval_orig(self):
        return pd.read_csv(self.data_path + "performance_table_all_features_test_set.csv")
    
    def _read_eval_subsets(self):
        return pd.read_csv(self.data_path + "performance_table_all_subsets.csv")
    
    def _read_make_ds(self):
        return pd.read_csv(self.data_path + "dataset_stats.csv")
    
    def _generate_data_path(self, data_type, params, file_filter):
        if data_type == "item_lvl":
            return self._get_newest_non_empty_dir_in_dir(self.base_path + "data/create_datasets/", 
                                                         params, file_filter)
        elif data_type == "eval_orig":
            return self._get_newest_non_empty_dir_in_dir(self.base_path + "reports/evaluate_original_features/", 
                                                         params, file_filter)
        elif data_type == "eval_subsets":
            return self._get_newest_non_empty_dir_in_dir(self.base_path + "reports/identify_feature_subsets/", 
                                                         params, file_filter)
        elif data_type == "make_ds":
            return self._get_newest_non_empty_dir_in_dir(self.base_path + "reports/make_datasets/", 
                                                         params, file_filter)
        else:
            raise ValueError("data_type not recognized: ", data_type)
        
    def _get_newest_non_empty_dir_in_dir(self, path, extra_filters=[], file_filter=""):
        dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
        
        if extra_filters:
            for extra_filter in extra_filters:
                non_empty_dir_names = [d for d in non_empty_dir_names if extra_filter in d]
        if file_filter:
            non_empty_dir_names = [d for d in non_empty_dir_names if os.path.isfile(path+d+'/'+file_filter)]
        
        # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
        timestamps = [d.split("___")[0] for d in non_empty_dir_names]
        timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
        if not timestamps:
            print("\nNo non-empty dir found in: ", path, "among: ", non_empty_dir_names, "\ntimestamps:", timestamps)
            return None
        
        newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
        return path + newest_dir_name + "/"

        