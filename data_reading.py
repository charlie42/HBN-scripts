import pandas as pd
import os, datetime
from joblib import load
from dsutils import file_utils

class DataReader:
    
    BASE_PATH = "../diagnosis_predictor_data_archive/"
    DATA_TYPE_TO_PATH_MAPPING = {
        "item_lvl": "data/create_datasets/",
        "subscale_scores": "data/create_datasets/",
        "eval_orig": "reports/evaluate_original_models/",
        "eval_subsets": "reports/evaluate_models_on_feature_subsets/",
        "eval_subsets_one_subset": "reports/evaluate_models_on_feature_subsets/",
        "make_ds": "reports/create_data_reports/",
        "estimators_on_subsets": "models/identify_feature_subsets/",
        "thresholds": "reports/evaluate_models_on_feature_subsets/",
        "manual_scoring": None,
        "compare_orig_vs_subsets": None,
        "compare_orig_vs_subsets_learning": None,
        "what_improves_LD": None,
        "sum_score_aurocs": None,
        "sum_score_aurocs_free": None,
        "learning_improvements": None,
        "saturation": None,
    }
    PARAM_TO_PATH_MAPPING = {
        "only_parent_report": "only_parent_report__1",
        "parent_and_sr": "only_parent_report__0",
        "multiple_assessments": "first_assessment_to_drop",
        "all_assessments": "only_free_assessments__0",
        "free_assessments": "only_free_assessments__1",
        "only_learning_diags": "learning?__1",
        "learning_and_consensus_diags": "learning?__0",
        "NIH": "NIH?__1",
        "no_NIH": "NIH?__0",
        #"fix_n_all": "fix_n__1", # rename to fix_n_all__1 when new data is generated
        "fix_n_all": "fix_n_all__1", 
        "fix_n_learning": "fix_n_learning__1",
    }
    FILE_FILTER_MAPPING = {
        "eval_orig_test_set_file": "performance_table_all_features_test_set.csv",
    }

    def __init__(self):
        pass                  
        
    def read_data(self, data_type, params=[], file_filter="", filename=""):

        self.params = [self.PARAM_TO_PATH_MAPPING[param] for param in params]
        self.file_filter = self.FILE_FILTER_MAPPING[file_filter] if file_filter else ""
        self.data_type_path = self.DATA_TYPE_TO_PATH_MAPPING[data_type]

        if self.data_type_path is not None:
            self.data_path = self._generate_data_path()
            print("Reading data from: ", self.data_path)

        if data_type == "item_lvl":
            return self._read_item_lvl()
        elif data_type == "subscale_scores":
            return self._read_subscale_scores()
        elif data_type == "eval_orig":
            return self._read_eval_orig()
        elif data_type == "eval_subsets":
            return self._read_eval_subsets()
        elif data_type == "make_ds":
            return self._read_make_ds()
        elif data_type == "manual_scoring":
            return self._read_manual_scoring()
        elif data_type == "compare_orig_vs_subsets":
            return self._read_compare_orig_vs_subsets()
        elif data_type == "compare_orig_vs_subsets_learning":
            return self._read_compare_orig_vs_subsets_learning()
        elif data_type == "what_improves_LD":
            return self._read_what_improves_LD()
        elif data_type == "estimators_on_subsets":
            return self._read_estimators_on_subsets()
        elif data_type == "thresholds":
            return self._read_thresholds(filename)
        elif data_type == "sum_score_aurocs":
            return self._read_sum_score_aurocs()
        elif data_type == "eval_subsets_one_subset":
            return self._read_eval_subsets(one_subset=True)
        elif data_type == "learning_improvements":
            return self._read_learning_improvements(all_diags=True)
        elif data_type == "saturation":
            return self._read_saturation()
        else:
            raise ValueError("data_type not recognized: ", data_type)
        
    def _read_item_lvl(self):
        return pd.read_csv(self.data_path + "item_lvl_new.csv")
    
    def _read_subscale_scores(self):
        return pd.read_csv(self.data_path + "subscale_scores.csv")
    
    def _read_eval_orig(self):
        return pd.read_csv(self.data_path + "performance_table_all_features_test_set.csv", index_col=0)
    
    def _read_eval_subsets(self, one_subset=False):
        if one_subset:
            return pd.read_csv(self.data_path + "perf-on-subsets-test-set-one-threshold-optimal-nb-features.csv", index_col=1)
        else:
            return pd.read_csv(self.data_path + "auc-on-subsets-test-set.csv", index_col=0)
    
    def _read_make_ds(self):
        return pd.read_csv(self.data_path + "dataset_stats.csv", index_col=0)
    
    def _read_manual_scoring(self):
        #return pd.read_csv("output/manual_scoring_analysis/manual_subsale_scores_vs_ml.csv", index_col=0)
        return pd.read_csv("output/manual_scoring_analysis/manual_subsale_scores_vs_ml_learning.csv", index_col=0)
    
    def _read_compare_orig_vs_subsets(self):
        return pd.read_csv("output/compare_orig_vs_subsets.csv", index_col=0)
    
    def _read_compare_orig_vs_subsets_learning(self):
        return pd.read_csv("output/compare_orig_vs_subsets_learning.csv", index_col=0)
    
    def _read_what_improves_LD(self):
        return pd.read_csv("output/what_improves_LD.csv", index_col=0)
    
    def _read_estimators_on_subsets(self):
        return load(self.data_path + "estimators-on-subsets.joblib")
    
    def _read_thresholds(self, filename):
        return pd.read_csv(self.data_path + "sens-spec-on-subsets-test-set-optimal-nb-features/" + filename, index_col=0)
    
    def _read_sum_score_aurocs(self):
        return pd.read_csv("output/sum_score_merged.csv", index_col=0)
    
    def _read_learning_improvements(self, all_diags=False):
        if all_diags:
            return pd.read_csv("output/learning_improvements_all_diags.csv", index_col=0)
        else:
            return pd.read_csv("output/learning_improvements.csv", index_col=0)
        
    def _read_saturation(self):
        return file_utils.read_dict_of_dfs_from_dir("output/saturation_dfs/", index_col="Number of features")
    
    def _generate_data_path(self):
        return self._get_newest_non_empty_dir_in_dir(
            self.BASE_PATH + self.data_type_path, 
            self.params, 
            self.file_filter)
        
    def _get_newest_non_empty_dir_in_dir(self, path, extra_filters=[], file_filter=""):
        dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if dir_names == []:
            raise ValueError("No dirs found in: ", path)
            
        non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
        if non_empty_dir_names == []:
            raise ValueError("No non-empty dirs found in: ", path)
        
        if extra_filters:
            for extra_filter in extra_filters:
                non_empty_dir_names = [d for d in non_empty_dir_names if extra_filter in d]
        if non_empty_dir_names == []:
            raise ValueError("No non-empty dirs found in: ", path, " with extra_filters: ", extra_filters)
        
        if file_filter:
            non_empty_dir_names = [d for d in non_empty_dir_names if os.path.isfile(path+d+'/'+file_filter)]
        if non_empty_dir_names == []:
            raise ValueError("No non-empty dirs found in: ", path, " with file_filter: ", file_filter)
        
        # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
        timestamps = [d.split("___")[0] for d in non_empty_dir_names]
        timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
        
        if not timestamps:
            raise ValueError("No timestamps found in: ", path, "among: ", non_empty_dir_names)
        
        newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
        return path + newest_dir_name + "/"