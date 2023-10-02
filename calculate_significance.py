import numpy as np
from scipy.stats import mannwhitneyu

from data_reading import DataReader

class SignificanceCalculator:
    alpha = 0.05

    def __init__(self, data):
        self.data = self._remove_none_and_any_diags(data)

    def calculate_significance(self, method, features):
        if method == "mann-whitney":
            statistic, p_value = mannwhitneyu(data[features[0]], self.data[features[1]])
            self._print_significance(p_value)
            return p_value
        else:
            raise ValueError("Method not recognized: ", method)
        
    def _print_significance(self, p_value):
        if p_value < self.alpha:
            print("\nSignificant difference: ", p_value, " < ", self.alpha)
        else:
            print("\nNo significant difference: ", p_value, " >= ", self.alpha)

    def _remove_none_and_any_diags(self, data):
        try:
            result = data.drop("Diag.No Diagnosis Given", axis=0)
        except KeyError:
            result = data
            print("No 'Diag.No Diagnosis Given' in data")
        try:
            result = result.drop("Diag.Any Diag", axis=0)
        except KeyError:
            print("No 'Diag.Any Diag' in data")
        return result


if __name__ == "__main__":

    data_reader = DataReader()

    # Do tests for ML models 
    print("\nML Models:\n")

    data = data_reader.read_data(data_type="compare_orig_vs_subsets")
    print(data.columns)

    significance_calculator = SignificanceCalculator(data)
    method = "mann-whitney"

    # Compare means of "AUC all features all assessments parent and sr" and "Best subscale score"
    features = ["AUC all features all assessments parent and sr", "Best subscale score"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features all assessments only parent report"
    features = ["AUC all features all assessments parent and sr", "AUC all features all assessments only parent report"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments parent and sr"
    features = ["AUC all features all assessments parent and sr", "AUC all features free assessments parent and sr"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments only parent report"
    features = ["AUC all features all assessments parent and sr", "AUC all features free assessments only parent report"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Do tests for sum-scores of subsets
    print("\nSum-scores of subsets:\n")

    data = data_reader.read_data(data_type="sum_score_aurocs")
    print(data.columns)

    significance_calculator = SignificanceCalculator(data)
    method = "mann-whitney"

    # Compare means of "AUC all features all assessments parent and sr" and "Best subscale score"
    features = ["AUROC all assessments parent and sr", "Best subscale score"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features all assessments only parent report"
    features = ["AUROC all assessments parent and sr", "AUROC all assessments only parent report"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments parent and sr"
    features = ["AUROC all assessments parent and sr", "AUROC free assessments parent and sr"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments only parent report"
    features = ["AUROC all assessments parent and sr", "AUROC free assessments only parent report"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Do tests for learning improvements
    print("\nLearning improvements:\n")

    data = data_reader.read_data(data_type="learning_improvements")
    print(data.columns)

    ## Only learning diags
    print("DEBUG", data.index)
    learning_diags = [
        'Diag.Specific Learning Disorder with Impairment in Reading (test)',
        'Diag.NVLD without reading condition (test)',
        'Diag.Specific Learning Disorder with Impairment in Reading',
        'Diag.Specific Learning Disorder with Impairment in Written Expression (test)',
        'Diag.NVLD (test)', 
        'Diag.Specific Learning Disorder with Impairment in Mathematics (test)',
    ]

    data = data.loc[learning_diags]

    significance_calculator = SignificanceCalculator(data)
    method = "mann-whitney"

    # Compare means of "original" and "more assessments"
    features = ["original", "more assessments"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "original" and "more assessments and NIH"
    features = ["original", "more assessments and NIH"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)

    # Compare means of "more assessments " and "more assessments and NIH"
    features = ["more assessments", "more assessments and NIH"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, features)


