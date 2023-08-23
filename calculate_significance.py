import numpy as np
from scipy.stats import mannwhitneyu

from data_reading import DataReader

class SignificanceCalculator:
    alpha = 0.05

    def __init__(self):
        pass

    def calculate_significance(self, method, data, features):
        if method == "mann-whitney":
            statistic, p_value = mannwhitneyu(data[features[0]], data[features[1]])
            self._print_significance(p_value)
            return p_value
        else:
            raise ValueError("Method not recognized: ", method)
        
    def _print_significance(self, p_value):
        if p_value < self.alpha:
            print("\nSignificant difference: ", p_value, " < ", self.alpha)
        else:
            print("\nNo significant difference: ", p_value, " >= ", self.alpha)

if __name__ == "__main__":

    data_reader = DataReader()
    data = data_reader.read_data(data_type="compare_orig_vs_subsets")

    significance_calculator = SignificanceCalculator()
    method = "mann-whitney"

    # Compare means of "AUC all features all assessments parent and sr" and "Best subscale score"
    features = ["AUC all features free assessments only parent report", "Best subscale score"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, data, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features all assessments only parent report"
    features = ["AUC all features all assessments parent and sr", "AUC all features all assessments only parent report"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, data, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments parent and sr"
    features = ["AUC all features all assessments parent and sr", "AUC all features free assessments parent and sr"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, data, features)

    # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments only parent report"
    features = ["AUC all features all assessments parent and sr", "AUC all features free assessments only parent report"]
    print(f"Comparing: {features}:")
    significance_calculator.calculate_significance(method, data, features)