import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

class SignificanceCalculator:
    alpha = 0.05

    def __init__(self):
        pass

    def calculate_significance(self, data_1, data_2, method):
        """"
        Calculate significance of difference between means of two features
        """

        if method == "mann-whitney":
            statistic, p_value = mannwhitneyu(data_1, data_2)
            self._print_significance(p_value)
            return p_value
        else:
            raise ValueError("Method not recognized: ", method)
        
    def calculte_multitest(self, method, p_values):
        """"
        Correct p-values for multiple testing
        """
        return multipletests(p_values, alpha=self.alpha, method=method)[1]
        
    def _print_significance(self, p_value):
        if p_value < self.alpha:
            print("\nSignificant difference: ", p_value, " < ", self.alpha)
        else:
            print("\nNo significant difference: ", p_value, " >= ", self.alpha)

    def _remove_none_and_any_diags(self, data):
        """
        Remove "Diag.No Diagnosis Given" and "Diag.Any Diag" from data
        """
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

def make_data_list(data, col, is_list=False):
    if is_list:
        result = []
        for index, value in data[col].items():
            result += value
        return result
    else:
        return list(data[col])

if __name__ == "__main__":

    # Do tests for ML models 
    print("\nML Models:\n")
    ml_model_p_values = []

    from ast import literal_eval
    data = pd.read_csv(
        "output/cv/manual_vs_cv_ml.csv",
        converters={
            "CV ML scores at N": literal_eval,
            "CV sum scores at N": literal_eval,
        } # To read list as list and not str)
    )

    significance_calculator = SignificanceCalculator()
    method = "mann-whitney"

    data_1 = make_data_list(data, "Score of best scale")
    data_2 = make_data_list(data, "CV ML scores at N", is_list=True)
    print(f"Comparing: Score of best scale and ML scores at N:")
    ml_model_p_values.append(significance_calculator.calculate_significance(data_1, data_2, method))

    data_1 = make_data_list(data, "Score of best scale")
    data_2 = make_data_list(data, "CV sum scores at N", is_list=True)
    print(f"Comparing: Score of best scale and sum scores at N:")
    ml_model_p_values.append(significance_calculator.calculate_significance(data_1, data_2, method))

    # # Compare means of "AUC all features all assessments parent and sr" and "Best subscale score"
    # features = ["AUC all features all assessments parent and sr", "Best subscale score"]
    # print(f"Comparing: {features}:")
    # ml_model_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "AUC all features all assessments parent and sr" and "AUC all features all assessments only parent report"
    # features = ["AUC all features all assessments parent and sr", "AUC all features all assessments only parent report"]
    # print(f"Comparing: {features}:")
    # ml_model_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments parent and sr"
    # features = ["AUC all features all assessments parent and sr", "AUC all features free assessments parent and sr"]
    # print(f"Comparing: {features}:")
    # ml_model_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments only parent report"
    # features = ["AUC all features all assessments parent and sr", "AUC all features free assessments only parent report"]
    # print(f"Comparing: {features}:")
    # ml_model_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Correct p-values for multiple testing
    # print("\nCorrecting p-values for multiple testing:\n")
    # print("Before correction: ", ml_model_p_values)
    # ml_model_p_values_corrected = significance_calculator.calculte_multitest(
    #     method="bonferroni", 
    #     p_values=ml_model_p_values)
    # print("After correction: ", ml_model_p_values_corrected)


    # ################################################################


    # # Do tests for sum-scores of subsets
    # print("\nSum-scores of subsets:\n")
    # sum_scores_p_values = []

    # data = data_reader.read_data(data_type="sum_score_aurocs")

    # significance_calculator = SignificanceCalculator(data)
    # method = "mann-whitney"

    # # Compare means of "AUC all features all assessments parent and sr" and "Best subscale score"
    # features = ["AUROC all assessments parent and sr", "Best subscale score"]
    # print(f"Comparing: {features}:")
    # sum_scores_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "AUC all features all assessments parent and sr" and "AUC all features all assessments only parent report"
    # features = ["AUROC all assessments parent and sr", "AUROC all assessments only parent report"]
    # print(f"Comparing: {features}:")
    # sum_scores_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments parent and sr"
    # features = ["AUROC all assessments parent and sr", "AUROC free assessments parent and sr"]
    # print(f"Comparing: {features}:")
    # sum_scores_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "AUC all features all assessments parent and sr" and "AUC all features free assessments only parent report"
    # features = ["AUROC all assessments parent and sr", "AUROC free assessments only parent report"]
    # print(f"Comparing: {features}:")
    # sum_scores_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Correct p-values for multiple testing
    # print("\nCorrecting p-values for multiple testing:\n")
    # print("Before correction: ", sum_scores_p_values)
    # sum_scores_p_values_corrected = significance_calculator.calculte_multitest(
    #     method="bonferroni", 
    #     p_values=sum_scores_p_values)
    # print("After correction: ", sum_scores_p_values_corrected)


    # ################################################################


    # # Do tests for learning improvements
    # print("\nLearning improvements:\n")
    # learning_improvements_p_values = []

    # data = data_reader.read_data(data_type="learning_improvements")

    # ## Only learning diags
    # learning_diags = [
    #     'Diag.Specific Learning Disorder with Impairment in Reading (test)',
    #     'Diag.NVLD without reading condition (test)',
    #     'Diag.Specific Learning Disorder with Impairment in Reading',
    #     'Diag.Specific Learning Disorder with Impairment in Written Expression (test)',
    #     'Diag.NVLD (test)', 
    #     'Diag.Specific Learning Disorder with Impairment in Mathematics (test)',
    # ]

    # data = data.loc[learning_diags]

    # significance_calculator = SignificanceCalculator(data)
    # method = "mann-whitney"

    # # Compare means of "original" and "more assessments"
    # features = ["original", "more assessments"]
    # print(f"Comparing: {features}:")
    # learning_improvements_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "original" and "more assessments and NIH"
    # features = ["original", "more assessments and NIH"]
    # print(f"Comparing: {features}:")
    # learning_improvements_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Compare means of "more assessments " and "more assessments and NIH"
    # features = ["more assessments", "more assessments and NIH"]
    # print(f"Comparing: {features}:")
    # learning_improvements_p_values.append(significance_calculator.calculate_significance(method, features))

    # # Correct p-values for multiple testing
    # print("\nCorrecting p-values for multiple testing:\n")
    # print("Before correction: ", learning_improvements_p_values)
    # learning_improvements_p_values_corrected = significance_calculator.calculte_multitest(
    #     method="bonferroni", 
    #     p_values=learning_improvements_p_values)
    # print("After correction: ", learning_improvements_p_values_corrected)


