from data_reading import DataReader

import pandas as pd
from sklearn.metrics import roc_auc_score

class SumScorer():
    MAX_SUBSCALE_LENGTH = 27

    def __init__(self, params):
        self.params = params
        
    def make_sum_scores(self):
        # Read data
        self._read_data()
        
        # Make sum scores
        self._make_sum_scores()

        # Make AUROCS
        self._make_aurocs()
        
        # Save AUROCs
        self._save_aurocs()

    def _read_data(self):
        data_reader = DataReader()
        # Read item_level data
        self.item_lvl = data_reader.read_data(data_type = "item_lvl", 
                                         params = self.params)
        
        # Read estimators on subsets
        self.estimators_on_subsets = data_reader.read_data(data_type = "estimators_on_subsets",
                                                        params = self.params)
        self.diags = list(self.estimators_on_subsets.keys())
        
        # Read best manual subscales table
        self.manual_subscale_scores = data_reader.read_data(data_type = "manual_scoring")

    def _make_sum_scores(self):
        coefficients = self._make_coefficients_dfs()
        screener_lengths = self._make_screener_lengths()

        # Make sum scores
        self.sum_scores = {}
        for diag in self.diags:
            print(diag)
            screener_length = screener_lengths[diag]
            coef_df = coefficients[diag][screener_length]
            
            # Make sum score
            sum_score = self._make_sum_score(coef_df)
            self.sum_scores[diag] = sum_score

    def _make_aurocs(self):
        # Make AUROCs
        self.aurocs = {}
        for diag in self.diags:
            sum_scores = self.sum_scores[diag]
            auroc = roc_auc_score(self.item_lvl[diag], sum_scores)
            self.aurocs[diag] = auroc

    def _save_aurocs(self):
        # Make a df with AUROCs for sum scores, add lenght of best subscale, and performance of best subscale
        aurocs_df = pd.DataFrame.from_dict(self.aurocs, orient="index")
        aurocs_df.columns = ["AUROC"]
        aurocs_df = aurocs_df.merge(self.manual_subscale_scores[['Best subscale', "# of items in best subscale", "Best subscale score"]], left_index=True, right_index=True)
        aurocs_df = aurocs_df.sort_values(by="Best subscale score", ascending=False)

        # Save to csv
        aurocs_df.to_csv(f"output/sum_score_aurocs_{'_'.join(self.params)}.csv")

    def _make_sum_score(self, coef_df):
        # Sum up responses from self.item_lvl to items with a positive coefficient, and subtract responses from items with a negative coefficient, 
        # ignore magnitude of coefficient
        to_add_up = coef_df[coef_df["coef"] > 0].index.tolist()
        to_subtract = coef_df[coef_df["coef"] < 0].index.tolist()

        # Drop items that have range of values > 6
        to_add_up = [item for item in to_add_up if self.item_lvl[item].max() - self.item_lvl[item].min() <= 6]
        to_subtract = [item for item in to_subtract if self.item_lvl[item].max() - self.item_lvl[item].min() <= 6]

        print("to_add_up", to_add_up)

        sum_score = self.item_lvl[to_add_up].sum(axis=1) - self.item_lvl[to_subtract].sum(axis=1)
        return sum_score

    def _make_coefficients_dfs(self):
        coef_dfs = {}
        for diag in self.estimators_on_subsets.keys():
            coef_dfs[diag] = {}
            for subset in self.estimators_on_subsets[diag].keys():

                pipeline = self.estimators_on_subsets[diag][subset]
                estimator = self._get_estimator_from_pipeline(pipeline)
                estimator_base_model_name = self._get_base_model_name_from_estimator(estimator)
                feature_names = pipeline.named_steps["simpleimputer"].feature_names_in_

                # If model doesn't have coeffieicents, make values empty
                if estimator_base_model_name not in ["logisticregression", "svc"]:
                    raise ValueError("Model doesn't have coefficients: ", estimator_base_model_name)

                if estimator_base_model_name == "logisticregression":
                    coef = estimator.coef_[0]
                else:
                    coef = estimator.coef_[0]
                
                # Make df with coefficients
                coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
                coef_df = coef_df.set_index("feature")
                coef_dfs[diag][subset] = coef_df

        return coef_dfs
    
    def _make_screener_lengths(self):
        # Get # of items in the best subscale column value for each diag
        screener_lengths = {}
        for diag in self.manual_subscale_scores.index:
            screener_lengths[diag] = self.manual_subscale_scores.loc[diag, "# of items in best subscale"]
            if screener_lengths[diag] > self.MAX_SUBSCALE_LENGTH:
                screener_lengths[diag] = self.MAX_SUBSCALE_LENGTH
        
        return screener_lengths

    def _get_estimator_from_pipeline(self, pipeline):
        return pipeline.steps[-1][1]

    def _get_base_model_name_from_estimator(self, estimator):
        return estimator.__class__.__name__.lower()
    
    def _get_base_model_name_from_pipeline(self, pipeline):
        return self._get_base_model_name_from_estimator(self._get_estimator_from_pipeline(pipeline))


if __name__ == "__main__":
    sum_scorer = SumScorer(params = ["multiple_assessments", "all_assessments", "learning_and_consensus_diags"])
    sum_scorer.make_sum_scores()

    sum_scorer = SumScorer(params = ["multiple_assessments", "free_assessments", "learning_and_consensus_diags"])
    sum_scorer.make_sum_scores()