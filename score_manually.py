import pandas as pd
from sklearn.metrics import roc_auc_score

from helpers import get_newest_non_empty_dir_in_dir

numbers_of_items = {"SCQ,SCQ_Total": 40,
                    "ASSQ,ASSQ_Total": 27,
                    "ARI_P,ARI_P_Total_Score": 7,
                    "SWAN,SWAN_Total": 18,
                    "SRS,SRS_Total_T": 65,
                    "CBCL,CBCL_Total_T": 112,
                    "SCARED_P,SCARED_P_Total": 41,
                    "CIS_P,CIS_P_Score": 13,
                    "CIS_SR,CIS_SR_Score": 13,
                    "PCIAT,PCIAT_Total": 20,
                    "APQ_SR,APQ_SR_Total": 42,
                    "APQ_SR,APQ_SR_OPD": 42,
                    "APQ_P,APQ_P_Total": 42,
                    "APQ_P,APQ_P_OPD": 42,
                    "PSI,PSI_Total": 36,
                    "SWAN,SWAN_HY": 9,
                    "SWAN,SWAN_IN": 9,
                    "ICU_P,ICU_P_Total": 24,
                    "ICU_P,ICU_P_Callousness": 11,
                    "ICU_P,ICU_P_Uncaring": 8,
                    "ICU_P,ICU_P_Unemotional": 5,
                    "APQ_SR,APQ_SR_INV_D": 10, 
                    "APQ_SR,APQ_SR_INV_M": 10,
                    "APQ_SR,APQ_SR_PP": 6,
                    "APQ_SR,APQ_SR_PM": 10,
                    "APQ_SR,APQ_SR_ID": 6,
                    "APQ_SR,APQ_SR_CP": 3,
                    "APQ_P,APQ_P_INV": 10, 
                    "APQ_P,APQ_P_PP": 6,
                    "APQ_P,APQ_P_PM": 10,
                    "APQ_P,APQ_P_ID": 6,
                    "APQ_P,APQ_P_CP": 3,
                    "SRS,SRS_MOT_T": 11,
                    "SRS,SRS_AWR_T": 8,
                    "SRS,SRS_COG_T": 12,
                    "SRS,SRS_COM_T": 22,
                    "SRS,SRS_RRB_T": 12,
                    "SRS,SRS_SCI_T": 53, 
                    "SRS,SRS_DSMRRB_T": 12,
                    "CBCL,CBCL_WD_T": 8,
                    "CBCL,CBCL_AD_T": 13,
                    "CBCL,CBCL_AP_T": 10,
                    "CBCL,CBCL_AB_T": 18,
                    "CBCL,CBCL_RBB_T": 17,
                    "CBCL,CBCL_TP_T": 15,
                    "CBCL,CBCL_SC_T": 11,
                    "CBCL,CBCL_SP_T": 11,
                    "CBCL,CBCL_OP_T": 17,
                    "CBCL,CBCL_Int_T": 32,
                    "CBCL,CBCL_Ext_T": 35,
                    "CBCL,CBCL_C_T": 53, 
                    "SCARED_P,SCARED_P_GD": 9,
                    "SCARED_P,SCARED_P_PN": 13,
                    "SCARED_P,SCARED_P_SC": 7,
                    "SCARED_P,SCARED_P_SH": 4,
                    "SCARED_P,SCARED_P_SP": 8,
                    "PSI,PSI_DC_T": 10, # to confirm
                    "PSI,PSI_PCDI_T": 10, # to confirm
                    "PSI,PSI_PCDI_T": 10, # to confirm
                    "PSI,PSI_PD_T": 10, # to confirm
                    "C3SR,C3SR_AG_T": 10, # to confirm
                    "C3SR,C3SR_FR_T": 10, # to confirm
                    "C3SR,C3SR_HY_T": 10, # to confirm
                    "C3SR,C3SR_IN_T": 10, # to confirm
                    "C3SR,C3SR_LP_T": 10, # to confirm
                    "C3SR,C3SR_NI": 10, # to confirm
                    "C3SR,C3SR_PI": 10, # to confirm
                    "CCSC,CCSC_PFC": 5, # to confirm
                    "CCSC,CCSC_CDM": 5, # to confirm
                    "CCSC,CCSC_DPS": 5, # to confirm
                    "CCSC,CCSC_SU": 5, # to confirm
                    "CCSC,CCSC_AC": 5, # to confirm
                    "CCSC,CCSC_AA": 5, # to confirm
                    "CCSC,CCSC_REP": 5, # to confirm
                    "CCSC,CCSC_WT": 5, # to confirm
                    "CCSC,CCSC_PCR": 5, # to confirm
                    "CCSC,CCSC_CON": 5, # to confirm
                    "CCSC,CCSC_OPT": 5, # to confirm
                    "CCSC,CCSC_POS": 5, # to confirm
                    "CCSC,CCSC_REL": 5, # to confirm
                    "CCSC,CCSC_SS": 5, # to confirm
                    "CCSC,CCSC_SUPMF": 5, # to confirm
                    "CCSC,CCSC_SUPOA": 5, # to confirm
                    "CCSC,CCSC_SUPEER": 5, # to confirm
                    "CCSC,CCSC_SUPSIB": 5, # to confirm
                    }

def get_list_of_analysed_diags():
    print("Getting list of analysed diags")
    path = get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data_archive/reports/evaluate_models_on_feature_subsets/", [
        "only_parent_report__0",
        "first_assessment_to_drop", 
        "learning?__0",
        "only_free_assessments__0"])
    report = pd.read_csv(path + "auc-on-subsets-test-set.csv")
    
    diags = [x for x in report.columns if x.startswith("Diag.")]
    print("Analysed diags: ", diags)
 
    return diags

def find_auc_for_score_col(total_scores_data, subscale_scores_data, score_col, diag_col):
    if "Total" in score_col or "_OPD" in score_col or "CIS_P_Score" in score_col or "CIS_SR_Score" in score_col or "WHODAS_SR_Score" in score_col or "WHODAS_P_Score" in score_col: # Total scores
        auc = roc_auc_score(total_scores_data[diag_col], total_scores_data[score_col])    
    else:
        auc = roc_auc_score(subscale_scores_data[diag_col], subscale_scores_data[score_col])
    return auc

# Read prepared data

def read_subscales_data():

    #path = get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data_archive/data/create_datasets/", 
    #                                       ["first_assessment_to_drop",
    #                                        "only_free_assessments__0",
    #                                        "learning?__1"])
    path = "../diagnosis_predictor_data_archive/data/create_datasets/2023-11-10 10.04.10___only_parent_report__0___first_assessment_to_drop__CPIC___use_other_diags_as_input__0___only_free_assessments__0___learning?__1___NIH?__0___fix_n_all__0___fix_n_learning__0/"
                                           
    print("Reading input data from: ", path)
    total_scores_data = pd.read_csv(path+'total_scores.csv', index_col=0)
    subscale_scores_data = pd.read_csv(path+'subscale_scores.csv', index_col=0)

    print("total_scores_data: ", total_scores_data.shape)
    print("subscale_scores_data: ", subscale_scores_data.shape)

    return total_scores_data, subscale_scores_data

def get_score_cols(total_scores_data, subscale_scores_data):

    score_cols = ([x for x in total_scores_data.columns if not x.startswith("Diag") and not x.endswith("WAS_MISSING") and not "Barratt" in x and not "preg_symp" in x] 
    + [x for x in subscale_scores_data.columns if not x.startswith("Diag") and not x.endswith("WAS_MISSING")  and not "Barratt" in x and not "preg_symp" in x])

    return score_cols

# Get manual AUC scores for each diag and each scale/subscale
def get_manual_scores(total_scores_data, subscale_scores_data, diags):
    score_cols = get_score_cols(total_scores_data, subscale_scores_data)
    
    all_manual_scores = {}
    for diag_col in diags:
        print("Calculating AUC for: ", diag_col)
        scores_for_diag = []
        for score_col in score_cols:
            auc = find_auc_for_score_col(total_scores_data, subscale_scores_data, score_col, diag_col)
            scores_for_diag.append(auc)
        all_manual_scores[diag_col] = scores_for_diag

    all_manual_scores_df = pd.DataFrame.from_dict(all_manual_scores, orient='index', columns=score_cols)
    print(all_manual_scores_df)
    return all_manual_scores_df

# Make df with best scores for each diag (cols: Diag,Best score,AUC) 

def find_best_manual_score_for_diag(all_manual_scores_df, diags):
    best_manual_scores = []
    for diag_col in diags:
        best_manual_score = all_manual_scores_df.loc[diag_col].max()
        best_manual_score_col = all_manual_scores_df.loc[diag_col].idxmax()
        best_manual_scores.append([diag_col, best_manual_score_col, best_manual_score])
    best_manual_scores_df = pd.DataFrame(best_manual_scores, columns=["Diag","Best score","AUC"]).sort_values(by="AUC", ascending=False)
    print(best_manual_scores_df)
    best_manual_scores_df.to_csv("output/manual_scoring_analysis/best_manual_scores_learning.csv", float_format='%.3f')

    return best_manual_scores_df

# Compare with ML scores
def read_ml_scores(free=False, only_parent=False):
    path = "../diagnosis_predictor_data_archive/reports/evaluate_models_on_feature_subsets/"
    free_param = "only_free_assessments__1" if free else "only_free_assessments__0"
    only_parent_param = "only_parent_report__1" if only_parent else "only_parent_report__0"
    
    print("Reading ML scores from: ", path, " with params: ", free_param, only_parent_param)
    path_all_assessments = get_newest_non_empty_dir_in_dir(path, [
        only_parent_param,
        "first_assessment_to_drop",
        free_param,
        "learning?__0",
        "fix_n_all__1"
        ])
    print("Reading report from: ", path_all_assessments)
    ml_scores_all_assessments = pd.read_csv(path_all_assessments + "auc-on-subsets-test-set.csv")

    return ml_scores_all_assessments

def compare_ml_scores_with_best_manual_scores(best_manual_scores_df, 
                                              ml_scores_all_assessments_parent_and_sr, 
                                              ml_scores_free_assessments_parent_and_sr, 
                                              ml_scores_all_assessments_only_parent_report,
                                              ml_scores_free_assessments_only_parent_report,
                                              diags):
    ml_scores_at_num_features = {}
    for diag_col in diags:
        best_manual_score = best_manual_scores_df[best_manual_scores_df["Diag"] == diag_col]["AUC"].values[0]
        best_manual_score_subscale = best_manual_scores_df[best_manual_scores_df["Diag"] == diag_col]["Best score"].values[0]
        best_manual_score_subscale_short = best_manual_score_subscale.split(",")[1]
        number_of_items_in_best_manual_subscale = numbers_of_items[best_manual_score_subscale]

        number_of_items_to_check = number_of_items_in_best_manual_subscale if number_of_items_in_best_manual_subscale <= 27 else 27
        ml_score_at_number_of_items_of_best_manual_subscale_all_parent_and_sr = ml_scores_all_assessments_parent_and_sr[ml_scores_all_assessments_parent_and_sr["Number of features"] == number_of_items_to_check][diag_col].values[0]
        ml_score_at_number_of_items_of_best_manual_subscale_free_parent_and_sr = ml_scores_free_assessments_parent_and_sr[ml_scores_free_assessments_parent_and_sr["Number of features"] == number_of_items_to_check][diag_col].values[0]
        ml_score_at_number_of_items_of_best_manual_subscale_all_only_parent_report = ml_scores_all_assessments_only_parent_report[ml_scores_all_assessments_only_parent_report["Number of features"] == number_of_items_to_check][diag_col].values[0]
        ml_score_at_number_of_items_of_best_manual_subscale_free_only_parent_report = ml_scores_free_assessments_only_parent_report[ml_scores_free_assessments_only_parent_report["Number of features"] == number_of_items_to_check][diag_col].values[0]

        # Find number of items needed to reach performance of the best subscale    
        number_of_items_for_ml_score_of_best_manual_subscale_all_parent_and_sr = ml_scores_all_assessments_parent_and_sr[ml_scores_all_assessments_parent_and_sr[diag_col] >= best_manual_score]["Number of features"].min()
        number_of_items_for_ml_score_of_best_manual_subscale_free_parent_and_sr = ml_scores_free_assessments_parent_and_sr[ml_scores_free_assessments_parent_and_sr[diag_col] >= best_manual_score]["Number of features"].min()
        number_of_items_for_ml_score_of_best_manual_subscale_all_only_parent_report = ml_scores_all_assessments_only_parent_report[ml_scores_all_assessments_only_parent_report[diag_col] >= best_manual_score]["Number of features"].min()
        number_of_items_for_ml_score_of_best_manual_subscale_free_only_parent_report = ml_scores_free_assessments_only_parent_report[ml_scores_free_assessments_only_parent_report[diag_col] >= best_manual_score]["Number of features"].min()

        ml_scores_at_num_features[diag_col] = [
            best_manual_score_subscale_short, 
            best_manual_score, 
            number_of_items_in_best_manual_subscale, 
            ml_score_at_number_of_items_of_best_manual_subscale_all_parent_and_sr,
            number_of_items_for_ml_score_of_best_manual_subscale_all_parent_and_sr,
            ml_score_at_number_of_items_of_best_manual_subscale_free_parent_and_sr,
            number_of_items_for_ml_score_of_best_manual_subscale_free_parent_and_sr,
            ml_score_at_number_of_items_of_best_manual_subscale_all_only_parent_report, 
            number_of_items_for_ml_score_of_best_manual_subscale_all_only_parent_report,
            ml_score_at_number_of_items_of_best_manual_subscale_free_only_parent_report,
            number_of_items_for_ml_score_of_best_manual_subscale_free_only_parent_report,
        ]

    ml_scores_at_num_features_df = pd.DataFrame.from_dict(ml_scores_at_num_features, orient='index', columns=[
        "Best subscale", 
        "Best subscale score", 
        "# of items in best subscale", 
        "ML score at # of items of best subscale (all assessments)",
        "# of items to reach best subscale (all assessments)",
        "ML score at # of items of best subscale (free assessments)",
        "# of items to reach best subscale (free assessments)",
        "ML score at # of items of best subscale (only parent report)",
        "# of items to reach best subscale (only parent report)",
        "ML score at # of items of best subscale (free assessments, only parent report)",
        "# of items to reach best subscale (free assessments, only parent report)",
        ]).sort_values(by="Best subscale score", ascending=False)
    return ml_scores_at_num_features_df

def main():
    output_dir = "output/manual_scoring_analysis/"

    total_scores_data, subscale_scores_data = read_subscales_data() # Read prepared data: total and subscale scores and diagnoses from HBN data
    total_scores_data = total_scores_data.drop("ID", axis=1)
    subscale_scores_data = subscale_scores_data.drop("ID", axis=1)
    
    #diags = get_list_of_analysed_diags()
    #Diag.Oppositional Defiant Disorder,Diag.Generalized Anxiety Disorder,Diag.Autism Spectrum Disorder,Diag.No Diagnosis Given,Diag.ADHD-Combined Type,Diag.ADHD-Inattentive Type,Diag.Borderline Intellectual Functioning (test),Diag.Any Diag,Diag.Specific Learning Disorder with Impairment in Reading,
    #Diag.Processing Speed Deficit (test),Diag.Specific Learning Disorder with Impairment in Reading (test),Diag.NVLD without reading condition (test),Diag.Language Disorder,Diag.Social Anxiety (Social Phobia),Diag.NVLD (test),Diag.Specific Learning Disorder with Impairment in Mathematics (test),Diag.Specific Phobia,Diag.Specific Learning Disorder with Impairment in Written Expression (test)
    diags = [
        "Diag.Oppositional Defiant Disorder",
        "Diag.Generalized Anxiety Disorder",
        "Diag.Autism Spectrum Disorder",
        "Diag.No Diagnosis Given", ###
        "Diag.ADHD-Combined Type",
        "Diag.ADHD-Inattentive Type",
        "Diag.Borderline Intellectual Functioning (test)", ###
        "Diag.Any Diag",
        "Diag.Specific Learning Disorder with Impairment in Reading",
        "Diag.Processing Speed Deficit (test)", ###
        "Diag.Specific Learning Disorder with Impairment in Reading (test)",
        "Diag.NVLD without reading condition (test)",
        "Diag.Language Disorder", ###
        "Diag.Social Anxiety (Social Phobia)",
        "Diag.NVLD (test)",
        "Diag.Specific Learning Disorder with Impairment in Mathematics (test)",
        "Diag.Specific Phobia",
        "Diag.Specific Learning Disorder with Impairment in Written Expression (test)"
    ]

    all_manual_scores_df = get_manual_scores(total_scores_data, subscale_scores_data, diags)
    all_manual_scores_df.T.to_csv(output_dir + "manual_subscale_scores_learning.csv", float_format='%.3f')

    best_manual_scores_df = find_best_manual_score_for_diag(all_manual_scores_df, diags)

    ml_scores_all_assessments_parent_and_sr = read_ml_scores(free=False, only_parent=False)
    ml_scores_all_assessments_only_parent_report = read_ml_scores(free=False, only_parent=True)
    ml_scores_free_assessments_parent_and_sr = read_ml_scores(free=True, only_parent=False)
    ml_scores_free_assessments_only_parent_report = read_ml_scores(free=True, only_parent=True)

    print("DEBUG best_manual_scores_df")
    print(best_manual_scores_df)
    print("DEBUG ml_scores_all_assessments_parent_and_sr")
    print(ml_scores_all_assessments_parent_and_sr.columns)
    print("DEBUG ml_scores_all_assessments_only_parent_report")
    print(ml_scores_all_assessments_only_parent_report.columns)
    print("DEBUG ml_scores_free_assessments_parent_and_sr")
    print(ml_scores_free_assessments_parent_and_sr.columns)
    print("DEBUG ml_scores_free_assessments_only_parent_report")
    print(ml_scores_free_assessments_only_parent_report.columns)
    print(diags)

    comparison_table = compare_ml_scores_with_best_manual_scores(
        best_manual_scores_df, 
        ml_scores_all_assessments_parent_and_sr, 
        ml_scores_all_assessments_only_parent_report,
        ml_scores_free_assessments_parent_and_sr,
        ml_scores_free_assessments_only_parent_report, 
        diags)
    print(comparison_table)
    comparison_table.to_csv(output_dir + "manual_subsale_scores_vs_ml_learning.csv", float_format='%.3f')

if __name__ == "__main__":
    main()