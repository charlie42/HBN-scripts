import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_admin_cols(full):
    # Remove uninteresting columns
    columns_to_drop = []

    column_suffixes_to_drop = ["Administration", "Data_entry", "Days_Baseline", "START_DATE", "Season", "Site", "Study", "Year", "Commercial_Use", "Release_Number"]
    for suffix in column_suffixes_to_drop:
        cols_w_suffix = [x for x in full.columns if suffix in x]
        columns_to_drop.extend(cols_w_suffix)

    present_columns_to_drop = full.filter(columns_to_drop)
    full = full.drop(present_columns_to_drop, axis = 1)
    return full 

def get_ID_from_EID(full, EID_cols):

    # Get only EID cols
    full_for_EID_check = full[EID_cols]

    # In EID cols df, fill missing EIDs with EIDs from other questionnaires 
    full_for_EID_check = full_for_EID_check.ffill(axis=1).bfill(axis=1)

    # Drop lines with different EID within one row
    full = full[full_for_EID_check.eq(full_for_EID_check.iloc[:, 0], axis=0).all(1)]

    # Fill ID field with the first non-null questionnaire-specific EID
    full["ID"] = full_for_EID_check.iloc[:, 0]

    return full

# Drop rows with underscores in ID (NDARZZ007YMP_1, NDARAA075AMK_Visit_1)
def drop_rows_w_underscore_in_id(full):

    rows_with_underscore_in_id = full[full["ID"].str.contains("_")]
    non_empty_columns_in_underscore = rows_with_underscore_in_id.columns[
        ~rows_with_underscore_in_id.isna().all()
    ].tolist() 
    non_empty_questionnaires_in_underscore = set([x.split(",")[0] for x in non_empty_columns_in_underscore])
    
    non_empty_questionnaires_in_underscore.remove("Identifiers")
    non_empty_questionnaires_in_underscore.remove("ID")
    full_wo_underscore = full[~full["ID"].str.contains("_")]

    # Drop questionnaires present in rows with underscores from data ({'DailyMeds', 'TRF', 'TRF_P', 'TRF_Pre'})
    for questionnaire in non_empty_questionnaires_in_underscore:
        full_wo_underscore = full_wo_underscore.drop(full_wo_underscore.filter(regex=(questionnaire+",")), axis=1)

    return full_wo_underscore

def prepare_data(full):

    # Replace NaN (currently ".") values with np.nan
    full = full.replace(".", np.nan)

    # Drop first row (doesn't have ID)
    full = full.iloc[1: , :]

    # Drop empty columns
    full = full.dropna(how='all', axis=1)

    full = remove_admin_cols(full)

    # Get ID columns (contain quetsionnaire names, e.g. 'ACE,EID', will be used to check if an assessment is filled)
    EID_cols = [x for x in full.columns if ",EID" in x]

    # Get ID col from EID cols
    full = get_ID_from_EID(full, EID_cols)

    full_wo_underscore = drop_rows_w_underscore_in_id(full)

    # Drop questionnaires present in rows with underscores from data from list of ID columns
    EID_cols = [x for x in EID_cols if 'TRF' not in x]
    EID_cols = [x for x in EID_cols if 'DailyMeds' not in x]

    # Drop TRF and DailyMeds from data
    full_wo_underscore = full_wo_underscore.drop(full_wo_underscore.filter(regex=("TRF,")), axis=1)
    full_wo_underscore = full_wo_underscore.drop(full_wo_underscore.filter(regex=("DailyMeds,")), axis=1)

    return full_wo_underscore

def get_assessment_answer_count(full_wo_underscore, EID_cols):
    assessment_answer_counts = full_wo_underscore[EID_cols].count().sort_values(ascending=False).to_frame()
    assessment_answer_counts["Ratio"] = assessment_answer_counts[0]/full_wo_underscore["ID"].nunique()*100
    assessment_answer_counts.columns = ["N of Participants", "% of Participants Filled"]
    return assessment_answer_counts

def get_relevant_id_cols_by_popularity(assessment_answer_counts, relevant_assessment_list):

    relevant_EID_list = [x+",EID" for x in relevant_assessment_list]

    # Get list of assessments sorted by popularity
    EID_columns_by_popularity = assessment_answer_counts.index

    # Get relevant ID columns sorted by popularity    
    EID_columns_by_popularity = [x for x in EID_columns_by_popularity if x in relevant_EID_list]

    return EID_columns_by_popularity

def get_cumul_number_of_examples_df(full_wo_underscore, EID_columns_by_popularity):
    cumul_number_of_examples_list = []
    for i in range(1, len(EID_columns_by_popularity)+1):
        columns = EID_columns_by_popularity[0:i] # top i assessments
        cumul_number_of_examples = full_wo_underscore[columns].notnull().all(axis=1).sum()
        min_age_among_non_null = full_wo_underscore[full_wo_underscore[columns].notnull().all(axis=1)]["Basic_Demos,Age"].astype(float).min()
        cumul_number_of_examples_list.append([cumul_number_of_examples, [x.split(",")[0] for x in columns], min_age_among_non_null])
    cumul_number_of_examples_df = pd.DataFrame(cumul_number_of_examples_list)
    cumul_number_of_examples_df.columns = ("Respondents", "Assessments", "Min Age")
    cumul_number_of_examples_df["N of Assessments"] = cumul_number_of_examples_df["Assessments"].str.len()
    cumul_number_of_examples_df["Last Assessment"] = cumul_number_of_examples_df["Assessments"].str[-1]
    return cumul_number_of_examples_df

def plot_comul_number_of_examples(cumul_number_of_examples_df, data_statistics_dir):
    plt.figure(figsize=(16,8))
    plt.xticks(cumul_number_of_examples_df["N of Assessments"])
    plt.scatter(cumul_number_of_examples_df["N of Assessments"], cumul_number_of_examples_df["Respondents"])
    # Add vertical lines for each point
    for i in range(0, len(cumul_number_of_examples_df)):
        plt.axvline(x=cumul_number_of_examples_df["N of Assessments"][i], color='gray', linestyle='--')
    plt.xlabel("Number of Assessments")
    plt.ylabel("Number of Respondents")
    plt.title("Cumulative Number of Respondents with Complete Data")
    plt.savefig(data_statistics_dir+'figures/cumul_assessment_distrib.png')  

def get_columns_until_dropped(full_wo_underscore, EID_columns_until_dropped):
    columns_until_dropped = []
    assessments_until_dropped = [x.split(",")[0]+"," for x in EID_columns_until_dropped]
    for assessment in assessments_until_dropped:
        columns = [column for column in full_wo_underscore.columns if column.startswith(assessment)]
        columns_until_dropped.extend(columns)
    return columns_until_dropped

def get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped, columns_until_dropped):
    diag_colunms = ["Diagnosis_ClinicianConsensus,DX_01", "Diagnosis_ClinicianConsensus,DX_02", "Diagnosis_ClinicianConsensus,DX_03", 
        "Diagnosis_ClinicianConsensus,DX_04", "Diagnosis_ClinicianConsensus,DX_05", "Diagnosis_ClinicianConsensus,DX_06", 
        "Diagnosis_ClinicianConsensus,DX_07", "Diagnosis_ClinicianConsensus,DX_08", "Diagnosis_ClinicianConsensus,DX_09", 
        "Diagnosis_ClinicianConsensus,DX_10"]
    data_up_to_dropped = full_wo_underscore.loc[full_wo_underscore[EID_columns_until_dropped].dropna(how="any").index][columns_until_dropped+["ID"]+diag_colunms]

    return data_up_to_dropped

def get_cumul_df(data, relevent_assessments_list):
    EID_cols = [x for x in data.columns if ",EID" in x]

    # Check how many people filled each assessments
    assessment_answer_counts = get_assessment_answer_count(data, EID_cols)

    # Get relevant ID columns sorted by popularity
    EID_columns_by_popularity = get_relevant_id_cols_by_popularity(assessment_answer_counts, relevent_assessments_list)    

    # Get cumulative distribution of assessments: number of people who took all top 1, top 2, top 3, etc. popular assessments 
    cumul_number_of_examples_df = get_cumul_number_of_examples_df(data, EID_columns_by_popularity)

    return cumul_number_of_examples_df

def get_columns_until_dropped(full_wo_underscore, EID_columns_until_dropped):
    columns_until_dropped = []
    assessments_until_dropped = [x.split(",")[0]+"," for x in EID_columns_until_dropped]
    for assessment in assessments_until_dropped:
        columns = [column for column in full_wo_underscore.columns if column.startswith(assessment)]
        columns_until_dropped.extend(columns)
    return columns_until_dropped

def get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped, columns_until_dropped):
    diag_colunms = ["Diagnosis_ClinicianConsensus,DX_01", "Diagnosis_ClinicianConsensus,DX_02", "Diagnosis_ClinicianConsensus,DX_03", 
        "Diagnosis_ClinicianConsensus,DX_04", "Diagnosis_ClinicianConsensus,DX_05", "Diagnosis_ClinicianConsensus,DX_06", 
        "Diagnosis_ClinicianConsensus,DX_07", "Diagnosis_ClinicianConsensus,DX_08", "Diagnosis_ClinicianConsensus,DX_09", 
        "Diagnosis_ClinicianConsensus,DX_10"]
    data_up_to_dropped = full_wo_underscore.loc[full_wo_underscore[EID_columns_until_dropped].dropna(how="any").index][columns_until_dropped+["ID"]+diag_colunms]

    return data_up_to_dropped
    
def get_missing_values_df(data_up_to_dropped):
    missing_report_up_to_dropped = data_up_to_dropped.isna().sum().to_frame(name="Amount missing")
    missing_report_up_to_dropped["Persentage missing"] = missing_report_up_to_dropped["Amount missing"]/data_up_to_dropped["ID"].nunique() * 100
    missing_report_up_to_dropped = missing_report_up_to_dropped[~missing_report_up_to_dropped.index.str.contains("Diagnosis_ClinicianConsensus")] # remove dx because it's expected to be missing
    missing_report_up_to_dropped = missing_report_up_to_dropped[missing_report_up_to_dropped["Persentage missing"] > 0]
    return missing_report_up_to_dropped[missing_report_up_to_dropped["Persentage missing"] > 0].sort_values(ascending=False, by="Amount missing")

def remove_cols_w_missing_over_n(data_up_to_dropped, n, missing_values_df):
    cols_to_remove = list(missing_values_df[missing_values_df["Persentage missing"] > n].index)
    data_up_to_dropped = data_up_to_dropped.drop(cols_to_remove, axis=1)
    return data_up_to_dropped

def add_missingness_markers(data_up_to_dropped, n, missing_values_df):
    missing_cols_to_mark = list(missing_values_df[(missing_values_df["Persentage missing"] <= 40) & (missing_values_df["Persentage missing"] > n)].index)
    for col in missing_cols_to_mark:
        data_up_to_dropped[col+ "_WAS_MISSING"] = data_up_to_dropped[col].isna()
    return data_up_to_dropped

def transform_dx_cols(data_up_to_dropped):
    og_diag_cols = [x for x in data_up_to_dropped.columns if "DX_" in x]

    # Get list of diagnoses
    diags = []
    for col in og_diag_cols:
        diags.extend(list(data_up_to_dropped[col].value_counts().index))
    diags = list(set(diags))
    diags.remove(' ')

    # Make new columns
    for diag in diags:
        data_up_to_dropped["Diag." + util.remove_chars_forbidden_in_file_names(diag)] = (data_up_to_dropped[og_diag_cols] == diag).any(axis=1)
        
    # Drop original diag columns
    data_up_to_dropped = data_up_to_dropped.drop(og_diag_cols, axis=1)

    return data_up_to_dropped

def transform_devhx_eduhx_cols(data_up_to_dropped):

    list_of_preg_symp_cols = [x for x in data_up_to_dropped.columns if "preg_symp" in x]
    
    # If any of the preg_symp columns are 1, then the preg_symp column is 1
    data_up_to_dropped["preg_symp"] = (data_up_to_dropped[list_of_preg_symp_cols] == 1).any(axis=1)

    # Drop original preg_symp columns
    data_up_to_dropped = data_up_to_dropped.drop(list_of_preg_symp_cols, axis=1) 

    data_up_to_dropped = data_up_to_dropped.drop(["PreInt_EduHx,NeuroPsych", "PreInt_EduHx,IEP", "PreInt_EduHx,learning_disability", "PreInt_EduHx,EI", "PreInt_EduHx,CPSE"], axis=1)

    return data_up_to_dropped

def get_data_up_to_assessment(data, first_assessment_to_drop, relevent_assessments_list):

    EID_cols = [x for x in data.columns if ",EID" in x]
    assessment_answer_counts = get_assessment_answer_count(data, EID_cols)

    # Get relevant ID columns sorted by popularity
    EID_columns_by_popularity = get_relevant_id_cols_by_popularity(assessment_answer_counts, relevent_assessments_list)    

    # List of most popular assessments until the first one from the drop list 
    EID_columns_until_dropped = [x for x in EID_columns_by_popularity[:EID_columns_by_popularity.index(first_assessment_to_drop+",EID")]]

    # Get data up to the dropped assessment
    # Get only people who took the most popular assessments until the first one from the drop list 
    columns_until_dropped = get_columns_until_dropped(data, EID_columns_until_dropped)
    data_up_to_dropped = get_data_up_to_dropped(data, EID_columns_until_dropped, columns_until_dropped)

    # Remove EID columns: not needed anymore
    data_up_to_dropped = data_up_to_dropped.drop(EID_columns_until_dropped, axis=1)
    
    return data_up_to_dropped

def replace_with_dict_otherwise_nan(data, cols, dict):
    for col in cols:
        data.loc[~data[col].isin(dict.keys()), col] = np.nan # Replace any other possible values with NaN
        data[col] = data[col].astype(str).replace(dict)
    return data