import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import helpers 

from data_reading import DataReader

if __name__ == "__main__":

    path = helpers.get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data/data/create_datasets/", 
                                            ["first_assessment_to_drop",
                                                "only_free_assessments__0"])
    df = pd.read_csv(path+'item_lvl.csv')

    # ASD:
    #       "(0.46*) ASSQ,ASSQ_11: uses language freely but fails to make adjustments to fit social contexts or the needs of different listeners - 0=No, 1=Somewhat, 2=Yes",
    #       "(0.38*) SRS,SRS_29: 29. Is regarded by other children as odd or weird. - 0= Not True, 1= Sometimes True, 2= Often True, 3= Almost Always True",
    #       "(0.36*) SympChck,CSC_51P: 51. Often has a difficult time making eye contact(Past) - 0=No, 1=Yes",
    #       "(0.40*) SRS,SRS_28: 28. Thinks or talks about the same thing over and over. - 0= Not True, 1= Sometimes True, 2= Often True, 3= Almost Always True",
    #       "(0.28*) SCQ,SCQ_15: 15. Does she/he ever have any mannerisms or odd ways of moving her/his hands or fingers, such as flapping or moving her/his fingers in front or her/his eyes? - 0= No, 1= Yes",
    #       "(-0.23*) SDQ,SDQ_18: Often lies or cheats - 0=Not True, 1=Somewhat True, 2=Certainly True",
    #       "(-0.32*) Basic_Demos,Sex:  - ",
    #       "(0.28*) SRS,SRS_42: 42. Seems overly sensitive to sounds, textures, or smells. - 0= Not True, 1= Sometimes True, 2= Often True, 3= Almost Always True",
    #       "(-0.30*) CBCL,CBCL_87: 87. Sudden changes in mood or feelings - 0=Not true, 1=Somewhat or sometimes true, 2=Very true or often true",
    #       "(0.29*) ASSQ,ASSQ_16: can be with other children but only on his/her terms - 0=No, 1=Somewhat, 2=Yes",
    #       "(0.21*) PreInt_DevHx,temp_11: Problems with social relatedness - 0= No, 1= Yes",
    #       "(0.33*) SRS,SRS_48: 48. Has a sense of humor, understands jokes. - 0= Not True, 1= Sometimes True, 2= Often True, 3= Almost Always True"

    cols_to_add = ["ASSQ,ASSQ_11", "SRS,SRS_29", "SympChck,CSC_51P", "SRS,SRS_28", "SCQ,SCQ_15", "SRS,SRS_42", "ASSQ,ASSQ_16", "PreInt_DevHx,temp_11", "SRS,SRS_48"]
    cols_to_subtract = ["SDQ,SDQ_18", "CBCL,CBCL_87"]
    diag = "Diag.Autism Spectrum Disorder"

    diag_df = df[cols_to_add + cols_to_subtract + [diag]]

    diag_df["sum_score"] = diag_df[cols_to_add].sum(axis=1) - diag_df[cols_to_subtract].sum(axis=1)

    auc = roc_auc_score(diag_df[diag], diag_df["sum_score"]) 
    print(auc) #0.9, 0.91 with coefficients

    # MDD:
    #   "(0.62*) Basic_Demos,Age:  - ",
    #   "(0.54*) SympChck,CSC_01P: 1. Feels sad and down most days for at least 1 week(past) - 0=No, 1=Yes",
    #   "(0.46*) CBCL,CBCL_103: 103. Unhappy, sad, or depressed - 0=Not true, 1=Somewhat or sometimes true, 2=Very true or often true",
    #   "(0.25*) Basic_Demos,Sex:  - ",
    #   "(0.21*) CBCL,CBCL_54: 54. Overtired without good reason - 0=Not true, 1=Somewhat or sometimes true, 2=Very true or often true",
    #   "(0.21*) Barratt,financialsupport:  - ",
    #   "(0.20*) PreInt_DevHx,puberty: Has your child shown adult sexual body development (puberty)? - 0= No, 1= Yes",
    #   "(0.20*) PreInt_EduHx,absent_other: Other - 0= 0 days, 1= 1 day, 2= 2 days, 3= 3 or 4 days, 4= 5 or 6 days, 5= 7 or more days"

    cols_to_add = ["SympChck,CSC_01P", "CBCL,CBCL_103", "CBCL,CBCL_54"]
    cols_to_subtract = ["Barratt,financialsupport", "PreInt_DevHx,puberty", "PreInt_EduHx,absent_other"]
    diag = "Diag.Major Depressive Disorder"

    diag_df = df[cols_to_add + cols_to_subtract + [diag]]

    diag_df["sum_score"] = diag_df[cols_to_add].sum(axis=1) - diag_df[cols_to_subtract].sum(axis=1)

    auc = roc_auc_score(diag_df[diag], diag_df["sum_score"])

    print(auc) # 0.65, 0.89 with coefficients (due to variables of different magnitude (PreInt_EduHx,absent_other)?)

    # ADHD Combined:
    #   "(0.33*) SDQ,SDQ_02: Restless, overactive, cannot stay still for long - 0=Not True, 1=Somewhat True, 2=Certainly True",
    #   "(0.25*) SWAN,SWAN_16: 16. Reflects on questions (controls blurting out answers) - -3= Far above average,-2= Above average,-1= Slightly above average,0= Average,1= Slightly below average,2= Above average,3= Far above average",
    #   "(0.28*) SWAN,SWAN_11: 11. Stays seated (when required by class rules or social conventions) - -3= Far above average,-2= Above average,-1= Slightly above average,0= Average,1= Slightly below average,2= Above average,3= Far above average",
    #   "(0.23*) SWAN,SWAN_17: 17. Awaits turn (stands in line and takes turns) - -3= Far above average,-2= Above average,-1= Slightly above average,0= Average,1= Slightly below average,2= Above average,3= Far above average",
    #   "(0.24*) SympChck,CSC_37P: 37. Often acts before thinking(Past) - 0=No, 1=Yes",
    #   "(0.20*) SRS,SRS_56: 56. Walks in between two people who are talking. - 0= Not True, 1= Sometimes True, 2= Often True, 3= Almost Always True",
    #   "(0.20*) CBCL,CBCL_104: 104. Unusually loud - 0=Not true, 1=Somewhat or sometimes true, 2=Very true or often true",
    #   "(-0.11*) SCARED_P,SCARED_P_41: 41. My child is shy - 0=Not True or Hardly Ever True, 1=Somewhat True or Sometimes True, 2=Very True or Often True",
    #   "(-0.10*) Basic_Demos,Sex:  - "

    cols_to_add = ["SDQ,SDQ_02", "SWAN,SWAN_16", "SWAN,SWAN_11", "SWAN,SWAN_17", "SympChck,CSC_37P", "SRS,SRS_56", "CBCL,CBCL_104"]
    cols_to_subtract = ["SCARED_P,SCARED_P_41"]
    diag = "Diag.ADHD-Combined Type"

    diag_df = df[cols_to_add + cols_to_subtract + [diag]]

    diag_df["sum_score"] = diag_df[cols_to_add].sum(axis=1) - diag_df[cols_to_subtract].sum(axis=1)

    auc = roc_auc_score(diag_df[diag], diag_df["sum_score"])

    print(auc) #0.87, 0.86 with coefficients