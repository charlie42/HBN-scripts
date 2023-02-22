items = ["C3SR,C3SR_33",
	"PreInt_EduHx,strength_english",
	"C3SR,C3SR_13",
	"NIH_Scores,NIH7_List",
	"PreInt_EduHx,repeated_grades",
	"C3SR,C3SR_38",
	"SCARED_SR,SCARED_SR_21",
	"CCSC,CCSC_46",
	"PreInt_EduHx,weakness_english",
	"PreInt_EduHx,strength_math",
	"CIS_SR,CIS_SR_06",
	"CIS_SR,CIS_SR_09",
	"PreInt_DevHx,complications",
	"CIS_SR,CIS_SR_13",
	"PreInt_DevHx,puberty",
	"SCARED_SR,SCARED_SR_10",
	"SCARED_SR,SCARED_SR_11"

    "NIH_Scores,NIH7_List",
	"PSI,PSI_18",
	"PreInt_EduHx,weakness_math",
	"PreInt_EduHx,repeated_grades",
	"PreInt_EduHx,recent_grades",
	"NIH_Scores,NIH7_Flanker",
	"APQ_SR,APQ_SR_30",
	"C3SR,C3SR_09",
	"SRS,SRS_63",
	"SRS,SRS_57",
	"APQ_SR,APQ_SR_01"

    "NIH_Scores,NIH7_List",
	"NIH_Scores,NIH7_Card",
	"PreInt_EduHx,recent_grades",
	"PreInt_DevHx,temp_02",
	"Basic_Demos,Age",
	"CBCL,CBCL_11",
	"SDQ,SDQ_09",
	"CBCL,CBCL_03",
	"CCSC,CCSC_28",
	"CBCL,CBCL_66",
	"CBCL,CBCL_89",
	"SCQ,SCQ_27",
	"SRS,SRS_31",
	"APQ_SR,APQ_SR_15",
	"WHODAS_SR,WHODAS_SR_09",
	"CBCL,CBCL_28",
	"DTS,DTS_14",
	"CCSC,CCSC_17",
	"SCQ,SCQ_35",
	"PSI,PSI_18",
	"C3SR,C3SR_24",
	"ICU_P,ICU_P_17",
	"ESWAN,SocAnx_04A",
	"PreInt_DevHx,labor_dur_WAS_MISSING",
	"PreInt_DevHx,complications",
	"PreInt_DevHx,preg_dur",
	"WHODAS_SR,WHODAS_SR_03",
	"SRS,SRS_09",
	"SWAN,SWAN_10",
	"APQ_P,APQ_P_04",
	"PreInt_DevHx,growth_concerns",
	"SCQ,SCQ_10",
	"ICU_P,ICU_P_20",
	"ICU_P,ICU_P_13",
	"CBCL,CBCL_16",
	"APQ_SR,APQ_SR_11A_WAS_MISSING",
	"Basic_Demos,Sex",
	"SympChck,CSC_55hC",
	"PCIAT,PCIAT_17",
	"PreInt_EduHx,strength_science",
	"CBCL,CBCL_79",
	"SympChck,CSC_29C",
	"CBCL,CBCL_108",
	"SCARED_SR,SCARED_SR_19",
	"SCQ,SCQ_25",
	"SympChck,CSC_40P",
	"SRS,SRS_21",
	"SCARED_SR,SCARED_SR_32",
	"PreInt_DevHx,temp_07",
	"CBCL,CBCL_80",
	"APQ_SR,APQ_SR_04A_WAS_MISSING",
	"C3SR,C3SR_20",
	"SCQ,SCQ_40",
	"SWAN,SWAN_09",
	"SympChck,CSC_08P",
	"C3SR,C3SR_19",
	"SRS,SRS_15",
	"PreInt_EduHx,absent_late",
	"SympChck,CSC_42P",
	"CBCL,CBCL_06",
	"APQ_SR,APQ_SR_14",
	"SWAN,SWAN_01",
	"PreInt_EduHx,afterschoolteams",
	"CIS_SR,CIS_SR_09",
	"PreInt_EduHx,strength_history",
	"APQ_P,APQ_P_03",
	"NIH_Scores,NIH7_Complete",
	"APQ_SR,APQ_SR_07",
	"CBCL,CBCL_90",
	"PSI,PSI_33",
	"APQ_P,APQ_P_24",
	"RBS,RBS_14",
	"CIS_P,CIS_P_02",
	"SympChck,CSC_03P",
	"CBCL,CBCL_39",
	"MFQ_P,MFQ_P_05",
	"PCIAT,PCIAT_16",
	"MFQ_P,MFQ_P_30",
	"SympChck,CSC_55iP",
	"PreInt_EduHx,tutor",
	"CCSC,CCSC_SUPOA",
	"CCSC,CCSC_01",
	"PreInt_DevHx,hospital_dur",
	"ASSQ,ASSQ_10",
	"ICU_P,ICU_P_10",
	"CBCL,CBCL_87",
	"SCQ,SCQ_11",
	"SCQ,SCQ_21",
	"SRS,SRS_12",
	"RBS,RBS_11"
]

# Get unique assessments
unique_assessments = list(set([x.split(",")[0] for x in items]))
print(unique_assessments)