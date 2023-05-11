all_assessments = ['Basic_Demos', 'PreInt_EduHx', 'PreInt_DevHx', 'NIH_Scores', 'WIAT', 'SympChck', 'SCQ', 'Barratt', 'ASSQ', 'ARI_P', 'SDQ', 'SWAN', 'SRS', 'CBCL', 'ICU_P', 'APQ_P', 'PCIAT', 'DTS', 'ESWAN', 'MFQ_P', 'APQ_SR', 'WISC', 'WHODAS_P', 'CIS_P', 'PSI', 'RBS', 'PhenX_Neighborhood', 'WHODAS_SR', 'CIS_SR', 'SCARED_SR', 'C3SR', 'CCSC']

used_assessments = ['PreInt_DevHx', 'SRS', 'SCQ', 'NIH_Scores', 'SCARED_SR', 'WHODAS_SR', 'ASSQ', 'DTS', 'CCSC', 'APQ_SR', 'SympChck', 'RBS', 'SWAN', 'PCIAT', 'CBCL', 'PreInt_EduHx', 'APQ_P', 'MFQ_P', 'ESWAN', 'CIS_SR', 'CIS_P', 'C3SR', 'SDQ', 'Basic_Demos', 'ICU_P', 'PSI']

# Get the difference between the two lists
diff = list(set(all_assessments) - set(used_assessments))
print(diff)