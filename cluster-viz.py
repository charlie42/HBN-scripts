from joblib import load

score_dict = load("input/scores_objects_lr_debug.joblib")

for diag in score_dict:
    print(diag)
    for n in score_dict[diag]["perf_on_features"]:
        #print(score_dict[diag]["perf_on_features"][n].keys())
        print(n, ": ", score_dict[diag]["perf_on_features"][n]["auc"])