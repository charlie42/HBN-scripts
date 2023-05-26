import datetime, os

def get_newest_non_empty_dir_in_dir(path, extra_filters=[], file_filter=""):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    
    if extra_filters:
        for extra_filter in extra_filters:
            non_empty_dir_names = [d for d in non_empty_dir_names if extra_filter in d]
    if file_filter:
        non_empty_dir_names = [d for d in non_empty_dir_names if os.path.isfile(path+d+'/'+file_filter)]
    
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)