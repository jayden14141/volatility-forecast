import pandas as pd

FEATURES = ["rv1", "rv5", "rv21"]
TARGETS  = ["y_rv1", "y_rv5", "y_rv21"]

def load_work_frame(path, features=FEATURES, targets=TARGETS):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    work = df.dropna(subset=list(features) + list(targets)).copy()
    return work