import numpy as np


def fit_har(X_train, y_train):
    """Fit a HAR model via OLS (numpy least squares).

    X_train: (n, 3) array of [rv1, rv5, rv21] (daily/weekly/monthly components).
    y_train: (n,)   array of future RV target.

    Returns beta: (4,) array = [intercept, beta_d, beta_w, beta_m].
    Prepends a column of ones so lstsq also solves for the intercept.
    Closed form: beta_hat = (X'X)^-1 X'y, which lstsq computes stably.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])   # intercept column
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def predict_har(beta, X):
    """Apply fitted HAR betas to features.

    beta: (4,) = [intercept, beta_d, beta_w, beta_m].
    X:    (m, 3) array of [rv1, rv5, rv21].
    Returns yhat: (m,) predictions.
    """
    X = np.asarray(X, dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta