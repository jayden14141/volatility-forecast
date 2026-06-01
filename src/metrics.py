import numpy as np

def rmse(y_true, y_pred):
    r"""Root mean squared error.

    RMSE = sqrt( (1/n) * Σ_i (y_true_i - y_pred_i)^2 )
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))