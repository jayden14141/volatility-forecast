import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def qlike(y_true, y_pred, eps=1e-12):
    """QLIKE loss for a single volatility forecast.

    QLIKE(y, yhat) = y/yhat - log(y/yhat) - 1. It is minimized at
    yhat = y, and unlike MSE it stays a consistent ranking criterion
    even when the volatility target is a noisy proxy of the true
    variance (Patton 2011, "Volatility forecast comparison using
    imperfect volatility proxies", Journal of Econometrics).

    Both arguments must be positive (RV scale); eps guards the log
    against zeros.
    """
    y_pred = max(float(y_pred), eps)
    y_true = max(float(y_true), eps)
    r = y_true / y_pred
    return r - np.log(r) - 1.0
