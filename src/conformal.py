import numpy as np


def frozen_har_residuals(X, y, n_train=1008):
    """
    Reproduce Stage 06's single-deployed HAR and return signed residual stream.

    Fit OLS ONCE on the first n_train rows (the '2011-15 frozen mapping'),
    then forward-predict the entire series with NO retraining. The residual
    stream therefore inherits the same drift that Stage 06 measured.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)   HAR design matrix (rv1, rv5, rv21), no intercept col.
    y : np.ndarray, shape (T,)     target (future h-day RV).
    n_train : int                  frozen-fit window (default 1008 = f00 test start anchor).

    Returns
    -------
    yhat : np.ndarray, shape (T,)  frozen-HAR point forecast over the full series.
    resid : np.ndarray, shape (T,) signed residual y - yhat (raw, sign preserved).
    beta : np.ndarray, shape (p+1,) fitted coefficients [intercept, b1, b5, b21].
    """
    Xtr = np.column_stack([np.ones(n_train), X[:n_train]])  # add intercept
    ytr = y[:n_train]
    # OLS closed form: beta = (X'X)^-1 X'y  (fit once, frozen thereafter)
    beta = np.linalg.solve(Xtr.T @ Xtr, Xtr.T @ ytr)

    Xfull = np.column_stack([np.ones(len(X)), X])
    yhat = Xfull @ beta
    resid = y - yhat
    return yhat, resid, beta


def rolling_sigma(rv1, k=21):
    """
    Local scale estimate = trailing rolling std of rv1, shifted by 1.

    sigma_t uses only information up to t-1 (shift(1)) -> leakage-free,
    preserving the exchangeability needed for normalized-conformal coverage.

    Parameters
    ----------
    rv1 : np.ndarray, shape (T,)   1-day realized vol series.
    k : int                        rolling window (default 21 = HAR lookback / GARCH half-life).

    Returns
    -------
    sigma : np.ndarray, shape (T,) trailing rolling std; first k entries are NaN.
    """
    T = len(rv1)
    sigma = np.full(T, np.nan)
    for t in range(k, T):
        sigma[t] = np.std(rv1[t - k:t], ddof=1)  # window [t-k, t-1], excludes t
    return sigma


def split_conformal_coverage(resid_cal, resid_test, alpha=0.1):
    """
    Split conformal with absolute-residual score (constant-width band).

    Calibrate q on |resid_cal|, then measure realized coverage on test:
    fraction of test points whose |resid| falls within the band half-width q.

    score s_i = |resid_i|;  q = ceil((n+1)(1-alpha))-th smallest s_i.
    Covered  <=>  |resid_test| <= q   (band is [yhat - q, yhat + q]).

    Returns
    -------
    coverage : float   realized fraction covered on test.
    q : float          calibrated band half-width.
    width : float      full band width = 2q (constant across test points).
    """
    s = np.abs(resid_cal)
    n = len(s)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:                       # finite-sample edge: band = infinite
        q = np.inf
    else:
        q = np.sort(s)[rank - 1]       # rank-th smallest (1-indexed)
    coverage = np.mean(np.abs(resid_test) <= q)
    return coverage, q, 2 * q


def normalized_conformal_coverage(resid_cal, sigma_cal,
                                  resid_test, sigma_test, alpha=0.1):
    """
    Normalized (locally-adaptive) conformal: score scaled by local sigma.

    score s_i = |resid_i| / sigma_i;  q = ceil((n+1)(1-alpha))-th smallest s_i.
    Band at test point = [yhat - q*sigma_test, yhat + q*sigma_test], so the
    width breathes with local volatility. sigma must be leakage-free
    (no calibration label) to preserve the coverage guarantee.

    Returns
    -------
    coverage : float        realized fraction covered on test.
    q : float               calibrated quantile of normalized scores.
    mean_width : float       average full band width = 2*q*mean(sigma_test).
    """
    s = np.abs(resid_cal) / sigma_cal
    n = len(s)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        q = np.inf
    else:
        q = np.sort(s)[rank - 1]
    covered = np.abs(resid_test) <= q * sigma_test
    coverage = np.mean(covered)
    mean_width = 2 * q * np.mean(sigma_test)
    return coverage, q, mean_width


def aci_stream_coverage(resid, sigma, test_blocks, cal_window=252,
                        alpha=0.1, gamma=0.01, a_lo=1e-3, a_hi=1-1e-3):
    """
    Adaptive Conformal Inference (Gibbs & Candes 2021) run as one continuous
    online stream, with coverage aggregated per fold afterward.

    Normalized score s_i = |resid_i| / sigma_i. At each step t (over the union
    of all test blocks, in time order) the band uses the (1 - alpha_t) quantile
    of the trailing cal_window normalized scores; alpha_t is then updated by
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1[Y_t not covered].
    alpha_t carries across fold boundaries (true online behavior); it is NOT
    reset per fold. Coverage is then averaged within each fold's test indices.

    Parameters
    ----------
    resid, sigma : np.ndarray, shape (T,)   signed residuals and local scale.
    test_blocks : dict {fold:int -> np.ndarray of absolute test indices}.
    cal_window : int    trailing calibration length for each step's quantile.
    alpha : float       target miscoverage (nominal coverage = 1-alpha).
    gamma : float       ACI step size.
    a_lo, a_hi : float  clip bounds on alpha_t to keep the quantile finite.

    Returns
    -------
    fold_cov : dict {fold -> realized coverage on that fold's test block}.
    alpha_path : np.ndarray   alpha_t over the processed stream (diagnostics).
    """
    # ordered union of all test indices across folds
    all_test = np.sort(np.concatenate([idx for idx in test_blocks.values()]))
    score = np.abs(resid) / sigma

    alpha_t = alpha
    alpha_path = []
    covered_at = {}  # absolute index -> bool

    for t in all_test:
        cal = score[t - cal_window:t]            # trailing window, excludes t
        cal = cal[~np.isnan(cal)]
        q = np.quantile(cal, 1 - alpha_t)        # (1 - alpha_t) quantile
        covered = np.abs(resid[t]) <= q * sigma[t]
        covered_at[t] = bool(covered)

        err = 0.0 if covered else 1.0
        alpha_t = np.clip(alpha_t + gamma * (alpha - err), a_lo, a_hi)
        alpha_path.append(alpha_t)

    fold_cov = {
        fold: np.mean([covered_at[t] for t in idx])
        for fold, idx in test_blocks.items()
    }
    return fold_cov, np.array(alpha_path)

# ── Bayesian conjugate linear regression for HAR prediction intervals ──
# Normal-Inverse-Gamma conjugate prior -> analytic Student-t posterior predictive.
# NO MCMC. Fit ONCE on the first `n_cal` rows (same frozen information set as 06/07),
# then produce prediction intervals fold-by-fold without refitting.

import numpy as np
from scipy import stats


def bayes_nig_fit(X_cal, y_cal, tau=1e-3, a0=1.0, b0=1.0):
    """Fit Normal-Inverse-Gamma posterior on calibration data (frozen once).

    Prior: beta | sigma^2 ~ N(0, sigma^2 * (tau*I)^-1),  sigma^2 ~ Inv-Gamma(a0, b0).
    Weakly-informative: tau small -> prior barely shrinks (posterior ~ OLS).

    Returns dict of posterior params consumed by bayes_predict_interval.
    """
    X_cal = np.asarray(X_cal, float)
    y_cal = np.asarray(y_cal, float)
    n, d = X_cal.shape

    Lambda0 = tau * np.eye(d)          # prior precision matrix
    mu0 = np.zeros(d)                  # prior mean (0 -> ridge-like shrinkage)

    Lambda_n = X_cal.T @ X_cal + Lambda0                      # posterior precision
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    mu_n = Lambda_n_inv @ (X_cal.T @ y_cal + Lambda0 @ mu0)   # posterior mean (ridge point est)

    a_n = a0 + n / 2.0
    b_n = b0 + 0.5 * (
        y_cal @ y_cal + mu0 @ Lambda0 @ mu0 - mu_n @ Lambda_n @ mu_n
    )

    return {
        "mu_n": mu_n,               # posterior mean of beta        (d,)
        "Lambda_n_inv": Lambda_n_inv,  # posterior cov scaffold      (d, d)
        "a_n": a_n,                 # Inv-Gamma shape  -> dof = 2*a_n
        "b_n": b_n,                 # Inv-Gamma scale
    }


def bayes_predict_interval(post, X_new, alpha=0.1):
    """Student-t posterior predictive interval for each row of X_new.

    Predictive: y* ~ t_nu( mean = x*'mu_n,
                           scale^2 = (b_n/a_n) * (1 + x*' Lambda_n_inv x*) ),
                nu = 2*a_n.
    scale^2 splits into aleatoric (b_n/a_n) + epistemic (x*' Lambda_n_inv x*) parts.

    Returns (lower, upper, y_hat) each shape (len(X_new),).
    """
    X_new = np.asarray(X_new, float)
    mu_n = post["mu_n"]
    Lambda_n_inv = post["Lambda_n_inv"]
    a_n, b_n = post["a_n"], post["b_n"]

    nu = 2.0 * a_n
    y_hat = X_new @ mu_n

    # per-row epistemic quadratic form  x*' Lambda_n_inv x*
    quad = np.einsum("ij,jk,ik->i", X_new, Lambda_n_inv, X_new)
    scale = np.sqrt((b_n / a_n) * (1.0 + quad))

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=nu)
    lower = y_hat - t_crit * scale
    upper = y_hat + t_crit * scale
    return lower, upper, y_hat