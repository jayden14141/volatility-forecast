"""Uncertainty quantification for the HAR volatility forecaster (stage 07).

All interval methods here wrap the SAME model as stage 06: a static HAR
fit once by OLS on the first 1008 rows and then used with no retraining.
Sharing one static model is what makes the stage-06 drift z-scores and
the stage-07 coverage gaps directly comparable per fold.

Four interval families, in order of adaptivity:
  1. split_conformal_coverage       -- constant-width, distribution-free.
  2. normalized_conformal_coverage  -- width scales with a local sigma.
  3. aci_stream_coverage            -- online alpha_t update
                                       (Adaptive Conformal Inference,
                                       Gibbs & Candès 2021).
  4. bayes_nig_fit / bayes_predict_interval -- parametric Student-t
                                       intervals from a Normal-Inverse-
                                       Gamma posterior, fit once.

Working hypothesis being tested: under concept drift, exchangeability
breaks, so the static-calibration methods (split conformal, Bayesian)
under-cover, while the online method (ACI) recovers long-run coverage.
"""

import numpy as np
from scipy import stats


def static_har_residuals(X, y, n_train=1008):
    """Reproduce stage 06's static HAR and return its residual stream.

    Fit OLS once on the first n_train rows, then forward-predict the
    entire series with no retraining. The residual stream therefore
    shows the same drift that stage 06 measured -- stages 06 and 07
    describe one identical model.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)   HAR features (rv1, rv5, rv21), no intercept column.
    y : np.ndarray, shape (T,)     target (future h-day RV).
    n_train : int                  training window (default 1008 = f00 test start).

    Returns
    -------
    yhat : np.ndarray, shape (T,)   static-HAR point forecast over the full series.
    resid : np.ndarray, shape (T,)  signed residual y - yhat.
    beta : np.ndarray, shape (p+1,) fitted coefficients [intercept, b1, b5, b21].
    """
    Xtr = np.column_stack([np.ones(n_train), X[:n_train]])
    ytr = y[:n_train]
    # OLS normal equations: beta = (X'X)^-1 X'y, fit once
    beta = np.linalg.solve(Xtr.T @ Xtr, Xtr.T @ ytr)

    Xfull = np.column_stack([np.ones(len(X)), X])
    yhat = Xfull @ beta
    resid = y - yhat
    return yhat, resid, beta


def rolling_sigma(rv1, k=21):
    """Local scale estimate: trailing rolling std of rv1, excluding today.

    sigma[t] is computed from rv1[t-k : t], i.e. information up to t-1
    only, so it is leakage-free and can be used as a normalization
    factor without breaking the conformal coverage guarantee.

    Parameters
    ----------
    rv1 : np.ndarray, shape (T,)   1-day realized vol series.
    k : int                        rolling window length (default 21 days).

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
    """Split conformal prediction with absolute-residual score
    (constant-width band).

    Calibrate the band half-width q on |resid_cal|, then measure
    realized coverage on the test residuals.

    score s_i = |resid_i|;  q = ceil((n+1)(1-alpha))-th smallest s_i.
    A test point is covered iff |resid_test| <= q, i.e. the interval is
    [yhat - q, yhat + q].

    Returns
    -------
    coverage : float   realized fraction covered on test.
    q : float          calibrated band half-width.
    width : float      full band width = 2q (constant across test points).
    """
    s = np.abs(resid_cal)
    n = len(s)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:                       # finite-sample edge case: infinite band
        q = np.inf
    else:
        q = np.sort(s)[rank - 1]       # rank-th smallest (1-indexed)
    coverage = np.mean(np.abs(resid_test) <= q)
    return coverage, q, 2 * q


def normalized_conformal_coverage(resid_cal, sigma_cal,
                                  resid_test, sigma_test, alpha=0.1):
    """Normalized (locally adaptive) conformal: score scaled by local sigma.

    score s_i = |resid_i| / sigma_i;  q = ceil((n+1)(1-alpha))-th smallest s_i.
    The interval at a test point is [yhat - q*sigma, yhat + q*sigma], so
    the width tracks local volatility. sigma must be computed without
    using the current label (see rolling_sigma) or the coverage
    guarantee is lost.

    Returns
    -------
    coverage : float      realized fraction covered on test.
    q : float             calibrated quantile of normalized scores.
    mean_width : float    average full band width = 2*q*mean(sigma_test).
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
    """Adaptive Conformal Inference (Gibbs & Candès 2021) run as one
    continuous online stream, with coverage aggregated per fold afterward.

    Normalized score s_i = |resid_i| / sigma_i. At each step t (over the
    union of all test blocks, in time order) the band uses the
    (1 - alpha_t) quantile of the trailing cal_window normalized scores;
    alpha_t is then updated by
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),
        err_t = 1[Y_t not covered].
    alpha_t carries across fold boundaries (true online behavior); it is
    NOT reset per fold. Coverage is then averaged within each fold.

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


def bayes_nig_fit(X_cal, y_cal, tau=1e-3, a0=1.0, b0=1.0):
    """Fit a Normal-Inverse-Gamma posterior on calibration data (fit once).

    Conjugate to a Gaussian likelihood, so the posterior is analytic (no
    MCMC needed).
    Prior: beta | sigma^2 ~ N(0, sigma^2 (tau*I)^-1),
           sigma^2 ~ Inv-Gamma(a0, b0).
    A small tau makes the prior weak, so the posterior mean is close to
    the OLS estimate.
    Note on scale: for tiny RV-scale targets keep a0, b0 small (e.g.
    1e-3, 1e-6); otherwise the prior dominates b_n/a_n and the intervals
    inflate toward ~100% coverage.

    Parameters
    ----------
    X_cal : np.ndarray, shape (n, d)   design matrix INCLUDING an intercept column.
    y_cal : np.ndarray, shape (n,)     calibration target.
    tau : float                        prior precision scale (small = weak prior).
    a0, b0 : float                     Inv-Gamma prior shape / scale.

    Returns
    -------
    post : dict with
        mu_n : (d,)            posterior mean of beta (ridge-like point estimate).
        Lambda_n_inv : (d, d)  inverse posterior precision (parameter covariance
                               up to the sigma^2 factor).
        a_n : float            posterior Inv-Gamma shape (predictive dof = 2*a_n).
        b_n : float            posterior Inv-Gamma scale (noise variance = b_n/a_n).
    """
    X_cal = np.asarray(X_cal, float)
    y_cal = np.asarray(y_cal, float)
    n, d = X_cal.shape

    Lambda0 = tau * np.eye(d)          # prior precision matrix
    mu0 = np.zeros(d)                  # prior mean (0 -> ridge-like shrinkage)

    Lambda_n = X_cal.T @ X_cal + Lambda0                      # posterior precision
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    mu_n = Lambda_n_inv @ (X_cal.T @ y_cal + Lambda0 @ mu0)   # posterior mean

    a_n = a0 + n / 2.0
    b_n = b0 + 0.5 * (
        y_cal @ y_cal + mu0 @ Lambda0 @ mu0 - mu_n @ Lambda_n @ mu_n
    )

    return {
        "mu_n": mu_n,
        "Lambda_n_inv": Lambda_n_inv,
        "a_n": a_n,
        "b_n": b_n,
    }


def bayes_predict_interval(post, X_new, alpha=0.1):
    """Student-t posterior predictive interval for each row of X_new.

    Predictive: y* ~ t_nu( mean = x*'mu_n,
                           scale^2 = (b_n/a_n) * (1 + x*' Lambda_n_inv x*) ),
                nu = 2*a_n.
    The scale decomposes into
        aleatoric  = b_n / a_n            (observation noise, fixed at fit time),
        epistemic  = x*' Lambda_n_inv x*  (parameter uncertainty; grows as x*
                                           moves away from the calibration data).

    Parameters
    ----------
    post : dict     output of bayes_nig_fit.
    X_new : np.ndarray, shape (m, d)   query design matrix (same intercept layout).
    alpha : float   target miscoverage (nominal coverage = 1-alpha).

    Returns
    -------
    lower, upper, y_hat : np.ndarray, each shape (m,).
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
