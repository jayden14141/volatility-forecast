# src/drift.py
# Covariate-shift detection utilities: per-feature 2-sample tests.
# Compares a reference window (train) against a test window for distribution shift.

import numpy as np
from scipy.stats import ks_2samp


def ks_drift(ref, test):
    """Kolmogorov-Smirnov 2-sample test for distribution shift.

    KS statistic D = sup_x |F_ref(x) - F_test(x)|, the max gap between the
    two empirical CDFs. Non-parametric: detects any shift in distribution
    shape, not just mean. Returns (D, p_value); small p_value => shift.

    ref, test: 1D arrays of one feature's values in each window.
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    ref = ref[~np.isnan(ref)]
    test = test[~np.isnan(test)]
    stat, p = ks_2samp(ref, test)
    return float(stat), float(p)


def psi(ref, test, n_bins=10, eps=1e-6):
    """Population Stability Index between reference and test windows.

    PSI = sum_i (p_test_i - p_ref_i) * ln(p_test_i / p_ref_i),
    over bins defined on the REFERENCE distribution's quantiles.
    Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 large shift.

    Bins are reference-quantile based so each ref bin holds ~equal mass;
    eps floors empty bins to keep the log finite.
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    ref = ref[~np.isnan(ref)]
    test = test[~np.isnan(test)]

    # Bin edges from reference quantiles; open the outer edges to catch
    # test-side values beyond the reference range.
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_cnt = np.histogram(ref, bins=edges)[0].astype(float)
    test_cnt = np.histogram(test, bins=edges)[0].astype(float)

    p_ref = ref_cnt / ref_cnt.sum()
    p_test = test_cnt / test_cnt.sum()
    p_ref = np.clip(p_ref, eps, None)
    p_test = np.clip(p_test, eps, None)

    return float(np.sum((p_test - p_ref) * np.log(p_test / p_ref)))