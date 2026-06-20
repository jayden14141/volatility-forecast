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

def ddm(stream, warn_sigma=2.0, drift_sigma=3.0, min_n=100):
    """Drift Detection Method (Gama et al. 2004) on a binary error stream.

    stream : 1D array of 0/1, where 1 = model lost to the baseline at that step.
    Monitors the Bernoulli loss rate p_i and flags when its lower-confidence
    floor (p_min + s_min) is breached by `*_sigma` standard deviations.
    Resets state after a drift (fresh concept).
    """
    import numpy as np
    e = np.asarray(stream, dtype=float)
    n = len(e)
    p_min, s_min = np.inf, np.inf
    warn, drift = [], []
    trace = np.full(n, np.nan)
    seen, errs = 0, 0.0
    for i in range(n):
        seen += 1
        errs += e[i]
        p_i = errs / seen
        s_i = np.sqrt(p_i * (1.0 - p_i) / seen)
        trace[i] = p_i + s_i
        if seen < min_n:
            continue
        if p_i + s_i < p_min + s_min:        # track running minimum
            p_min, s_min = p_i, s_i
        if p_i + s_i >= p_min + drift_sigma * s_min:
            drift.append(i)
            p_min, s_min = np.inf, np.inf     # reset to fresh concept
            seen, errs = 0, 0.0
        elif p_i + s_i >= p_min + warn_sigma * s_min:
            warn.append(i)
    return {"warning": warn, "drift": drift, "trace": trace}


def adwin(stream, delta=0.002, min_n=30):
    """ADWIN (Bifet & Gavalda 2007), simple exact variant for a [0,1] stream.

    Keeps an explicit window; after each new value, tries every split
    W = W0 . W1 and drops the stale prefix W0 when the sub-window means
    differ by more than the Hoeffding cut. Records the index where a drop
    (= detected change) occurs and the running window width.

    Exact O(n^2) worst case; fine up to a few thousand points. Production
    ADWIN uses exponential-histogram buckets for O(log n) memory/time.
    """
    import numpy as np
    x = np.asarray(stream, dtype=float)
    n = len(x)
    window = []
    drift = []
    width = np.full(n, np.nan, dtype=float)
    for i in range(n):
        window.append(float(x[i]))
        if len(window) < min_n:
            width[i] = len(window)
            continue
        cut_found = True
        while cut_found and len(window) >= 2:
            cut_found = False
            W = len(window)
            total = sum(window)
            n0, s0 = 0, 0.0
            for k in range(1, W):             # split after position k (prefix = window[:k])
                n0 += 1
                s0 += window[k - 1]
                n1 = W - n0
                mu0 = s0 / n0
                mu1 = (total - s0) / n1
                m = 1.0 / (1.0 / n0 + 1.0 / n1)        # harmonic mean
                delta_prime = delta / W
                eps_cut = np.sqrt(np.log(4.0 / delta_prime) / (2.0 * m))
                if abs(mu0 - mu1) > eps_cut:
                    window = window[k:]        # drop stale prefix
                    drift.append(i)
                    cut_found = True
                    break
        width[i] = len(window)
    return {"drift": drift, "width": width}