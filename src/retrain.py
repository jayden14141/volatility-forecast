"""Retraining-policy simulator for the HAR volatility model (stage 08).

The h=21 target for day s is realized volatility over days s+1..s+21,
so the forecast error for day s can only be computed 21 trading days
later. simulate_retrain() enforces this label delay: a retrain decision
made at day i may only look at losses for days s <= i - LABEL_DELAY.
Violating this would let the trigger react to information that does not
exist yet (look-ahead bias) and invalidate every policy comparison.

References
----------
ADWIN : Bifet & Gavaldà (2007), "Learning from Time-Changing Data with
        Adaptive Windowing", SIAM SDM.
QLIKE : Patton (2011), "Volatility forecast comparison using imperfect
        volatility proxies", Journal of Econometrics.
"""

import numpy as np
import pandas as pd

from src.har import fit_har, predict_har
from src.metrics import qlike

FEATURES = ['rv1', 'rv5', 'rv21']
TARGET = 'y_rv21'
LABEL_DELAY = 21


def fit_har_frame(frame):
    """OLS HAR fit on a work-frame slice with FEATURES + TARGET columns.

    Returns beta = [intercept, b_rv1, b_rv5, b_rv21].
    """
    return fit_har(frame[FEATURES].values, frame[TARGET].values)


def predict_har_row(beta, row):
    """One-day HAR forecast from a single work-frame row."""
    return predict_har(beta, [[row['rv1'], row['rv5'], row['rv21']]])[0]


def simulate_retrain(df, trigger_fn, train_window=1008, start=1008,
                     label_delay=LABEL_DELAY, min_fit_rows=50):
    """Simulate a retraining policy day by day, without look-ahead.

    At each out-of-sample day i (integer position, i >= start):
      1. The loss for day s = i - label_delay matures today and joins
         the visible loss history.
      2. trigger_fn decides whether to retrain. It only sees matured
         losses (days s <= i - label_delay), never anything newer.
      3. If it returns True, HAR is refit on the trailing train_window
         rows ending at i - label_delay -- the newest row whose h=21
         label is fully observed by day i.
      4. The current model forecasts day i; that forecast's loss will
         become visible at day i + label_delay.

    trigger_fn signature: (i, loss_history, visible_indices, df) -> bool
      i               : integer position of the current day
      loss_history    : np.ndarray of matured QLIKE losses, oldest first
      visible_indices : row positions matching loss_history
      df              : full work frame (features are observable same
                        day; labels are NOT -- triggers must not read
                        df[TARGET] at rows newer than i - label_delay)

    Returns a per-day log DataFrame. .attrs['newest_visible'] records
    the newest loss index the trigger could see each day, so a unit
    test can verify the no-look-ahead invariant.
    """
    n = len(df)
    beta = fit_har_frame(df.iloc[:start - label_delay])   # initial model: static HAR on first `start` rows

    matured = {}      # s -> QLIKE loss, present only once s + label_delay <= i
    pending = {}      # s -> (y_pred, y_true), waiting for the label to mature
    log = []
    newest_visible = []
    n_retrains = 0
    model_age = 0

    for i in range(start, n):
        row = df.iloc[i]

        # (1) the single loss maturing today becomes visible
        s = i - label_delay
        if s >= start and s in pending:
            yp, yt = pending.pop(s)
            matured[s] = qlike(yt, yp)

        # (2) trigger sees matured losses only (all keys <= i - label_delay)
        visible_indices = sorted(matured.keys())
        newest_visible.append(visible_indices[-1] if visible_indices else -1)
        loss_history = np.array([matured[k] for k in visible_indices])
        retrain_triggered = bool(trigger_fn(i, loss_history, visible_indices, df))

        # (3) refit on the trailing window ending at the last observable label
        if retrain_triggered:
            end = i - label_delay
            begin = max(0, end - train_window)
            if end - begin >= min_fit_rows:
                beta = fit_har_frame(df.iloc[begin:end])
                n_retrains += 1
                model_age = 0

        # (4) forecast today; loss stays pending until the label matures
        yp = predict_har_row(beta, row)
        yt = row[TARGET]
        pending[i] = (yp, yt)

        log.append({'date': df.index[i], 'i': i, 'y_pred': yp, 'y_true': yt,
                    'retrain_triggered': retrain_triggered,
                    'n_retrains': n_retrains, 'model_age': model_age})
        model_age += 1

    out = pd.DataFrame(log).set_index('date')
    out['loss'] = [qlike(t.y_true, t.y_pred) for t in out.itertuples()]
    out.attrs['newest_visible'] = newest_visible
    out.attrs['start'] = start
    out.attrs['label_delay'] = label_delay
    return out


class AdwinStream:
    """Streaming ADWIN change detector (Bifet & Gavaldà 2007).

    Same math as the batch adwin() in src/drift.py -- Hoeffding cut with
    the delta/W correction -- but stateful: call update(x) once per new
    value; it returns True when a change is detected. Call reset() after
    acting on a detection (e.g. after retraining) so the window restarts
    on the new regime.

    The defaults reproduce the plain two-sided detector ("adwin_raw").
    Three optional constraints make it a one-sided, minimum-evidence
    variant ("adwin_directional"):
      one_sided : only fire when the recent mean is HIGHER than the old
                  mean (loss got worse; improvement is not a reason to
                  retrain).
      gap_min   : require at least this mean gap (effect size floor).
      n1_min    : require at least this many points in the recent
                  sub-window (don't fire on a handful of bad days).

    This is the exact O(n^2)-worst-case variant with an explicit window;
    production ADWIN uses exponential-histogram buckets for O(log n).
    """

    def __init__(self, delta=0.002, min_n=30, one_sided=False, gap_min=0.0, n1_min=1):
        self.delta = delta
        self.min_n = min_n
        self.one_sided = one_sided
        self.gap_min = gap_min
        self.n1_min = n1_min
        self.window = []
        self.last_cut = None

    def reset(self):
        self.window = []

    def update(self, x):
        self.window.append(float(x))
        if len(self.window) < self.min_n:
            return False
        detected = False
        cut_found = True
        while cut_found and len(self.window) >= 2:
            cut_found = False
            W = len(self.window)
            total = sum(self.window)
            n0, s0 = 0, 0.0
            for k in range(1, W):
                n0 += 1
                s0 += self.window[k - 1]
                n1 = W - n0
                mu0 = s0 / n0
                mu1 = (total - s0) / n1
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                delta_prime = self.delta / W
                eps_cut = np.sqrt(np.log(4.0 / delta_prime) / (2.0 * m))
                gap = (mu1 - mu0) if self.one_sided else abs(mu1 - mu0)
                if gap > eps_cut and gap >= self.gap_min and n1 >= self.n1_min:
                    self.last_cut = (mu0, mu1, n0, n1)
                    self.window = self.window[k:]
                    detected = True
                    cut_found = True
                    break
        return detected
