import numpy as np
import pandas as pd

FEATURES = ['rv1', 'rv5', 'rv21']
TARGET = 'y_rv21'
LABEL_DELAY = 21


def fit_har(sub):
    """OLS HAR on a slice with FEATURES + TARGET. Returns beta [b0, b_rv1, b_rv5, b_rv21]."""
    X = np.column_stack([np.ones(len(sub))] + [sub[f].values for f in FEATURES])
    y = sub[TARGET].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_har(beta, row):
    return beta[0] + beta[1] * row['rv1'] + beta[2] * row['rv5'] + beta[3] * row['rv21']


def qlike(y_true, y_pred, eps=1e-12):
    """QLIKE loss: y/yhat - ln(y/yhat) - 1. Positive-only (RV scale)."""
    y_pred = max(float(y_pred), eps)
    y_true = max(float(y_true), eps)
    r = y_true / y_pred
    return r - np.log(r) - 1.0


def simulate_retrain(df, trigger_fn, train_window=1008, start=1008,
                     label_delay=LABEL_DELAY, min_fit_rows=50):
    """
    Causal day-by-day retrain simulation.

    At OOS day tau (integer position i >= start):
      1. reveal the loss L[i-label_delay] (it matures exactly today)
      2. trigger_fn sees ONLY the revealed-loss buffer: {L[s] : s <= i-label_delay}
      3. if fired, refit HAR on rolling window ending at last observable label (i-label_delay)
      4. forecast y_pred[i] with current model; stash its loss as pending (matures at i+label_delay)

    Returns per-day log DataFrame; engine records newest visible loss index in .attrs
    for a causality unit-test.
    """
    n = len(df)
    beta = fit_har(df.iloc[:start])          # initial = frozen HAR on first `start` rows

    revealed = {}     # s -> L[s], present only once s + label_delay <= i
    pending = {}      # s -> (y_pred, y_true), awaiting maturity
    log = []
    newest_visible = []
    n_retrains = 0
    model_age = 0

    for i in range(start, n):
        row = df.iloc[i]

        # (1) reveal the single loss maturing today
        s = i - label_delay
        if s >= start and s in pending:
            yp, yt = pending.pop(s)
            revealed[s] = qlike(yt, yp)

        # (2) trigger sees revealed buffer (all keys <= i - label_delay by construction)
        vis = sorted(revealed.keys())
        newest_visible.append(vis[-1] if vis else -1)
        loss_stream = np.array([revealed[k] for k in vis])
        fire = bool(trigger_fn(i, loss_stream, vis, df))

        # (3) retrain on rolling window ending at last observable label
        if fire:
            end = i - label_delay
            begin = max(0, end - train_window)
            if end - begin >= min_fit_rows:
                beta = fit_har(df.iloc[begin:end])
                n_retrains += 1
                model_age = 0

        # (4) forecast + stash pending loss
        yp = predict_har(beta, row)
        yt = row[TARGET]
        pending[i] = (yp, yt)

        log.append({'date': df.index[i], 'i': i, 'y_pred': yp, 'y_true': yt,
                    'retrained': fire, 'n_retrains': n_retrains, 'model_age': model_age})
        model_age += 1

    out = pd.DataFrame(log).set_index('date')
    out['loss'] = [qlike(t.y_true, t.y_pred) for t in out.itertuples()]
    out.attrs['newest_visible'] = newest_visible
    out.attrs['start'] = start
    out.attrs['label_delay'] = label_delay
    return out

class AdwinStream:
    """Streaming port of src/drift.py adwin -- identical math (Hoeffding cut,
    delta/W correction), but stateful: one update(x) call per value, returns
    True when a change is detected. reset() clears the window (post-retrain)."""

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