import numpy as np


def walk_forward_splits(n_samples, min_train=1008, test_size=252, embargo=21):
    """Expanding-window walk-forward splits with an embargo gap.

    Fold k: train = [0, test_start - embargo), test = [test_start,
    test_start + test_size), where test_start = min_train + k * test_size.

    The embargo drops the last `embargo` training rows because the h=21
    target at row t is built from days t+1..t+21: without the gap, the
    last training labels would overlap the first test days (leakage).
    With embargo = 21 the newest training label ends exactly at
    test_start - 1.
    """
    test_start = min_train
    while test_start + test_size <= n_samples:
        train_end = test_start - embargo
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_start + test_size)
        yield train_idx, test_idx
        test_start += test_size
