import numpy as np

def walk_forward_splits(n_samples, min_train=1008, test_size=252, embargo=21):
    test_start = min_train
    while test_start + test_size <= n_samples:
        train_end = test_start - embargo          # purge last `embargo` rows
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, test_start + test_size)
        yield train_idx, test_idx
        test_start += test_size