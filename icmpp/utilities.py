from sklearn.preprocessing import normalize as _normalize
import numpy as np


def normalize(values):
    if values.ndim == 1:
        values = _normalize(values.reshape(1, -1), 'l1').flatten()
    else:
        values = _normalize(values, 'l1')

    return values


def pad_like(values, like_values):
    if count := like_values.size - values.size:
        values = np.pad(values, (0, count))

    return values
