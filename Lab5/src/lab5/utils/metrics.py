import numpy as np


def cluster_purity(y_true, y_pred, k):
    total = 0
    for cluster_id in range(k):
        members = y_true[y_pred == cluster_id]
        if len(members) > 0:
            total += np.bincount(members).max()
    return total / len(y_true)

