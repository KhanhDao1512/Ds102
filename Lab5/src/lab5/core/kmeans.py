import numpy as np


def squared_distances(x, centers):
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def kmeans(x, k, max_iter=100, tol=1e-6, seed=0):
    """K-means implemented with NumPy only."""
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    history = []

    for _ in range(max_iter):
        distances = squared_distances(x, centers)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()

        for cluster_id in range(k):
            members = x[new_labels == cluster_id]
            if len(members) == 0:
                new_centers[cluster_id] = x[rng.integers(len(x))]
            else:
                new_centers[cluster_id] = members.mean(axis=0)

        objective = np.sum((x - new_centers[new_labels]) ** 2)
        history.append(float(objective))

        shift = np.linalg.norm(new_centers - centers)
        if np.array_equal(new_labels, labels) or shift < tol:
            centers = new_centers
            labels = new_labels
            break

        centers = new_centers
        labels = new_labels

    return labels, centers, history

