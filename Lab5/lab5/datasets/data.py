import numpy as np


def generate_gaussian_data(specs, seed=0):
    """Generate 2D Gaussian toy data.

    specs is a list of (n_points, mean, covariance).
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label, (n, mean, cov) in enumerate(specs):
        points = rng.multivariate_normal(np.array(mean, dtype=float), np.array(cov, dtype=float), n)
        xs.append(points)
        ys.append(np.full(n, label, dtype=int))
    return np.vstack(xs), np.concatenate(ys)

