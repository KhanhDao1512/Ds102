import numpy as np

from lab5.core.kmeans import kmeans


def log_gaussian_pdf(x, means, covariances):
    """Return log N(x | mean_k, covariance_k) with shape (n_samples, k)."""
    n_features = x.shape[1]
    logs = []
    for mean, cov in zip(means, covariances):
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + np.eye(n_features) * 1e-6
            sign, logdet = np.linalg.slogdet(cov)
        diff = x - mean
        solved = np.linalg.solve(cov, diff.T).T
        mahalanobis = np.sum(diff * solved, axis=1)
        logs.append(-0.5 * (n_features * np.log(2 * np.pi) + logdet + mahalanobis))
    return np.column_stack(logs)


def logsumexp(a, axis=1, keepdims=False):
    max_a = np.max(a, axis=axis, keepdims=True)
    out = max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def gmm_em(x, k, max_iter=80, tol=1e-5, seed=0, reg=1e-6):
    """Gaussian Mixture Model trained by EM, implemented with NumPy only."""
    n_samples, n_features = x.shape
    init_labels, means, _ = kmeans(x, k, max_iter=30, seed=seed)
    covariances = np.zeros((k, n_features, n_features))
    weights = np.zeros(k)

    global_cov = np.cov(x.T) + np.eye(n_features) * reg
    for cluster_id in range(k):
        members = x[init_labels == cluster_id]
        if len(members) <= n_features:
            covariances[cluster_id] = global_cov
            weights[cluster_id] = 1.0 / k
        else:
            covariances[cluster_id] = np.cov(members.T) + np.eye(n_features) * reg
            weights[cluster_id] = len(members) / n_samples

    log_likelihoods = []
    responsibilities = np.full((n_samples, k), 1.0 / k)

    for _ in range(max_iter):
        weighted_logs = log_gaussian_pdf(x, means, covariances) + np.log(weights + 1e-15)
        sample_logs = logsumexp(weighted_logs, axis=1, keepdims=True)
        responsibilities = np.exp(weighted_logs - sample_logs)
        log_likelihood = float(sample_logs.sum())
        log_likelihoods.append(log_likelihood)

        nk = responsibilities.sum(axis=0) + 1e-15
        weights = nk / n_samples
        means = (responsibilities.T @ x) / nk[:, None]

        for cluster_id in range(k):
            diff = x - means[cluster_id]
            weighted = diff * responsibilities[:, cluster_id][:, None]
            covariances[cluster_id] = (weighted.T @ diff) / nk[cluster_id]
            covariances[cluster_id] += np.eye(n_features) * reg

        if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return {
        "weights": weights,
        "means": means,
        "covariances": covariances,
        "responsibilities": responsibilities,
        "log_likelihoods": log_likelihoods,
        "labels": responsibilities.argmax(axis=1),
    }
