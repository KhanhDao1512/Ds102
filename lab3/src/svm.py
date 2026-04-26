import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, lr=0.001, lambda_=0.01, epochs=10):
        self.lr = lr
        self.lambda_ = lambda_
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def _init_params(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def _hinge_loss(self, X, y):
        margins = 1 - y * (X @ self.w + self.b)
        loss = 0.5 * np.dot(self.w, self.w) + self.lambda_ * np.sum(np.maximum(0, margins))
        return loss / X.shape[0]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_params(n_features)

        # 👉 class weight (fix imbalance)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)

        w_pos = n_samples / (2 * n_pos)
        w_neg = n_samples / (2 * n_neg)

        pbar = tqdm(range(self.epochs), desc="Training")

        for epoch in pbar:
            indices = np.random.permutation(n_samples)
            X, y = X[indices], y[indices]

            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                # 👉 chọn weight theo class
                weight = w_pos if yi == 1 else w_neg

                condition = yi * (np.dot(xi, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - weight * self.lambda_ * yi * xi)
                    self.b -= self.lr * (-weight * self.lambda_ * yi)

            loss = self._hinge_loss(X, y)
            self.loss_history.append(loss)
            pbar.set_postfix(loss=loss)

    def predict(self, X):
        linear_output = X @ self.w + self.b
        return np.where(linear_output >= 0, 1, -1)

    def evaluate(self, X, y, metrics_fn):
        y_pred = self.predict(X)
        return metrics_fn(y, y_pred)