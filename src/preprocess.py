import numpy as np

def preprocess(X):
    # scale về [0,1]
    X = X / 255.0

    # standardize (rất quan trọng)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    return X