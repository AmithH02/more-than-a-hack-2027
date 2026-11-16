"""A small, deterministic logistic regression in numpy, stored as plain JSON.

No randomness and no third-party learner, so a model refitted on the same data gives the same
numbers, and the committed JSON is all that predict() needs.

    model = fit_logistic(X, y)          # y in {0, 1}
    p = predict_proba(model, X)          # (n,) probability of 1
"""

import numpy as np


def fit_logistic(X, y, steps=500, rate=0.5, l2=1e-3, names=None):
    """Full-batch gradient descent on standardised features; returns a JSON-able dict."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = np.column_stack([(X - mean) / scale, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        w -= rate * (Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0])
    return {"names": list(names) if names is not None else None, "mean": mean.tolist(),
            "scale": scale.tolist(), "weights": w[:-1].tolist(), "bias": float(w[-1]),
            "n": int(len(y)), "positive_share": float(y.mean())}


def predict_proba(model, X):
    X = np.asarray(X, dtype=np.float64)
    z = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(model["weights"]) + model["bias"])))
