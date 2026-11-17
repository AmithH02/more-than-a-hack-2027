"""A small, deterministic neural network for a yes/no probability, in numpy, stored as plain JSON.

One hidden layer of tanh units on standardised features, trained full-batch with Adam on the
log-loss from a fixed-seed start. No mini-batches and no shuffling, so refitting on the same data
gives the same numbers, and the committed JSON is all predict_proba() needs. A drop-in for
possession/learn.py's fit_logistic / predict_proba.

    model = fit_mlp(X, y, hidden=32)
    p = predict_proba(model, X)
"""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_mlp(X, y, hidden=32, steps=400, rate=0.01, l2=1e-4, seed=0, names=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0, 1 / np.sqrt(k), size=(k, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, 1 / np.sqrt(hidden), size=hidden),
        "b2": np.array([np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))]),
    }
    m = {key: np.zeros_like(v) for key, v in params.items()}
    v = {key: np.zeros_like(val) for key, val in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        H = np.tanh(Z @ params["W1"] + params["b1"])
        p = _sigmoid(H @ params["w2"] + params["b2"][0])
        d_out = (p - y) / n                                    # dLoss/dlogit
        grads = {
            "w2": H.T @ d_out + l2 * params["w2"],
            "b2": np.array([d_out.sum()]),
        }
        d_hidden = np.outer(d_out, params["w2"]) * (1 - H ** 2)
        grads["W1"] = Z.T @ d_hidden + l2 * params["W1"]
        grads["b1"] = d_hidden.sum(axis=0)
        for key in params:
            m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
            v[key] = beta2 * v[key] + (1 - beta2) * grads[key] ** 2
            m_hat = m[key] / (1 - beta1 ** step)
            v_hat = v[key] / (1 - beta2 ** step)
            params[key] = params[key] - rate * m_hat / (np.sqrt(v_hat) + eps)
    return {"kind": "mlp", "names": list(names) if names is not None else None,
            "mean": mean.tolist(), "scale": scale.tolist(),
            "W1": params["W1"].tolist(), "b1": params["b1"].tolist(),
            "w2": params["w2"].tolist(), "b2": float(params["b2"][0]),
            "n": int(n), "positive_share": float(y.mean())}


def predict_proba(model, X):
    Z = (np.asarray(X, dtype=np.float64) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    H = np.tanh(Z @ np.asarray(model["W1"]) + np.asarray(model["b1"]))
    return _sigmoid(H @ np.asarray(model["w2"]) + model["b2"])
