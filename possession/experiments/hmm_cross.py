"""hmm_touch plus a few hand-made products of its features, so logistic regression can combine
evidence the way the network did (hmm_mlp) without its freedom to memorise the training matches.

    python -m possession.bench train hmm_cross
"""

import numpy as np

from possession.experiments import hmm_touch
from possession.hmm_models import HmmModels

PAIRS = (("last_fresh_r2", "near_margin"), ("next_fresh_r2", "near_margin"),
         ("last_r2", "next_r2"), ("near_margin", "near_any"), ("side_next", "log_until"),
         ("side_last", "log_since"), ("last_fresh_r2", "mean_speed"), ("ball_dx", "last_r2"))

GRID = {"w": [0.4, 0.7], "switch": [0.5, 1.0], "dead": [1.0], "dead_min": [1]}
DEFAULT = {"w": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 1}  # replaced by `bench train`


def features(match):
    X, names = hmm_touch.features(match)
    index = {name: i for i, name in enumerate(names)}
    crosses = [X[:, index[a]] * X[:, index[b]] for a, b in PAIRS]
    return np.column_stack([X] + crosses), names + tuple("{}*{}".format(a, b) for a, b in PAIRS)


MODEL = HmmModels("hmm_cross", features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
