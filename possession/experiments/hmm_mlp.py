"""hmm_touch's features and HMM, with a small neural network instead of logistic regression for
the per-frame probabilities (possession/mlp.py: one hidden layer of 32 tanh units, deterministic).
Lets the evidence combine (e.g. "home touched it last" and "an away player is now on the ball")
without hand-made crosses.

    python -m possession.bench train hmm_mlp
"""

from possession.experiments import hmm_touch
from possession.hmm_models import HmmModels

GRID = {"w": [0.4, 0.7, 1.0], "switch": [0.5, 1.0], "dead": [1.0], "dead_min": [1]}
DEFAULT = {"w": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 1}  # replaced by `bench train`

MODEL = HmmModels("hmm_mlp", hmm_touch.features, learner="mlp", hidden=32, steps=400, rate=0.01)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
