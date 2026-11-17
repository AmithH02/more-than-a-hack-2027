"""hmm_touch with separate evidence weights for frames where the ball is tracked (`w`) and where
it is hidden (`w_hidden`): the hidden-ball models are less sure, so their frames may deserve to
count for less (or more) against the cost of a change. Same features and models as hmm_touch,
refitted under this name.

    python -m possession.bench train hmm_split_w
"""

from possession.experiments import hmm_touch
from possession.hmm_models import HmmModels

GRID = {"w": [0.5, 0.7, 1.0], "w_hidden": [0.2, 0.4, 0.7, 1.0], "switch": [0.5],
        "dead": [1.0], "dead_min": [1]}
DEFAULT = {"w": 0.7, "w_hidden": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 1}

MODEL = HmmModels("hmm_split_w", hmm_touch.features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
