"""hmm_touch plus "who restarts": when the ball goes out, the other team takes the throw-in, goal
kick or corner. Two passes, from tracking only:
  1. decode with hmm_touch to find the stoppages;
  2. for each live frame after a stoppage, the side opposite the last control before it, fading
     over about 5 s; separately, the same hint only where the ball was seen off the pitch around
     the stoppage (a stoppage without the ball going out is often a foul, where the rule is weaker);
then refit the models with these features and decode again.

On the training matches pass 1 is in-sample, so its stoppages are a little cleaner there than on
validation; validation decides whether the hint helps.

    python -m possession.bench train hmm_restart
"""

import numpy as np

from possession.experiments import hmm_touch, hmm_viterbi
from possession.hmm_models import HmmModels

FADE = 125      # frames (5 s): the hint halves after this long
LOOK = 50       # frames (2 s) before a stoppage in which the ball may have been seen out
CAP = 750

GRID = {"w": [0.4, 0.7], "switch": [0.5, 1.0], "dead": [1.0], "dead_min": [1]}
DEFAULT = {"w": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 1}  # replaced by `bench train`


def dead_runs(labels):
    """[(start, stop)] maximal runs of "D"."""
    hit = np.concatenate([[0], (np.asarray(labels) == "D").astype(np.int8), [0]])
    edges = np.diff(hit)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def restart_hints(first_pass, last_control, ball_out):
    """(hint, hint_out, log_since) per frame.

    first_pass: (n,) "H"/"A"/"D" labels; last_control: (n,) +1 home / -1 away / 0, the side of the
    last control at or before each frame; ball_out: (n,) bool, the ball seen off the pitch.
    """
    n = len(first_pass)
    hint, hint_out = np.zeros(n), np.zeros(n)
    since = np.full(n, float(CAP))
    runs = dead_runs(first_pass)
    for i, (start, stop) in enumerate(runs):
        side = -last_control[start - 1] if start > 0 else 0.0
        went_out = bool(ball_out[max(start - LOOK, 0):stop].any())
        end = runs[i + 1][0] if i + 1 < len(runs) else n
        t = np.arange(end - stop)
        fade = 1.0 / (1.0 + t / FADE)
        hint[stop:end] = side * fade
        hint_out[stop:end] = side * fade * went_out
        since[stop:end] = np.minimum(t, CAP)
    return hint, hint_out, np.log1p(since)


def restart_features(match):
    first_pass = hmm_touch.predict(match)
    ball, names = hmm_viterbi.ball_features(match)
    col = {name: ball[:, i] for i, name in enumerate(names)}
    seen = ~np.isnan(np.asarray(match.ball_xy)[:, 0])
    reach, margin = hmm_touch.STRENGTHS[0]
    home = seen & (col["near_home"] <= reach) & (col["near_margin"] <= -margin)
    away = seen & (col["near_away"] <= reach) & (col["near_margin"] >= margin)
    last_control = hmm_touch.carry(home.astype(float) - away)[0]
    out = seen & (col["ball_out"] > 0)
    hint, hint_out, log_since = restart_hints(first_pass, last_control, out)
    return np.column_stack([hint, hint_out, log_since]), ("restart", "restart_out",
                                                          "log_since_restart")


def features(match):
    X, names = hmm_touch.features(match)
    R, rnames = restart_features(match)
    return np.column_stack([X, R]), names + rnames


MODEL = HmmModels("hmm_restart", features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
