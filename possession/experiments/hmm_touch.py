"""hmm_context plus "who touched it last": possession sticks with the side that last controlled the
ball, through passes in flight and tracking gaps, so the evidence says which side that was.

Diagnosis on the training matches (scripts/bench_errors.py hmm_context): most true possessions are
short (1451 of 1814 under 10 s) and few of those were found; even with the ball tracked, the side
was wrong on 12-13% of frames; and short false stoppages split possessions. So, on top of
hmm_context's features:
  - control evidence at two strengths (within 2 m with a 1 m margin; within 3 m with 0.5 m);
  - for each, the side of the last control before the frame and of the next one after it, how
    long ago / until, and whether the two agree;
  - the ball's movement along the pitch over ±0.2 s (towards which goal it is travelling);
and `dead_min` in the GRID: a stoppage must last at least that many frames (possession/hmm.py).
All from tracking; the whole match is known when it is labelled, so looking ahead is fair.

    python -m possession.bench train hmm_touch
"""

import numpy as np

from possession.experiments import hmm_context, hmm_viterbi
from possession.hmm_models import HmmModels

STRENGTHS = ((2.0, 1.0), (3.0, 0.5))   # (reach, margin) in metres
CAP = 750                               # frames: longer than 30 s is simply long
LAG = 5                                 # frames either side for the ball's movement

GRID = {"w": [0.4, 0.7], "switch": [0.5, 1.0], "dead": [0.75, 1.0], "dead_min": [1, 50, 100]}
DEFAULT = {"w": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 100}  # chosen by `bench train`


def carry(side):
    """(last side, frames since, next side, frames until) of the nonzero entries of `side`."""
    n = len(side)
    has = side != 0
    last = hmm_context.last_index(has)
    nxt = hmm_context.next_index(has)
    t = np.arange(n)
    last_side = np.where(last >= 0, side[np.clip(last, 0, n - 1)], 0.0)
    next_side = np.where(nxt < n, side[np.clip(nxt, 0, n - 1)], 0.0)
    since = np.minimum(np.where(last >= 0, t - last, CAP), CAP)
    until = np.minimum(np.where(nxt < n, nxt - t, CAP), CAP)
    return last_side, since, next_side, until


def touch_features(match):
    ball, names = hmm_viterbi.ball_features(match)
    col = {name: ball[:, i] for i, name in enumerate(names)}
    seen = ~np.isnan(np.asarray(match.ball_xy)[:, 0])
    columns, out = [], []
    for reach, margin in STRENGTHS:
        home = seen & (col["near_home"] <= reach) & (col["near_margin"] <= -margin)
        away = seen & (col["near_away"] <= reach) & (col["near_margin"] >= margin)
        side = home.astype(float) - away
        last_side, since, next_side, until = carry(side)
        tag = "r{:g}".format(reach)
        for name, values in (("last", last_side), ("next", next_side),
                             ("agree", last_side * next_side),
                             ("log_since", np.log1p(since)), ("log_until", np.log1p(until)),
                             ("last_fresh", last_side / (1 + since / 25.0)),
                             ("next_fresh", next_side / (1 + until / 25.0))):
            columns.append(values)
            out.append("{}_{}".format(name, tag))

    x = np.where(seen, col["ball_x"], np.nan)
    n = len(x)
    ahead = np.r_[x[LAG:], np.full(LAG, np.nan)]
    behind = np.r_[np.full(LAG, np.nan), x[:-LAG]]
    dx = np.nan_to_num(np.clip(ahead - behind, -20, 20))
    columns.append(dx)
    out.append("ball_dx")
    assert len(columns[0]) == n
    return np.column_stack(columns), tuple(out)


def features(match):
    X, names = hmm_context.all_features(match)
    T, tnames = touch_features(match)
    return np.nan_to_num(np.column_stack([X, T])), names + tnames


MODEL = HmmModels("hmm_touch", features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
