"""hmm_touch plus team movement over time and dead-ball cues.

Diagnosis on the training matches (scripts/bench_errors.py hmm_touch): with the ball hidden, the
side is wrong on 13-19% of live frames; with it tracked, 11% of dead-ball frames are called live
(stoppage edges). So, on top of hmm_touch's features:
  - team movement: centred means over ±2 s and ±5 s of which side is moving forward (vx_diff),
    where the sides are (gap_x), how fast everyone moves (mean_speed, share_running);
  - ball stillness: how long the ball has been below 1 m/s before the frame and stays so after;
  - set-piece spots: the ball near a corner, the centre spot, a penalty spot, or on a touchline.

    python -m possession.bench train hmm_rich
"""

import numpy as np

from possession import team_shape
from possession.experiments import hmm_context, hmm_touch
from possession.hmm_models import HmmModels

WINDOWS = (50, 125)        # frames either side: ±2 s and ±5 s
SHAPE = ("vx_diff", "gap_x", "mean_speed", "share_running")
STILL = 1.0                # m/s
CAP = 750

GRID = {"w": [0.4, 0.7], "switch": [0.5, 1.0], "dead": [1.0], "dead_min": [1]}
DEFAULT = {"w": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 1}  # replaced by `bench train`


def run_lengths(mask):
    """(frames the mask has been True up to t, frames it stays True from t), 0 where False."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    t = np.arange(n)
    start = hmm_context.last_index(~mask)            # last False at or before t
    stop = hmm_context.next_index(~mask)             # first False at or after t
    before = np.where(mask, t - start, 0)
    after = np.where(mask, stop - t, 0)
    return np.minimum(before, CAP), np.minimum(after, CAP)


def spot_distance(x, y, length, width):
    """Distance from the ball to the nearest corner, centre spot or penalty spot."""
    spots = np.array([(0, 0), (0, width), (length, 0), (length, width), (length / 2, width / 2),
                      (11, width / 2), (length - 11, width / 2)])
    d = np.sqrt((x[:, None] - spots[:, 0]) ** 2 + (y[:, None] - spots[:, 1]) ** 2)
    return d.min(axis=1)


def rich_features(match):
    table = team_shape.of(match)
    columns, names = [], []
    for name in SHAPE:
        values = np.nan_to_num(table[name])
        for half in WINDOWS:
            columns.append(hmm_context.centred_mean(values, half))
            names.append("{}_w{}".format(name, half))

    ball = np.asarray(match.ball_xy, dtype=np.float64)
    seen = ~np.isnan(ball[:, 0])
    speed = np.asarray(match.ball_speed, dtype=np.float64)
    still = seen & (np.nan_to_num(speed, nan=99.0) < STILL)
    before, after = run_lengths(still)
    length, width = team_shape.pitch_of(match)
    x, y = np.nan_to_num(ball[:, 0]), np.nan_to_num(ball[:, 1])
    spot = np.where(seen, np.minimum(spot_distance(x, y, length, width), 20.0), 20.0)
    line = np.where(seen, np.minimum(np.minimum(np.abs(y), np.abs(y - width)), 20.0), 20.0)
    for name, values in (("log_still_before", np.log1p(before)),
                         ("log_still_after", np.log1p(after)),
                         ("spot_dist", spot), ("near_spot", (spot < 2.0).astype(float)),
                         ("touchline_dist", line), ("on_touchline", (line < 1.0) * seen)):
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(name)
    return np.column_stack(columns), tuple(names)


def features(match):
    X, names = hmm_touch.features(match)
    R, rnames = rich_features(match)
    return np.nan_to_num(np.column_stack([X, R])), names + rnames


MODEL = HmmModels("hmm_rich", features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
