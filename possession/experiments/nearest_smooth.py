"""The organisers' baseline on the Match's arrays: the team of the player nearest the ball, D where
the ball is not tracked, then a centred majority vote of +/- `window` frames.

Same method as starter/baseline.py (whose smoothing it imports), so it runs on BAR-ATH and on the
other matches in possession/external.py alike. `window` is chosen on the training matches by
`python -m possession.bench train nearest_smooth` (GRID below); DEFAULT is the organisers' 50.
"""

import numpy as np

from starter.baseline import AWAY, HOME, NOBODY, smooth  # organisers': imported, never edited

GRID = {"window": [0, 12, 25, 50, 75, 100, 125, 175, 250]}
DEFAULT = {"window": 50}


def nearest_team_codes(ball_xy, track_xy, track_team):
    """(n,) int8: HOME / AWAY for the side of the nearest tracked player, NOBODY without a ball."""
    out = np.full(len(ball_xy), NOBODY, dtype=np.int8)
    seen = ~np.isnan(ball_xy[:, 0])
    distance = np.linalg.norm(track_xy[seen] - ball_xy[seen, None, :], axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argmin(distance, axis=1)
    anyone = np.isfinite(distance.min(axis=1))
    team = np.where(np.asarray(track_team)[nearest] == "H", HOME, AWAY).astype(np.int8)
    out[np.flatnonzero(seen)[anyone]] = team[anyone]
    return out


def predict_with(match, window):
    return smooth(nearest_team_codes(match.ball_xy, match.track_xy, match.track_team), window)


def predict(match):
    return predict_with(match, **DEFAULT)
