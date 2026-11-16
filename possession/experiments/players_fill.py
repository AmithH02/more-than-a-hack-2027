"""The baseline, with the players filling in where the tracking has lost the ball.

Where the ball is tracked: the side of the nearest player, as in nearest_smooth. Where it is not,
two small models read the players instead (possession/team_shape.py: each side's centre, spread,
forward movement, how many are running):
  - alive: is play going on, or is it stopped?  If stopped, D.
  - home:  if play is on, which side has the ball?
Then the same centred majority vote as the baseline.

The models are fitted on the bench's training matches only (ADR 0003), with the ball hidden and
the players broken into tracklets like the ATD, and only on the frames where the ball is hidden,
because those are the frames they are used on. They are stored in possession/models/ and refitted
by:

    python -m possession.bench train players_fill
"""

import json
import os

import numpy as np

from possession import learn, team_shape
from possession.experiments.nearest_smooth import nearest_team_codes
from starter.baseline import AWAY, HOME, NOBODY, smooth  # organisers': imported, never edited

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "players_fill.json")
EVERY = 5   # fit on every 5th hidden frame: neighbouring frames are near copies, and it is 5x faster

GRID = {"window": [25, 50, 75, 125], "alive_at": [0.4, 0.5, 0.6]}
DEFAULT = {"window": 50, "alive_at": 0.6}  # chosen by `bench train` (experiments/metrics/bench)


def fit(matches, path=MODEL_PATH, say=print):
    """Fit both models on the hidden-ball frames of `matches` (they carry labels) and save them."""
    rows, alive, home = [], [], []
    for match in matches:
        X, names = team_shape.features(match)
        hidden = np.flatnonzero(np.isnan(match.ball_xy[:, 0]))[::EVERY]
        rows.append(X[hidden])
        alive.append(match.labels[hidden] != "D")
        home.append(match.labels[hidden] == "H")
    X, alive, home = np.concatenate(rows), np.concatenate(alive), np.concatenate(home)
    models = {
        "about": "players_fill models, fitted by `python -m possession.bench train players_fill` "
                 "on the bench's training matches (ADR 0003). Hidden-ball frames only.",
        "matches": sorted({repr(m)[14:-1] for m in matches}),
        "alive": learn.fit_logistic(X, alive, names=names),
        "home": learn.fit_logistic(X[alive], home[alive], names=names),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(models, handle, indent=1)
        handle.write("\n")
    say("  fitted on {} hidden-ball frames ({} alive); saved {}".format(
        len(X), int(alive.sum()), os.path.relpath(path)))
    _load.cache = None
    return models


def _load(path=MODEL_PATH):
    if getattr(_load, "cache", None) is None:
        with open(path) as handle:
            _load.cache = json.load(handle)
    return _load.cache


def gap_codes(match, alive_at, models=None):
    """(n,) int8 for every frame: HOME / AWAY / NOBODY as the players say; used where the ball is
    hidden."""
    models = models or _load()
    X, _ = team_shape.features(match)
    alive = learn.predict_proba(models["alive"], X) >= alive_at
    home = learn.predict_proba(models["home"], X) >= 0.5
    return np.where(alive, np.where(home, HOME, AWAY), NOBODY).astype(np.int8)


def predict_with(match, window, alive_at):
    codes = nearest_team_codes(match.ball_xy, match.track_xy, match.track_team)
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
    codes = np.where(hidden, gap_codes(match, alive_at), codes).astype(np.int8)
    return smooth(codes, window)


def predict(match):
    return predict_with(match, **DEFAULT)
