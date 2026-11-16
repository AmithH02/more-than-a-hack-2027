"""players_fill, with the dead ball found from Smart Tagging and the tracking (possession/dead_ball.py).

  1. players_fill's labels: nearest player where the ball is tracked, the team-shape models in
     its gaps, the ±window majority vote.
  2. Every dead stretch the dead-ball module finds overrides them with D: each half until its
     kick-off, each goal until the kick-off after it, each tagged set piece from the ball going
     out (or the players slowing) to its restart, and the stoppages Smart Tagging missed.
  3. After each restart found, play is alive for `live_after_s` seconds (unless a new stoppage
     starts): the side Smart Tagging tags, else the first side players_fill gives after it.

The players_fill models are reused as they are (possession/models/players_fill.json). Settings
are chosen on the bench's training matches:

    python -m possession.bench train players_dead_ball [--version 3]
"""

import numpy as np

from possession import dead_ball
from possession.experiments import players_fill

GRID = {"dead_at": [None, 1.0, 1.2], "still_dead_s": [None, 3.0], "live_after_s": [0, 3]}
DEFAULT = {"dead_at": 1.2, "still_dead_s": 3.0, "live_after_s": 3}
LIVE_AT = 1.6        # m/s: players faster than this on average are playing (hysteresis top)
MIN_DEAD_S = 8.0     # s: the shortest untagged slow stretch called dead
SIDE_LOOK_S = 30     # s: how far after a restart to look for players_fill's side


def _base(match):
    """players_fill's labels with its own defaults, kept on the match for the grid search."""
    cached = getattr(match, "_players_fill_labels", None)
    if cached is None:
        cached = np.asarray(players_fill.predict(match)).astype("<U1")
        try:
            match._players_fill_labels = cached
        except AttributeError:
            pass
    return cached


def combine(base, stretches, live_after_s):
    """players_fill's labels with the dead stretches and restarts laid over them."""
    labels = np.array(base, dtype="<U1", copy=True)
    labels[stretches.dead] = "D"
    if not live_after_s:
        return labels
    n = len(labels)
    for r in stretches.restarts:
        a = r.frame
        b = min(n, a + int(live_after_s * dead_ball.FPS))
        later = np.flatnonzero(stretches.dead[a:b])
        b = a + int(later[0]) if len(later) else b
        if b <= a:
            continue
        side = r.side
        if side is None:
            ahead = base[a:min(n, a + SIDE_LOOK_S * dead_ball.FPS)]
            alive = ahead[(ahead == "H") | (ahead == "A")]
            side = alive[0] if len(alive) else None
        if side is None:
            continue
        if r.side is not None:
            labels[a:b] = side
        else:
            span = labels[a:b]
            span[span == "D"] = side
    return labels


def predict_with(match, dead_at, still_dead_s, live_after_s):
    stretches = dead_ball.detect(match, dead_at=dead_at, live_at=LIVE_AT, min_dead_s=MIN_DEAD_S,
                                 still_dead_s=still_dead_s)
    return combine(_base(match), stretches, live_after_s)


def predict(match):
    return predict_with(match, **DEFAULT)
