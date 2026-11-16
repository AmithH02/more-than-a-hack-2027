"""hmm_viterbi with context: each frame's evidence also sees the frames around it.

The same 3-state HMM and Viterbi (possession/hmm.py) as hmm_viterbi. What changes is the evidence:
  - control evidence (as in #5's prototype: a side's nearest player within REACH metres of the ball,
    the other side's at least MARGIN further), averaged over centred windows of ±0.5 s and ±2 s;
  - how often the ball is tracked around the frame, and its mean speed there;
  - on hidden-ball frames: how long since the ball was last seen and until it is seen again, and
    which side was nearest to it at each end of the gap;
  - both model pairs (ball tracked, ball hidden) are our own, fitted with these features; the
    hidden-ball pair no longer reuses players_fill's.
The whole match is known when we label it, so centred windows and the far end of a gap are fair:
they use tracking only, never labels.

    python -m possession.bench train hmm_context
"""

import json
import os
import weakref

import numpy as np

from possession import hmm, learn, team_shape
from possession.experiments import hmm_viterbi

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "hmm_context.json")
EVERY = 5
EPS = hmm_viterbi.EPS
REACH, MARGIN = 2.0, 1.0       # m: control evidence, as in #5's prototype
WINDOWS = (12, 50)             # frames either side: ±0.5 s and ±2 s
GAP_CAP = 750                  # frames (30 s): longer than this is simply long

GRID = {"w": [0.2, 0.4, 0.7], "switch": [0.5, 1.0], "dead": [0.75, 1.0]}
DEFAULT = {"w": 0.4, "switch": 0.5, "dead": 0.75}  # replaced by what `bench train` chooses


# Features ----------------------------------------------------------------------------------------

def centred_mean(values, half):
    """Mean of `values` over [t - half, t + half], shrinking at the ends."""
    values = np.asarray(values, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(values)])
    n = len(values)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def last_index(mask):
    """(n,) the last index <= t where mask is True, -1 if none."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def next_index(mask):
    """(n,) the first index >= t where mask is True, n if none."""
    n = len(mask)
    rev = last_index(mask[::-1])[::-1]
    return np.where(rev >= 0, n - 1 - rev, n)


def context_features(match):
    """(X, names): windowed control evidence, ball activity and gap context, per frame."""
    ball, names = hmm_viterbi.ball_features(match)
    col = {name: ball[:, i] for i, name in enumerate(names)}
    seen = ~np.isnan(np.asarray(match.ball_xy)[:, 0])
    n = len(seen)
    margin = col["near_margin"]            # home distance minus away distance
    ev_home = seen & (col["near_home"] <= REACH) & (margin <= -MARGIN)
    ev_away = seen & (col["near_away"] <= REACH) & (margin >= MARGIN)
    side = np.where(seen, np.sign(-margin), 0.0)     # +1 home nearer, -1 away nearer

    columns, out_names = [], []
    for half in WINDOWS:
        for name, values in (("ev_home", ev_home), ("ev_away", ev_away), ("seen", seen),
                             ("speed", np.where(seen, col["ball_speed"], 0.0)),
                             ("side", side)):
            columns.append(centred_mean(values, half))
            out_names.append("{}_w{}".format(name, half))

    last, nxt = last_index(seen), next_index(seen)
    since = np.where(last >= 0, np.arange(n) - last, GAP_CAP)
    until = np.where(nxt < n, nxt - np.arange(n), GAP_CAP)
    side_last = np.where(last >= 0, side[np.clip(last, 0, n - 1)], 0.0)
    side_next = np.where(nxt < n, side[np.clip(nxt, 0, n - 1)], 0.0)
    for name, values in (("log_since", np.log1p(np.minimum(since, GAP_CAP))),
                         ("log_until", np.log1p(np.minimum(until, GAP_CAP))),
                         ("side_last", side_last), ("side_next", side_next),
                         ("ev_now", ev_home.astype(float) - ev_away)):
        columns.append(values)
        out_names.append(name)
    return np.column_stack(columns), tuple(out_names)


def all_features(match):
    ball, bn = hmm_viterbi.ball_features(match)
    shape, sn = team_shape.features(match)
    ctx, cn = context_features(match)
    return np.nan_to_num(np.column_stack([ball, shape, ctx])), bn + sn + cn


# Fitting -----------------------------------------------------------------------------------------

def fit(matches, path=None, say=print):
    """Fit both model pairs (ball tracked, ball hidden) and the transition rates on `matches`."""
    path = path or MODEL_PATH
    parts = {"seen": ([], [], []), "hidden": ([], [], [])}
    for match in matches:
        X, names = all_features(match)
        hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
        for key, mask in (("seen", ~hidden), ("hidden", hidden)):
            rows = np.flatnonzero(mask)[::EVERY]
            parts[key][0].append(X[rows])
            parts[key][1].append(match.labels[rows] != "D")
            parts[key][2].append(match.labels[rows] == "H")

    models = {
        "about": "hmm_context models, fitted by `python -m possession.bench train hmm_context` on "
                 "the bench's training matches (ADR 0003).",
        "names": list(names),
    }
    for key, (rows, alive, home) in parts.items():
        X, alive, home = np.concatenate(rows), np.concatenate(alive), np.concatenate(home)
        if alive.sum() == 0 or alive.all() or len(X) < 2:
            # nothing to learn from (e.g. the ball is never hidden): use the other pair's frames
            other = parts["seen" if key == "hidden" else "hidden"]
            X, alive, home = (np.concatenate(other[0]), np.concatenate(other[1]),
                              np.concatenate(other[2]))
        models[key] = {"alive": learn.fit_logistic(X, alive, names=names),
                       "home": learn.fit_logistic(X[alive], home[alive], names=names)}
        say("  {}: fitted on {} frames ({} alive)".format(key, len(X), int(alive.sum())))

    halves, ids = [], set()
    for match in matches:
        if match.match_id not in ids:
            ids.add(match.match_id)
            halves += [match.labels[first:last + 1] for first, last in match.blocks]
    labels = np.concatenate(halves)
    models["matches"] = sorted(ids)
    models["rates"] = hmm.fit_rates(halves)
    models["prior"] = {s: float(np.mean(labels == s)) for s in hmm.STATES}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(models, handle, indent=1)
        handle.write("\n")
    say("  saved {}".format(os.path.relpath(path)))
    _load.cache = None
    _STATE_LOGP.clear()
    return models


def _load(path=None):
    if getattr(_load, "cache", None) is None:
        with open(path or MODEL_PATH) as handle:
            _load.cache = json.load(handle)
    return _load.cache


# Prediction --------------------------------------------------------------------------------------

_STATE_LOGP = weakref.WeakKeyDictionary()


def state_logp(match):
    if match in _STATE_LOGP:
        return _STATE_LOGP[match]
    models = _load()
    X, _ = all_features(match)
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
    p_alive = np.where(hidden, learn.predict_proba(models["hidden"]["alive"], X),
                       learn.predict_proba(models["seen"]["alive"], X))
    p_home = np.where(hidden, learn.predict_proba(models["hidden"]["home"], X),
                      learn.predict_proba(models["seen"]["home"], X))
    p_alive, p_home = np.clip(p_alive, EPS, 1 - EPS), np.clip(p_home, EPS, 1 - EPS)
    logp = np.log(np.column_stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive]))
    _STATE_LOGP[match] = logp
    return logp


def predict_with(match, w, switch, dead):
    models = _load()
    log_prior = np.log([models["prior"][s] for s in hmm.STATES])
    log_emit = w * (state_logp(match) - log_prior)
    T = hmm.transitions(models["rates"], switch=switch, dead=dead)
    return hmm.decode(log_emit, T, team_shape.blocks_of(match))


def predict(match):
    return predict_with(match, **DEFAULT)
