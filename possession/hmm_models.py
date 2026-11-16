"""The machinery shared by the HMM experiments: fit two model pairs, turn them into per-frame state
probabilities, decode with Viterbi. An experiment only says which features it uses.

    MODEL = HmmModels("hmm_touch", features)       # features(match) -> (X, names)
    fit = MODEL.fit                                 # the bench's hook: fit(matches, say)
    predict_with = MODEL.predict_with               # (match, w, switch, dead[, dead_min])

Two logistic pairs (possession/learn.py), one for frames where the ball is tracked and one for
frames where it is hidden; each pair gives p(alive) and p(home | alive). Emissions are
w * (log p(state) - log prior(state)); transitions start from the rates counted in the training
labels, scaled by `switch` and `dead` (possession/hmm.py). `dead_min` > 1 makes a stoppage last at
least that many frames (a chain of D sub-states).
"""

import json
import os
import weakref

import numpy as np

from possession import hmm, learn, mlp, team_shape

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EVERY = 5
EPS = 1e-4


class HmmModels:
    def __init__(self, name, features, learner="logistic", **learner_options):
        self.name = name
        self.features = features
        self.learner = learner                  # "logistic" (learn.py) or "mlp" (mlp.py)
        self.learner_options = learner_options
        self.path = os.path.join(MODELS_DIR, name + ".json")
        self._models = None
        self._logp = weakref.WeakKeyDictionary()

    # Fitting -------------------------------------------------------------------------------------

    def fit(self, matches, path=None, say=print):
        parts = {"seen": ([], [], []), "hidden": ([], [], [])}
        names = None
        for match in matches:
            X, names = self.features(match)
            hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
            for key, mask in (("seen", ~hidden), ("hidden", hidden)):
                rows = np.flatnonzero(mask)[::EVERY]
                parts[key][0].append(X[rows])
                parts[key][1].append(match.labels[rows] != "D")
                parts[key][2].append(match.labels[rows] == "H")

        models = {"about": "{} models, fitted by `python -m possession.bench train {}` on the "
                           "bench's training matches (ADR 0003).".format(self.name, self.name),
                  "names": list(names)}
        for key in ("seen", "hidden"):
            rows, alive, home = parts[key]
            X, alive, home = np.concatenate(rows), np.concatenate(alive), np.concatenate(home)
            if len(X) < 2 or alive.all() or not alive.any():
                other = parts["seen" if key == "hidden" else "hidden"]
                X, alive, home = (np.concatenate(p) for p in other)
            models[key] = {"alive": self._fit(X, alive, names),
                           "home": self._fit(X[alive], home[alive], names)}
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

        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(models, handle, indent=1)
            handle.write("\n")
        say("  saved {}".format(os.path.relpath(path)))
        self.reset()
        return models

    def _fit(self, X, y, names):
        if self.learner == "mlp":
            return mlp.fit_mlp(X, y, names=names, **self.learner_options)
        return learn.fit_logistic(X, y, names=names)

    @staticmethod
    def _proba(model, X):
        if model.get("kind") == "mlp":
            return mlp.predict_proba(model, X)
        return learn.predict_proba(model, X)

    def reset(self):
        self._models = None
        self._logp = weakref.WeakKeyDictionary()

    def load(self):
        if self._models is None:
            with open(self.path) as handle:
                self._models = json.load(handle)
        return self._models

    # Prediction ----------------------------------------------------------------------------------

    def state_logp(self, match):
        """(n, 3) log p(H), log p(A), log p(D) per frame."""
        if match in self._logp:
            return self._logp[match]
        models = self.load()
        X, _ = self.features(match)
        hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])

        def both(kind):
            return np.where(hidden, self._proba(models["hidden"][kind], X),
                            self._proba(models["seen"][kind], X))

        p_alive = np.clip(both("alive"), EPS, 1 - EPS)
        p_home = np.clip(both("home"), EPS, 1 - EPS)
        logp = np.log(np.column_stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive]))
        self._logp[match] = logp
        return logp

    def predict_with(self, match, w, switch, dead, dead_min=1):
        models = self.load()
        log_prior = np.log([models["prior"][s] for s in hmm.STATES])
        log_emit = w * (self.state_logp(match) - log_prior)
        T = hmm.transitions(models["rates"], switch=switch, dead=dead)
        return hmm.decode(log_emit, T, team_shape.blocks_of(match), dead_min=dead_min)
