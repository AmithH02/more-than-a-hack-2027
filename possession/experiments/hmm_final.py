"""The final candidate: hmm_touch's method and settings, refitted on every non-test match.

hmm_touch was chosen on validation (spellF1 0.453; every later variant within noise, see
experiments/26-hmm.md). Its settings were chosen on the training matches and are frozen here; the
models are refitted on train + validation (5 matches, degradation v2, seeds 0-2), so the matches
that decided the method also teach it. Test is untouched until `bench test`, run once.

    python -m possession.experiments.hmm_final      # refit possession/models/hmm_final.json
    python -m possession.bench test hmm_final        # once
    python -m possession run hmm_final               # BAR-ATH
"""

import sys

from possession import external
from possession.experiments import hmm_touch
from possession.hmm_models import HmmModels

SEEDS = (0, 1, 2)
DEFAULT = dict(hmm_touch.DEFAULT)   # {"w": 0.7, "switch": 0.5, "dead": 1.0, "dead_min": 100}

MODEL = HmmModels("hmm_final", hmm_touch.features)
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)


def refit(say=print):
    """Fit on train + validation, every degradation seed; checks the data first."""
    sys.path.insert(0, external.REPO + "/scripts")
    import check_data
    import fetch_idsse
    problems = fetch_idsse.check()
    if problems:
        raise SystemExit("\n".join(problems))
    check_data.check_data()
    matches = [m for seed in SEEDS for split in ("train", "val")
               for m in external.split(split, degrade=seed)]
    say("refitting hmm_final on {} ({} match copies)".format(
        ", ".join(external.SPLITS["train"] + external.SPLITS["val"]), len(matches)))
    return MODEL.fit(matches, say=say)


if __name__ == "__main__":
    refit()
