"""Score an experiment on other matches that have a full ground truth (possession/external.py).

    python -m possession.bench val   <experiment>      # score on the validation match(es)
    python -m possession.bench train <experiment>      # pick the experiment's GRID settings on the
                                                       #   training matches, then score them on val
    python -m possession.bench test  <experiment>      # the held-out matches: once per candidate
    options: --seeds 0 1 2   degradation seeds (default 0 1 2)
             --clean         score on the clean DFL tracking instead (an upper bound)
             --version 3     degradation version (default degrade.VERSION; 3 hides the ball
                             mostly in stoppages, as the ATD does); saved under "<split>_v3"
             --set window=75 run with these settings instead of the experiment's DEFAULT

An experiment is benchable as it is: predict(match) must use only attributes ExternalMatch has.
To be trainable it also defines GRID ({name: [values]}) and predict_with(match, **settings). An
experiment with learned parts also defines fit(matches, say): `train` calls it first, on the
training matches (which carry labels), and the experiment saves what it learnt in the repo.

Scores are the official ones (evaluation/possession_metrics.evaluate) after the same fill as the
runner's step 5, averaged over matches and degradation seeds. Results go to
experiments/metrics/bench/<experiment>.json, one entry per split, and are printed.
"""

import argparse
import importlib
import itertools
import json
import os
import sys
import time

import numpy as np

from possession import REPO, external
from possession.fill import fill

import possession_metrics as pm  # noqa: E402  (evaluation/ is put on the path by possession)

METRICS_DIR = os.path.join(REPO, "experiments", "metrics", "bench")
HEADLINE = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1",
            "team_acc_both_alive")


def load_experiment(name):
    return importlib.import_module("possession.experiments.{}".format(name))


def score(labels, match):
    """The official metrics of `labels` against the match's ground truth, after filling X."""
    filled = fill(labels, blocks=match.blocks)
    result = pm.evaluate(pm.label_codes(match.labels), pm.label_codes(filled), bootstrap_n=0)
    return {key: float(result[key]) for key in HEADLINE}


def run_on(matches, predict):
    """[(match, scores)] for each match."""
    out = []
    for match in matches:
        labels = np.asarray(predict(match))
        if labels.shape != (match.n_frames,):
            raise ValueError("{} returned {} labels for {} frames".format(
                match, labels.shape, match.n_frames))
        out.append((match, score(labels, match)))
    return out


def mean(rows, key):
    return float(np.mean([scores[key] for _, scores in rows]))


def matches_for(split, seeds, clean, version=None):
    if clean:
        return external.split(split)
    return [m for seed in seeds for m in external.split(split, degrade=seed, version=version)]


def report(title, rows, say):
    say(title)
    for match, s in rows:
        say("  {:<28} spellF1 {:.3f}  macroF1 {:.3f}  spells {:>4.0f}/{:<4.0f} alive/dead {:.3f}"
            "  team {:.3f}".format(repr(match)[14:-1], s["spell_f1_mean"], s["macro_f1"],
                                   s["n_pred_spells"], s["n_gt_spells"], s["alive_dead_f1"],
                                   s["team_acc_both_alive"]))
    say("  {:<28} spellF1 {:.3f}  macroF1 {:.3f}".format(
        "MEAN", mean(rows, "spell_f1_mean"), mean(rows, "macro_f1")))


def train(module, matches, say):
    """The GRID setting with the best mean spellF1 on `matches`; ties go to the first listed."""
    grid = getattr(module, "GRID", None)
    if not grid or not hasattr(module, "predict_with"):
        raise SystemExit("{} has no GRID and predict_with(); nothing to train".format(
            module.__name__))
    if hasattr(module, "fit"):
        say("  fitting {}'s models".format(module.__name__.rsplit(".", 1)[-1]))
        module.fit(matches, say=say)
    names = list(grid)
    table = []
    for values in itertools.product(*(grid[n] for n in names)):
        settings = dict(zip(names, values))
        rows = run_on(matches, lambda m: module.predict_with(m, **settings))
        table.append((settings, mean(rows, "spell_f1_mean"), mean(rows, "macro_f1")))
        say("  {:<40} spellF1 {:.3f}  macroF1 {:.3f}".format(
            json.dumps(settings), table[-1][1], table[-1][2]))
    best = max(table, key=lambda row: row[1])
    say("  chosen: {}".format(json.dumps(best[0])))
    return best[0], table


def save(experiment, split, entry, metrics_dir=METRICS_DIR):
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, "{}.json".format(experiment))
    stored = {}
    if os.path.isfile(path):
        with open(path) as handle:
            stored = json.load(handle)
    stored[split] = entry
    with open(path, "w") as handle:
        json.dump(stored, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def parse_settings(pairs):
    out = {}
    for pair in pairs or ():
        name, _, value = pair.partition("=")
        out[name] = json.loads(value)
    return out


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(prog="python -m possession.bench",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("split", choices=("train", "val", "test"))
    parser.add_argument("experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--set", nargs="+", metavar="NAME=VALUE")
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args(argv)

    # The other matches, and BAR-ATH's ATD whose ball gaps the degradation copies.
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import check_data
    import fetch_idsse
    problems = fetch_idsse.check()
    if problems:
        raise SystemExit("\n".join(problems + ["run: python scripts/fetch_idsse.py"]))
    if not args.clean:
        check_data.check_data()

    module = load_experiment(args.experiment)
    started = time.time()
    from possession import degrade
    version = args.version or degrade.VERSION
    regime = "clean" if args.clean else "degraded v{} seeds {}".format(version, args.seeds)
    entry = {"regime": regime}

    settings = parse_settings(args.set)
    if args.split == "train":
        say("training {} on {} ({})".format(args.experiment, ", ".join(external.SPLITS["train"]),
                                            regime))
        settings, table = train(module, matches_for("train", args.seeds, args.clean, version),
                                say)
        entry["train_grid"] = [{"settings": s, "spell_f1_mean": f, "macro_f1": m}
                               for s, f, m in table]
        split = "val"
    else:
        split = args.split

    if settings:
        predict = lambda m: module.predict_with(m, **settings)  # noqa: E731
    else:
        predict = module.predict
        settings = getattr(module, "DEFAULT", {})
    rows = run_on(matches_for(split, args.seeds, args.clean, version), predict)
    report("{} on {} with {} ({})".format(args.experiment, split, json.dumps(settings), regime),
           rows, say)
    entry.update({
        "settings": settings,
        "split": split,
        "spell_f1_mean": mean(rows, "spell_f1_mean"),
        "macro_f1": mean(rows, "macro_f1"),
        "per_match": [dict(match=m.match_id, degraded=m.degraded, **s) for m, s in rows],
    })
    key = args.split if not args.clean else args.split + "_clean"
    if not args.clean and version != degrade.VERSION:
        key += "_v{}".format(version)
    path = save(args.experiment, key, entry)
    say("saved to {} ({:.0f} s)".format(os.path.relpath(path, REPO), time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
