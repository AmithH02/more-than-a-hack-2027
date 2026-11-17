"""Where an experiment loses points on the bench's TRAINING matches: read-only diagnostics.

    python scripts/bench_errors.py hmm_context            # degraded v2, seed 0
    python scripts/bench_errors.py hmm_context --seed 1

Training matches only, so looking at the errors never peeks at validation or test (ADR 0003).
Prints, summed over the training matches:
  - the H / A / D confusion, separately where the ball is tracked and where it is hidden;
  - true and predicted possessions by length, and how many are matched at IoU 0.5 (the middle
    of spellF1's 0.3-0.7 range), so missed short spells and spurious ones show up;
  - true and predicted stoppages (D runs) by length.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from possession import external  # noqa: E402
from possession.bench import load_experiment  # noqa: E402
import possession_metrics as pm  # noqa: E402  (evaluation/ is put on the path by possession)

BUCKETS = ((0, 50, "<2 s"), (50, 250, "2-10 s"), (250, 750, "10-30 s"), (750, 10 ** 9, ">30 s"))


def runs_of(labels, value):
    """[(start, stop)] maximal runs of `value`."""
    hit = np.concatenate([[0], (labels == value).astype(np.int8), [0]])
    edges = np.diff(hit)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bucket(length):
    for lo, hi, name in BUCKETS:
        if lo <= length < hi:
            return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    module = load_experiment(args.experiment)
    conf = {"tracked": np.zeros((3, 3), int), "hidden": np.zeros((3, 3), int)}
    spells = {name: {"true": 0, "true_hit": 0, "pred": 0, "pred_hit": 0} for *_, name in BUCKETS}
    dead = {name: {"true": 0, "pred": 0} for *_, name in BUCKETS}
    index = {"H": 0, "A": 1, "D": 2}

    for match in external.split("train", degrade=args.seed):
        pred = np.asarray(module.predict(match))
        truth = np.asarray(match.labels)
        hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
        for key, mask in (("tracked", ~hidden), ("hidden", hidden)):
            t = np.array([index[x] for x in truth[mask]])
            p = np.array([index[x] for x in pred[mask]])
            np.add.at(conf[key], (t, p), 1)

        runs = [(0, len(truth))]
        g = pm.possession_spells(pm.label_codes(truth), runs)
        q = pm.possession_spells(pm.label_codes(pred), runs)
        overlaps = pm.spell_overlaps(g, q)
        g_hit = pm.matched_gt_flags(overlaps, len(g), 0.5)
        q_hit = np.zeros(len(q), bool)
        used = set()
        for iou, gi, qi in overlaps:
            if iou < 0.5:
                break
            if gi in used or q_hit[qi]:
                continue
            used.add(gi)
            q_hit[qi] = True
        for (a, b, _), hit in zip(g, g_hit):
            spells[bucket(b - a)]["true"] += 1
            spells[bucket(b - a)]["true_hit"] += int(hit)
        for (a, b, _), hit in zip(q, q_hit):
            spells[bucket(b - a)]["pred"] += 1
            spells[bucket(b - a)]["pred_hit"] += int(hit)
        for key, labels in (("true", truth), ("pred", pred)):
            for a, b in runs_of(labels, "D"):
                dead[bucket(b - a)][key] += 1

    print("{} on train, degraded seed {}".format(args.experiment, args.seed))
    for key, cm in conf.items():
        print("\nconfusion where the ball is {} ({} frames); rows true, columns predicted, "
              "share of the row".format(key, cm.sum()))
        for i, name in enumerate("HAD"):
            row = cm[i] / max(cm[i].sum(), 1)
            print("  {}  H {:.2f}  A {:.2f}  D {:.2f}   ({} frames)".format(
                name, row[0], row[1], row[2], cm[i].sum()))
    print("\npossessions by length: true (matched at IoU 0.5) / predicted (matched)")
    for *_, name in BUCKETS:
        s = spells[name]
        print("  {:<8} true {:>5} ({:>4} = {:>3.0f}%)   pred {:>5} ({:>4} = {:>3.0f}%)".format(
            name, s["true"], s["true_hit"], 100 * s["true_hit"] / max(s["true"], 1),
            s["pred"], s["pred_hit"], 100 * s["pred_hit"] / max(s["pred"], 1)))
    print("\nstoppages (D runs) by length: true / predicted")
    for *_, name in BUCKETS:
        print("  {:<8} true {:>5}   pred {:>5}".format(name, dead[name]["true"], dead[name]["pred"]))


if __name__ == "__main__":
    main()
