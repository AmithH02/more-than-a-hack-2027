# 0003: Judging methods on other matches that have a full ground truth

- Status: accepted (slice 1)
- Date: 2026-11-16
- Issue: #20

## Context

We may not tune on the public five minutes, and label-free proxies cannot tell us whether a change
gets possessions right. External data about other games is allowed (RULES.md, FAQ.md). The IDSSE
dataset (Bassek, Weber, Rein & Memmert, 2025, Scientific Data; CC BY 4.0) has seven full Bundesliga
and 2. Bundesliga matches with 25 Hz tracking, the DFL's per-frame ball status and ball possession,
and the official event data.

## Decision

- **Data**: `scripts/fetch_idsse.py` downloads it to `data/external/idsse/` (gitignored) and checks
  it against `scripts/idsse_checksums.json`. `possession/bench.py` refuses to run if that fails.
- **Same interface**: `possession/external.py` gives each match as an `ExternalMatch` with the
  attributes experiments read from the `Match` (same units and pitch frame), plus `labels` and
  `blocks` (the two halves). So any experiment that uses only tracking runs here unchanged.
- **Labels**: `D` where the DFL's BallStatus is 0, otherwise `H`/`A` from its BallPossession. This
  is the provider's possession, a close cousin of ours (CONTEXT.md), not the same definition.
  Rebuilding H/A from the event data to match "controlled possession" is a later slice.
- **Degradation**: experiments are scored on the clean tracking made to look like the ATD
  (`possession/degrade.py`), using only statistics of the ATD's own inputs in the graded halves,
  never its labels. Three seeds per match by default; a seed's ball mask is the same in every
  version.
  - Version 1 (#20) hides the ball, in alternating visible/hidden runs drawn from the ATD's run
    lengths.
  - Version 2 (#22, the default) also breaks the players into tracklets. How many are missing at
    each frame is copied from a random stretch of the ATD's per-frame shortfall; which ones follows
    random priorities redrawn at ATD tracklet lengths; each visible stretch goes into the first
    free slot of its side (24 home, 22 away, a frame's gap between tracklets). Per-frame counts
    match the ATD (mean 20.4 against 20.5, 5th percentile 14); tracklets come out shorter (median
    15 s against 35 s), so v2 is somewhat harsher than the ATD.
- **Learned parts**: an experiment may define `fit(matches, say)`. `bench train` calls it on the
  training matches before the settings grid, and the experiment saves what it learnt as a small
  file in `possession/models/`, committed, so `predict()` and the submission need no training
  run.
- **Split**, fixed before any score was read:
  train `J03WMX J03WN1 J03WOH J03WQQ`, validation `J03WOY`, test `J03WPY J03WR9`.
  Training picks an experiment's settings (its `GRID`, by mean spellF1); validation compares
  candidates; test is run once per candidate we intend to submit.
- **Scores** are the organisers' own (`evaluation/possession_metrics.evaluate`) after the runner's
  fill, and are saved to `experiments/metrics/bench/<experiment>.json`.
- **Viewer**: `python -m possession.viz --external <match> --experiments <names>` shows a match's
  degraded tracking, its full ground truth and each experiment's labels on the pitch.

## Consequences

- A first check that the bench transfers: on the training matches the organisers' baseline is best
  at a 50-frame window, the window that is best on the real match's full ground truth, at a similar
  level (spellF1 0.22 against 0.24).
- The scores measure agreement with the DFL's labelling under a simple degradation. A method can
  overfit either.
- **Where the gaps fall is not modelled, and it matters** (found in #22). The bench hides the ball
  at random, so about half the hidden frames are live play. BAR-ATH's real gaps are not random:
  inside them the players move at 1.1 m/s on average, against 2.1 m/s where the ball is tracked,
  and Smart Tagging puts most of them in crowded-box and goal phases. They are mostly stoppages.
  So the bench overstates what filling gaps is worth on BAR-ATH. Next version: draw gap positions
  from the kind of moments the ATD loses the ball in.
- One validation match (three seeds) is a small sample: differences under about 0.02 there are
  not evidence.
