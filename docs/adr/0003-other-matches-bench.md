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
  never its labels. Three seeds per match by default; a seed's ball mask is the same in versions
  1 and 2.
  - Version 1 (#20) hides the ball, in alternating visible/hidden runs drawn from the ATD's run
    lengths.
  - Version 2 (#22, the default) also breaks the players into tracklets. How many are missing at
    each frame is copied from a random stretch of the ATD's per-frame shortfall; which ones follows
    random priorities redrawn at ATD tracklet lengths; each visible stretch goes into the first
    free slot of its side (24 home, 22 away, a frame's gap between tracklets). Per-frame counts
    match the ATD (mean 20.4 against 20.5, 5th percentile 14); tracklets come out shorter (median
    15 s against 35 s), so v2 is somewhat harsher than the ATD.
  - Version 3 (#24, `--version 3`, saved under `<split>_v3`) moves the ball's gaps to where the
    ATD loses it: 85% of stoppages (the DFL's dead runs) are hidden, except up to 2 s at each end,
    and live play loses the ball in ATD-length gaps (up to 10 s) on 10% of its frames. The labels
    place the gaps; no experiment sees them. The players are as in v2. It also hides the DFL's
    frozen ball: during a stoppage the DFL keeps the ball still (often beyond a line), which makes
    ball cues for dead play far too easy on v1/v2.
- **Stand-in Smart Tagging** (#24): the IDSSE matches have none, so `ExternalMatch` builds set-piece
  windows like BAR-ATH's from the DFL's own restarts in `event.xml` (`external.stand_in_windows`):
  35 s long (5 s for kick-offs), starting 4-8 s before the restart (8-16 s for free kicks, 1-4 s
  for kick-offs), as measured on BAR-ATH's inputs; a share of each type dropped so the counts look
  like BAR-ATH's (throw-ins and goal kicks 60% kept, free kicks 20%, corners and kick-offs all),
  half of the team tags replaced by N/A, and a GOALS phase from 30 s before each goal to its
  kick-off. Seeded by the match. They answer `smart_tagging_events` and the painted
  `smart_tagging_*` attributes, as the Match does.
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
  So the bench overstates what filling gaps is worth on BAR-ATH. Version 3 (#24) places them
  mostly in stoppages; its shares (85% of stoppages, 10% of live play) are a judgement from the
  BAR-ATH evidence above, not measured.
- One validation match (three seeds) is a small sample: differences under about 0.02 there are
  not evidence.
