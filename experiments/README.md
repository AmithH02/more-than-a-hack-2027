# Experiments

There is no leaderboard, so this folder is our record of what we tried and how it looked.

## One file per experiment

`experiments/<issue>-<slug>.md`, e.g. `experiments/12-hysteresis-control.md`. Separate files mean
parallel sessions never edit the same file. Each record has:

- **Issue and commit**: the issue number and the commit that produced the result, so it can be
  regenerated.
- **Method**: what changed against the previous best, in a few lines.
- **Command**: the exact command that produced the CSV.
- **Proxy metrics**: the label-free checks (see [CONTEXT.md](../CONTEXT.md#scoring)), e.g. number of
  spells, spell-length distribution, flips per minute, dead-ball share, agreement with Smart
  Tagging.
- **Validation**: whether `evaluation/validate_submission.py` passed.
- **Verdict**: kept, dropped or superseded, and why.

## What gets committed

- These Markdown records, and `experiments/metrics/<experiment>.json`, the proxy report the
  runner writes on every run (`python -m possession run <experiment>`, step 7). Its thresholds are
  provisional until #5; a warning in it is a prompt to look, never a verdict.
- **Never** experiment CSVs (they go in the gitignored `outputs/`). Only the final
  `<team_name>.csv` is committed.
- **Never** a score against the public 5-minute ground truth used to choose between experiments.
  That is a pipeline sanity check only (RULES.md).

## Summary

| Issue | Experiment | Key proxy metrics | Verdict |
|---|---|---|---|
| — | `baseline_smooth50` (organisers') | published full-match spellF1 0.2433 | the floor to beat |
| #20 | `nearest_smooth` (the same baseline, window from the bench) | bench val spellF1 0.184, macroF1 0.455 (degradation v1); 0.176 / 0.444 (v2) | the bench's reference point |
| #22 | `players_fill` (ball gaps read from the team shape) | bench val spellF1 0.253, macroF1 0.521 (v2); BAR-ATH: Smart Tagging agreement 0.727 vs 0.714, sub-second possessions 15% vs 21% | candidate; the bench overstates its gain on BAR-ATH |
| #24 | `players_dead_ball` (dead stretches from Smart Tagging and the tracking over `players_fill`) | bench val spellF1 0.327 / macroF1 0.755 (v2), 0.334 / 0.752 (v3, ball lost in stoppages) vs `players_fill` 0.253 / 0.276; BAR-ATH: dead share 42% vs 26%, Smart Tagging agreement 0.761 vs 0.727, second half pre-kick-off all D | candidate |
