# #24 `players_dead_ball`: dead ball from Smart Tagging and the tracking

- **Issue**: #24 (builds on #22, `players_fill`)
- **Method**: `players_fill`'s labels, overridden with `D` on the dead stretches that
  `possession/dead_ball.py` finds: each half until its kick-off, each goal until the kick-off after
  it, each Smart Tagging set piece from its onset (ball seen out, or players slowing) to its
  restart (found by the Laws), and untagged stoppages (ball seen out, ball lying still for 3 s,
  players slow for 8 s). Play is alive for 3 s after each restart, for Smart Tagging's side if
  tagged. Details in `docs/TECHNICAL.md`.
- **Commands**:
  `python -m possession.bench train players_dead_ball --version 3` (picks the settings),
  `python -m possession.bench val players_dead_ball [--version 3]`,
  `python -m possession run players_dead_ball` (BAR-ATH)
- **Metrics**: `experiments/metrics/bench/players_dead_ball.json`,
  `experiments/metrics/bench/players_fill.json` (adds `val_v3`), `experiments/metrics/players_dead_ball.json`
- **Validation**: `validate_submission.py` passes on `outputs/players_dead_ball.csv`.

## New on the bench

- **Stand-in Smart Tagging** for the IDSSE matches, from the DFL's own set-piece events
  (ADR 0003): BAR-ATH-shaped windows (35 s, 5 s for kick-offs, restart 4-8 s in), thinned to
  BAR-ATH's counts, half the team tags dropped.
- **Degradation v3**: the ball hidden mostly in stoppages, as the ATD loses it. It also hides the
  DFL's frozen ball: during the DFL's dead frames the ball does not move at all (speed exactly 0,
  74% of the time beyond a line), which flatters any ball-based dead-ball cue under v1/v2.

## Bench, validation (J03WOY, 3 seeds)

| | `players_fill` | `players_dead_ball` |
|---|---|---|
| spellF1, degradation v2 | 0.253 | **0.327** |
| macroF1, v2 | 0.521 | 0.755 |
| spellF1, degradation v3 (ball lost in stoppages) | 0.276 | **0.334** |
| macroF1, v3 | 0.657 | 0.752 |

Every seed improves under both versions (v3 per seed: 0.341 / 0.323 / 0.339 against
0.280 / 0.271 / 0.277). Test not run.

- **Settings** (`train --version 3`): `dead_at` 1.2 m/s, `still_dead_s` 3 s, `live_after_s` 3 s;
  the twelve settings span 0.333-0.345 on train, so they barely matter. Under v2, training prefers
  no slow-player rule and no live period (0.329 against 0.327), noise.
- **What the windows add**: on validation, emptying the stand-in Smart Tagging costs 0.006 (v2)
  and 0.011 (v3). Most of the gain is from the untagged stoppages and the kick-offs.
- **The detector alone** (validation, mean of seeds): 92-93% of the frames it calls dead are dead;
  it finds 70% (v2) and 60% (v3) of the dead frames. Restarts in windows: about half within 1 s of
  the DFL's, about 85% within 3 s (training matches, v3); the DFL's labels turn alive about 1 s
  after the ball is struck, so part of that is a definition gap.

## BAR-ATH, label-free (graded halves)

| | `players_fill` | `players_dead_ball` |
|---|---|---|
| possessions | 560 | 499 |
| median possession | 5.2 s | 4.4 s |
| sub-second possessions | 14.6% | 15.4% |
| dead-ball share | 25.9% | 42.0% |
| `D` where the ball is tracked | 0% | 17% |
| `D` in the first 4 s of set-piece windows | 39% | 84% |
| agreement with Smart Tagging | 0.727 | 0.761 |
| second half, before its kick-off (2,961 frames) | 92% H/A | **all `D`** |

Restarts found in the 59 set-piece windows: 35 from the ball, 2 from the formation, 19 from the
players speeding up, 3 from the median offset; 48 untagged stoppages. The kick-offs are found
at frames 11187 (the published first-half kick-off is 11185) and 113769 (published 113742), both
at run time. One goal's stoppage comes out at 161 s (onset from the players, restart from the
median offset); the other three at 63-82 s.

## Verdict

Candidate, clearly better than `players_fill` on the bench under both degradations (+0.06 to
+0.07 spellF1, every seed) and on every BAR-ATH proxy except sub-second possessions (flat). The
dead-ball share of 42% is in the Bundesliga's range (41-52% on the training matches) but has no
BAR-ATH check beyond proxies. Not run on test; the owner decides the submission.

Next: the restart's side (who takes it) from the formation when Smart Tagging says N/A; a
goal-line check (ball in the net) for goal onsets; sub-second spells right after restarts.
