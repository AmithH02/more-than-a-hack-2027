# #22 `players_fill`: read the players where the tracking has lost the ball

- **Issue**: #22 (builds on the bench, #20 / ADR 0003)
- **Method**: the baseline (nearest player, ±50-frame majority vote), except on ball gaps. There,
  two logistic regressions on side-level team-shape features (`possession/team_shape.py`) decide
  first whether play is alive (else `D`) and then which side has the ball. Fitted on the hidden-ball
  frames of the four training matches (3 seeds), under degradation v2.
- **Commands**:
  `python -m possession.bench train players_fill` (fits `possession/models/players_fill.json`,
  picks `window` and `alive_at`), `python -m possession run players_fill` (BAR-ATH)
- **Metrics**: `experiments/metrics/bench/players_fill.json`, `experiments/metrics/players_fill.json`

## Feasibility probe (before building, clean players, ball hidden at random, validation)

| On hidden frames | Today's rule | Players |
|---|---|---|
| Alive or dead | 48% (always dead) | 84% |
| Which side, gap under 4 s | 74% (last-seen side) | 82% |
| Which side, gap over 10 s | 55% (last-seen side) | 83% |
| Ball position | — | median error 12 m |

## Bench, validation (J03WOY, degradation v2)

| seed | `nearest_smooth` | `players_fill` |
|---|---|---|
| 0 | 0.192 | 0.276 |
| 1 | 0.174 | 0.239 |
| 2 | 0.164 | 0.244 |
| **mean spellF1** | **0.176** | **0.253** |
| mean macroF1 | 0.444 | 0.521 |

Training chose `window` 50 and `alive_at` 0.6; windows of 25 and 50 are within 0.003, and
`alive_at` barely matters. Test not run.

## BAR-ATH, label-free (graded halves)

| | `nearest_smooth` | `players_fill` |
|---|---|---|
| possessions | 594 | 560 |
| median possession | 4.6 s | 5.2 s |
| sub-second possessions | 20.9% | about 15% |
| dead-ball share | 32.3% | 25.9% |
| agreement with Smart Tagging | 0.714 | 0.727 |

## What we learnt

- **A bug found on BAR-ATH**: telling which way each side attacks from the kick-off formation
  failed on BAR-ATH's second half, whose block does not start at a kick-off. Replaced by each
  side's deepest player; it agrees with the kick-off rule on all 14 Bundesliga halves.
- **BAR-ATH's gaps are mostly stoppages.** The alive model calls only about 14% of BAR-ATH's gap
  frames alive. Not a calibration fault: where the ball is tracked, BAR-ATH's players move like the
  Bundesliga's (2.1 m/s), but inside its gaps they move at 1.1 m/s, slower than in any live
  Bundesliga state. The bench hides the ball at random (about half live play), so it overstates
  this experiment's gain on BAR-ATH.

## Verdict

Candidate. Clearly better on the bench (+0.077, every seed) and better on every BAR-ATH proxy,
but by less on BAR-ATH than the bench suggests. Next: place the bench's gaps where the ATD loses
the ball, then re-measure before choosing the submission.
