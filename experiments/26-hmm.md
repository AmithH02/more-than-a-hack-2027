# #26 A hidden Markov model for possession (`hmm_*`)

- **Issue**: #26. Built on #22 (`players_fill`: team shape, `learn.py`, degradation v2).
- **Method** (`hmm_touch`, refitted as `hmm_final`): per frame, two logistic models give
  p(alive) and p(home | alive), one pair for frames where the ball is tracked and one where it is
  hidden. A 3-state HMM (H, A, D) turns them into emissions, `w * (log p(state) - log prior)`, and
  Viterbi picks the best sequence over each half, with transitions from the rates counted in the
  training labels scaled by `switch` and `dead`. Code: `possession/hmm.py` (decoder),
  `possession/hmm_models.py` (shared fitting and prediction), `possession/experiments/hmm_*.py`.
- **Features** (all from tracking): each side's nearest player and the margin, ball speed, ball
  off the pitch, where along it, the team shape (#22); control evidence averaged over ±0.5 s and
  ±2 s, how long the ball has been missing and until it is back, and which side was nearest at
  each end; and, the step that mattered most, **the side of the last control before the frame
  and of the next one after it**, at two strengths, with how long ago and until.
- **Commands**: `python -m possession.bench train hmm_touch`;
  `python -m possession.experiments.hmm_final` then `python -m possession.bench test hmm_final`
  (once); `python scripts/bench_errors.py <experiment>` (training matches only).
- **Metrics**: `experiments/metrics/bench/hmm_*.json`.

## Validation (J03WOY, degradation v2), settings chosen on train

| experiment | what it adds | seed 0 | seed 1 | seed 2 | **spellF1** | macroF1 | verdict |
|---|---|---|---|---|---|---|---|
| `players_fill` (#22) | reference | 0.276 | 0.239 | 0.244 | 0.253 | 0.521 | |
| `hmm_viterbi` | HMM decoding | 0.349 | 0.332 | 0.325 | 0.335 | 0.774 | kept, superseded |
| `hmm_context` | windowed and gap context | 0.390 | 0.367 | 0.366 | 0.374 | 0.790 | kept, superseded |
| **`hmm_touch`** | last and next control | 0.464 | 0.464 | 0.431 | **0.453** | 0.815 | **the method** |
| `hmm_mlp` | a small network instead of logistic | 0.478 | 0.441 | 0.487 | 0.469 | 0.833 | not kept: within noise, seeds mixed, train 0.678 (memorises) |
| `hmm_split_w` | separate weight for hidden frames | 0.469 | 0.461 | 0.438 | 0.456 | 0.814 | not kept: noise |
| `hmm_restart` | who restarts after a stoppage | 0.461 | 0.460 | 0.430 | 0.450 | 0.814 | not kept: next control already says it |
| `hmm_rich` | team movement over time, dead-ball cues | 0.463 | 0.456 | 0.439 | 0.453 | 0.817 | not kept: no change |
| `hmm_cross` | hand-made feature products | 0.463 | 0.453 | 0.436 | 0.451 | 0.815 | not kept: no change |

## Test (J03WPY, J03WR9), run once, `hmm_final`

`hmm_touch` with its settings frozen (`w` 0.7, `switch` 0.5, `dead` 1.0, `dead_min` 100), models
refitted on train + validation: **spellF1 0.479, macroF1 0.838**, every match and seed between
0.45 and 0.51. Refitting reproduces `possession/models/hmm_final.json` byte for byte.

## What we learnt

- **Short possessions decide spellF1.** On the training matches 1451 of 1814 true possessions
  last under 10 s. `hmm_context` matched 4% of those under 2 s; `hmm_touch` 14%, and 41% of those
  of 2-10 s (from 28%).
- **Possession follows the last control, not the nearest player.** The DFL keeps it with the side
  that last controlled the ball through passes in flight; giving the model the side of the last
  and next control was worth +0.08 on validation, more than anything else.
- **Decoding beats smoothing.** Replacing the majority vote with Viterbi was worth +0.08 on its
  own, and made `D` reachable while the ball is visible (macroF1 0.52 to 0.77).
- **Four training matches limit flexible models.** The network fits train far better (0.678)
  but not validation.
- **Tuning knobs are flat** near the chosen settings (within 0.003 on train), and a minimum
  stoppage length barely matters once the evidence is good.

## Caveats

- The bench hides the ball at random; BAR-ATH loses it mostly in stoppages (#22), so the bench
  likely overstates scores there. Degradation v3 would re-measure.
- The DFL's possession is a close cousin of the challenge's, not the same definition.
- BAR-ATH label-free proxies: pending (`python -m possession run hmm_final`).

## Verdict

`hmm_final` is our candidate. Whether it becomes the team's submission is for #5.
