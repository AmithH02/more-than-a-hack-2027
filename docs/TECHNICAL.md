# Technical document

The third deliverable ([SUBMISSION.md](../SUBMISSION.md)). Kept up to date as we go: every PR that
changes the method updates the relevant section. At submission it is copied to `<team_name>.md`.
Markdown, not PDF: `*.pdf` is gitignored in this repo.

## Approach

_What we built, and why that rather than something else. To be filled once #5 decides the method._

## Assets used, and dropped

| Asset | Used? | How / why not |
|---|---|---|
| Tactical ATD (tracking) | | |
| Smart Tagging | candidate #24 | set-piece windows (type, team) and GOALS phases bound where restarts are searched for |
| Homography | | |
| Tactical video | | |
| Broadcast video and ATD | no (so far) | not synchronised to the scored timeline |

## Training data

_Every labelled dataset used: source, who labelled it, how much. None from the provided game.
Write "none" if the method uses no labels._

- **IDSSE open Bundesliga data** (Bassek, Weber, Rein & Memmert, 2025, *Scientific Data*;
  CC BY 4.0; figshare): seven full matches of the 2022/23 Bundesliga and 2. Bundesliga, about
  1.0 million frames at 25 Hz. Labels are the DFL's own per-frame ball status and ball possession,
  produced by the provider's data collection, not by us. Split into four training, one validation
  and two test matches (ADR 0003). The tracking is degraded to look like the ATD before any
  method sees it: the ball is hidden in runs drawn from the ATD's own gap lengths, and the players
  are broken into ATD-like tracklets (degradation v2).
- Candidate `players_fill` (#22) fits two small logistic regressions on the training matches'
  hidden-ball frames (about 130,000 frames, every fifth): alive or dead, and which side has the
  ball, from side-level team-shape features. Their parameters are committed in
  `possession/models/players_fill.json`.

- Candidate `players_dead_ball` (#24) reuses those models and learns nothing new; its three
  settings were picked on the training matches under degradation v3 (`bench train --version 3`).
  To measure it the bench gives the IDSSE matches stand-in Smart Tagging windows made from the
  DFL's own set-piece events (ADR 0003).

## Dead ball (candidate `players_dead_ball`, #24)

`possession/dead_ball.py` finds dead stretches from Smart Tagging and the tracking, and they
override `players_fill`'s labels with `D`. Nothing is a frame number found by looking at BAR-ATH:
every step runs on the match's own inputs, with thresholds from the Laws or physics, or settings
chosen on the training matches.

1. **Kick-offs**: `D` from each half's first graded frame to the kick inside its Kick Off (Start)
   window (a ball lying still within 4 m of the centre spot, then struck at over 5 m/s). After a
   goal, `D` from the onset (step 3, not before the GOALS phase starts, up to 3 minutes back) to
   the kick inside the Kick Off (After Goal) window. A window belongs to the half it ends in.
2. **Restart finder**, inside every set-piece window, by the Laws: a throw-in is the ball coming
   back over a touchline (or a still ball within 2.5 m of one, then thrown); a corner a still ball
   within 4 m of a corner flag, then struck; a goal kick the same in the goal area (plus 2 m); a
   free kick a still ball anywhere, then struck. Without the ball: a thrower standing beyond a
   touchline, or a taker at a corner flag, for a second (the restart is when they leave); else the
   players speeding up past 1.6 m/s after being slower than 1.2 m/s; else the median offset of the
   restarts found for that type in the same match.
3. **Onset**: from the restart, walk back (up to 60 s, never past the previous restart) to the
   first frame the ball was seen out of play after the last proof of live play (a ball faster than
   5 m/s inside the pitch); without a ball seen out, to where the players' mean speed rose above
   1.6 m/s; without that, the median stoppage of that type.
4. **Untagged stoppages**: the ball seen over a line (by more than 1 m) until the next restart
   candidate; the ball lying still (under 1 m/s) inside the pitch for 3 s; the players' mean speed
   under 1.2 m/s until it is back over 1.6 m/s, for at least 8 s, with no proof of live play
   inside (its end snapped to a set kick within 3 s).
5. **Combined**: these stretches are `D`; for 3 s after each restart play is alive, for the side
   Smart Tagging tags if it does, else the first side `players_fill` gives.

On BAR-ATH the second half's graded block starts about two minutes before its kick-off; that
stretch is now `D` (step 1).

## Possession rules implemented

_Where we drew the line on loose balls, challenges, deflections, aerial duels and restarts. Use the
terms in [CONTEXT.md](../CONTEXT.md)._

## Limitations

_Where it fails, and what we would fix with more time._

## Reproduction

```bash
pip install -r requirements.txt                            # pinned versions
python scripts/check_data.py                               # the data matches the committed checksums (#2)
python -m possession run north_london_tactics              # regenerates "North London Tactics.csv"
python evaluation/validate_submission.py "North London Tactics.csv"
```

`python -m possession run` checks the data itself and stops if it fails, runs the chosen
experiment, fills each `X` inside a half from the labels around it (never across halftime), writes
the CSV and checks it as read back from disk with the official format check (ADR 0002). It then
prints a label-free proxy report for the graded halves (possessions, their lengths, label changes
per minute, dead-ball and home shares, agreement with Smart Tagging's in-possession phases), with
warnings for values outside provisional ranges, and saves it to
`experiments/metrics/<experiment>.json`. For the submission it also runs the organisers'
`evaluation/validate_submission.py` on the file. Only the format checks can fail a run. Add `--rebuild-cache` to rebuild every
prepared input item instead of loading it from `outputs/cache/`; the result must be byte-identical
either way.

## Pre-submission checklist

- [ ] A fresh clone plus the data bundle regenerates `<team_name>.csv` byte for byte.
- [ ] `validate_submission.py` passes on the exact file being uploaded.
- [ ] The public 5-minute score looks sane (pipeline check only; never used to choose).
- [ ] No data, video or ground truth is tracked in git.
- [ ] Every section above is filled in and matches the code.
- [ ] CSV and document named after the team.
