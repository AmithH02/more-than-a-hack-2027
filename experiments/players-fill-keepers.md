# `players_fill_keepers`: keepers put on the right side

- **Issue**: none yet (built on the `22-players-fill` branch)
- **Method**: `players_fill`, with the nearest player's side read from `match.track_team_fixed`
  instead of `match.track_team`. Each tracklet of a keeper slot (Track 24 home, Track 46 away in
  the ATD's metadata) is given to the side defending the end it stands in; ends come from each
  half's kick-off in Smart Tagging (`possession/keepers.py`). Same models, same settings.
- **Command**: `python -m possession run players_fill_keepers` (BAR-ATH)
- **Metrics**: `experiments/metrics/players_fill_keepers.json`
- **Validation**: the runner's official format check passed.

## What the rule changes on BAR-ATH (graded frames)

| | |
|---|---|
| Tracked slot-frames given the other side | 17,405 (home keeper slot 1,013; away keeper slot 16,392) |
| Frames whose nearest player changes side | 1,306 of 94,214 with a ball |
| Outfield tracklets of 10 s+ living in the other side's keeper area | 3 (left alone) |

## Proxy metrics

| | `players_fill` | `players_fill_keepers` |
|---|---|---|
| possessions | 560 | 537 |
| median possession | 5.24 s | 5.28 s |
| sub-second possessions | 14.6% | 15.1% |
| home share | 0.568 | 0.575 |
| dead-ball share | 25.9% | 25.9% |
| agreement with Smart Tagging | 0.727 | 0.734 |

## What the bench can and cannot say

The bench's matches have the provider's own sides, keepers included, and no keeper slots, so
`track_team_fixed` is `track_team` there and the two experiments score the same. The rule's effect
is measured on BAR-ATH only, by label-free proxies.

## Verdict

Candidate. 23 fewer possessions and more agreement with Smart Tagging, with the dead-ball share
unchanged; sub-second possessions are slightly up. Not checked by eye in the viewer yet.
