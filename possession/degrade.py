"""Make clean tracking look like the ATD, so that what we measure on other matches transfers.

Version 1 hides the ball only. Within BAR-ATH's graded halves the ATD loses the ball on 41% of
frames, in long stretches (about 150 gaps, median about 10 s, the longest several minutes). We
take the lengths of the ATD's own ball-present and ball-missing runs (inputs only, no labels) and
lay a fresh alternating sequence of them, drawn at random, over each half of the other match.

Not yet modelled (later slices): where the gaps fall (they are probably not random: dead balls,
high balls, crowds), players dropping out, and tracklets that are not players (46 slots for 22
players, about 20 tracked per frame).
"""

import numpy as np

VERSION = 1


def atd_ball_runs(match=None):
    """(present run lengths, missing run lengths) of the ATD ball within the graded halves."""
    from possession.fill import half_blocks
    from possession.match import Match

    match = match or Match(log=lambda *args: None)
    present, missing = [], []
    for first, last in half_blocks():
        gone = np.isnan(match.ball_xy[first:last + 1, 0])
        edges = np.flatnonzero(np.diff(gone.astype(np.int8))) + 1
        bounds = np.concatenate(([0], edges, [len(gone)]))
        for start, end in zip(bounds[:-1], bounds[1:]):
            (missing if gone[start] else present).append(end - start)
    return np.array(present), np.array(missing)


def ball_mask(n, present_runs, missing_runs, rng):
    """(n,) bool, True where the ball is hidden: alternating runs drawn from the two pools, starting
    at a random point of a present run so that every half does not begin with the ball visible."""
    hidden = np.zeros(n, dtype=bool)
    position = -int(rng.integers(0, rng.choice(present_runs)))
    visible = True
    while position < n:
        length = int(rng.choice(present_runs if visible else missing_runs))
        if not visible:
            hidden[max(position, 0):max(position + length, 0)] = True
        position += length
        visible = not visible
    return hidden


_RUNS = None


def degrade(match, seed, runs=None):
    """A copy of an ExternalMatch with the ball hidden like the ATD's; the same seed, the same mask."""
    global _RUNS
    if runs is None:
        _RUNS = _RUNS or atd_ball_runs()
        runs = _RUNS
    present, missing = runs
    rng = np.random.default_rng([seed, sum(map(ord, match.match_id))])
    hidden = np.zeros(match.n_frames, dtype=bool)
    for first, last in match.blocks:
        hidden[first:last + 1] = ball_mask(last - first + 1, present, missing, rng)
    ball_xy = match.ball_xy.copy()
    ball_speed = match.ball_speed.copy()
    ball_xy[hidden] = np.nan
    ball_speed[hidden] = np.nan
    return match.with_arrays("v{}/seed{}".format(VERSION, seed), ball_xy=ball_xy,
                             ball_speed=ball_speed)
