"""Put the keepers on the right side: the ATD often files a keeper under the other team.

The ATD's metadata marks one slot of each side as the keeper's (position_type "Goalkeeper"; on
BAR-ATH, Track 24 for home and Track 46 for away). The two keepers wear colours of their own, and
the tracker mixes them up: a keeper's tracklet often lands in the *other* side's keeper slot, at
the far end of the pitch from where that side's keeper stands, sometimes while the right slot is
tracking the same keeper too (the keeper split in two). A slot's side is fixed in the metadata, so
every such tracklet is given to the wrong team.

The rule: each tracklet of a keeper slot belongs to the side that defends the end it stands in.
  - Which side defends which end is read from the kick-off of each half (Smart Tagging's
    "Kick Off (Start)"): at kick-off every player is in their own half, so each side's median x
    tells which goal it defends. Keeper slots are left out of that median.
  - A tracklet is a run of frames on which the slot is tracked without a break. Its end is where
    its median x lies, cut at the halves, so a tracklet that runs over halftime is judged in each
    half on its own. A duplicate is judged like any other tracklet, so both copies of a split
    keeper go to the keeper's side.
  - Outfield slots are left alone.

    low = [home_defends_low_x(track_xy, track_team, frame, keeper_slots) for frame in kickoffs]
    team = fix(track_xy, track_team, keeper_slots, halves, low, length)      # (n, slots) "H"/"A"

On BAR-ATH this is match.track_team_fixed. Only 3 outfield tracklets of 10 s or more live in the
other side's keeper area, so nothing is gained by judging outfield slots too.
"""

import numpy as np

VERSION = 1    # bump whenever fix() returns something different: it is part of the cache key
KICKOFF_WINDOW = 12   # frames each side of the tagged kick-off frame used for the medians


def runs(tracked):
    """[(start, end)] of the runs of True in a 1-D bool array, end exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], tracked.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def home_defends_low_x(track_xy, track_team, frame, keeper_slots=()):
    """True when the home side defends x = 0 at kick-off `frame`: its outfield players' median x
    is lower than the away side's, over a second around the kick-off."""
    team = np.asarray(track_team)
    outfield = np.ones(len(team), dtype=bool)
    outfield[list(keeper_slots)] = False
    window = slice(max(frame - KICKOFF_WINDOW, 0), frame + KICKOFF_WINDOW + 1)
    x = np.asarray(track_xy[window, :, 0], dtype=np.float64)
    home = np.nanmedian(x[:, outfield & (team == "H")])
    away = np.nanmedian(x[:, outfield & (team == "A")])
    if np.isnan(home) or np.isnan(away):
        raise ValueError("no players of one side tracked around kick-off frame {}".format(frame))
    return bool(home < away)


def half_spans(halves, n_frames):
    """Every frame given to one half: half k runs from the first frame of its block up to the first
    frame of the next block (the first half also takes the frames before it), so tracklets in the
    breaks are judged too."""
    starts = [first for first, _ in halves]
    bounds = [0] + starts[1:] + [n_frames]
    return list(zip(bounds[:-1], bounds[1:]))


def fix(track_xy, track_team, keeper_slots, halves, home_low, length):
    """(n_frames, n_slots) "<U1": each slot's side on each frame, keeper slots' tracklets given to
    the side defending the end they stand in.

    track_xy      (n, slots, 2) positions in metres, NaN where not tracked
    track_team    (slots,) "H"/"A" from the metadata
    keeper_slots  indices of the keeper slots
    halves        [(first, last)] graded blocks, one per half, in order
    home_low      one bool per half: the home side defends x = 0 (home_defends_low_x)
    length        pitch length in metres
    """
    team = np.asarray(track_team)
    n = len(track_xy)
    out = np.repeat(team[None, :], n, axis=0).astype("<U1")
    for (start, end), low in zip(half_spans(halves, n), home_low):
        low_side, high_side = ("H", "A") if low else ("A", "H")
        for slot in keeper_slots:
            x = np.asarray(track_xy[start:end, slot, 0], dtype=np.float64)
            for s, e in runs(~np.isnan(x)):
                end_x = np.median(x[s:e])
                out[start + s:start + e, slot] = low_side if end_x < length / 2 else high_side
    return out


def changed(track_team, fixed):
    """(n,) int: how many tracked-or-not slots changed side on each frame (for reports)."""
    return (np.asarray(fixed) != np.asarray(track_team)[None, :]).sum(axis=1)
