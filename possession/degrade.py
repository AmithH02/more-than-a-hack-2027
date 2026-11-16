"""Make clean tracking look like the ATD, so that what we measure on other matches transfers.

Everything is copied from statistics of BAR-ATH's own ATD inputs within its graded halves (never
its labels), and laid at random over the other match. The same seed gives the same result, and
the ball mask of a seed is the same in every version.

Version 1, the ball. The ATD loses it on 41% of frames, in long stretches (about 150 gaps, median
about 10 s). Alternating visible/hidden runs are drawn from the ATD's own run lengths.

Version 2 adds the players. The ATD tracks about 20.5 of 22 players per frame, but losses come in
bursts (5% of frames have 14 or fewer), and a player's track is broken into tracklets (median
35 s) that land in any free slot of its side (24 home slots, 22 away). So:
  - how many players are missing at each frame is copied from a random stretch of the ATD's own
    per-frame shortfall;
  - which ones: each player gets a random priority, redrawn at intervals drawn from the ATD's
    tracklet lengths, and the highest-priority players present are the ones hidden;
  - every visible stretch, cut again where its priority is redrawn, becomes one tracklet, put in
    the first free slot of its side, as the ATD does; a tracklet with no free slot is lost.
Slots are therefore tracklets, not players, exactly as in the ATD.

Version 3 moves the ball's gaps to where the ATD loses it: mostly in stoppages. On BAR-ATH the
players move at 1.1 m/s inside the ball's gaps against 2.1 m/s where it is tracked, and the
players_fill alive model calls only about 14% of gap frames alive. So, with the players as in
version 2:
  - each stoppage (a run of the DFL's dead frames) is hidden with probability STOP_HIDDEN, from a
    moment after it starts to a moment before the restart (each drawn up to EDGE_S), so the ball
    going out and the restart itself are sometimes seen;
  - live play loses the ball in gaps drawn from the ATD's gap lengths of up to LIVE_GAP_MAX frames,
    laid at random until LIVE_HIDDEN of live frames are hidden.
This uses the labels to place the gaps: it simulates the ATD, and no experiment sees the labels.
Its ball mask differs from versions 1 and 2.

Not modelled yet: identity switches inside a tracklet, and objects tracked that are not players.
"""

import numpy as np

VERSION = 2          # the default; 3 is chosen with `python -m possession.bench --version 3`
STOP_HIDDEN = 0.85   # share of stoppages whose ball is lost (version 3)
EDGE_S = 2.0         # s: the ball is seen for up to this long at each end of a hidden stoppage
LIVE_HIDDEN = 0.10   # share of live frames whose ball is lost (version 3)
LIVE_GAP_MAX = 250   # frames: live-play gaps are drawn from ATD gaps no longer than this
FULL_SIDES = 22
MIN_VISIBLE = 25  # frames


# ATD statistics ----------------------------------------------------------------------------------

def _runs(flags):
    """[(start, end, value)] of the runs of a 1-D bool array."""
    edges = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(flags)]))
    return [(s, e, bool(flags[s])) for s, e in zip(bounds[:-1], bounds[1:])]


def atd_stats(match=None):
    """Everything the degradation copies from the ATD, as a dict of arrays."""
    from possession.fill import half_blocks
    from possession.match import Match

    match = match or Match(log=lambda *args: None)
    present, missing, tracklets, shortfall = [], [], [], []
    for first, last in half_blocks():
        gone = np.isnan(match.ball_xy[first:last + 1, 0])
        for s, e, hidden in _runs(gone):
            (missing if hidden else present).append(e - s)
        tracked = ~np.isnan(match.track_xy[first:last + 1, :, 0])
        for j in range(tracked.shape[1]):
            tracklets += [e - s for s, e, on in _runs(tracked[:, j]) if on]
        shortfall.append(np.clip(FULL_SIDES - tracked.sum(axis=1), 0, None))
    team = np.asarray(match.track_team)
    return {"ball_present": np.array(present), "ball_missing": np.array(missing),
            "tracklets": np.array(tracklets), "shortfall": np.concatenate(shortfall),
            "slots_home": int((team == "H").sum()), "slots_away": int((team == "A").sum())}


# The ball ----------------------------------------------------------------------------------------

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


def stoppage_mask(labels, blocks, missing_runs, rng):
    """(n,) bool, True where the ball is hidden (version 3): most stoppages, some live play."""
    labels = np.asarray(labels)
    n = len(labels)
    hidden = np.zeros(n, dtype=bool)
    edge = int(EDGE_S * 25)
    for first, last in blocks:
        for s, e, dead in _runs(labels[first:last + 1] == "D"):
            if dead and rng.random() < STOP_HIDDEN:
                a = first + s + int(rng.integers(0, edge + 1))
                b = first + e - int(rng.integers(0, edge + 1))
                hidden[a:max(a, b)] = True
    live = labels != "D"
    pool = missing_runs[missing_runs <= LIVE_GAP_MAX]
    pool = pool if len(pool) else np.array([25])
    target = LIVE_HIDDEN * live.sum()
    live_hidden = 0
    while live_hidden < target:
        length = int(rng.choice(pool))
        start = int(rng.integers(0, max(1, n - length)))
        newly = live[start:start + length] & ~hidden[start:start + length]
        hidden[start:start + length] |= live[start:start + length]
        live_hidden += int(newly.sum())
    return hidden


# The players -------------------------------------------------------------------------------------

def segment_ids(n, lengths, rng):
    """(n,) int: consecutive segments with lengths drawn from `lengths`, numbered 0, 1, ..."""
    ids = np.zeros(n, dtype=np.int64)
    position, k = -int(rng.integers(0, rng.choice(lengths))), 0
    while position < n:
        end = position + int(rng.choice(lengths))
        ids[max(position, 0):max(end, 0)] = k
        position, k = end, k + 1
    return ids


def player_mask(present, shortfall, tracklet_lengths, rng):
    """(n, players) bool, True where a player present in the clean data is hidden, and the
    (n, players) segment ids that cut tracklets."""
    n, players = present.shape
    start = int(rng.integers(0, len(shortfall)))
    deficit = np.resize(np.roll(shortfall, -start), n)
    keep = np.maximum(FULL_SIDES - deficit, 0)
    to_hide = np.maximum(present.sum(axis=1) - keep, 0)

    segments = np.stack([segment_ids(n, tracklet_lengths, rng) for _ in range(players)], axis=1)
    draws = rng.random((segments.max() + 1, players))
    priority = np.take_along_axis(draws, segments, axis=0)
    priority[~present] = -np.inf
    order = np.argsort(-priority, axis=1)                     # most likely to be hidden first
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(players)[None, :], axis=1)
    hidden = present & (rank < to_hide[:, None])
    # A player shown for under a second between two hidden stretches stays hidden: the shortfall
    # changes about 18 times a minute and would otherwise make the ATD's tracks flicker, which
    # they do not (a tenth of its tracklets are shorter than 2.6 s, almost none under 1 s).
    for j in range(players):
        for s, e, on in _runs(present[:, j] & ~hidden[:, j]):
            if on and e - s < MIN_VISIBLE and s > 0 and e < n:
                hidden[s:e, j] = present[s:e, j]
    return hidden, segments


def allocate(track_xy, track_speed, visible, segments, sides, slots_home, slots_away):
    """Cut each player's visible stretches into tracklets (also where its segment changes) and put
    each in the first free slot of its side. Returns slot arrays and the slots' sides."""
    n = len(track_xy)
    slot_team = np.array(["H"] * slots_home + ["A"] * slots_away)
    out_xy = np.full((n, len(slot_team), 2), np.nan, dtype=np.float32)
    out_speed = np.full((n, len(slot_team)), np.nan, dtype=np.float32)
    pieces = []
    for j in range(track_xy.shape[1]):
        cut = visible[:, j].copy()
        change = np.flatnonzero(np.diff(segments[:, j])) + 1
        on = np.flatnonzero(cut)
        if not len(on):
            continue
        breaks = np.union1d(np.flatnonzero(np.diff(on) > 1) + 1, np.searchsorted(on, change))
        for piece in np.split(on, breaks):
            if len(piece):
                pieces.append((int(piece[0]), int(piece[-1]) + 1, j))
    pieces.sort()
    free_at = np.zeros(len(slot_team), dtype=np.int64)
    for start, end, j in pieces:
        candidates = np.flatnonzero((slot_team == sides[j]) & (free_at <= start))
        if not len(candidates):
            continue  # every slot of that side is busy: the tracklet is lost
        slot = candidates[0]
        out_xy[start:end, slot] = track_xy[start:end, j]
        out_speed[start:end, slot] = track_speed[start:end, j]
        free_at[slot] = end + 1  # a slot is empty for at least a frame between two tracklets
    return out_xy, out_speed, slot_team


# The whole degradation ---------------------------------------------------------------------------

_STATS = None


def degrade(match, seed, version=VERSION, stats=None):
    """A copy of an ExternalMatch made to look like the ATD; the same seed, the same result."""
    global _STATS
    if stats is None:
        _STATS = _STATS or atd_stats()
        stats = _STATS
    key = sum(map(ord, match.match_id))
    rng = np.random.default_rng([seed, key])
    hidden = np.zeros(match.n_frames, dtype=bool)
    for first, last in match.blocks:
        hidden[first:last + 1] = ball_mask(last - first + 1, stats["ball_present"],
                                           stats["ball_missing"], rng)
    if version >= 3:
        srng = np.random.default_rng([seed, key, 3])
        hidden = stoppage_mask(match.labels, match.blocks, np.asarray(stats["ball_missing"]), srng)
    ball_xy = match.ball_xy.copy()
    ball_speed = match.ball_speed.copy()
    ball_xy[hidden] = np.nan
    ball_speed[hidden] = np.nan
    changes = {"ball_xy": ball_xy, "ball_speed": ball_speed}

    if version >= 2:
        prng = np.random.default_rng([seed, key, 2])  # its own stream: the ball mask is unchanged
        present = ~np.isnan(match.track_xy[:, :, 0])
        hide, segments = player_mask(present, stats["shortfall"], stats["tracklets"], prng)
        xy, speed, team = allocate(match.track_xy, match.track_speed, present & ~hide, segments,
                                   match.track_team, stats["slots_home"], stats["slots_away"])
        changes.update(track_xy=xy, track_speed=speed, track_team=team)
    return match.with_arrays("v{}/seed{}".format(version, seed), **changes)
