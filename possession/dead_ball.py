"""Dead-ball stretches from Smart Tagging and the tracking: kick-offs, restarts, stoppage onsets
and the stoppages Smart Tagging missed.

    stretches = detect(match, **settings)     # Stretches: a dead mask, the restarts found
    stretches.dead                            # (n,) bool, True where the ball is out of play
    stretches.restarts                        # [Restart(frame, kind, side, how)]

Everything is found at run time from the match's own inputs: its tracking (ball and players), its
Smart Tagging set-piece windows (`match.smart_tagging_events`) and its halves (`blocks`). Nothing
is a frame number. Thresholds come from the Laws of the Game and simple physics (named below), or
are settings chosen by `python -m possession.bench train` on the training matches only.

The steps, each its own function:
  1. kick_offs():  dead from each half's first frame, and from each goal, to the kick inside the
     kick-off window (still ball near the centre spot, then struck).
  2. restart_in_window(): inside each set-piece window, the restart by type, by the Laws: a throw-in
     comes back over a touchline, a corner is kicked from the corner arc, a goal kick from the goal
     area, a kick-off from the centre spot, a free kick from wherever the ball lies still. From the
     ball when it is seen, else from the formation (a thrower standing beyond the touchline, a
     taker at a corner flag), else from the players speeding up, else the median offset of the
     restarts found for that type in the same match.
  3. onset(): walking back from a restart to the ball going out of play or, without a visible ball,
     to where the players slowed down.
  4. untagged(): stoppages Smart Tagging missed: the ball seen out of play, the ball lying still,
     and long stretches of slow players, with the same edges.
"""

from collections import namedtuple

import numpy as np

FPS = 25

# Physics and the Laws ------------------------------------------------------------------------------
STILL_MS = 1.0        # m/s: a ball slower than this is lying still (tracking jitter included)
KICK_MS = 5.0         # m/s: a ball faster than this has been played
STILL_BEFORE_KICK = 10  # frames (0.4 s) of still ball, out of the second before, make a set kick
OUT_MARGIN = 1.0      # m beyond a line before the ball counts as out (whole ball over, plus noise)
CENTRE_R = 4.0        # m from the centre spot for a kick-off (the spot, plus tracking error)
CORNER_R = 4.0        # m from a corner flag for a corner (the 1 m arc, plus tracking error)
GOAL_AREA = (5.5 + 2.0, 9.16 + 2.0)  # m: depth and half-width of the goal area, plus 2 m error
TOUCH_BAND = 2.5      # m from a touchline for a throw-in taken on the line
GAP_REAPPEAR = FPS    # frames: a ball seen again after this long unseen may be a restart
MAX_BACK_S = 60       # s: the longest stoppage walked back over (a goal: up to its GOALS phase)
MAX_AHEAD_S = 60      # s: the longest an untagged stoppage waits for its restart
LIVE_PROOF = 5        # frames of a fast ball inside the pitch that prove play was on

Restart = namedtuple("Restart", "frame kind side how")
Stretches = namedtuple("Stretches", "dead restarts onsets")

KICK_OFF_START = "Kick Off (Start)"
KICK_OFF_GOAL = "Kick Off (After Goal)"


# Signals -----------------------------------------------------------------------------------------

def ball_speed(ball_xy, span=2):
    """(n,) m/s from positions span frames either side; NaN where the ball is not seen at both."""
    xy = np.asarray(ball_xy, dtype=np.float64)
    out = np.full(len(xy), np.nan)
    if len(xy) > 2 * span:
        out[span:-span] = np.linalg.norm(xy[2 * span:] - xy[:-2 * span], axis=1) / (2 * span / FPS)
    return out


def out_of_play(ball_xy, pitch, margin=OUT_MARGIN):
    """(n,) bool: the ball is seen beyond a line by more than `margin`."""
    x, y = np.asarray(ball_xy, dtype=np.float64).T
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def rolling_mean(values, frames):
    """Centred mean over `frames`, NaN filled by interpolation first (edges: nearest value)."""
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    if not ok.any():
        return np.zeros_like(v)
    v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
    if frames <= 1:
        return v
    pad = frames // 2
    padded = np.concatenate([np.full(pad, v[0]), v, np.full(frames - pad - 1, v[-1])])
    c = np.cumsum(np.concatenate([[0.0], padded]))
    return (c[frames:] - c[:-frames]) / frames


def activity(match, smooth_s=2.0):
    """(n,) m/s: all players' mean speed (team shape), smoothed over `smooth_s` seconds."""
    return rolling_mean(match.team_shape["mean_speed"], int(smooth_s * FPS))


def runs(flags):
    """[(start, end)] half-open runs of True in a 1-D bool array."""
    f = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(f.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


class Ball:
    """What the ball says, computed once: speed, still, out, and restart candidates."""

    def __init__(self, ball_xy, pitch):
        self.xy = np.asarray(ball_xy, dtype=np.float64)
        self.pitch = tuple(pitch)
        self.seen = ~np.isnan(self.xy[:, 0])
        self.speed = ball_speed(self.xy)
        with np.errstate(invalid="ignore"):
            self.still = self.speed < STILL_MS
            self.fast = self.speed > KICK_MS
        self.out = out_of_play(self.xy, self.pitch)
        length, width = self.pitch
        x, y = self.xy.T
        with np.errstate(invalid="ignore"):
            inside = (x > 0) & (x < length) & (y > 0) & (y < width)
        self.live_proof = self.fast & inside
        self.candidates = self._candidates()

    def _candidates(self):
        """[(frame, x, y, how)] sorted: set kicks ("kick": still, then struck), throws back over a
        touchline ("throw"), and the ball seen again, moving, after a gap ("reappear")."""
        out = []
        n = len(self.xy)
        still_count = np.concatenate([[0], np.cumsum(self.still)])
        rises = np.flatnonzero(self.fast[1:] & ~self.fast[:-1]) + 1
        for t in rises:
            lo = max(0, t - FPS)
            if still_count[t] - still_count[lo] >= STILL_BEFORE_KICK:
                last_still = lo + int(np.flatnonzero(self.still[lo:t])[-1])
                out.append((last_still + 1, self.xy[last_still, 0], self.xy[last_still, 1], "kick"))
        width = self.pitch[1]
        y = self.xy[:, 1]
        with np.errstate(invalid="ignore"):
            beyond = (y < -OUT_MARGIN) | (y > width + OUT_MARGIN)
            on = (y >= 0) & (y <= width)
        beyond_recent = np.convolve(beyond.astype(int), np.ones(FPS // 2, dtype=int))[:n] > 0
        crossing = np.concatenate([[False], on[1:] & beyond_recent[:-1]])
        for t, _ in runs(crossing):
            out.append((int(t), self.xy[t, 0], self.xy[t, 1], "throw"))
        for start, end in runs(~self.seen):
            if end - start >= GAP_REAPPEAR and end < n and self.fast[end:end + FPS // 2].any():
                out.append((int(end), self.xy[end, 0], self.xy[end, 1], "reappear"))
        out.sort(key=lambda c: c[0])
        return out


# Where each restart is taken (the Laws) -------------------------------------------------------------

def at_place(kind, x, y, pitch):
    """Is a ball at (x, y) where a restart of this kind is taken?"""
    length, width = pitch
    if kind in (KICK_OFF_START, KICK_OFF_GOAL):
        return np.hypot(x - length / 2, y - width / 2) < CENTRE_R
    if kind == "Corner Kick":
        return min(np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)) < CORNER_R
    if kind == "Goal Kick":
        return (min(x, length - x) < GOAL_AREA[0]) and abs(y - width / 2) < GOAL_AREA[1]
    if kind == "Throw-in":
        return min(y, width - y) < TOUCH_BAND
    if kind == "Penalty":
        return min(np.hypot(x - 11, y - width / 2), np.hypot(x - length + 11, y - width / 2)) < 3
    return True  # a free kick is taken where the offence was


def formation_restart(kind, track_xy, start, end, pitch):
    """A restart frame from where the players stand, or None: a thrower beyond a touchline, a corner
    taker at a flag, each for at least a second; the restart is when that ends."""
    length, width = pitch
    xy = np.asarray(track_xy[start:end], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        if kind == "Throw-in":
            there = (xy[..., 1] < -0.5) | (xy[..., 1] > width + 0.5)
        elif kind == "Corner Kick":
            d = np.min([np.hypot(xy[..., 0] - cx, xy[..., 1] - cy)
                        for cx in (0, length) for cy in (0, width)], axis=0)
            there = d < 2.0
        else:
            return None
    best = None
    for j in range(there.shape[1]):
        for s, e in runs(there[:, j]):
            if e - s >= FPS and (best is None or s < best[0]):
                best = (s, e)
    return None if best is None else start + int(best[1])


def activity_rise(act, start, end, dead_at, live_at):
    """The first frame in [start, end) where the players speed up past live_at after being slower
    than dead_at in the ten seconds before; None if they never do."""
    lo = max(0, start - 10 * FPS)
    slow_seen = False
    for t in range(lo, min(end, len(act))):
        if act[t] < dead_at:
            slow_seen = True
        elif slow_seen and t >= start and act[t] >= live_at:
            return t
    return None


def restart_in_window(kind, start, end, ball, track_xy, act, dead_at, live_at):
    """(frame, how) of the restart inside the window [start, end), or (None, None)."""
    for frame, x, y, how in ball.candidates:
        if frame < start:
            continue
        if frame >= end:
            break
        if how == "throw" and kind != "Throw-in":
            continue
        if at_place(kind, x, y, ball.pitch):
            return frame, "ball"
    frame = formation_restart(kind, track_xy, start, end, ball.pitch)
    if frame is not None:
        return frame, "formation"
    frame = activity_rise(act, start, end, dead_at, live_at)
    if frame is not None:
        return frame, "players"
    return None, None


# Onsets --------------------------------------------------------------------------------------------

def onset(restart, floor, ball, act, live_at, fallback=None, max_back_s=MAX_BACK_S):
    """First dead frame before `restart`, not before `floor`.

    With the ball seen out of play in the stretch, it walks back from the last out frame over
    anything that is not a fast ball inside the pitch, and returns the first out frame after the
    last such proof of live play. Else it walks back while the players are slow (and the ball
    gives no proof of live play) and returns where that began. Else restart - fallback, or None
    when no fallback is given.
    """
    lo = max(floor, restart - max_back_s * FPS)
    if restart <= lo:
        return restart
    out = np.flatnonzero(ball.out[lo:restart])
    if len(out):
        last_out = lo + int(out[-1])
        proof = np.flatnonzero(ball.live_proof[lo:last_out])
        after = int(proof[-1]) + 1 if len(proof) else 0
        return lo + int(out[np.searchsorted(out, after)])
    t = restart - 1
    quiet = lambda i: act[i] < live_at and not ball.live_proof[i]  # noqa: E731
    if t >= lo and quiet(t):
        while t > lo and quiet(t - 1):
            t -= 1
        return t
    return None if fallback is None else max(lo, restart - fallback)


# The steps -----------------------------------------------------------------------------------------

def _windows(match, kinds):
    out = []
    for event in getattr(match, "smart_tagging_events", []) or []:
        kind = event.get("tags", {}).get("Type")
        if event.get("code", "").upper() == "SET PIECES" and kind in kinds:
            out.append((int(event["start_frame"]), int(event["end_frame"]), kind,
                        _side(event, match)))
    return sorted(out)


def _side(event, match):
    team = event.get("tags", {}).get("Team")
    names = tuple(getattr(match, "team_names", ()) or ())
    if len(names) == 2 and team == names[0]:
        return "H"
    if len(names) == 2 and team == names[1]:
        return "A"
    return None


def _goal_phases(match):
    return sorted((int(e["start_frame"]), int(e["end_frame"]))
                  for e in getattr(match, "smart_tagging_events", []) or []
                  if e.get("code", "").upper() == "GOALS")


def _block_of(frame, blocks):
    for first, last in blocks:
        if first <= frame <= last:
            return first, last
    return None


def detect(match, dead_at=1.2, live_at=1.6, min_dead_s=8.0, still_dead_s=3.0, blocks=None):
    """Stretches for the whole match. Settings (m/s and seconds):

    dead_at, live_at   players' mean speed below which a stoppage may start, and above which play
                       has resumed (hysteresis); dead_at None turns the untagged slow stretches off
    min_dead_s         the shortest untagged slow stretch called dead
    still_dead_s       a ball lying still inside the pitch this long is dead (None: off)
    """
    from possession.team_shape import blocks_of, pitch_of

    blocks = blocks if blocks is not None else blocks_of(match)
    pitch = pitch_of(match)
    n = match.n_frames
    ball = Ball(match.ball_xy, pitch)
    act = activity(match)
    live = live_at
    dead = np.zeros(n, dtype=bool)
    restarts, onsets = [], []

    # Steps 1 and 2: every tagged window, kick-offs included.
    kinds = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Penalty",
             KICK_OFF_START, KICK_OFF_GOAL)
    found = []
    for start, end, kind, side in _windows(match, kinds):
        # A window belongs to the half it ends in, and is cut to it: a half's kick-off window may
        # start before the half's first frame.
        block = _block_of(end - 1, blocks)
        if block is None:
            continue
        start, end = max(start, block[0]), min(end, block[1] + 1)
        frame, how = restart_in_window(kind, start, end, ball, match.track_xy, act,
                                       dead_at or 1.2, live)
        found.append([start, end, kind, side, frame, how])
    offsets = {}
    for start, _, kind, _, frame, _ in found:
        if frame is not None:
            offsets.setdefault(kind, []).append(frame - start)
    for row in found:
        if row[4] is None:
            start, end, kind = row[:3]
            pool = offsets.get(kind) or [o for v in offsets.values() for o in v] or [6 * FPS]
            row[4], row[5] = min(end - 1, start + int(np.median(pool))), "median"

    goals = _goal_phases(match)
    durations = {}
    for start, end, kind, side, frame, how in sorted(found, key=lambda r: r[4]):
        block = _block_of(frame, blocks)
        if block is None:
            continue
        floor = block[0]
        previous = [r.frame for r in restarts if r.frame < frame]
        if previous:
            floor = max(floor, previous[-1] + 1)
        if kind == KICK_OFF_START:
            on = block[0]  # nothing before a half's kick-off is play
        elif kind == KICK_OFF_GOAL:
            # A goal's stoppage (celebration, checks) may be long, but not before its GOALS phase.
            phase = [g for g in goals if g[0] <= frame <= g[1] + 2 * FPS]
            lo = max(floor, phase[-1][0]) if phase else floor
            on = onset(frame, lo, ball, act, live, max_back_s=3 * MAX_BACK_S)
        else:
            on = onset(frame, floor, ball, act, live)
        restarts.append(Restart(frame, kind, side, how))
        onsets.append(on)
        durations.setdefault(kind, []).append(frame - on if on is not None else None)

    # Onsets with no evidence: the median stoppage of that type found with evidence.
    for i, r in enumerate(restarts):
        if onsets[i] is None:
            pool = [d for d in durations.get(r.kind, []) if d] or [10 * FPS]
            onsets[i] = max(_block_of(r.frame, blocks)[0], r.frame - int(np.median(pool)))
        dead[onsets[i]:r.frame] = True

    # Step 4: what Smart Tagging missed.
    extra = untagged(ball, act, blocks, dead, dead_at, live, min_dead_s, still_dead_s)
    for on, frame in extra:
        dead[on:frame] = True
        restarts.append(Restart(frame, "untagged", None, "untagged"))
        onsets.append(on)

    graded = np.zeros(n, dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    dead &= graded
    order = np.argsort([r.frame for r in restarts], kind="stable")
    return Stretches(dead, [restarts[i] for i in order], [onsets[i] for i in order])


def untagged(ball, act, blocks, dead, dead_at, live_at, min_dead_s, still_dead_s):
    """[(onset, restart)] of stoppages not already dead."""
    n = len(dead)
    found = []
    taken = dead.copy()

    def restart_after(t, limit):
        for frame, _, _, how in ball.candidates:
            if t <= frame < limit:
                return frame
        rise = activity_rise(act, t, limit, dead_at or 1.2, live_at)
        return rise

    for first, last in blocks:
        end_block = last + 1
        # The ball seen out of play.
        for s, e in runs(ball.out[first:end_block]):
            s, e = first + s, first + e
            if taken[s:e].any() or e - s < 3:
                continue
            frame = restart_after(e, min(end_block, e + MAX_AHEAD_S * FPS))
            if frame is None:
                continue
            on = onset(e, first, ball, act, live_at, fallback=0)
            on = min(on, s)
            if not taken[on:frame].any():
                found.append((on, frame))
                taken[on:frame] = True
        # The ball lying still inside the pitch.
        if still_dead_s:
            still_in = ball.still & ~ball.out
            for s, e in runs(still_in[first:end_block]):
                s, e = first + s, first + e
                if e - s >= still_dead_s * FPS and not taken[s:e].any():
                    found.append((s, e))
                    taken[s:e] = True
        # Slow players, with hysteresis, with no proof of live play inside.
        if dead_at:
            state, start = False, None
            for t in range(first, end_block):
                if not state and act[t] < dead_at:
                    state, start = True, t
                elif state and act[t] >= live_at:
                    state = False
                    _add_slow(found, taken, ball, start, t, min_dead_s)
            if state:
                _add_slow(found, taken, ball, start, end_block, min_dead_s)
    return found


def _add_slow(found, taken, ball, start, end, min_dead_s):
    if end - start < min_dead_s * FPS or taken[start:end].any():
        return
    if ball.live_proof[start:end].sum() >= LIVE_PROOF:
        return
    # Snap the end to a set kick just after the players speed up, when the ball shows one.
    for frame, _, _, how in ball.candidates:
        if end - 3 * FPS <= frame <= end + 3 * FPS and how == "kick":
            end = frame
            break
    found.append((start, end))
    taken[start:end] = True
