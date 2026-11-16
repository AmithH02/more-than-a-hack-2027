"""Other matches with a full ground truth: the IDSSE Bundesliga open data, as Match-like objects.

    from possession.external import load, SPLITS
    match = load("J03WR9")                 # clean DFL tracking, read once and cached
    match = load("J03WR9", degrade=0)      # the same match with the ball hidden like the ATD
    match.labels                           # (n_frames,) "H" / "A" / "D", the provider's own

The IDSSE dataset (Bassek, Weber, Rein & Memmert, 2025, Scientific Data; CC BY 4.0) has seven full
matches of the 2022/23 Bundesliga and 2. Bundesliga: player and ball positions at 25 Hz, the DFL's
per-frame ball status (in or out of play) and ball possession (home or away), and the official
event data. scripts/fetch_idsse.py downloads it into data/external/idsse/ (gitignored).

An ExternalMatch answers the attributes experiments use from the Match (ball_xy, ball_speed,
track_xy, track_speed, track_team, n_frames, the Smart Tagging ones), in the same units and the
same pitch frame: metres, origin at a corner, x along the length. Its frame axis is the two halves
back to back with nothing between, so `blocks` replaces fill.half_blocks() for it.

The DFL's possession is not exactly ours (CONTEXT.md): it is the provider's own definition, so
scores here measure agreement with a close cousin of the challenge's labelling, not the real thing.
Smart Tagging does not exist for these matches. In its place, `smart_tagging_events` (and the
painted `smart_tagging_*` attributes) are stand-in set-piece windows made from the DFL's own
set-piece events (event.xml), shaped like BAR-ATH's Smart Tagging: see stand_in_windows().
"""

import datetime
import json
import os
import re

import numpy as np

from possession import REPO
from possession.match import NO_TEAM_TAG, paint_smart_tagging

DATA_DIR = os.path.join(REPO, "data", "external", "idsse")
CACHE_DIR = os.path.join(REPO, "outputs", "cache", "idsse")
PARSE_VERSION = 1

# Fixed before any score was looked at. Training picks settings, validation compares candidates,
# and test is run once per candidate we intend to submit. Two Düsseldorf home games go to test so
# that neither validation nor training sees all of one home side's matches.
SPLITS = {
    "train": ("J03WMX", "J03WN1", "J03WOH", "J03WQQ"),
    "val": ("J03WOY",),
    "test": ("J03WPY", "J03WR9"),
}
MATCHES = tuple(m for split in SPLITS.values() for m in split)

HALVES = ("firstHalf", "secondHalf")
FPS = 25
_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")?'
                    r' D="[-\d.]+" S="([-\d.]+)"(?:[^>]*BallPossession="(\d)" BallStatus="(\d)")?')
_ATTR = re.compile(r'(\w+)="([^"]*)"')


# Set pieces as the DFL's event.xml names them, and as BAR-ATH's Smart Tagging types them.
SET_PIECE_TAGS = {"ThrowIn": "Throw-in", "CornerKick": "Corner Kick", "GoalKick": "Goal Kick",
                  "FreeKick": "Free Kick", "Penalty": "Penalty", "KickOff": "Kick Off"}
SET_PIECES_VERSION = 1
_EVENT = re.compile(r'<Event ([^>]*)>\s*<(\w+)([^>]*)>')


def match_dir(match_id, data_dir=DATA_DIR):
    return os.path.join(data_dir, match_id)


# Parsing -----------------------------------------------------------------------------------------

def read_meta(path):
    """(home team id, away team id, home name, away name, pitch (length, width))."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    general = dict(_ATTR.findall(re.search(r"<General [^>]*>", text).group(0)))
    env = dict(_ATTR.findall(re.search(r"<Environment [^>]*>", text).group(0)))
    return (general["HomeTeamId"], general["GuestTeamId"], general["HomeTeamName"],
            general["GuestTeamName"], (float(env["PitchX"]), float(env["PitchY"])))


def parse_tracking(path, home_id, away_id, pitch):
    """Arrays for the two halves back to back, from the DFL position file.

    The ball's frames define each half; a player's frame outside them is dropped. DFL positions are
    metres from the centre spot; they are shifted so the origin is a corner, as the Match has it.
    """
    sets = {}  # (half, team, person) -> list of (N, x, y, speed[, possession, status])
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FRAME.match(line)
                current.append(m.groups())
            elif line.startswith("<FrameSet "):
                half, team, person = _FRAMESET.match(line).groups()
                current = sets.setdefault((half, team, person), [])

    length, width = pitch
    # Two matches also track the referees; they are nobody's player.
    sets = {key: v for key, v in sets.items() if key[1] != "referee"}
    people = sorted({(team, person) for (_, team, person) in sets if team != "BALL"},
                    key=lambda tp: (tp[0] != home_id, tp[1]))
    column = {tp: j for j, tp in enumerate(people)}
    side = np.array(["H" if team == home_id else "A" for team, _ in people], dtype="<U1")
    if set(t for t, _ in people) - {home_id, away_id}:
        raise ValueError("{}: a FrameSet belongs to neither team".format(path))

    halves = []
    for half in HALVES:
        ball = [v for (h, team, _), v in sets.items() if h == half and team == "BALL"]
        if len(ball) != 1:
            raise ValueError("{}: expected one ball FrameSet in {}".format(path, half))
        ball = np.array(ball[0], dtype=object)
        n = ball[:, 0].astype(np.int64)
        first = n[0]
        if not np.array_equal(n, np.arange(first, first + len(n))):
            raise ValueError("{}: ball frames of {} are not consecutive".format(path, half))
        count = len(n)
        track_xy = np.full((count, len(people), 2), np.nan, dtype=np.float32)
        track_speed = np.full((count, len(people)), np.nan, dtype=np.float32)
        for (h, team, person), values in sets.items():
            if h != half or team == "BALL":
                continue
            v = np.array([row[:4] for row in values], dtype=np.float64)
            idx = v[:, 0].astype(np.int64) - first
            keep = (idx >= 0) & (idx < count)
            j = column[(team, person)]
            track_xy[idx[keep], j, 0] = v[keep, 1] + length / 2
            track_xy[idx[keep], j, 1] = v[keep, 2] + width / 2
            track_speed[idx[keep], j] = v[keep, 3]
        possession = ball[:, 4].astype(str)
        status = ball[:, 5].astype(str)
        labels = np.where(status == "1", np.where(possession == "1", "H", "A"), "D")
        halves.append({
            "ball_xy": np.stack([ball[:, 1].astype(np.float64) + length / 2,
                                 ball[:, 2].astype(np.float64) + width / 2], axis=1)
                         .astype(np.float32),
            "ball_speed": ball[:, 3].astype(np.float32),
            "track_xy": track_xy,
            "track_speed": track_speed,
            "labels": labels.astype("<U1"),
        })

    out = {key: np.concatenate([h[key] for h in halves]) for key in halves[0]}
    out["track_team"] = side
    out["half_lengths"] = np.array([len(h["labels"]) for h in halves])
    return out


def _time(text):
    return datetime.datetime.fromisoformat(text)


def half_start_times(tracking_path):
    """[time of the first ball frame of each half], read from the DFL position file."""
    starts = {}
    with open(tracking_path, encoding="utf-8") as handle:
        want = None
        for line in handle:
            if want is not None:
                starts[want] = _time(dict(_ATTR.findall(line))["T"])
                want = None
            elif line.startswith("<FrameSet ") and 'TeamId="BALL"' in line:
                want = _FRAMESET.match(line).group(1)
    return [starts[half] for half in HALVES]


def read_set_pieces(event_path, starts, half_lengths, home_id):
    """The DFL's restarts as [{type, side, frame, decision_frame}], in our frame axis (the halves
    back to back), sorted by frame.

    frame is the restart (EventTime: the ball is played), decision_frame when the referee gave it
    (DecisionTimestamp, near the ball going out; None for kick-offs). side is "H" or "A". A kick-off
    with a GameSection starts a half ("Kick Off (Start)"); the others follow a goal.
    """
    with open(event_path, encoding="utf-8") as handle:
        text = handle.read()
    offsets = [0, int(half_lengths[0])]

    def frame_of(when):
        half = 1 if when >= starts[1] else 0
        frame = offsets[half] + int(round((when - starts[half]).total_seconds() * FPS))
        return int(np.clip(frame, offsets[half], offsets[half] + int(half_lengths[half]) - 1))

    out = []
    for event_attrs, tag, attrs in _EVENT.findall(text):
        if tag not in SET_PIECE_TAGS:
            continue
        event, attrs = dict(_ATTR.findall(event_attrs)), dict(_ATTR.findall(attrs))
        kind = SET_PIECE_TAGS[tag]
        if tag == "KickOff":
            kind += " (Start)" if "GameSection" in attrs else " (After Goal)"
            side = "N"
        else:
            side = "H" if attrs.get("Team") == home_id else "A"
        decision = attrs.get("DecisionTimestamp")
        out.append({"type": kind, "side": side, "frame": frame_of(_time(event["EventTime"])),
                    "decision_frame": frame_of(_time(decision)) if decision else None})
    return sorted(out, key=lambda e: e["frame"])


def goal_frames(event_path, starts, half_lengths):
    """Frames of the goals (the shots with a SuccessfulShot), sorted."""
    with open(event_path, encoding="utf-8") as handle:
        blocks = handle.read().split("<Event ")[1:]
    out = []
    for block in blocks:
        if "<SuccessfulShot" not in block:
            continue
        when = _time(dict(_ATTR.findall(block[:block.find(">")]))["EventTime"])
        half = 1 if when >= starts[1] else 0
        offset = 0 if half == 0 else int(half_lengths[0])
        out.append(offset + int(round((when - starts[half]).total_seconds() * FPS)))
    return sorted(out)


def _set_pieces_cached(match_id, arrays, data_dir, cache_dir):
    """{set_pieces, goals} for a match, or None without an event.xml; kept as a small JSON."""
    folder = match_dir(match_id, data_dir)
    event_path = os.path.join(folder, "event.xml")
    if not os.path.isfile(event_path):
        return None
    path = os.path.join(cache_dir, "{}.set_pieces.v{}.json".format(match_id, SET_PIECES_VERSION))
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    home = read_meta(os.path.join(folder, "meta.xml"))[0]
    starts = half_start_times(os.path.join(folder, "tracking.xml"))
    out = {"set_pieces": read_set_pieces(event_path, starts, arrays["half_lengths"], home),
           "goals": goal_frames(event_path, starts, arrays["half_lengths"])}
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle)
    return out


# Stand-in Smart Tagging ---------------------------------------------------------------------------

# Window length (s) and where it starts before the restart (s, drawn uniformly), as measured on
# BAR-ATH's own Smart Tagging inputs (label-free: the ball reappears and speeds up, the players
# speed up): 35 s windows whose restart comes 4-8 s in (8-16 s for free kicks), 5 s for kick-offs.
WINDOW_S = {"Kick Off (Start)": 5, "Kick Off (After Goal)": 5}
WINDOW_DEFAULT_S = 35
LEAD_S = {"Free Kick": (8, 16), "Kick Off (Start)": (1, 4), "Kick Off (After Goal)": (1, 4)}
LEAD_DEFAULT_S = (4, 8)
# Share of each type kept, to mimic what Smart Tagging misses: BAR-ATH has 26 throw-ins, 12
# corners, 10 goal kicks, 5 free kicks and every kick-off, where the seven Bundesliga matches
# average 44, 8, 17 and 25.
KEEP = {"Throw-in": 0.6, "Corner Kick": 1.0, "Goal Kick": 0.6, "Free Kick": 0.2,
        "Penalty": 1.0, "Kick Off (Start)": 1.0, "Kick Off (After Goal)": 1.0}
# Smart Tagging tags the team on about half its set pieces (27 of BAR-ATH's 53 that are not
# kick-offs; the rest say N/A).
SIDE_KEPT = 0.5
GOAL_LEAD_S = 30  # a GOALS phase starts this long before the goal and ends 1 s into the kick-off


def stand_in_windows(set_pieces, goals, match_id, n_frames, team_names, seed=0):
    """Smart Tagging-like events ({code, start_frame, end_frame, tags}) for an IDSSE match, as the
    organisers' loader gives them; the same match and seed give the same windows."""
    rng = np.random.default_rng([seed, sum(map(ord, match_id)), 24])
    name = {"H": team_names[0], "A": team_names[1], "N": NO_TEAM_TAG}
    out = []
    for piece in set_pieces:
        kind = piece["type"]
        keep = rng.random() < KEEP.get(kind, 1.0)
        lead = rng.uniform(*LEAD_S.get(kind, LEAD_DEFAULT_S))
        side = piece["side"] if rng.random() < SIDE_KEPT else "N"
        if not keep:
            continue
        start = max(0, piece["frame"] - int(round(lead * FPS)))
        end = min(n_frames, start + WINDOW_S.get(kind, WINDOW_DEFAULT_S) * FPS)
        out.append({"code": "SET PIECES", "start_frame": start, "end_frame": end,
                    "tags": {"Type": kind, "Team": name[side]}})
    kick_offs = [e for e in out if e["tags"]["Type"] == "Kick Off (After Goal)"]
    for goal in goals:
        after = [e for e in kick_offs if e["start_frame"] > goal]
        if after:
            out.append({"code": "GOALS", "start_frame": max(0, goal - GOAL_LEAD_S * FPS),
                        "end_frame": after[0]["start_frame"] + FPS,
                        "tags": {"Team": NO_TEAM_TAG}})
    return sorted(out, key=lambda e: e["start_frame"])


def _cached(match_id, data_dir, cache_dir):
    path = os.path.join(cache_dir, "{}.v{}.npz".format(match_id, PARSE_VERSION))
    if os.path.isfile(path):
        with np.load(path) as stored:
            return {key: stored[key] for key in stored.files}
    folder = match_dir(match_id, data_dir)
    home, away, home_name, away_name, pitch = read_meta(os.path.join(folder, "meta.xml"))
    arrays = parse_tracking(os.path.join(folder, "tracking.xml"), home, away, pitch)
    arrays["team_names"] = np.array([home_name, away_name])
    arrays["pitch"] = np.array(pitch)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return arrays


# The match ---------------------------------------------------------------------------------------

class ExternalMatch:
    """One IDSSE match, answering what an experiment reads from the Match, plus its labels."""

    def __init__(self, match_id, arrays, degraded=None):
        self.match_id = match_id
        self.degraded = degraded
        self._a = arrays
        self.n_frames = len(arrays["labels"])
        n1 = int(arrays["half_lengths"][0])
        self.blocks = [(0, n1 - 1), (n1, self.n_frames - 1)]
        self.team_names = tuple(str(t) for t in arrays["team_names"])
        self.pitch = tuple(float(p) for p in arrays["pitch"])
        self._team_shape = None
        self._smart_tagging = None

    def __repr__(self):
        return "ExternalMatch({}{})".format(
            self.match_id, "" if self.degraded is None else ", degraded={}".format(self.degraded))

    labels = property(lambda self: self._a["labels"])
    ball_xy = property(lambda self: self._a["ball_xy"])
    ball_speed = property(lambda self: self._a["ball_speed"])
    track_xy = property(lambda self: self._a["track_xy"])
    track_speed = property(lambda self: self._a["track_speed"])
    track_team = property(lambda self: self._a["track_team"])

    @property
    def team_shape(self):
        """The same as Match.team_shape, computed on first use and kept on this object."""
        if self._team_shape is None:
            from possession import team_shape
            self._team_shape = team_shape.compute(self.track_xy, self.track_team, self.blocks,
                                                  self.pitch)
        return self._team_shape

    @property
    def set_pieces(self):
        """The DFL's restarts ([{type, side, frame, decision_frame}]); [] without an event.xml."""
        return (self._a.get("set_pieces") or {}).get("set_pieces", [])

    @property
    def goals(self):
        return (self._a.get("set_pieces") or {}).get("goals", [])

    @property
    def smart_tagging_events(self):
        """Stand-in Smart Tagging (stand_in_windows), in the format of Match.smart_tagging_events."""
        if self._smart_tagging is None:
            events = stand_in_windows(self.set_pieces, self.goals, self.match_id, self.n_frames,
                                      self.team_names)
            painted = paint_smart_tagging(events, self.team_names, self.n_frames)
            self._smart_tagging = (events, painted)
        return self._smart_tagging[0]

    @property
    def smart_tagging_codes(self):
        self.smart_tagging_events
        return self._smart_tagging[1]["codes"]

    @property
    def smart_tagging_team(self):
        self.smart_tagging_events
        return self._smart_tagging[1]["team"]

    @property
    def smart_tagging_phase(self):
        return self.smart_tagging_team != ""

    def with_arrays(self, degraded, **changes):
        arrays = dict(self._a)
        arrays.update(changes)
        return ExternalMatch(self.match_id, arrays, degraded)


def load(match_id, degrade=None, data_dir=DATA_DIR, cache_dir=CACHE_DIR, version=None):
    """An ExternalMatch; with `degrade` (a seed), the ball is hidden like the ATD's (degrade.py)."""
    if match_id not in MATCHES:
        raise ValueError("unknown IDSSE match {!r}; known: {}".format(match_id, ", ".join(MATCHES)))
    arrays = _cached(match_id, data_dir, cache_dir)
    arrays["set_pieces"] = _set_pieces_cached(match_id, arrays, data_dir, cache_dir)
    match = ExternalMatch(match_id, arrays)
    if degrade is None:
        return match
    from possession import degrade as _degrade
    return _degrade.degrade(match, seed=degrade, version=version or _degrade.VERSION)


def split(name, degrade=None, **kwargs):
    """Every match of a split ("train", "val", "test"), in order."""
    return [load(m, degrade=degrade, **kwargs) for m in SPLITS[name]]
