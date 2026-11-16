"""What the players say about the ball when the tracking has lost it: per-frame team-shape features.

Only side-level aggregates (centre, spread, forward movement, how many are running), so they need
no player identities and work on ATD tracklets as well as on clean tracking. x is turned so that
the home side always attacks towards +x (each side's deepest player tells which goal it defends).

    match.team_shape["mean_speed"]   # (n_frames,) float64, NaN where it cannot be told
    X, names = features(match)       # all columns as a matrix for a model, NaN replaced by 0

Every column, in metres, m/s or counts; x along the length (home attacking +x), y across:
    home_x, away_x                 each side's centre along the pitch
    gap_x, gap_y                   home centre minus away centre
    home_spread_x/_y, away_spread_x/_y   each side's spread (standard deviation)
    spread_diff_x/_y               home spread minus away spread
    home_vx, away_vx, vx_diff      each side's mean forward speed over the last second, and the gap
    mean_speed, share_running      all players' mean speed, and the share faster than 4 m/s
    all_x, all_y, all_spread_x/_y  centre and spread of all players together
    n_home, n_away                 players (track slots) tracked on each side
"""

import numpy as np

import games

VERSION = 1  # bump whenever compute() returns something different: it is part of the cache key
FPS = 25
LAG = 25            # velocities over one second
SPRINT = 4.0        # m/s: a player faster than this counts as running
NAMES = (
    "home_x", "away_x", "gap_x", "gap_y",
    "home_spread_x", "home_spread_y", "away_spread_x", "away_spread_y",
    "spread_diff_x", "spread_diff_y",
    "home_vx", "away_vx", "vx_diff", "mean_speed", "share_running",
    "all_x", "all_y", "all_spread_x", "all_spread_y",
    "n_home", "n_away",
)


def blocks_of(match):
    """The halves as [(first, last)]: the match's own, or BAR-ATH's."""
    blocks = getattr(match, "blocks", None)
    if blocks is not None:
        return blocks
    from possession.fill import half_blocks
    return half_blocks()


def pitch_of(match):
    pitch = getattr(match, "pitch", None)
    return tuple(pitch) if pitch is not None else tuple(games.GAMES[games.GAME]["pitch"])


def _side(values, mask, reduce):
    picked = np.where(mask[None, :, None] if values.ndim == 3 else mask[None, :], values, np.nan)
    with np.errstate(all="ignore"), _quiet():
        return reduce(picked, axis=1)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def home_attacks_plus_x(track_xy, home, away, blocks, n_frames):
    """(n,) +1 where the home side attacks towards +x, -1 where towards -x, half by half.

    Each side's deepest player (its keeper, nearly always) stays near the goal it defends, whoever
    dominates the half: so the side whose lowest-x player is lower, over the half, defends x = 0.
    Not the kick-off formation: a half's first frames need not be the kick-off (BAR-ATH's second
    half block starts with 14 players tracked, away from the kick-off).
    """
    sign = np.ones(n_frames)
    for first, last in blocks:
        x = track_xy[first:last + 1, :, 0]
        with _quiet():
            home_low = np.nanmedian(_side(x, home, np.nanmin))
            away_low = np.nanmedian(_side(x, away, np.nanmin))
        sign[first:last + 1] = 1.0 if home_low < away_low else -1.0
    return sign


def steady(tracked, lag):
    """(n, slots) bool: the slot is tracked on every frame from t - lag to t, so a velocity over
    `lag` frames compares one tracklet with itself."""
    gaps = np.cumsum(~tracked, axis=0)                         # untracked frames in [0, t]
    before = np.vstack([np.zeros((lag + 1, tracked.shape[1]), dtype=gaps.dtype),
                        gaps[:-lag - 1]])                      # untracked frames in [0, t - lag - 1]
    ok = (gaps - before) == 0
    ok[:lag] = False
    return ok


def compute(track_xy, track_team, blocks, pitch):
    """{name: (n,) float64} for every name in NAMES, NaN where it cannot be told (no player of
    that side tracked, or no steady velocity). The Match's `team_shape` item and ExternalMatch's
    attribute are both this."""
    xy = np.array(track_xy, dtype=np.float64)
    team = np.asarray(track_team)
    home, away = team == "H", team == "A"
    n = len(xy)
    length = pitch[0]
    sign = home_attacks_plus_x(xy, home, away, blocks, n)
    xy[..., 0] = np.where(sign[:, None] > 0, xy[..., 0], length - xy[..., 0])

    tracked = ~np.isnan(xy[..., 0])
    hc, ac = _side(xy, home, np.nanmean), _side(xy, away, np.nanmean)
    hs, as_ = _side(xy, home, np.nanstd), _side(xy, away, np.nanstd)
    both = home | away
    allc, alls = _side(xy, both, np.nanmean), _side(xy, both, np.nanstd)

    ok = steady(tracked, LAG)
    past = np.vstack([np.full((LAG,) + xy.shape[1:], np.nan), xy[:-LAG]])
    v = np.where(ok[..., None], (xy - past) / (LAG / FPS), np.nan)
    speed = np.linalg.norm(v, axis=2)
    hv, av = _side(v, home, np.nanmean), _side(v, away, np.nanmean)
    with np.errstate(all="ignore"), _quiet():
        mean_speed = np.nanmean(speed, axis=1)
        running = np.nanmean(np.where(np.isnan(speed), np.nan, speed > SPRINT), axis=1)

    columns = (
        hc[:, 0], ac[:, 0], hc[:, 0] - ac[:, 0], hc[:, 1] - ac[:, 1],
        hs[:, 0], hs[:, 1], as_[:, 0], as_[:, 1], hs[:, 0] - as_[:, 0], hs[:, 1] - as_[:, 1],
        hv[:, 0], av[:, 0], hv[:, 0] - av[:, 0], mean_speed, running,
        allc[:, 0], allc[:, 1], alls[:, 0], alls[:, 1],
        (tracked & home).sum(axis=1), (tracked & away).sum(axis=1),
    )
    return {name: np.asarray(column, dtype=np.float64) for name, column in zip(NAMES, columns)}


def of(match):
    """The match's team shape: its own attribute (cached for the Match), or computed here."""
    table = getattr(match, "team_shape", None)
    if table is not None:
        return table
    return compute(match.track_xy, match.track_team, blocks_of(match), pitch_of(match))


def features(match, names=NAMES):
    """(X, names): the named columns as one (n, k) matrix for a model, NaN replaced by 0."""
    table = of(match)
    return np.nan_to_num(np.column_stack([table[name] for name in names])), tuple(names)
