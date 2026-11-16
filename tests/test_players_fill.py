"""Degradation v2, team-shape features, the small learner and players_fill, on synthetic data."""

import numpy as np

from possession import degrade, external, learn, team_shape
from possession.experiments import players_fill

from tests.test_external import STATS, write_match


def synthetic_match(n=600, seed=0):
    """Two sides of 11 on a 100 x 60 pitch; home defends x = 0 in the first half, x = 100 after."""
    rng = np.random.default_rng(seed)
    half = n // 2
    xy = np.empty((n, 22, 2), dtype=np.float32)
    base = np.r_[np.linspace(5, 45, 11), np.linspace(55, 95, 11)]
    for t in range(n):
        x = base if t < half else 100 - base
        xy[t, :, 0] = x + rng.normal(0, 1, 22)
        xy[t, :, 1] = rng.uniform(5, 55, 22)
    labels = np.where(np.arange(n) % 100 < 20, "D", np.where(np.arange(n) % 200 < 100, "H", "A"))
    ball = xy[:, 5].copy()
    arrays = {"labels": labels.astype("<U1"), "ball_xy": ball,
              "ball_speed": np.ones(n, dtype=np.float32), "track_xy": xy,
              "track_speed": np.ones((n, 22), dtype=np.float32),
              "track_team": np.array(["H"] * 11 + ["A"] * 11),
              "half_lengths": np.array([half, n - half]),
              "team_names": np.array(["Home", "Away"]), "pitch": np.array([100.0, 60.0])}
    return external.ExternalMatch("J03WOY", arrays)


# Degradation v2 ---------------------------------------------------------------------------------

def test_v2_keeps_the_v1_ball_mask_and_is_seeded():
    match = synthetic_match()
    v1 = degrade.degrade(match, seed=3, version=1, stats=STATS)
    v2 = degrade.degrade(match, seed=3, version=2, stats=STATS)
    again = degrade.degrade(match, seed=3, version=2, stats=STATS)
    assert np.array_equal(np.isnan(v1.ball_xy[:, 0]), np.isnan(v2.ball_xy[:, 0]))
    np.testing.assert_array_equal(v2.track_xy, again.track_xy)
    assert (v2.labels == match.labels).all()


def test_v2_slots_hold_their_own_side_and_match_the_shortfall():
    match = synthetic_match()
    stats = dict(STATS, shortfall=np.array([0, 1, 2, 1]))
    v2 = degrade.degrade(match, seed=0, version=2, stats=stats)
    assert list(v2.track_team) == ["H", "H", "H", "A", "A"]
    tracked = ~np.isnan(v2.track_xy[:, :, 0])
    assert 0 < tracked.sum(axis=1).max() <= 5        # never more than the slots
    # every slot position comes from a player of the slot's side: home x < 50 in the first half
    first_half = v2.track_xy[:300]
    home_x = first_half[:, :3, 0][~np.isnan(first_half[:, :3, 0])]
    away_x = first_half[:, 3:, 0][~np.isnan(first_half[:, 3:, 0])]
    assert home_x.max() < 50 < away_x.min()


def test_allocate_leaves_a_frame_between_tracklets_in_a_slot():
    n = 20
    xy = np.zeros((n, 2, 2), dtype=np.float32)
    visible = np.zeros((n, 2), dtype=bool)
    visible[0:10, 0] = True
    visible[10:20, 1] = True       # starts the frame the first ends: needs a second slot
    segments = np.zeros((n, 2), dtype=np.int64)
    out, _, team = degrade.allocate(xy, np.zeros((n, 2)), visible, segments,
                                    np.array(["H", "H"]), 2, 0)
    tracked = ~np.isnan(out[:, :, 0])
    assert tracked[:10, 0].all() and tracked[10:, 1].all() and not tracked[10:, 0].any()


# Team shape ------------------------------------------------------------------------------------

def test_direction_follows_each_sides_deepest_player():
    match = synthetic_match()
    team = np.asarray(match.track_team)
    sign = team_shape.home_attacks_plus_x(match.track_xy, team == "H", team == "A",
                                          match.blocks, match.n_frames)
    assert (sign[:300] == 1).all() and (sign[300:] == -1).all()


def test_features_turn_the_second_half_round():
    X, names = team_shape.features(synthetic_match())
    home_x = X[:, list(names).index("home_x")]
    assert abs(home_x[:300].mean() - home_x[300:].mean()) < 2   # same after turning round
    assert X.shape == (600, len(names)) and np.isfinite(X).all()


def test_steady_needs_an_unbroken_lag():
    tracked = np.ones((10, 1), dtype=bool)
    tracked[4, 0] = False
    ok = team_shape.steady(tracked, 3)
    assert list(ok[:, 0]) == [False, False, False, True, False, False, False, False, True, True]


# Learner and experiment -------------------------------------------------------------------------

def test_logistic_separates_and_round_trips():
    X = np.r_[np.zeros((50, 1)), np.ones((50, 1))]
    y = np.r_[np.zeros(50), np.ones(50)]
    model = learn.fit_logistic(X, y)
    p = learn.predict_proba(model, X)
    assert (p[:50] < 0.5).all() and (p[50:] > 0.5).all()
    assert learn.fit_logistic(X, y) == model                    # deterministic


def test_players_fill_fits_saves_and_predicts(tmp_path):
    match = degrade.degrade(synthetic_match(), seed=0, version=2, stats=STATS)
    path = str(tmp_path / "model.json")
    models = players_fill.fit([match], path=path, say=lambda *a: None)
    assert set(models) >= {"alive", "home"}
    codes = players_fill.gap_codes(match, alive_at=0.5, models=models)
    assert codes.shape == (match.n_frames,)
    players_fill._load.cache = models
    try:
        labels = players_fill.predict_with(match, window=10, alive_at=0.5)
    finally:
        players_fill._load.cache = None
    assert labels.shape == (match.n_frames,) and set(labels) <= {"H", "A", "D"}
