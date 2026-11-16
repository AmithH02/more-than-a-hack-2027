"""The other matches (possession/external.py), their degradation and the bench, on a tiny synthetic
DFL file; never touches real data."""

import numpy as np

from possession import bench, degrade, external
from possession.experiments import nearest_smooth

META = """<?xml version="1.0"?>
<PutDataRequest><MatchInformation>
<General HomeTeamId="T-H" GuestTeamId="T-A" HomeTeamName="Home FC" GuestTeamName="Away FC" />
<Environment PitchX="100.00" PitchY="60.00" />
</MatchInformation></PutDataRequest>
"""


def frameset(half, team, person, rows):
    lines = ['<FrameSet GameSection="{}" MatchId="M" TeamId="{}" PersonId="{}">'.format(
        half, team, person)]
    for row in rows:
        if team == "BALL":
            n, x, y, poss, status = row
            lines.append('<Frame N="{}" T="t" X="{:.2f}" Y="{:.2f}" Z="0.10" D="0.00" S="1.00" '
                         'A="0.00" M="1" BallPossession="{}" BallStatus="{}"/>'.format(
                             n, x, y, poss, status))
        else:
            n, x, y = row
            lines.append('<Frame N="{}" T="t" X="{:.2f}" Y="{:.2f}" D="0.00" S="2.00" A="0.00" '
                         'M="1"/>'.format(n, x, y))
    lines.append("</FrameSet>")
    return "\n".join(lines)


def write_match(folder):
    folder.mkdir(parents=True)
    (folder / "meta.xml").write_text(META)
    parts = ['<?xml version="1.0"?>\n<PutDataRequest><Positions>']
    # First half, frames 100-109: home has it, then a stoppage, then away.
    ball1 = [(100 + i, -10.0 + i, 0.0, 1 if i < 6 else 2, 0 if 4 <= i < 6 else 1)
             for i in range(10)]
    parts.append(frameset("firstHalf", "BALL", "B", ball1))
    parts.append(frameset("firstHalf", "T-H", "P1", [(100 + i, -10.0 + i, 0.5) for i in range(10)]))
    parts.append(frameset("firstHalf", "T-A", "P2", [(100 + i, 20.0, 0.0) for i in range(10)]))
    parts.append(frameset("firstHalf", "referee", "R", [(100 + i, 0.0, 0.0) for i in range(10)]))
    # Second half, frames 500-505; P3 comes on, P1 is gone.
    ball2 = [(500 + i, 5.0, 5.0, 2, 1) for i in range(6)]
    parts.append(frameset("secondHalf", "BALL", "B", ball2))
    parts.append(frameset("secondHalf", "T-A", "P2", [(500 + i, 5.0, 5.5) for i in range(6)]))
    parts.append(frameset("secondHalf", "T-H", "P3", [(498 + i, 40.0, 0.0) for i in range(8)]))
    parts.append("</Positions></PutDataRequest>\n")
    (folder / "tracking.xml").write_text("\n".join(parts))


def test_parse_and_labels(tmp_path, monkeypatch):
    write_match(tmp_path / "data" / "J03WOY")
    match = external.load("J03WOY", data_dir=str(tmp_path / "data"),
                          cache_dir=str(tmp_path / "cache"))
    assert match.n_frames == 16
    assert match.blocks == [(0, 9), (10, 15)]
    assert match.team_names == ("Home FC", "Away FC")
    assert "".join(match.labels) == "HHHHDDAAAA" + "AAAAAA"
    assert list(match.track_team) == ["H", "H", "A"]  # P1, P3, P2; no referee
    # Centre-origin metres become corner-origin metres: x + 50, y + 30.
    np.testing.assert_allclose(match.ball_xy[0], [40.0, 30.0])
    np.testing.assert_allclose(match.track_xy[0, 0], [40.0, 30.5])
    assert np.isnan(match.track_xy[10:, 0]).all()      # P1 is not in the second half
    assert not np.isnan(match.track_xy[10:, 1]).any()  # P3's frames before 500 are dropped
    assert match.smart_tagging_codes.size == 0
    # Cached: a second load reads the npz and gives the same arrays.
    again = external.load("J03WOY", data_dir=str(tmp_path / "data"),
                          cache_dir=str(tmp_path / "cache"))
    assert (again.labels == match.labels).all()


STATS = {"ball_present": np.array([3]), "ball_missing": np.array([2]),
         "tracklets": np.array([4, 6]), "shortfall": np.array([20, 21, 21, 20]),
         "slots_home": 3, "slots_away": 2}


def test_degrade_v1_hides_only_the_ball_and_is_seeded(tmp_path):
    write_match(tmp_path / "data" / "J03WOY")
    match = external.load("J03WOY", data_dir=str(tmp_path / "data"),
                          cache_dir=str(tmp_path / "cache"))
    a = degrade.degrade(match, seed=1, version=1, stats=STATS)
    b = degrade.degrade(match, seed=1, version=1, stats=STATS)
    hidden = np.isnan(a.ball_xy[:, 0])
    assert hidden.any() and not hidden.all()
    assert (hidden == np.isnan(b.ball_xy[:, 0])).all()
    assert not np.isnan(match.ball_xy).any()  # the original is untouched
    np.testing.assert_array_equal(a.track_xy, match.track_xy)
    assert (a.labels == match.labels).all()


def test_ball_mask_alternates_runs():
    hidden = degrade.ball_mask(1000, np.array([7]), np.array([3]), np.random.default_rng(0))
    edges = np.flatnonzero(np.diff(hidden.astype(int))) + 1
    lengths = np.diff(edges)
    assert set(lengths[hidden[edges[:-1]]]) == {3}
    assert set(lengths[~hidden[edges[:-1]]]) == {7}


def test_bench_scores_the_baseline(tmp_path):
    write_match(tmp_path / "data" / "J03WOY")
    match = external.load("J03WOY", data_dir=str(tmp_path / "data"),
                          cache_dir=str(tmp_path / "cache"))
    perfect = bench.score(match.labels, match)
    assert perfect["spell_f1_mean"] == 1.0 and perfect["macro_f1"] == 1.0
    labels = nearest_smooth.predict_with(match, window=0)
    assert labels[0] == "H" and labels[-1] == "A"
    assert 0.0 <= bench.score(labels, match)["spell_f1_mean"] <= 1.0
