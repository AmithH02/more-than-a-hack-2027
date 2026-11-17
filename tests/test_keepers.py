"""The keeper rule on tiny synthetic tracking; never touches the match data."""

import numpy as np

from possession import keepers
from possession import match as m
from possession.experiments.nearest_smooth import nearest_team_codes
from starter.baseline import AWAY, HOME

LENGTH = 100.0
NAN = np.nan


def tracking(columns):
    """(n, slots, 2) from one list of x per slot; y is the middle of the pitch."""
    x = np.array(columns, dtype=np.float32).T
    return np.stack([x, np.full_like(x, 34.0)], axis=2)


def test_runs():
    assert keepers.runs(np.array([True, True, False, True])) == [(0, 2), (3, 4)]
    assert keepers.runs(np.array([False, False])) == []


def test_sides_at_kickoff_ignore_the_keeper_slots():
    # Home outfield at 30, away outfield at 70; the home keeper slot stands at the away end.
    xy = tracking([[30] * 3, [95] * 3, [70] * 3, [5] * 3])
    team = np.array(["H", "H", "A", "A"])
    assert keepers.home_defends_low_x(xy, team, 1, keeper_slots=(1, 3)) is True
    assert keepers.home_defends_low_x(LENGTH - xy, team, 1, keeper_slots=(1, 3)) is False


def test_a_keeper_tracklet_takes_the_side_defending_its_end():
    # Slot 1 is the home keeper slot, slot 3 the away one. Home defends x = 0 in the first half
    # (frames 0-3) and x = 100 in the second (frames 4-7).
    home_keeper = [5, 5, NAN, 95, 95, 95, NAN, 5]     # tracklets at: low, high | high, low
    away_keeper = [95, 95, 95, 95, 5, 5, 5, 5]
    xy = tracking([[30] * 8, home_keeper, [70] * 8, away_keeper])
    team = np.array(["H", "H", "A", "A"])
    fixed = keepers.fix(xy, team, (1, 3), [(0, 3), (4, 7)], [True, False], LENGTH)
    assert fixed.shape == (8, 4)
    assert "".join(fixed[:, 1]) == "HHHAHHHA"         # untracked frames keep the slot's side
    assert "".join(fixed[:, 3]) == "AAAAAAAA"
    assert (fixed[:, 0] == "H").all() and (fixed[:, 2] == "A").all()   # outfield slots untouched
    assert keepers.changed(team, fixed).tolist() == [0, 0, 0, 1, 0, 0, 0, 1]


def test_a_tracklet_over_halftime_is_judged_in_each_half():
    xy = tracking([[5] * 6])
    fixed = keepers.fix(xy, np.array(["H"]), (0,), [(0, 2), (3, 5)], [True, False], LENGTH)
    assert "".join(fixed[:, 0]) == "HHHAAA"


def test_both_copies_of_a_split_keeper_go_to_the_same_side():
    xy = tracking([[6] * 4, [7] * 4])
    fixed = keepers.fix(xy, np.array(["H", "A"]), (0, 1), [(0, 3)], [True], LENGTH)
    assert (fixed == "H").all()


def test_nearest_player_reads_sides_per_frame():
    ball = np.array([[10.0, 34.0], [10.0, 34.0]], dtype=np.float32)
    xy = tracking([[11, 11], [60, 60]])
    per_frame = np.array([["A", "H"], ["H", "A"]])
    assert nearest_team_codes(ball, xy, per_frame).tolist() == [AWAY, HOME]
    assert nearest_team_codes(ball, xy, np.array(["H", "A"])).tolist() == [HOME, HOME]


META = """<Root><Teams><Team id="t1"><Name>Home</Name></Team><Team id="t2"><Name>Away</Name></Team>
</Teams><Players>
  <Player id="1" teamId="t1"><ProviderPlayerParameters><ProviderParameter>
    <Name>position_type</Name><Value>Unknown</Value></ProviderParameter></ProviderPlayerParameters>
  </Player>
  <Player id="2" teamId="t2"><ProviderPlayerParameters><ProviderParameter>
    <Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters>
  </Player>
</Players></Root>"""


def test_read_keeper_slots(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text(META)
    assert m.read_keeper_slots(str(path)).tolist() == [1]


def test_kickoff_frames_picks_the_start_nearest_each_half():
    codes = np.array(["BUILD UP", m.KICKOFF_CODE])
    phase = np.zeros((40, 2), dtype=bool)
    phase[3:6, 1] = True
    phase[22:25, 1] = True
    assert m.kickoff_frames(codes, phase, [(4, 15), (20, 39)]) == [3, 22]
