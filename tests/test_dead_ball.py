"""The dead-ball module, the stand-in Smart Tagging, degradation v3 and players_dead_ball, on
synthetic data only."""

import numpy as np

from possession import dead_ball, degrade, external
from possession.experiments import players_dead_ball

from tests.test_external import STATS

PITCH = (100.0, 60.0)
FPS = dead_ball.FPS


class Scene:
    """A Match-like object built frame by frame: players moving at a set speed, a ball."""

    def __init__(self, n, blocks=None):
        self.n_frames = n
        self.blocks = blocks or [(0, n - 1)]
        self.pitch = PITCH
        self.team_names = ("Home FC", "Away FC")
        self.ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
        self.track_xy = np.full((n, 4, 2), np.nan, dtype=np.float32)
        self.track_xy[:, :, 0] = [30, 40, 60, 70]
        self.track_xy[:, :, 1] = 30.0
        self.track_team = np.array(["H", "H", "A", "A"])
        self.speed = np.full(n, 2.5)
        self.smart_tagging_events = []

    @property
    def team_shape(self):
        return {"mean_speed": self.speed}

    def window(self, kind, start, seconds, team="N/A"):
        self.smart_tagging_events.append({"code": "SET PIECES", "start_frame": start,
                                          "end_frame": start + seconds * FPS,
                                          "tags": {"Type": kind, "Team": team}})

    def still_then_kicked(self, at, x, y, still=2 * FPS, after=2 * FPS, speed=15.0):
        """The ball lies still at (x, y) until `at`, then travels along x at `speed` m/s."""
        self.ball_xy[at - still:at] = (x, y)
        steps = np.arange(after)[:, None] * np.array([speed / FPS, 0.0])
        self.ball_xy[at:at + after] = np.array([x, y]) + steps


# Signals ----------------------------------------------------------------------------------------

def test_ball_speed_and_out_of_play():
    xy = np.zeros((20, 2))
    xy[:, 0] = np.arange(20) * 0.4      # 10 m/s
    xy[15:, 1] = -3.0                   # over a touchline
    speed = dead_ball.ball_speed(xy)
    assert np.isnan(speed[:2]).all() and np.isclose(speed[5], 10.0)
    out = dead_ball.out_of_play(xy, PITCH)
    assert not out[:15].any() and out[15:].all()


def test_runs_and_rolling_mean():
    assert dead_ball.runs([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    smoothed = dead_ball.rolling_mean([np.nan, 1.0, np.nan, 3.0], 1)
    np.testing.assert_allclose(smoothed, [1.0, 1.0, 2.0, 3.0])


def test_at_place_follows_the_laws():
    assert dead_ball.at_place("Kick Off (Start)", 50, 30, PITCH)
    assert not dead_ball.at_place("Kick Off (Start)", 60, 30, PITCH)
    assert dead_ball.at_place("Corner Kick", 99.5, 0.5, PITCH)
    assert not dead_ball.at_place("Corner Kick", 90, 5, PITCH)
    assert dead_ball.at_place("Goal Kick", 5, 33, PITCH)
    assert not dead_ball.at_place("Goal Kick", 20, 30, PITCH)
    assert dead_ball.at_place("Throw-in", 40, 59.5, PITCH)
    assert not dead_ball.at_place("Throw-in", 40, 30, PITCH)
    assert dead_ball.at_place("Free Kick", 40, 30, PITCH)


def test_ball_candidates_find_a_set_kick_a_throw_and_a_reappearance():
    n = 40 * FPS
    scene = Scene(n)
    scene.still_then_kicked(5 * FPS, 50, 30)
    t = 15 * FPS                                      # thrown back in over the y = 0 line
    scene.ball_xy[t - FPS:t] = (40, -2.0)
    scene.ball_xy[t:t + FPS] = np.c_[np.full(FPS, 40), np.linspace(0.5, 8, FPS)]
    scene.ball_xy[30 * FPS:32 * FPS] = np.c_[np.linspace(20, 40, 2 * FPS), np.full(2 * FPS, 30)]
    ball = dead_ball.Ball(scene.ball_xy, PITCH)
    hows = {}
    for frame, _, _, how in ball.candidates:
        hows.setdefault(how, frame)
    assert abs(hows["kick"] - 5 * FPS) <= 2
    assert hows["throw"] == t
    assert hows["reappear"] == 30 * FPS


# Restarts and onsets ----------------------------------------------------------------------------

def test_restart_in_window_needs_the_right_place():
    scene = Scene(40 * FPS)
    scene.still_then_kicked(10 * FPS, 70, 30)         # a free kick in midfield
    ball = dead_ball.Ball(scene.ball_xy, PITCH)
    act = scene.speed
    assert dead_ball.restart_in_window("Free Kick", 5 * FPS, 40 * FPS, ball, scene.track_xy,
                                       act, 1.2, 1.6)[1] == "ball"
    # Not a corner: no ball at a corner, no taker at a flag, the players never slowed.
    assert dead_ball.restart_in_window("Corner Kick", 5 * FPS, 40 * FPS, ball, scene.track_xy,
                                       act, 1.2, 1.6) == (None, None)


def test_formation_finds_a_thrower_beyond_the_touchline():
    scene = Scene(30 * FPS)
    scene.track_xy[5 * FPS:9 * FPS, 0] = (40, -1.5)
    frame = dead_ball.formation_restart("Throw-in", scene.track_xy, 0, 30 * FPS, PITCH)
    assert frame == 9 * FPS


def test_activity_rise():
    act = np.r_[np.full(100, 2.5), np.full(100, 0.8), np.full(100, 2.5)]
    assert dead_ball.activity_rise(act, 150, 300, 1.2, 1.6) == 200
    assert dead_ball.activity_rise(np.full(300, 2.5), 0, 300, 1.2, 1.6) is None


def test_onset_walks_back_to_the_ball_going_out():
    scene = Scene(60 * FPS)
    play = np.arange(10 * FPS, 20 * FPS)
    scene.ball_xy[play] = np.c_[20 + (play - play[0]) * 0.3, np.full(len(play), 30)]
    scene.ball_xy[20 * FPS:22 * FPS] = (101.5, 30)  # over the goal line, seen out
    ball = dead_ball.Ball(scene.ball_xy, PITCH)
    on = dead_ball.onset(40 * FPS, 0, ball, scene.speed, 1.6)
    assert on == 20 * FPS


def test_onset_from_the_players_slowing_without_a_ball():
    scene = Scene(60 * FPS)
    scene.speed[25 * FPS:40 * FPS] = 0.8
    ball = dead_ball.Ball(scene.ball_xy, PITCH)
    assert dead_ball.onset(40 * FPS, 0, ball, scene.speed, 1.6) == 25 * FPS
    scene.speed[:] = 2.5
    assert dead_ball.onset(40 * FPS, 0, ball, scene.speed, 1.6) is None
    assert dead_ball.onset(40 * FPS, 0, ball, scene.speed, 1.6, fallback=50) == 40 * FPS - 50


# The whole detection ----------------------------------------------------------------------------

def test_detect_kick_off_set_piece_and_untagged_stoppage():
    n = 200 * FPS
    scene = Scene(n, blocks=[(0, n - 1)])
    # The half starts 20 s before its kick-off; the kick-off window starts 2 s before the kick.
    scene.speed[:20 * FPS] = 0.7
    scene.still_then_kicked(20 * FPS, 50, 30)
    scene.window("Kick Off (Start)", 18 * FPS, 5)
    # A corner: the ball goes out over the goal line at 60 s, is kicked from the corner at 75 s.
    scene.ball_xy[59 * FPS:60 * FPS] = np.c_[np.linspace(80, 99, FPS), np.full(FPS, 20)]
    scene.ball_xy[60 * FPS:62 * FPS] = (102.0, 20)
    scene.speed[60 * FPS:75 * FPS] = 0.9
    scene.still_then_kicked(75 * FPS, 99.5, 0.5, speed=-15.0)   # back into the pitch
    scene.window("Corner Kick", 70 * FPS, 35, team="Home FC")
    # An untagged stoppage: players slow for 20 s, nothing tagged.
    scene.speed[130 * FPS:150 * FPS] = 0.6
    st = dead_ball.detect(scene, dead_at=1.2, live_at=1.6, min_dead_s=8, still_dead_s=None)
    assert st.dead[:20 * FPS - 2].all() and not st.dead[21 * FPS:55 * FPS].any()
    assert st.dead[60 * FPS:75 * FPS - 2].all() and not st.dead[76 * FPS:130 * FPS].any()
    assert st.dead[131 * FPS:149 * FPS].all() and not st.dead[151 * FPS:].any()
    kinds = {r.kind: r for r in st.restarts}
    assert kinds["Corner Kick"].side == "H" and kinds["Corner Kick"].how == "ball"
    assert "untagged" in kinds


def test_detect_falls_back_on_the_median_offset():
    n = 200 * FPS
    scene = Scene(n)
    for start in (10 * FPS, 60 * FPS):                # two throw-ins seen, 6 s into their windows
        scene.ball_xy[start + 5 * FPS:start + 6 * FPS] = (40, -2.0)
        scene.ball_xy[start + 6 * FPS:start + 7 * FPS] = np.c_[np.full(FPS, 40),
                                                               np.linspace(0.5, 8, FPS)]
        scene.window("Throw-in", start, 35)
    scene.window("Throw-in", 120 * FPS, 35)           # one with no evidence at all
    st = dead_ball.detect(scene, dead_at=None, still_dead_s=None)
    unseen = [r for r in st.restarts if r.frame >= 120 * FPS]
    assert unseen[0].how == "median" and abs(unseen[0].frame - 126 * FPS) <= 3


def test_window_is_cut_to_the_half_it_ends_in():
    n = 100 * FPS
    scene = Scene(n, blocks=[(0, 50 * FPS - 1), (50 * FPS, n - 1)])
    scene.still_then_kicked(52 * FPS, 50, 30)
    scene.window("Kick Off (Start)", 48 * FPS, 5)     # starts in the first half
    st = dead_ball.detect(scene, dead_at=None, still_dead_s=None)
    ko = [r for r in st.restarts if r.kind == "Kick Off (Start)"][0]
    assert ko.frame >= 50 * FPS
    assert st.dead[50 * FPS:52 * FPS - 2].all() and not st.dead[:50 * FPS].any()


# The experiment ---------------------------------------------------------------------------------

def test_combine_overrides_with_dead_and_forces_live_after_restarts():
    base = np.array(list("HHHHHHDDDDHHHH"))
    dead = np.zeros(14, dtype=bool)
    dead[1:4] = True
    stretches = dead_ball.Stretches(dead, [dead_ball.Restart(4, "Throw-in", "A", "ball"),
                                           dead_ball.Restart(6, "untagged", None, "untagged")],
                                    [1, 5])
    out = players_dead_ball.combine(base, stretches, live_after_s=4 / FPS)
    assert "".join(out[:4]) == "HDDD"
    assert out[4] == "A"                               # the side Smart Tagging gave
    labels = players_dead_ball.combine(base, stretches, live_after_s=0)
    assert "".join(labels) == "HDDDHHDDDDHHHH"
    longer = players_dead_ball.combine(base, stretches, live_after_s=10)
    assert "D" not in "".join(longer[4:])             # alive for 10 s after each restart


# Stand-in Smart Tagging and degradation v3 --------------------------------------------------------

EVENTS = """<?xml version="1.0"?>
<PutDataRequest>
<Event MatchId="M" EventId="1" EventTime="2022-09-10T20:00:00.000+02:00">
 <KickOff GameSection="firstHalf" TeamLeft="T-H" TeamRight="T-A">
  <Play Team="T-H"/>
 </KickOff>
</Event>
<Event MatchId="M" EventId="2" EventTime="2022-09-10T20:00:10.000+02:00">
 <ThrowIn Side="left" Team="T-A" DecisionTimestamp="2022-09-10T20:00:04.000+02:00">
  <Play Team="T-A"/>
 </ThrowIn>
</Event>
<Event MatchId="M" EventId="3" EventTime="2022-09-10T20:00:20.000+02:00">
 <ShotAtGoal Team="T-H">
  <SuccessfulShot CurrentResult="1:0"/>
 </ShotAtGoal>
</Event>
<Event MatchId="M" EventId="4" EventTime="2022-09-10T21:00:02.000+02:00">
 <KickOff TeamLeft="T-H" TeamRight="T-A">
  <Play Team="T-A"/>
 </KickOff>
</Event>
</PutDataRequest>
"""


def test_read_set_pieces_and_goals(tmp_path):
    import datetime
    path = tmp_path / "event.xml"
    path.write_text(EVENTS)
    tz = datetime.timezone(datetime.timedelta(hours=2))
    starts = [datetime.datetime(2022, 9, 10, 20, 0, 0, tzinfo=tz),
              datetime.datetime(2022, 9, 10, 21, 0, 0, tzinfo=tz)]
    pieces = external.read_set_pieces(str(path), starts, [1000, 1000], "T-H")
    assert [(p["type"], p["side"], p["frame"]) for p in pieces] == [
        ("Kick Off (Start)", "N", 0), ("Throw-in", "A", 250), ("Kick Off (After Goal)", "N", 1050)]
    assert pieces[1]["decision_frame"] == 100
    assert external.goal_frames(str(path), starts, [1000, 1000]) == [500]


def test_stand_in_windows_are_shaped_like_smart_tagging():
    pieces = [{"type": "Throw-in", "side": "A", "frame": 2000 + 1000 * i, "decision_frame": None}
              for i in range(40)]
    pieces.append({"type": "Kick Off (After Goal)", "side": "N", "frame": 60000,
                   "decision_frame": None})
    args = (pieces, [59000], "J03WOY", 70000, ("Home FC", "Away FC"))
    events = external.stand_in_windows(*args)
    assert events == external.stand_in_windows(*args)          # seeded
    throws = [e for e in events if e["tags"].get("Type") == "Throw-in"]
    assert 10 < len(throws) < 40                               # some dropped, like Smart Tagging
    for e in throws:
        assert e["end_frame"] - e["start_frame"] == 35 * 25
        restart = min((p["frame"] for p in pieces if p["frame"] >= e["start_frame"]))
        assert 4 * 25 <= restart - e["start_frame"] <= 8 * 25
        assert e["tags"]["Team"] in ("Away FC", "N/A")
    goals = [e for e in events if e["code"] == "GOALS"]
    assert len(goals) == 1 and goals[0]["start_frame"] < 59000 < goals[0]["end_frame"]


def test_degrade_v3_hides_the_ball_mostly_in_stoppages():
    n = 3000
    labels = np.where((np.arange(n) // 300) % 2 == 0, "H", "D").astype("<U1")
    arrays = {"labels": labels, "ball_xy": np.ones((n, 2), dtype=np.float32),
              "ball_speed": np.ones(n, dtype=np.float32),
              "track_xy": np.ones((n, 2, 2), dtype=np.float32),
              "track_speed": np.ones((n, 2), dtype=np.float32),
              "track_team": np.array(["H", "A"]), "half_lengths": np.array([1500, 1500]),
              "team_names": np.array(["Home", "Away"]), "pitch": np.array([100.0, 60.0])}
    match = external.ExternalMatch("J03WOY", arrays)
    v3 = degrade.degrade(match, seed=0, version=3, stats=STATS)
    hidden = np.isnan(v3.ball_xy[:, 0])
    dead = labels == "D"
    assert hidden[dead].mean() > 0.5 and hidden[~dead].mean() < 0.3
    again = degrade.degrade(match, seed=0, version=3, stats=STATS)
    assert np.array_equal(hidden, np.isnan(again.ball_xy[:, 0]))
