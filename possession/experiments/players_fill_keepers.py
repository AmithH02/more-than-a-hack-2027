"""players_fill, with the keepers put on the right side (possession/keepers.py).

The same method and the same models as players_fill. The only difference: where the ball is
tracked, the nearest player's side is read from match.track_team_fixed, in which each tracklet of
a keeper slot belongs to the side defending the end it stands in, instead of the side the ATD's
metadata gives the slot.

On the bench's matches the sides are already right, so this scores exactly as players_fill there;
its effect can only be seen on BAR-ATH, through the proxy report.
"""

import numpy as np

from possession.experiments.nearest_smooth import nearest_team_codes
from possession.experiments.players_fill import DEFAULT, GRID, gap_codes  # noqa: F401
from starter.baseline import smooth  # organisers': imported, never edited


def predict_with(match, window, alive_at):
    codes = nearest_team_codes(match.ball_xy, match.track_xy, match.track_team_fixed)
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
    codes = np.where(hidden, gap_codes(match, alive_at), codes).astype(np.int8)
    return smooth(codes, window)


def predict(match):
    return predict_with(match, **DEFAULT)
