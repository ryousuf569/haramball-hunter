import numpy as np
from physics.engine import global_to_local
from environment.grid import PC_NX, PC_NY, PC_CELL_SIZE

GOAL = [105, 34]

def nearest_grid_cell(position):
    # Global position -> (i, j)
    local = global_to_local(np.atleast_2d(np.asarray(position, dtype=float)))
    idx = np.floor(local / PC_CELL_SIZE).astype(int)

    idx[:, 0] = np.clip(idx[:, 0], 0, PC_NX - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, PC_NY - 1)

    return idx

def scoring_probability(position):
    # S(r) - constants pulled from Spearman paper

    d = np.linalg.norm(position - GOAL, axis=1)

    a = -0.14 * np.sqrt(d)
    e = np.exp(a)
    b = 0.93 * e

    return b ** 0.48

# Cell-index offsets covering AREA_RADIUS around a cell
AREA_RADIUS = 3.0
_r = int(AREA_RADIUS // PC_CELL_SIZE)
_di, _dj = np.meshgrid(np.arange(-_r, _r + 1), np.arange(-_r, _r + 1), indexing="ij")
_within = (_di ** 2 + _dj ** 2) * PC_CELL_SIZE ** 2 <= AREA_RADIUS ** 2
AREA_DI, AREA_DJ = _di[_within], _dj[_within]

def pcf_in_area(position, pcf_grid):
    # mean pitch control over the cells within AREA_RADIUS of the player.
    # pcf_grid is step()'s pc_att, already reshaped to (PC_NX, PC_NY).

    idx = nearest_grid_cell(position)
    ii = np.clip(idx[:, 0, None] + AREA_DI, 0, PC_NX - 1)
    jj = np.clip(idx[:, 1, None] + AREA_DJ, 0, PC_NY - 1)

    return pcf_grid[ii, jj].mean(axis=1)

def nearest_defender_distance(players, position):
    # metres to the closest defender, keeper included

    dmask = players["team"] == "defender"
    delta = players["position"][dmask] - position
    d2 = np.einsum("ij,ij->i", delta, delta)

    return float(np.sqrt(d2.min())) > 3.0

def check_shot_opening(players, ball, pcf_att):

    holder_id = ball.get("holder_id")
    if holder_id is None:
        return False

    holder = players[players["id"] == holder_id][0]
    if holder["team"] != "attacker":
        return False

    position = np.atleast_2d(np.asarray(holder["position"], dtype=float))

    # Both thresholds come from environment/calibration/
    return bool(pcf_in_area(position, pcf_att)[0] >= 0.30
                and scoring_probability(position)[0] >= 0.74
                and nearest_defender_distance(players, position[0]))
