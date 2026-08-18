from collections import namedtuple

import numpy as np
from physics.engine import global_to_local
from environment.grid import PC_NX, PC_NY, PC_CELL_SIZE, make_ppcf_grid

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
AREA_CELLS = len(AREA_DI)


def area_cells(position):
    local = global_to_local(np.atleast_2d(np.asarray(position, dtype=float)))
    idx = np.floor(local / PC_CELL_SIZE).astype(int)

    ii = idx[:, 0, None] + AREA_DI
    jj = idx[:, 1, None] + AREA_DJ
    valid = (ii >= 0) & (ii < PC_NX) & (jj >= 0) & (jj < PC_NY)
    return np.clip(ii, 0, PC_NX - 1), np.clip(jj, 0, PC_NY - 1), valid


def _masked_mean(vals, ii, jj, valid, cell_mask):
    vals = np.where(valid, vals, 0.0)
    if cell_mask is not None:
        vals = np.where(np.asarray(cell_mask)[ii, jj], vals, 0.0)
    return vals.sum(axis=1) / AREA_CELLS


def pcf_in_area(position, pcf_grid, cell_mask=None):
    # mean pitch control over the cells within AREA_RADIUS of the player.
    # pcf_grid is step()'s pc_att, already reshaped to (PC_NX, PC_NY).
    # cell_mask, if given, is a (PC_NX, PC_NY) bool: cells outside it count 0.
    ii, jj, valid = area_cells(position)
    return _masked_mean(np.asarray(pcf_grid)[ii, jj], ii, jj, valid, cell_mask)


def own_pcf_in_area(position, pcf_own, cell_mask=None):
    position = np.atleast_2d(np.asarray(position, dtype=float))
    ii, jj, valid = area_cells(position)
    who = np.arange(position.shape[0])[:, None]
    return _masked_mean(np.asarray(pcf_own)[ii, jj, who], ii, jj, valid,
                        cell_mask)

# Spearman's S(r) constants, named so the inverse cannot drift from the forward.
_S_A, _S_B, _S_C = 0.93, 0.14, 0.48


def radius_for_p(p):
    # Distance from goal at which scoring_probability equals p, in metres
    p = float(p)
    return (-np.log(p ** (1.0 / _S_C) / _S_A) / _S_B) ** 2


def p_for_radius(r):
    # scoring_probability at r metres from goal, the inverse of the above
    return float((_S_A * np.exp(-_S_B * np.sqrt(float(r)))) ** _S_C)

# Measured under random play, over the disc's cells: mean control is ~0.26
# wherever the disc is put and whatever radius it has (0.23-0.28 over centres
# 76-84 and radii 10-15), peaking near 0.5. A quarter of the space is simply
# what attackers own in the final third against a compact block, so the
# threshold sits above that mean rather than at a half.
ZONE_X = 86.0
ZONE_Y = 34.0
ZONE_RADIUS = 8.0
ZONE_PC_MIN = 0.45 # mean attacker control over the zone's cells

Zone = namedtuple("Zone", ("centre", "radius", "mask", "n_cells"))


def make_zone(x=ZONE_X, y=ZONE_Y, radius=ZONE_RADIUS):
    centres = make_ppcf_grid().reshape(PC_NX, PC_NY, 2)
    centre = np.array([float(x), float(y)])
    d = np.linalg.norm(centres - centre, axis=-1)
    mask = d <= float(radius)
    return Zone(centre=centre, radius=float(radius), mask=mask,
                n_cells=int(mask.sum()))


def ball_in_zone(position, zone):
    delta = np.asarray(position, dtype=float) - zone.centre
    return bool(delta @ delta <= zone.radius ** 2)


def zone_control(pc_att, zone):
    # Mean attacker control over the zone
    if zone.n_cells == 0:
        return 0.0
    return float(np.asarray(pc_att)[zone.mask].sum() / zone.n_cells)


def success_gate(players, ball, pc_att, zone):
    holder_id = ball.get("holder_id")
    if ball.get("state") != "held" or holder_id is None:
        return None, None

    holder = players[players["id"] == holder_id]
    if holder.size == 0 or holder["team"][0] != "attacker":
        return None, None

    return ball_in_zone(ball["position"], zone), zone_control(pc_att, zone)


def check_success(players, ball, pc_att, zone, pc_min=ZONE_PC_MIN):
    in_zone, control = success_gate(players, ball, pc_att, zone)
    if in_zone is None:
        return False
    return bool(in_zone and control >= pc_min)
