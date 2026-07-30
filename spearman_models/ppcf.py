import numpy as np
import math as m
from spearman_models.tti import TTI_vec, intercept_probability_vec

attacker_control_rate = 4.30
defender_control_rate_multiplier = 1.72
defender_control_rate = attacker_control_rate * defender_control_rate_multiplier
integration_timestep = 0.08
integration_horizon = 10

def PPCF_grid(targets, players, ball_pos=None):
    # targets: (n_cells, 2). players: structured array as in pitch.py.
    positions = np.asarray(players['position'], dtype=float)   # (n_players, 2)
    velocities = np.asarray(players['velocity'], dtype=float)  # (n_players, 2)

    if ball_pos is not None:
        ball_pos = np.asarray(ball_pos, dtype=float).reshape(1, 2)
        all_targets = np.vstack([targets, ball_pos])       # (n_cells + 1, 2)
        tti_all = TTI_vec(positions, velocities, all_targets)
        players['i_p'] = tti_all[-1]                       # exact TTI to ball
        tti = tti_all[:-1]                                 # (n_cells, n_players)
    else:
        tti = TTI_vec(positions, velocities, targets) # (n_cells, n_players)

    # per-player control rate lambda, mirroring the scalar team branch
    lam = np.where(players['team'] == 'attacker',
                   attacker_control_rate, defender_control_rate)  # (n_players,)

    n_cells = targets.shape[0]
    n = len(players)
    PPCF_array = np.zeros((n_cells, n))
    PPCF_total = np.zeros(n_cells)

    t = 0
    while t < integration_horizon and PPCF_total.min() < 0.99:
        snapshot = 1 - PPCF_total # (n_cells,)

        f = intercept_probability_vec(t, tti) # (n_cells, n_players)
        increments = snapshot[:, None] * f * lam[None, :] * integration_timestep

        # Per-cell early stop: cells already at/above 0.99 at the top of this
        # iteration would have broken out of the scalar loop, so they receive
        # no further increment.
        active = (PPCF_total < 0.99)[:, None] # (n_cells, 1)
        increments = np.where(active, increments, 0.0)

        PPCF_array += increments
        PPCF_total = PPCF_array.sum(axis=1)
        t += integration_timestep
    return PPCF_array
