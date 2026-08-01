import numpy as np
import math as m
from physics.tti import TTI_vec, intercept_probability_vec

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

    # Per-cell early stop. A cell at/above 0.99 would have broken out of the
    # scalar loop, so it takes no further increment
    # Previously this was a full-grid computation with the inactive rows
    # multiplied out to zero by np.where
    active = np.ones(n_cells, dtype=bool)

    t = 0
    while t < integration_horizon and active.any():
        tti_a = tti[active] # (n_active, n_players)
        snapshot_a = 1 - PPCF_total[active] # (n_active,)

        f = intercept_probability_vec(t, tti_a) # (n_active, n_players)
        increments = snapshot_a[:, None] * f * lam[None, :] * integration_timestep

        PPCF_array[active] += increments
        PPCF_total[active] = PPCF_array[active].sum(axis=1)

        active &= (PPCF_total < 0.99)
        t += integration_timestep
    return PPCF_array
