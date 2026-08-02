# Scenario builders for the PPCF golden fixture. Used only by generate_golden.py.
# test_ppcf_golden.py replays raw arrays from the .npz instead of importing this,
# so it stays independent of the attacker/defender policies.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import player_dt  # noqa: E402


def build_scenarios():
    from environment.lowblock_env import make_initial_world, make_ppcf_grid, step

    scenarios = {}
    targets = make_ppcf_grid()  # production grid: 31 x 34 cells @ 2m = 1054

    # kick-off: everyone at rest, so all 20 players hit TTI's v_parallel == 0 branch
    players, ball, attacker_ids, rng, dstate = make_initial_world(
        n_att=10, n_def=10, seed=0)
    scenarios["tick0"] = {
        "players": players.copy(),
        "targets": targets,
        "ball_pos": np.asarray(ball["position"], dtype=float),
    }

    # same state down the ball_pos=None branch, which skips the i_p cache
    scenarios["tick0_no_ball"] = {
        "players": players.copy(),
        "targets": targets,
        "ball_pos": None,
    }

    # mid-episode: players spread and moving, the longest loop seen (93 iterations)
    for tick in range(40):
        players, ball, _ = step(players, ball, attacker_ids, dstate, tick,
                                ppcf_grid=targets)
    scenarios["tick40"] = {
        "players": players.copy(),
        "targets": targets,
        "ball_pos": np.asarray(ball["position"], dtype=float),
    }

    # edge cases the sim states don't reliably produce
    edge = np.zeros(4, dtype=player_dt)
    edge["id"] = np.arange(1, 5)
    edge["team"] = ["attacker", "attacker", "defender", "defender"]
    edge["position"] = [[10.0, 10.0], [50.0, 40.0], [10.0, 10.5], [90.0, 60.0]]
    # player 3 sprints away from the grid -> TTI's Case B braking branch
    edge["velocity"] = [[0.0, 0.0], [5.0, 0.0], [-5.0, -5.0], [0.0, 0.0]]
    edge_targets = np.array([
        [10.0, 10.0],    # exactly on player 1 -> D == 0
        [10.0, 10.5],    # exactly on player 3 -> D == 0, defender rate
        [0.0, 0.0],      # far corner, slow convergence
        [105.0, 68.0],   # opposite far corner
        [52.5, 34.0],    # midpoint
    ])
    scenarios["edge_cases"] = {
        "players": edge,
        "targets": edge_targets,
        "ball_pos": np.array([50.0, 40.0]),
    }

    return scenarios
