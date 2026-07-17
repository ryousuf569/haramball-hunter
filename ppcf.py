import numpy as np
import math as m

attacker_control_rate = 4.30
defender_control_rate_multiplier = 1.72
defender_control_rate = attacker_control_rate * defender_control_rate_multiplier
integration_timestep = 0.05
integration_horizon = 10

player_dt = np.dtype([('position', 'f4', (2,)), ('velocity', 'f4', (2,)), ('team', 'U8')])

# toy implementation
players = np.array([
    ([10.0, 25.5], [1.2, -0.5], 'attacker'),
    ([14.2, 11.0], [-0.8, 2.1], 'defender')
], dtype=player_dt)