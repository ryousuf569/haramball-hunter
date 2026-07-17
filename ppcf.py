import numpy as np
import math as m
from tti import derive_t, TTI, intercept_probability

attacker_control_rate = 4.30
defender_control_rate_multiplier = 1.72
defender_control_rate = attacker_control_rate * defender_control_rate_multiplier
integration_timestep = 0.05
integration_horizon = 10

player_dt = np.dtype([('position', 'f4', (2,)), ('velocity', 'f4', (2,)), ('team', 'U8'), ('tti', 'f4')])

players = np.array([
    ([5, -5], [2.0, -1.0], 'attacker', 0),
    ([-10.0, 10.0], [-1.0, 1.0], 'defender', 0)
], dtype=player_dt)

def PPCF(target, players):

    for i in players:
        pos = i[0]
        vel = i[1]
        i[3] = TTI(pos, vel, target)

    t = 0
    n = len(players)
    PPCF_array = np.zeros(n)
    PPCF_total = 0.0

    while t < integration_horizon and PPCF_total < 0.99:
        snapshot = 1 - PPCF_total 

        increments = np.zeros(n)
        for j, p in enumerate(players):
            lam = attacker_control_rate if p['team'] == 'attacker' else defender_control_rate
            f = intercept_probability(t, p['position'], p['velocity'], target, p['tti'])
            increments[j] = snapshot * f * lam * integration_timestep

        PPCF_array += increments
        PPCF_total = PPCF_array.sum()
        t += integration_timestep

    return PPCF_array
    
target = [0, 0]
print(PPCF(target=np.array(target), players=players))
