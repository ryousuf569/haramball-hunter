import numpy as np

v_max = 5
a_max = 7
reaction_time = 0.54
intercept_uncertainty = 0.45
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

def derive_t(d_mag, v_parallel):
    t_1 = (v_max - v_parallel) / a_max
    A = (v_parallel * t_1) 
    B = (0.5 * a_max * (t_1 * t_1))
    d_1 = A + B

    if d_1 > d_mag:
        coeff = [0.5 * a_max, v_parallel, -d_mag]
        roots = np.roots(coeff)
        real_positive_roots = roots[(np.isreal(roots)) & (roots > 0)]
        t = real_positive_roots[0] if len(real_positive_roots) > 0 else None
    if d_1 < d_mag:
        t_2 = (d_mag - d_1)/v_max
        t = t_1 + t_2

    return t

def TTI(position, velocity, target_location):

    distance = target_location - position
    D = np.linalg.norm(distance)
    
    if D == 0:
        return 0
    else:
        u = distance/D

    v_u_dot = np.dot(velocity, u)
    u_mag_squared = np.dot(u, u)
    v_parallel = v_u_dot / np.sqrt(u_mag_squared)

    if v_parallel >= 0: # player stationary or moving toward target
        t = derive_t(d_mag=D, v_parallel=v_parallel)
    elif v_parallel < 0: # player moving away from the target
        t_brake = abs(v_parallel) / a_max
        d_brake = (abs(v_parallel) * t_brake) - (0.5 * a_max * (t_brake * t_brake))
        D_prime = D + d_brake
        t = derive_t(d_mag=D_prime, v_parallel=0) + t_brake

    expected_reaction = t + reaction_time
    return t, expected_reaction

test_cases = [
    # (label, position, velocity, target, expected_kinematic_t, expected_tau_exp)
    ("Case 1 - stationary, far",        (0.0, 0.0), (0.0, 0.0), (10.0, 0.0), 2.3571, 2.8971),
    ("Case 2 - stationary, short",      (0.0, 0.0), (0.0, 0.0), (1.5, 0.0),  0.6547, 1.1947),
    ("Case 3 - toward, below v_max",    (0.0, 0.0), (3.0, 0.0), (10.0, 0.0), 2.0571, 2.5971),
    ("Case 4 - toward, at v_max",       (0.0, 0.0), (5.0, 0.0), (10.0, 0.0), 2.0000, 2.5400),
    ("Case 5 - moving away",            (0.0, 0.0), (-2.0, 0.0),(10.0, 0.0), 2.7000, 3.2400),
]

for label, pos, vel, target, expected_t, expected_tau in test_cases:
    position = np.array(pos)
    velocity = np.array(vel)
    target_location = np.array(target)

    result = TTI(position, velocity, target_location)

    print(f"{label}")
    print(f"  got: {result}")
    print(f"  expected kinematic t ≈ {expected_t}, expected tau_exp ≈ {expected_tau}")
    print()