import numpy as np
import math as m

v_max = 5
a_max = 7
reaction_time = 0.54
intercept_uncertainty = 0.45

# Vectorized functions

def derive_t_vec(d_mag, v_parallel):
    t_1 = (v_max - v_parallel) / a_max
    A = (v_parallel * t_1)
    B = (0.5 * a_max * (t_1 * t_1))
    d_1 = A + B

    # The scalar code used np.roots(...) then picked the single real positive
    # root. For this quadratic with a_max > 0 and d_mag > 0 the discriminant
    # v_parallel^2 + 2*a_max*d_mag is strictly greater than v_parallel^2, so
    # exactly one root is positive -- the '+' root below. That is precisely the
    # root np.roots' filter selected, so the closed form is identical (and it
    # vectorizes, unlike np.roots which is scalar-only).
    disc = v_parallel * v_parallel + 2.0 * a_max * d_mag
    t_quad = (-v_parallel + np.sqrt(disc)) / a_max
    t_2 = (d_mag - d_1) / v_max
    t_lin = t_1 + t_2

    # Mask selects the branch elementwise. The scalar code used two separate
    # `if` statements (`> d_mag` and `< d_mag`) and left t undefined on the
    # exact `d_1 == d_mag` tie (a latent bug that would raise there). Selecting
    # on `d_1 > d_mag` routes ties to the linear branch, which cannot be hit by
    # the toy scenario; it only makes the vectorized code not crash on a tie
    # the scalar code never handled either.
    return np.where(d_1 > d_mag, t_quad, t_lin)


def TTI_vec(positions, velocities, targets):

    # distance[c, p, :] = target_c - position_p -> (n_cells, n_players, 2)
    distance = targets[:, None, :] - positions[None, :, :]
    D = np.linalg.norm(distance, axis=-1) # (n_cells, n_players)

    zero = (D == 0)
    D_safe = np.where(zero, 1.0, D)
    u = distance / D_safe[..., None] # (n_cells, n_players, 2)

    v_u_dot = np.sum(velocities[None, :, :] * u, axis=-1)   # (n_cells, n_players)
    u_mag_squared = np.sum(u * u, axis=-1)
    # u is the zero vector wherever D == 0, so we're guarding this divide the same way as
    # the one above
    u_mag = np.sqrt(u_mag_squared)
    v_parallel = np.divide(v_u_dot, u_mag, out=np.zeros_like(v_u_dot), where=u_mag > 0)

    # Case A (scalar: `if v_parallel >= 0`) stationary or moving toward target
    t_caseA = derive_t_vec(D, v_parallel)

    # Case B (scalar: `elif v_parallel < 0`) moving away
    t_brake = np.abs(v_parallel) / a_max
    d_brake = (np.abs(v_parallel) * t_brake) - (0.5 * a_max * (t_brake * t_brake))
    D_prime = D + d_brake
    t_caseB = derive_t_vec(D_prime, np.zeros_like(v_parallel)) + t_brake

    # mask below picks Case A vs Case B
    t = np.where(v_parallel >= 0, t_caseA, t_caseB)

    expected_reaction = t + reaction_time
    # D == 0 short-circuits to 0 (before adding reaction_time), as in scalar TTI.
    return np.where(zero, 0.0, expected_reaction)


def intercept_probability_vec(T, react_exp):
    a = -(m.pi / m.sqrt(3))
    b = (T - react_exp) / intercept_uncertainty
    denom = 1 + np.exp(a * b)

    return 1 / denom

'''test_cases = [
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
    print()'''