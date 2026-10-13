# PIDMultipliers, pure torch, no environment, so it sits in the push-triggered
# suite next to the physics tests.

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ppo_ablation import Constraint  # noqa: E402
from model.ppo_pid import PIDMultipliers  # noqa: E402

Z_INIT = 0.02

# the live spec: slow <= 0.75, crowd <= 1.0, success >= 0.4
CONSTRAINTS = [Constraint("slow", 0.75), Constraint("crowd_disc", 1.0),
               Constraint("success", 0.40, lower=True)]
SLOW, SUCC = 0, 2


def make(**gains):
    g = dict(kp=0.4, ki=0.12, kd=1.0, i_max=50.0, i_floor=25.0, d_ema=0.9)
    g.update(gains)
    return PIDMultipliers(CONSTRAINTS, Z_INIT, **g)


def test_simplex_closes():
    # at init and at arbitrary z
    mult = make()
    lam0, lam = mult()
    assert torch.isclose(lam0 + lam.sum(), torch.tensor(1.0), atol=1e-6)

    with torch.no_grad():
        mult.z.copy_(torch.tensor([7.5, -4.0, 2.25]))
    lam0, lam = mult()
    assert torch.isclose(lam0 + lam.sum(), torch.tensor(1.0), atol=1e-6)
    assert (lam >= 0).all() and lam0 >= 0


def test_z_init_is_the_zero_gain_fixed_point():
    # every departure from z_init elsewhere is then the controller, not
    # bookkeeping
    mult = make(kp=0.0, ki=0.0, kd=0.0)
    for rate in (1.0, 0.9, 0.2, 0.0):
        mult.update(torch.tensor([rate, 0.0, 0.0]))
        assert torch.allclose(mult.z, torch.full((3,), Z_INIT), atol=1e-6)


def test_constant_violation_raises_z_monotonically():
    # slow stuck at 1.0 against a 0.75 bound is a persistent +0.25 violation
    mult = make()
    J = torch.tensor([1.0, 0.0, 0.9]) # slow violated, crowd and success fine
    zs, lams = [], []
    for _ in range(30):
        mult.update(J)
        zs.append(float(mult.z[SLOW]))
        lams.append(float(mult()[1][SLOW]))

    assert all(b > a for a, b in zip(zs, zs[1:])), zs
    assert all(b > a for a, b in zip(lams, lams[1:])), lams
    # the integral is the only term moving here: e is constant and the EMA of a
    # constant signal has zero slope after the first update
    assert abs((zs[2] - zs[1]) - 0.12 * 0.25) < 1e-6
    # ...and that rate is the point of the gain choice: comparable to the ~0.03
    # per update Adam at mult_lr 0.03 puts into the same logit
    assert 0.02 < zs[2] - zs[1] < 0.05


def test_satisfied_constraint_lowers_z_toward_the_floor():
    # the anti-windup clamp must not turn into a floor on lam
    mult = make()
    J = torch.tensor([0.30, 0.0, 0.9]) # slow well inside its 0.75 bound
    zs = []
    for _ in range(200):
        mult.update(J)
        zs.append(float(mult.z[SLOW]))

    # strictly falling until the integral saturates, which at e = -0.45 takes
    # 25 / 0.45 = 56 updates; flat after that, not falling further
    assert all(b < a for a, b in zip(zs[:50], zs[1:50])), zs[:5]
    # floor = z_init + kp*e - ki*i_floor with e = 0.30 - 0.75
    expected = Z_INIT + 0.4 * (0.30 - 0.75) - 0.12 * 25.0
    assert abs(zs[-1] - expected) < 1e-5
    assert float(mult.I[SLOW]) == -25.0
    # the whole point: lam_slow gets out of the way. The scale-free statement is
    # the ratio to lam0, which is exactly exp(z_slow) on the simplex, so the
    # floor buys lam_slow / lam0 = exp(-3.16) and a larger i_floor buys more.
    lam0, lam = mult()
    assert float(lam[SLOW] / lam0) < 0.05
    assert float(lam[SLOW]) < 0.05


def test_i_max_clamp_holds():
    # the integral saturates and stops contributing, so z stops climbing
    mult = make(i_max=5.0)
    J = torch.tensor([1.0, 0.0, 0.9])
    for _ in range(500):
        mult.update(J)
    assert float(mult.I[SLOW]) == 5.0

    z_at_clamp = float(mult.z[SLOW])
    mult.update(J)
    assert abs(float(mult.z[SLOW]) - z_at_clamp) < 1e-6
    expected = Z_INIT + 0.4 * 0.25 + 0.12 * 5.0
    assert abs(z_at_clamp - expected) < 1e-5

    # and symmetrically on the lower clamp
    mult = make(i_floor=3.0)
    for _ in range(500):
        mult.update(torch.tensor([0.0, 0.0, 0.9]))
    assert float(mult.I[SLOW]) == -3.0


def test_derivative_is_rectified_and_only_damps():
    # D = max(0, rise of the EMA), so a falling rate contributes nothing
    mult = make(kp=0.0, ki=0.0, kd=10.0)
    mult.update(torch.tensor([0.2, 0.0, 0.9])) # primes the EMA
    assert abs(float(mult.z[SLOW]) - Z_INIT) < 1e-6

    mult.update(torch.tensor([0.9, 0.0, 0.9])) # rate jumps -> D > 0
    assert float(mult.z[SLOW]) > Z_INIT

    mult.update(torch.tensor([0.1, 0.0, 0.9])) # rate collapses -> D clamped to 0
    assert abs(float(mult.z[SLOW]) - Z_INIT) < 1e-6


def test_lower_bound_channel_reads_the_right_sign():
    # success below its floor is the violation, so the bootstrap drives the
    # other way round
    mult = make(kd=0.0)
    for _ in range(10):
        mult.update(torch.tensor([0.5, 0.0, 0.10])) # success 0.10 < 0.40 floor
    assert float(mult.z[SUCC]) > Z_INIT

    mult = make(kd=0.0)
    for _ in range(10):
        mult.update(torch.tensor([0.5, 0.0, 0.80])) # comfortably above it
    assert float(mult.z[SUCC]) < Z_INIT


def test_d_on_violation_flips_only_the_lower_bound_channel():
    # identical on the upper-bound costs, sign-flipped on the bootstrap
    rising = [torch.tensor([0.2, 0.3, 0.2]), torch.tensor([0.8, 0.9, 0.8])]
    a, b = make(kp=0.0, ki=0.0, kd=5.0), make(kp=0.0, ki=0.0, kd=5.0, d_on="violation")
    for J in rising:
        a.update(J)
        b.update(J)
    assert torch.allclose(a.z[:2], b.z[:2], atol=1e-6)
    # rising success: D on j raises lam_succ, D on the violation does not
    assert float(a.z[SUCC]) > Z_INIT
    assert abs(float(b.z[SUCC]) - Z_INIT) < 1e-6


def test_interface_matches_multipliers():
    # the rest of the algorithm only touches forward(), violation(), d and lower
    from model.ppo_ablation import Multipliers

    pid, adam = make(), Multipliers(CONSTRAINTS, Z_INIT)
    assert torch.equal(pid.d, adam.d)
    assert torch.equal(pid.lower, adam.lower)

    J = torch.tensor([0.9, 0.5, 0.2])
    assert torch.allclose(pid.violation(J), adam.violation(J))

    l0_p, lam_p = pid()
    l0_a, lam_a = adam()
    assert torch.allclose(l0_p, l0_a) and torch.allclose(lam_p, lam_a)
    # z is not an autograd leaf under PID; the controller writes it
    assert not pid.z.requires_grad
