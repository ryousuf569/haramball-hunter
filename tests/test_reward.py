# Structural tests for environment.reward's potential-based shaping.
# The telescoping identity is the strong one: it is exact, policy-independent,
# and a sign error, a misplaced gamma or a mishandled terminal all break it.
# Run: python tests/test_reward.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace  # noqa: E402

from environment.lowblock_env import (  # noqa: E402
    compute_attacker_ppcf,
    make_initial_world,
    make_ppcf_grid,
)
from environment.reward import (  # noqa: E402
    FAILURE,
    SUCCESS,
    TIMEOUT,
    RewardConfig,
    agent_potential,
    agent_shaping,
    build_zone_masks,
    make_pcf_state,
    phi,
    reset_agent_potential,
    reset_potential,
    step_reward,
)
from physics.ppcf import PPCF_grid  # noqa: E402

TOL = 1e-12

CFG = RewardConfig(alpha=1.0, beta=1.0, gamma=0.99, terminal_bonus=5.0)

GRID = make_ppcf_grid()
F3_MASK, HS_MASK = build_zone_masks(GRID)


def _world(seed=11):
    players, ball, _, _, _ = make_initial_world(seed=seed)
    return players, ball


def _ppcf(players):
    return PPCF_grid(GRID, players)


def _random_rollout(players, rng, n_steps, jitter=3.0):
    # A stand-in for "several different random policies": each rng gives a
    # different sequence of states from the same s0. The identity must not care
    # which sequence it was.
    states = []
    pos = np.asarray(players["position"], dtype=float).copy()
    for _ in range(n_steps):
        pos = pos + rng.normal(0.0, jitter, size=pos.shape)
        moved = players.copy()
        moved["position"] = pos
        states.append(moved)
    return states


def _episode_return(states, cfg, outcome=SUCCESS, discount_shaping_only=True):
    # sum_t gamma^t F_t over one episode, plus the components for reporting
    players0 = states[0]
    pcf_state = make_pcf_state()
    phi_0, _ = reset_potential(players0, _ppcf(players0), F3_MASK, HS_MASK,
                               pcf_state, cfg)

    total = 0.0
    for t, players in enumerate(states[1:]):
        last = (t == len(states) - 2)
        step_outcome = outcome if last else None
        _, parts = step_reward(players, _ppcf(players), F3_MASK, HS_MASK,
                               pcf_state, cfg, outcome=step_outcome)
        shaping = parts["shaping"] if discount_shaping_only else parts["reward"]
        total += (cfg.gamma ** t) * shaping

    return total, phi_0


def test_telescoping_identity():
    # sum_t gamma^t F_t == -Phi(s0) for every policy, from the same s0
    players0, _ = _world()
    totals = []
    for seed in (0, 1, 2, 3, 4):
        rng = np.random.default_rng(seed)
        n_steps = 3 + int(rng.integers(0, 6))  # policies differ in length too
        states = [players0] + _random_rollout(players0, rng, n_steps)
        total, phi_0 = _episode_return(states, CFG)
        totals.append(total)

        assert abs(total + phi_0) <= TOL, (
            f"seed {seed}: sum gamma^t F_t = {total:.15f} != -Phi(s0) = {-phi_0:.15f}")

    spread = max(totals) - min(totals)
    assert spread <= TOL, f"policy-dependent total (spread {spread:.3e})"
    return totals[0]


def test_telescoping_needs_terminal_zeroing():
    # the guard on the guard: without the zeroing the total is policy-dependent
    cfg = replace(CFG, zero_terminal_potential=False)
    players0, _ = _world()
    totals = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        states = [players0] + _random_rollout(players0, rng, 5)
        total, phi_0 = _episode_return(states, cfg)
        totals.append(total)

    spread = max(totals) - min(totals)
    assert spread > 1e-6, (
        "leaving Phi(terminal) non-zero should make the total policy-dependent; "
        f"got spread {spread:.3e} -- is the flag wired up?")


def test_static_state_living_cost():
    # a state that does not change pays F = (gamma - 1) * Phi(s), i.e. negative
    players0, _ = _world()
    ppcf_result = _ppcf(players0)
    phi_s, _ = phi(players0, ppcf_result, F3_MASK, HS_MASK, CFG)

    pcf_state = make_pcf_state()
    reset_potential(players0, ppcf_result, F3_MASK, HS_MASK, pcf_state, CFG)

    expected = (CFG.gamma - 1.0) * phi_s
    for _ in range(3):
        reward, parts = step_reward(players0, ppcf_result, F3_MASK, HS_MASK,
                                    pcf_state, CFG, outcome=None)
        assert abs(parts["shaping"] - expected) <= TOL, (
            f"static-state F = {parts['shaping']:.15f}, expected {expected:.15f}")
        assert reward < 0.0, "static-state reward should be a living cost"
    return phi_s, expected


def test_terminal_reward_is_signed_per_outcome():
    players0, _ = _world()
    ppcf_result = _ppcf(players0)

    got = {}
    for outcome in (SUCCESS, FAILURE, TIMEOUT):
        pcf_state = make_pcf_state()
        reset_potential(players0, ppcf_result, F3_MASK, HS_MASK, pcf_state, CFG)
        _, parts = step_reward(players0, ppcf_result, F3_MASK, HS_MASK,
                               pcf_state, CFG, outcome=outcome)
        got[outcome] = parts["terminal_reward"]
        assert parts["terminal"], f"{outcome} should be terminal"
        assert parts["phi_next"] == 0.0, f"{outcome}: terminal potential not zeroed"

    assert got[SUCCESS] == CFG.terminal_bonus, got
    assert got[FAILURE] == CFG.turnover_penalty, got
    assert got[TIMEOUT] == CFG.timeout_penalty, got

    # Timeout is the worst ending, because it is the one a policy can always
    # guarantee by recycling possession.
    assert got[TIMEOUT] < got[FAILURE] < got[SUCCESS], got

    # ...but not so much worse that giving the ball away beats playing on.
    # Playing on from success probability p is worth
    # 5p + turnover_penalty*(1-p); it has to dominate both degenerate endings
    # for every p, which pins timeout_penalty <= turnover_penalty and
    # turnover_penalty <= terminal_bonus.
    for p in (0.0, 0.05, 0.1, 0.5, 1.0):
        play_on = CFG.terminal_bonus * p + CFG.turnover_penalty * (1 - p)
        assert play_on >= CFG.turnover_penalty, (p, play_on)
        assert play_on > CFG.timeout_penalty, (
            f"stalling beats attacking at p={p}: {CFG.timeout_penalty} vs {play_on}")


def test_components_are_reported():
    players0, _ = _world()
    ppcf_result = _ppcf(players0)
    pcf_state = make_pcf_state()
    reset_potential(players0, ppcf_result, F3_MASK, HS_MASK, pcf_state, CFG)

    reward, parts = step_reward(players0, ppcf_result, F3_MASK, HS_MASK,
                                pcf_state, CFG)
    for key in ("reward", "shaping", "terminal_reward", "phi_prev", "phi_next",
                "pc_f3", "pc_hs", "terminal", "outcome"):
        assert key in parts, f"missing component {key}"
    assert abs(reward - (parts["shaping"] + parts["terminal_reward"])) <= TOL


def test_phi_scale_is_order_one():
    # the calibration guard: with normalization="mean" both zone values are
    # mean pitch control, so Phi <= alpha + beta and the (gamma-1)*Phi living
    # cost stays small next to terminal_bonus.
    for seed in range(8):
        players, _ = _world(seed=seed)
        phi_s, parts = phi(players, _ppcf(players), F3_MASK, HS_MASK, CFG)
        assert 0.0 <= parts["pc_f3"] <= 1.0, parts
        assert 0.0 <= parts["pc_hs"] <= 1.0, parts
        assert phi_s <= CFG.alpha + CFG.beta + TOL, phi_s


N_ATT = 10


def _pc_own(players, ball_pos):
    # The per-attacker breakdown, which is what the per-agent potential reads
    # now -- an attacker's own control, not the team's sum.
    return compute_attacker_ppcf(players, GRID, ball_pos, per_attacker=True)[1]


def _advanced_world(seed=11):
    # The sampled formation sits around x = 66, so most of it is behind the
    # final-third line and scores an honest zero under the masked potential.
    # Push it forward for the tests that need the term to be live.
    players, ball = _world(seed)
    players = players.copy()
    players["position"][:N_ATT, 0] += 22.0
    return players, ball


def _agent_rollout(rng, n_steps, jitter=3.0):
    players0, ball = _advanced_world()
    states = [players0] + _random_rollout(players0, rng, n_steps, jitter)
    return states, np.asarray(ball["position"], dtype=float)


def test_agent_shaping_telescopes_per_agent():
    # Same guarantee as the team term, made ten times over: the per-attacker
    # shaping is potential-based, so sum gamma^t F_i,t == -phi_i(s0) for each
    # agent independently, whatever the ten of them did in between.
    cfg = replace(CFG, agent_alpha=1.0)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        states, ball_pos = _agent_rollout(rng, 4 + int(rng.integers(0, 4)))

        pcf_state = make_pcf_state()
        phi_0 = reset_agent_potential(pcf_state, _pc_own(states[0], ball_pos),
                                      states[0]["position"][:N_ATT], F3_MASK)

        total = np.zeros(N_ATT)
        for t, players in enumerate(states[1:]):
            last = (t == len(states) - 2)
            f = agent_shaping(pcf_state, _pc_own(players, ball_pos),
                              players["position"][:N_ATT], cfg, F3_MASK,
                              terminal=last)
            total += (cfg.gamma ** t) * f

        assert np.abs(total + phi_0).max() <= TOL, (
            f"seed {seed}: per-agent totals {total} != {-phi_0}")


def test_agent_shaping_is_actually_per_agent():
    # The whole point: the ten numbers must differ. If they did not, this would
    # be the team scalar again under a new name.
    players, ball = _advanced_world()
    ball_pos = np.asarray(ball["position"], dtype=float)
    phi_i = agent_potential(_pc_own(players, ball_pos),
                            players["position"][:N_ATT], F3_MASK)
    assert phi_i.shape == (N_ATT,), phi_i.shape
    assert phi_i.std() > 1e-3, f"per-agent potentials are all but identical: {phi_i}"
    assert ((phi_i >= 0.0) & (phi_i <= 1.0)).all(), phi_i


def test_agent_potential_is_own_control_not_the_teams():
    # The bug this replaced: fed the SUMMED surface, an attacker inherited its
    # neighbours' control, so ten players stacked on one spot each scored ~1.0
    # and nothing rewarded occupying distinct space. Own-control splits.
    players, ball = _advanced_world()
    ball_pos = np.asarray(ball["position"], dtype=float)

    stacked = players.copy()
    spot = np.array([88.0, 34.0])
    stacked["position"][:N_ATT] = spot + np.linspace(-1.0, 1.0, N_ATT)[:, None]
    phi_stacked = agent_potential(_pc_own(stacked, ball_pos),
                                  stacked["position"][:N_ATT], F3_MASK)

    alone = players.copy()
    alone["position"][:N_ATT, 0] = 80.0
    alone["position"][:N_ATT, 1] = np.linspace(4.0, 64.0, N_ATT)
    alone["position"][0] = spot
    phi_alone = agent_potential(_pc_own(alone, ball_pos),
                                alone["position"][:N_ATT], F3_MASK)

    assert phi_stacked[0] < phi_alone[0], (
        f"crowding the same cells did not cost anything: stacked "
        f"{phi_stacked[0]:.3f} vs spread {phi_alone[0]:.3f}")


def test_agent_potential_is_zero_outside_the_final_third():
    # Retreating behind the final-third line must be worth nothing, however
    # much free grass is back there. This is the term that used to pay 0.71
    # behind halfway against 0.29 in the final third.
    players, ball = _world()
    ball_pos = np.asarray(ball["position"], dtype=float)
    deep = players.copy()
    deep["position"][:N_ATT, 0] = 30.0          # behind halfway, and off-grid
    deep["position"][:N_ATT, 1] = np.linspace(4.0, 64.0, N_ATT)
    phi_i = agent_potential(_pc_own(deep, ball_pos),
                            deep["position"][:N_ATT], F3_MASK)
    assert np.array_equal(phi_i, np.zeros(N_ATT)), phi_i

    # ... and the boundary ramps rather than stepping: a disc straddling the
    # line counts only the cells past it.
    edge = deep.copy()
    edge["position"][:N_ATT, 0] = 70.0
    phi_edge = agent_potential(_pc_own(edge, ball_pos),
                               edge["position"][:N_ATT], F3_MASK)
    inside = deep.copy()
    inside["position"][:N_ATT, 0] = 76.0
    phi_inside = agent_potential(_pc_own(inside, ball_pos),
                                 inside["position"][:N_ATT], F3_MASK)
    assert (phi_edge <= phi_inside + TOL).all(), (phi_edge, phi_inside)


def test_the_offside_term_pays_the_run_back():
    from environment.reward import OFFSIDE_W  # noqa: E402

    players, ball = _advanced_world()
    ball_pos = np.asarray(ball["position"], dtype=float)
    line = 80.0

    # Onside, the term is off and the potential is bit-identical.
    onside = players.copy()
    onside["position"][:N_ATT, 0] = 75.0
    pc = _pc_own(onside, ball_pos)
    assert np.array_equal(
        agent_potential(pc, onside["position"][:N_ATT], F3_MASK, line),
        agent_potential(pc, onside["position"][:N_ATT], F3_MASK))

    # Ten metres past it, it is exactly half of OFFSIDE_W (the 20m scale).
    off = onside.copy()
    off["position"][:N_ATT, 0] = 90.0
    pc_off = _pc_own(off, ball_pos)
    gap = (agent_potential(pc_off, off["position"][:N_ATT], F3_MASK)
           - agent_potential(pc_off, off["position"][:N_ATT], F3_MASK, line))
    assert np.allclose(gap, OFFSIDE_W * 0.5), gap

    def _shaped(states, line):
        pcf_state = make_pcf_state()
        reset_agent_potential(pcf_state, _pc_own(states[0], ball_pos),
                              states[0]["position"][:N_ATT], F3_MASK, line)
        return np.array([
            agent_shaping(pcf_state, _pc_own(p, ball_pos),
                          p["position"][:N_ATT], CFG, F3_MASK, line=line)
            for p in states[1:]]).sum(axis=0)

    def _walk(xs):
        out = []
        for x in xs:
            s = off.copy()
            s["position"][:N_ATT, 0] = x
            out.append(s)
        return out

    # Isolate the offside term by differencing against the same rollout without
    # it: the local-control part is identical in both, so what is left is what
    # the new term paid. Running back onside pays; running further off charges.
    backwards = _walk([90.0, 88.0, 86.0, 84.0])
    forwards = _walk([90.0, 92.0, 94.0, 96.0])
    paid_back = _shaped(backwards, line) - _shaped(backwards, None)
    paid_on = _shaped(forwards, line) - _shaped(forwards, None)

    assert (paid_back > 0).all(), paid_back
    assert (paid_on < 0).all(), paid_on


def test_agent_alpha_zero_switches_the_term_off():
    cfg = replace(CFG, agent_alpha=0.0)
    rng = np.random.default_rng(0)
    states, ball_pos = _agent_rollout(rng, 4)

    pcf_state = make_pcf_state()
    reset_agent_potential(pcf_state, _pc_own(states[0], ball_pos),
                          states[0]["position"][:N_ATT], F3_MASK)
    for players in states[1:]:
        f = agent_shaping(pcf_state, _pc_own(players, ball_pos),
                          players["position"][:N_ATT], cfg, F3_MASK)
        assert np.array_equal(f, np.zeros(N_ATT)), f


def main():
    failures = 0
    tests = [
        test_telescoping_identity,
        test_telescoping_needs_terminal_zeroing,
        test_static_state_living_cost,
        test_terminal_reward_is_signed_per_outcome,
        test_components_are_reported,
        test_phi_scale_is_order_one,
        test_agent_shaping_telescopes_per_agent,
        test_agent_shaping_is_actually_per_agent,
        test_agent_potential_is_own_control_not_the_teams,
        test_agent_potential_is_zero_outside_the_final_third,
        test_the_offside_term_pays_the_run_back,
        test_agent_alpha_zero_switches_the_term_off,
    ]
    for fn in tests:
        try:
            out = fn()
            extra = ""
            if fn is test_telescoping_identity:
                extra = f"  sum gamma^t F_t = {out:.9f}"
            elif fn is test_static_state_living_cost:
                extra = f"  Phi = {out[0]:.6f}, (gamma-1)Phi = {out[1]:.6f}"
            print(f"  PASS  {fn.__name__}{extra}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")

    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
