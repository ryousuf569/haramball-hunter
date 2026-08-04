# Integration tests for the reward wired into LowBlockEnv.
#
# tests/test_reward.py checks the shaping maths in isolation, on synthetic
# rollouts. This checks the wiring: that the env seeds Phi(s0) at reset, feeds
# the reward the same pitch-control surface step() built for the shot test, and
# labels terminals so the telescoping identity survives a real episode.
# Run: python tests/test_env_reward.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.lowblock_env import (  # noqa: E402
    FAILURE,
    SUCCESS,
    TIMEOUT,
    LowBlockEnv,
    compute_attacker_ppcf,
    make_vector_env,
)
from environment.reward import (  # noqa: E402
    attacker_control,
    build_zone_masks,
    phi,
    phi_from_pc_att,
)
from physics.ppcf import PPCF_grid  # noqa: E402

TOL = 1e-12


def test_env_telescoping_identity():
    # sum_t gamma^t F_t == -Phi(s0) through the real env, for whatever outcome
    # the episode happens to reach. This is the end-to-end guard: if the env fed
    # the reward a stale or wrongly-ordered surface, or forgot to seed Phi(s0),
    # or mislabelled the terminal, this is what breaks.
    outcomes = {}
    for seed in range(8):
        env = LowBlockEnv(max_ticks=120)
        _obs, info = env.reset(seed=seed)
        phi_0 = info["phi_0"]

        total = 0.0
        t = 0
        while True:
            _obs, _reward, terminated, truncated, step_info = env.step(None)
            total += (env.cfg.gamma ** t) * step_info["shaping"]
            t += 1
            if terminated or truncated:
                break

        outcome = step_info["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        assert abs(total + phi_0) <= TOL, (
            f"seed {seed} ({outcome}, {t} ticks): sum gamma^t F_t = {total:.15f} "
            f"!= -Phi(s0) = {-phi_0:.15f}")
    return outcomes


def test_step_surface_matches_a_fresh_ppcf():
    # The env hands the reward step()'s pc_att instead of recomputing PPCF. That
    # is only sound if flattening the (PC_NX, PC_NY) surface recovers the exact
    # cell order build_zone_masks was built on -- assert it rather than trust it.
    env = LowBlockEnv(max_ticks=50)
    env.reset(seed=3)
    for _ in range(5):
        env.step(None)

    pc_att = compute_attacker_ppcf(env.players, env.ppcf_grid,
                                   env.ball["position"])
    fresh = attacker_control(env.players,
                             PPCF_grid(env.ppcf_grid, env.players,
                                       env.ball["position"]))

    assert np.allclose(pc_att.reshape(-1), fresh, atol=1e-12), (
        "flattened pc_att does not match a fresh PPCF in grid order")

    f3, hs = build_zone_masks(env.ppcf_grid)
    phi_flat, _ = phi_from_pc_att(pc_att, f3, hs, env.cfg)
    phi_full, _ = phi(env.players,
                      PPCF_grid(env.ppcf_grid, env.players,
                                env.ball["position"]),
                      f3, hs, env.cfg)
    assert abs(phi_flat - phi_full) <= TOL, (
        f"Phi from the reused surface {phi_flat} != Phi from a fresh PPCF {phi_full}")


def test_terminal_labelling():
    # All three outcomes terminate -- timeout included, because reward.py zeroes
    # Phi(terminal) for it. Reporting timeout as truncated would let a learner
    # bootstrap V(s_T) on top and break the telescoping identity, so truncated
    # must stay False for every ending.
    env = LowBlockEnv(max_ticks=60)
    for seed in range(6):
        env.reset(seed=seed)
        while True:
            _obs, _r, terminated, truncated, info = env.step(None)
            if terminated or truncated:
                break
        outcome = info["outcome"]
        assert outcome in (SUCCESS, FAILURE, TIMEOUT), outcome
        assert terminated, f"{outcome} should terminate"
        assert not truncated, f"{outcome} reported truncated; nothing should be"
        assert info["phi_next"] == 0.0, f"{outcome}: terminal potential not zeroed"


def test_reset_is_seed_reproducible():
    a = LowBlockEnv(max_ticks=30)
    b = LowBlockEnv(max_ticks=30)
    obs_a, info_a = a.reset(seed=7)
    obs_b, info_b = b.reset(seed=7)
    assert np.array_equal(obs_a, obs_b), "same seed gave different initial obs"
    assert info_a["phi_0"] == info_b["phi_0"]

    for _ in range(10):
        oa, ra, _ta, _ua, _ia = a.step(None)
        ob, rb, _tb, _ub, _ib = b.step(None)
        assert np.array_equal(oa, ob) and ra == rb, "rollouts diverged"

    obs_c, _ = LowBlockEnv(max_ticks=30).reset(seed=8)
    assert not np.array_equal(obs_a, obs_c), "different seeds gave the same state"


def test_obs_within_declared_space():
    env = LowBlockEnv(max_ticks=40)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs), "reset obs outside declared space"
    for _ in range(40):
        obs, _r, term, trunc, _i = env.step(None)
        assert env.observation_space.contains(obs), "step obs outside declared space"
        if term or trunc:
            break


def _hold_action(env, velocity):
    vel = np.broadcast_to(np.asarray(velocity, dtype=np.float32),
                          (env.n_att, 2)).copy()
    return {"velocity": vel, "pass": 0}


def test_learned_attackers_bypass_the_baseline():
    # The whole point of scripted_attackers=False: baseline_attacker.py is a
    # defender test harness and must not be in the loop once the policy drives.
    # Detonate it and assert the env never reaches it.
    import environment.lowblock_env as env_mod

    def _boom(*_a, **_k):
        raise AssertionError("compute_attacker_targets was called with a policy "
                             "driving; the baseline is still wired in")

    env = LowBlockEnv(max_ticks=30, scripted_attackers=False)
    original = env_mod.compute_attacker_targets
    env_mod.compute_attacker_targets = _boom
    try:
        env.reset(seed=2)
        for _ in range(20):
            _obs, _r, term, _trunc, _i = env.step(_hold_action(env, [5.0, 0.0]))
            if term:
                break
    finally:
        env_mod.compute_attacker_targets = original


def test_commanded_velocity_reaches_the_integrator():
    # Attackers move straight off the physics: hold a constant target and the
    # integrator should ramp them onto it at A_MAX and hold there.
    env = LowBlockEnv(max_ticks=200, scripted_attackers=False)
    env.reset(seed=5)
    for _ in range(30):
        env.step(_hold_action(env, [5.0, 0.0]))

    att_vel = np.asarray(env.players["velocity"][:env.n_att], dtype=float)
    assert np.allclose(att_vel, [5.0, 0.0], atol=1e-3), (
        f"attacker velocities {att_vel} did not converge on the commanded target")


def test_pass_action_decodes_to_the_holder_slot():
    env = LowBlockEnv(max_ticks=50, scripted_attackers=False)
    env.reset(seed=1)

    holder_id = env.ball["holder_id"]
    holder_row = int(np.flatnonzero(env.attacker_ids == holder_id)[0])
    _vel, ball_idx = env._decode_action(_hold_action(env, [0.0, 0.0]) | {"pass": 3})

    assert ball_idx[holder_row] == 3, ball_idx
    assert np.all(np.delete(ball_idx, holder_row) == 0), (
        f"non-holder slots not HOLD: {ball_idx}")


def test_hold_action_never_releases_the_ball():
    # With the baseline gone nothing fires a pass on a cadence, so a policy that
    # always holds keeps the ball at its feet until a defender takes it.
    env = LowBlockEnv(max_ticks=120, scripted_attackers=False)
    env.reset(seed=4)
    holder_id = env.ball["holder_id"]

    for _ in range(60):
        _obs, _r, term, _trunc, info = env.step(_hold_action(env, [0.0, 0.0]))
        if term:
            assert info["outcome"] in (SUCCESS, FAILURE), info["outcome"]
            return
        assert env.ball["state"] == "held", (
            "ball left the holder's feet with pass=0 -- baseline cadence still firing?")
        assert env.ball["holder_id"] == holder_id, "holder changed without a pass"


def test_vector_env_matches_single():
    # SyncVectorEnv seeds env i with seed+i, so column i must reproduce a single
    # env run at that seed. Guards the reward being per-env state, not shared.
    n_envs = 3
    venv = make_vector_env(n_envs=n_envs, max_ticks=40)
    try:
        _obs, _info = venv.reset(seed=100)
        action = venv.action_space.sample()  # scripted attackers ignore it
        vec_rewards = []
        for _ in range(10):
            _obs, r, _term, _trunc, _info = venv.step(action)
            vec_rewards.append(r.copy())
    finally:
        venv.close()
    vec_rewards = np.array(vec_rewards)  # (10, n_envs)

    for i in range(n_envs):
        env = LowBlockEnv(max_ticks=40)
        env.reset(seed=100 + i)
        for t in range(10):
            _obs, r, _term, _trunc, _i = env.step(None)
            assert abs(r - vec_rewards[t, i]) <= TOL, (
                f"env {i} tick {t}: vector {vec_rewards[t, i]} != single {r}")


def main():
    failures = 0
    tests = [
        test_env_telescoping_identity,
        test_step_surface_matches_a_fresh_ppcf,
        test_terminal_labelling,
        test_reset_is_seed_reproducible,
        test_obs_within_declared_space,
        test_learned_attackers_bypass_the_baseline,
        test_commanded_velocity_reaches_the_integrator,
        test_pass_action_decodes_to_the_holder_slot,
        test_hold_action_never_releases_the_ball,
        test_vector_env_matches_single,
    ]
    for fn in tests:
        try:
            out = fn()
            extra = f"  outcomes={out}" if fn is test_env_telescoping_identity else ""
            print(f"  PASS  {fn.__name__}{extra}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")

    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
