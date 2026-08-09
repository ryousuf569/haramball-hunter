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
    ACTION_HEADS,
    FAILURE,
    N_DIRECTIONS,
    N_SPEEDS,
    SUCCESS,
    TIMEOUT,
    LowBlockEnv,
    compute_attacker_ppcf,
    make_vector_env,
    norm_pos,
    norm_vel,
    obs_clock_idx,
    obs_dim,
)
from environment.reward import (  # noqa: E402
    attacker_control,
    build_zone_masks,
    phi,
    phi_from_pc_att,
)
from physics.ppcf import PPCF_grid  # noqa: E402

TOL = 1e-12

FULL_SPEED = N_SPEEDS - 1        # speed_lookup[-1] == 1.0 * V_MAX
EAST = 0                          # direction_lookup[0] == [1, 0]
STAND = N_DIRECTIONS - 1          # direction_lookup[-1] == [0, 0]


def _action(env, direction, speed, ball=0):
    """(n_att, 3) with every attacker commanded the same thing."""
    return np.tile([direction, speed, ball], (env.n_att, 1)).astype(np.int64)


def _hold_action(env, direction=STAND, speed=0):
    return _action(env, direction, speed, ball=0)


def _probe_policy(env, seed, pass_every=8, east_bias=0.7):
    """A stand-in driver for the tests. Every attacker is learned now
    (attackers/baseline_attacker.py is a defender test harness and is not in the
    loop), so the invariants below are exercised the way training will exercise
    them: through decode_action, with the ball head varying tick to tick.

    Biased east -- toward the goal the block defends -- and passing on a cadence
    rather than every tick, because a uniform policy just mills around midfield
    and times out every episode. This one reaches all three outcomes, which is
    what makes the telescoping identity a test of the success bonus and the
    turnover path and not only of the timeout branch.
    """
    rng = np.random.default_rng(seed)
    state = {"t": 0}

    def act(mask):
        a = np.empty((env.n_att, ACTION_HEADS), dtype=np.int64)
        east = rng.random(env.n_att) < east_bias
        a[:, 0] = np.where(east, EAST, rng.integers(0, N_DIRECTIONS, env.n_att))
        a[:, 1] = FULL_SPEED
        a[:, 2] = 0
        # sample the ball head only from the entries the mask calls legal, and
        # only on a pass tick -- index 0 (HOLD) is legal everywhere, always.
        if state["t"] % pass_every == pass_every - 1:
            for i in range(env.n_att):
                legal = np.flatnonzero(mask[i])
                if legal.size > 1:
                    a[i, 2] = legal[rng.integers(1, legal.size)]
        state["t"] += 1
        return a

    return act


def test_env_telescoping_identity():
    # sum_t gamma^t F_t == -Phi(s0) through the real env, for whatever outcome
    # the episode happens to reach. This is the end-to-end guard: if the env fed
    # the reward a stale or wrongly-ordered surface, or forgot to seed Phi(s0),
    # or mislabelled the terminal, this is what breaks.
    #
    # The seeds are picked, not a range: at max_ticks=120 the probe policy ends
    # nearly every episode in a turnover, and the identity has a different shape
    # on the other two endings (success pays terminal_bonus on top of the
    # shaping; timeout zeroes Phi with no bonus). 5, 21 and 25 reach a shot
    # opening, 3, 29 and 34 run the clock out. The assert at the bottom is what
    # stops this from silently degrading back into seven turnovers.
    #
    # The probe passes on a slower cadence and sprints less than it used to.
    # Passes are contestable at every length now and are collected by whoever is
    # nearest where they land, so an east-biased sprint that releases every 8
    # ticks aims every ball at a receiver who has already run 13m off it: under
    # the old settings all seven seeds ended in a turnover and only the failure
    # branch was being tested.
    outcomes = {}
    for seed in (0, 1, 2, 5, 21, 25, 3, 29, 34):
        env = LowBlockEnv(max_ticks=120, scripted_attackers=False)
        _obs, info = env.reset(seed=seed)
        phi_0 = info["phi_0"]
        act = _probe_policy(env, seed, pass_every=40, east_bias=0.2)
        mask = info["action_mask"]

        total = 0.0
        t = 0
        while True:
            _obs, _reward, terminated, truncated, step_info = env.step(act(mask))
            mask = step_info["action_mask"]
            total += (env.cfg.gamma ** t) * step_info["shaping"]
            t += 1
            if terminated or truncated:
                break

        outcome = step_info["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        assert abs(total + phi_0) <= TOL, (
            f"seed {seed} ({outcome}, {t} ticks): sum gamma^t F_t = {total:.15f} "
            f"!= -Phi(s0) = {-phi_0:.15f}")

    assert set(outcomes) == {SUCCESS, FAILURE, TIMEOUT}, (
        f"identity only exercised on {sorted(outcomes)}; the terminal branches "
        "are no longer all covered")
    return outcomes


def test_step_surface_matches_a_fresh_ppcf():
    # The env hands the reward step()'s pc_att instead of recomputing PPCF. That
    # is only sound if flattening the (PC_NX, PC_NY) surface recovers the exact
    # cell order build_zone_masks was built on -- assert it rather than trust it.
    env = LowBlockEnv(max_ticks=50, scripted_attackers=False)
    _obs, info = env.reset(seed=3)
    act = _probe_policy(env, 3)
    for _ in range(5):
        _obs, _r, _t, _u, info = env.step(act(info["action_mask"]))

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
    env = LowBlockEnv(max_ticks=60, scripted_attackers=False)
    for seed in range(6):
        _obs, info = env.reset(seed=seed)
        act = _probe_policy(env, seed)
        while True:
            _obs, _r, terminated, truncated, info = env.step(
                act(info["action_mask"]))
            if terminated or truncated:
                break
        outcome = info["outcome"]
        assert outcome in (SUCCESS, FAILURE, TIMEOUT), outcome
        assert terminated, f"{outcome} should terminate"
        assert not truncated, f"{outcome} reported truncated; nothing should be"
        assert info["phi_next"] == 0.0, f"{outcome}: terminal potential not zeroed"


def test_reset_is_seed_reproducible():
    a = LowBlockEnv(max_ticks=30, scripted_attackers=False)
    b = LowBlockEnv(max_ticks=30, scripted_attackers=False)
    obs_a, info_a = a.reset(seed=7)
    obs_b, info_b = b.reset(seed=7)
    assert np.array_equal(obs_a, obs_b), "same seed gave different initial obs"
    assert info_a["phi_0"] == info_b["phi_0"]

    for _ in range(10):
        act = _hold_action(a, EAST, FULL_SPEED)
        oa, ra, _ta, _ua, _ia = a.step(act)
        ob, rb, _tb, _ub, _ib = b.step(act)
        assert np.array_equal(oa, ob) and ra == rb, "rollouts diverged"

    obs_c, _ = LowBlockEnv(max_ticks=30,
                           scripted_attackers=False).reset(seed=8)
    assert not np.array_equal(obs_a, obs_c), "different seeds gave the same state"


def test_same_seed_gives_an_identical_trajectory():
    # obs, state and mask all compared tick by tick. The state vector is the one
    # that catches a slot-ordering bug: if a slot were assigned by anything
    # episode-dependent, two runs of the same seed would still agree here, but
    # any run where the ordering key moves would scramble the critic's input --
    # so also assert the holder one-hot tracks the holder row it claims to.
    a = LowBlockEnv(max_ticks=80, scripted_attackers=False)
    b = LowBlockEnv(max_ticks=80, scripted_attackers=False)
    _oa, ia = a.reset(seed=21)
    _ob, ib = b.reset(seed=21)
    assert np.array_equal(ia["state"], ib["state"])
    assert np.array_equal(ia["action_mask"], ib["action_mask"])

    for t in range(60):
        act = _action(a, t % N_DIRECTIONS, t % N_SPEEDS, ball=t % a.ball_actions)
        oa, ra, ta, _ua, sa = a.step(act)
        ob, rb, tb, _ub, sb = b.step(act)
        assert np.array_equal(oa, ob), f"tick {t}: obs diverged"
        assert np.array_equal(sa["state"], sb["state"]), f"tick {t}: state diverged"
        assert np.array_equal(sa["action_mask"], sb["action_mask"]), (
            f"tick {t}: action_mask diverged")
        assert ra == rb and ta == tb

        row = a.holder_row()
        one_hot = sa["state"][4 * a.n_players + 2:4 * a.n_players + 2 + a.n_att]
        if row is None:
            assert one_hot.sum() == 0.0, (
                f"tick {t}: holder one-hot set with no attacker on the ball")
        else:
            assert one_hot.argmax() == row and one_hot.sum() == 1.0, (
                f"tick {t}: holder one-hot {one_hot.argmax()} != holder row {row}")
        if ta:
            break


def test_obs_state_and_mask_shapes():
    env = LowBlockEnv(max_ticks=50, scripted_attackers=False)
    obs, info = env.reset(seed=0)

    assert obs.shape == (env.n_att, obs_dim(env.n_players, env.n_att)), obs.shape
    assert obs.dtype == np.float32
    assert info["state"].shape == (99,), info["state"].shape
    assert info["state"].dtype == np.float32
    assert info["action_mask"].shape == (env.n_att, env.ball_actions)
    assert info["action_mask"].dtype == np.bool_

    # Never all-masked: a row of all-False sends a masked softmax to NaN.
    assert info["action_mask"].sum(axis=1).min() >= 1
    # The holder has real choices; everyone else has exactly HOLD.
    holder = env.holder_row()
    assert holder is not None, "ball should start at an attacker's feet"
    assert info["action_mask"][holder].sum() > 1
    assert np.all(np.delete(info["action_mask"], holder, axis=0).sum(axis=1) == 1)

    for t in range(40):
        obs, _r, term, _trunc, info = env.step(_hold_action(env))
        assert obs.shape == (env.n_att, obs_dim(env.n_players, env.n_att))
        assert info["state"].shape == (99,)
        assert info["action_mask"].shape == (env.n_att, env.ball_actions)
        assert info["action_mask"].sum(axis=1).min() >= 1, (
            f"tick {t}: a row was fully masked")
        if term:
            break


def test_remaining_time_is_in_obs_and_state():
    # Counts 1 -> 0 over the horizon in both encodings. Without it the same
    # board at tick 10 and tick 299 is one input. The obs column is looked up
    # rather than assumed to be last: the pitch-control tail sits after it.
    max_ticks = 25
    env = LowBlockEnv(max_ticks=max_ticks, scripted_attackers=False)
    clock = obs_clock_idx(env.n_players)
    obs, info = env.reset(seed=13)
    assert np.allclose(obs[:, clock], 1.0), obs[:, clock]
    assert info["state"][-1] == np.float32(1.0)

    for t in range(1, max_ticks + 1):
        obs, _r, term, _trunc, info = env.step(_hold_action(env))
        expected = np.float32(1.0 - t / max_ticks)
        assert np.allclose(obs[:, clock], expected), (
            f"tick {t}: obs clock {obs[0, clock]} != {expected}")
        assert np.allclose(info["state"][-1], expected)
        if term:
            break
    else:
        raise AssertionError("episode should have hit the horizon")


def test_actor_and_critic_share_one_normalization():
    # The failure this guards against does not look like a bug -- it looks like
    # "the value function won't converge". So compare the two vectors' shared
    # blocks entry for entry rather than trusting that both call the helpers.
    env = LowBlockEnv(max_ticks=60, scripted_attackers=False)
    env.reset(seed=6)
    for _ in range(15):
        _obs, _r, term, _trunc, _i = env.step(_action(env, EAST, FULL_SPEED))
        if term:
            break

    obs = env.obs()
    state = env.global_state()
    n4 = 4 * env.n_players

    # roster block: obs row carries it at [4 : 4+n4], state at [0 : n4]
    assert np.array_equal(obs[0, 4:4 + n4], state[:n4]), (
        "roster block differs between obs and state")
    # ball position
    assert np.array_equal(obs[0, 4 + n4:4 + n4 + 2], state[n4:n4 + 2]), (
        "ball position differs between obs and state")
    # and both are the raw world put through the shared helpers
    assert np.array_equal(
        state[:n4],
        np.concatenate([norm_pos(env.players["position"]).reshape(-1),
                        norm_vel(env.players["velocity"]).reshape(-1)]))

    # ego prefix is attacker i's own slot out of the roster block
    for i in range(env.n_att):
        assert np.array_equal(obs[i, :2], state[2 * i:2 * i + 2]), (
            f"attacker {i}: ego position != its roster slot")
        assert np.array_equal(
            obs[i, 2:4],
            state[2 * env.n_players + 2 * i:2 * env.n_players + 2 * i + 2]), (
            f"attacker {i}: ego velocity != its roster slot")


def test_offside_targets_are_masked_out():
    # 43% of all passing losses were offside turnovers -- a flat tax the policy
    # could not avoid, because the line is the second-largest defender x and
    # nothing in the observation said so. The holder should not be offered them.
    from defenders.turnover import check_offside  # noqa: E402

    env = LowBlockEnv(max_ticks=200, scripted_attackers=False)
    _obs, info = env.reset(seed=11)

    seen_illegal = 0
    for _ in range(120):
        row = env.holder_row()
        if row is not None:
            mask = env.action_mask()
            assert mask[row, 0], "HOLD must stay legal or the softmax goes NaN"
            holder_id = int(env.attacker_ids[row])
            mates = np.sort(env.attacker_ids[env.attacker_ids != holder_id])
            for k, tid in enumerate(mates):
                off = check_offside(env.players, holder_id, int(tid))
                assert mask[row, k + 1] == (not off), (
                    f"target {tid}: mask says {mask[row, k+1]}, offside={off}")
                seen_illegal += int(off)
        _obs, _r, term, _trunc, info = env.step(_action(env, EAST, FULL_SPEED))
        if term:
            break

    assert seen_illegal > 0, (
        "no offside target came up in 120 ticks; this test proved nothing")


def test_a_masked_pass_is_never_offered_even_with_everyone_beyond_the_line():
    # If every teammate is offside the row collapses to HOLD alone. That must
    # still be one legal action, not zero.
    env = LowBlockEnv(max_ticks=60, scripted_attackers=False)
    env.reset(seed=2)
    row = env.holder_row()
    env.players["position"][:env.n_att, 0] = 104.0
    env.players["position"][row, 0] = 60.0
    env.players["position"][env.n_att:, 0] = 70.0

    mask = env.action_mask()
    assert mask.sum(axis=1).min() >= 1
    assert mask[row].sum() == 1 and mask[row, 0], mask[row]


def test_offside_line_and_margin_are_in_the_obs():
    from defenders.turnover import offside_line  # noqa: E402

    env = LowBlockEnv(max_ticks=60, scripted_attackers=False)
    obs, _info = env.reset(seed=4)
    # the two columns sit just before the agent one-hot
    line_col = obs[:, -env.n_att - 2]
    margin_col = obs[:, -env.n_att - 1]

    line = offside_line(env.players)
    assert np.allclose(line_col, line / 105.0, atol=1e-6), (line_col[0], line)
    assert np.allclose(line_col, line_col[0]), "the line is shared by every row"

    own_x = np.asarray(env.players["position"][:env.n_att, 0], dtype=float)
    expected = np.clip((line - own_x) / 20.0, -1.0, 1.0)
    assert np.allclose(margin_col, expected, atol=1e-6)
    # and it actually separates the rows, or it is telling the policy nothing
    assert margin_col.std() > 0.0

    # an attacker beyond the line reads negative, one behind it reads positive
    ahead = own_x > line
    if ahead.any():
        assert (margin_col[ahead] < 0).all()
    assert (margin_col[~ahead] >= 0).all()


def test_agent_one_hot_distinguishes_the_rows():
    # Without it the shared weights are a function of where you are standing, so
    # two attackers at the same point are guaranteed the same action
    # distribution and no assignment of ten players to ten roles exists.
    env = LowBlockEnv(max_ticks=40, scripted_attackers=False)
    obs, info = env.reset(seed=11)
    tail = obs[:, -env.n_att:]
    assert np.array_equal(tail, np.eye(env.n_att, dtype=np.float32)), tail

    # And it must survive a step, and stack the same way for every attacker.
    for _ in range(10):
        obs, _r, term, _trunc, info = env.step(_hold_action(env))
        assert np.array_equal(obs[:, -env.n_att:],
                              np.eye(env.n_att, dtype=np.float32))
        if term:
            break

    # Two attackers standing in the same place still get distinct rows.
    env.players["position"][1] = env.players["position"][0]
    env.players["velocity"][1] = env.players["velocity"][0]
    rows = env.obs()
    assert not np.array_equal(rows[0], rows[1]), (
        "co-located attackers produce identical observations")


def test_agent_reward_is_per_attacker():
    env = LowBlockEnv(max_ticks=40, scripted_attackers=False)
    _obs, info = env.reset(seed=2)
    assert info["agent_reward"].shape == (env.n_att,), info["agent_reward"].shape
    assert np.array_equal(info["agent_reward"], np.zeros(env.n_att)), (
        "reset should seed the per-agent potential, not pay on it")

    for _ in range(15):
        _obs, _r, term, _trunc, info = env.step(_action(env, EAST, FULL_SPEED))
        ar = info["agent_reward"]
        assert ar.shape == (env.n_att,) and np.isfinite(ar).all(), ar
        if term:
            break
    assert ar.std() > 0.0, "every attacker was paid the same; is this a team scalar?"


def test_vector_env_shapes_hold_up():
    # SyncVectorEnv validates against the declared spaces, and batches the obs to
    # (n_envs, n_att, 92) -- the wrappers are single-agent and treat the agent
    # axis as part of the observation shape. That is what we want; it just means
    # the leading axis of everything out of here is n_envs.
    n_envs, n_steps = 6, 100
    OBS_W = obs_dim(21, 10)
    venv = make_vector_env(n_envs=n_envs, max_ticks=40,
                           scripted_attackers=False)
    try:
        obs, info = venv.reset(seed=0)
        assert obs.shape == (n_envs, 10, OBS_W), obs.shape
        assert np.asarray(info["state"]).shape == (n_envs, 99)
        assert np.asarray(info["action_mask"]).shape == (n_envs, 10, 10)

        for t in range(n_steps):
            action = venv.action_space.sample()
            assert action.shape == (n_envs, 10, ACTION_HEADS), action.shape
            obs, r, term, trunc, info = venv.step(action)
            assert obs.shape == (n_envs, 10, OBS_W), f"step {t}: {obs.shape}"
            assert r.shape == (n_envs,)
            assert term.shape == (n_envs,) and trunc.shape == (n_envs,)
            assert np.asarray(info["state"]).shape == (n_envs, 99)
            assert np.asarray(info["action_mask"]).shape == (n_envs, 10, 10)
    finally:
        venv.close()


def test_obs_within_declared_space():
    env = LowBlockEnv(max_ticks=40, scripted_attackers=False)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs), "reset obs outside declared space"
    assert env.state_space.contains(info["state"]), "reset state outside state_space"
    act = _probe_policy(env, 1)
    for _ in range(40):
        obs, _r, term, trunc, info = env.step(act(info["action_mask"]))
        assert env.observation_space.contains(obs), "step obs outside declared space"
        assert env.state_space.contains(info["state"]), (
            "step state outside state_space")
        if term or trunc:
            break


FULL_SPEED = N_SPEEDS - 1        # speed_lookup[-1] == 1.0 * V_MAX
EAST = 0                          # direction_lookup[0] == [1, 0]
STAND = N_DIRECTIONS - 1          # direction_lookup[-1] == [0, 0]


def _action(env, direction, speed, ball=0):
    """(n_att, 3) with every attacker commanded the same thing."""
    return np.tile([direction, speed, ball], (env.n_att, 1)).astype(np.int64)


def _hold_action(env, direction=STAND, speed=0):
    return _action(env, direction, speed, ball=0)


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
            _obs, _r, term, _trunc, _i = env.step(_action(env, EAST, FULL_SPEED))
            if term:
                break
    finally:
        env_mod.compute_attacker_targets = original


def test_commanded_velocity_reaches_the_integrator():
    # Attackers move straight off the physics: hold a constant target and the
    # integrator should ramp them onto it at A_MAX and hold there -- unless the
    # target runs them into a line, which now zeroes that velocity component
    # rather than leaving them pinned at x=105 advertising 5 m/s.
    env = LowBlockEnv(max_ticks=200, scripted_attackers=False)
    env.reset(seed=5)
    for _ in range(30):
        env.step(_action(env, EAST, FULL_SPEED))

    att_pos = np.asarray(env.players["position"][:env.n_att], dtype=float)
    att_vel = np.asarray(env.players["velocity"][:env.n_att], dtype=float)
    at_line = att_pos[:, 0] >= 105.0 - 1e-6
    assert at_line.any(), "seed 5 used to run at least one attacker into the line"

    assert np.allclose(att_vel[~at_line], [5.0, 0.0], atol=1e-3), (
        f"attacker velocities {att_vel[~at_line]} did not converge on the target")
    assert np.allclose(att_vel[at_line, 0], 0.0), (
        f"attackers held against the line still carry x-velocity {att_vel[at_line]}")


def test_pass_action_decodes_to_the_holder_slot():
    env = LowBlockEnv(max_ticks=50, scripted_attackers=False)
    env.reset(seed=1)

    holder_id = env.ball["holder_id"]
    holder_row = int(np.flatnonzero(env.attacker_ids == holder_id)[0])
    _vel, ball_idx = env.decode_action(_action(env, STAND, 0, ball=3))

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
        _obs, _r, term, _trunc, info = env.step(_hold_action(env))
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
    venv = make_vector_env(n_envs=n_envs, max_ticks=40,
                           scripted_attackers=False)
    # One fixed action for every env and tick, so the only thing that can differ
    # between the vector column and the single env is the env's own state.
    try:
        _obs, _info = venv.reset(seed=100)
        action = np.tile([EAST, FULL_SPEED, 0], (n_envs, 10, 1)).astype(np.int64)
        vec_rewards = []
        for _ in range(10):
            _obs, r, _term, _trunc, _info = venv.step(action)
            vec_rewards.append(r.copy())
    finally:
        venv.close()
    vec_rewards = np.array(vec_rewards)  # (10, n_envs)

    for i in range(n_envs):
        env = LowBlockEnv(max_ticks=40, scripted_attackers=False)
        env.reset(seed=100 + i)
        for t in range(10):
            _obs, r, _term, _trunc, _i = env.step(action[i])
            assert abs(r - vec_rewards[t, i]) <= TOL, (
                f"env {i} tick {t}: vector {vec_rewards[t, i]} != single {r}")


def main():
    failures = 0
    tests = [
        test_env_telescoping_identity,
        test_step_surface_matches_a_fresh_ppcf,
        test_terminal_labelling,
        test_reset_is_seed_reproducible,
        test_same_seed_gives_an_identical_trajectory,
        test_obs_within_declared_space,
        test_obs_state_and_mask_shapes,
        test_remaining_time_is_in_obs_and_state,
        test_actor_and_critic_share_one_normalization,
        test_offside_targets_are_masked_out,
        test_a_masked_pass_is_never_offered_even_with_everyone_beyond_the_line,
        test_offside_line_and_margin_are_in_the_obs,
        test_agent_one_hot_distinguishes_the_rows,
        test_agent_reward_is_per_attacker,
        test_vector_env_shapes_hold_up,
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
