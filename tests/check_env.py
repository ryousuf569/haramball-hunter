import argparse
import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from attackers.scripted_policy import make_policy              # noqa: E402
from environment.lowblock_env import (GAMMA, SUCCESS,          # noqa: E402
                                      LowBlockEnv,
                                      make_initial_world,
                                      world_step)

# Measured with run_episode at start_holder=0; see the probe notes
BASELINE = {"random": 0.007, "scripted": 0.42}


def check_gym_api():
    from gymnasium.utils.env_checker import check_env
    env = LowBlockEnv(start_holder=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_env(env, skip_render_check=True)
            ok = True
        except Exception as exc:
            ok = False
            print(f"  raised {type(exc).__name__}: {exc}")
    for w in caught:
        print(f"  warn: {w.message}")
    print(f"check_env: {'passed' if ok else 'FAILED'}  ({len(caught)} warnings)")
    return ok


def check_telescoping(n_episodes):
    # The DISCOUNTED sum of shaping telescopes to gamma^T * phi(s_T) - phi(s_0),
    # and phi is zeroed at the terminal, so it should equal -phi(s_0) exactly.
    env = LowBlockEnv(start_holder=0)
    worst = 0.0
    for e in range(n_episodes):
        env.reset(seed=e)
        phi0 = env.prev_phi
        total, t = 0.0, 0
        while True:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            total += (GAMMA ** t) * info["shaping"]
            t += 1
            if terminated or truncated:
                break
        worst = max(worst, abs(total - (-phi0)))
    ok = worst < 1e-4
    print(f"telescoping: {'passed' if ok else 'FAILED'}  worst |error| = {worst:.2e}")
    return ok


def check_equivalence(n_episodes):
    # The wrapper must not change the dynamics. Rebuild the same world from the
    # env's own episode seed, replay the identical actions through world_step,
    # and require bit-for-bit agreement. One episode says more here than a
    # hundred episodes of rate comparison.
    env = LowBlockEnv(start_holder=0)
    worst = 0.0
    for e in range(n_episodes):
        env.reset(seed=e)
        players, ball, aid, rng, dstate = make_initial_world(
            env.n_att, env.n_def, seed=env.ep_seed, start_holder=0)
        for tick in range(1, env.max_ticks + 1):
            a = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(a)
            players, ball, _pc, outcome = world_step(
                players, ball, aid, dstate, tick, rng,
                ppcf_grid=env.ppcf_grid, zone=env.zone,
                max_ticks=env.max_ticks,
                actions=(a[:, 0], a[:, 1], a[:, 2]))
            worst = max(worst, float(np.abs(
                np.asarray(players["position"], float)
                - np.asarray(env.players["position"], float)).max()))
            if terminated or truncated:
                assert (outcome is not None) == terminated, "outcome disagrees"
                break
    ok = worst == 0.0
    print(f"equivalence: {'passed' if ok else 'FAILED'}  "
          f"worst position delta = {worst:.3e} m")
    return ok


def _run(env, n_episodes, scripted):
    outcomes, ticks = [], []
    for e in range(n_episodes):
        env.reset(seed=10_000 + e)
        policy = make_policy(env.zone) if scripted else None
        t = 0
        while True:
            if policy is None:
                a = env.action_space.sample()
            else:
                d, s, b = policy(env.players, env.ball, env.attacker_ids)
                a = np.stack([d, s, b], axis=1)
            _, _, terminated, truncated, info = env.step(a)
            t += 1
            if terminated or truncated:
                break
        outcomes.append(info["outcome"])
        ticks.append(t)
    rate = float(np.mean([o == SUCCESS for o in outcomes]))
    return rate, float(np.mean(ticks))


def check_rates(n_episodes):
    env = LowBlockEnv(start_holder=0)
    ok = True
    for name, scripted in (("random", False), ("scripted", True)):
        rate, mean_ticks = _run(env, n_episodes, scripted)
        want = BASELINE[name]
        # 3 standard errors on n_episodes, floored so the random arm is not
        # asked for more precision than the sample can carry
        se = max(3.0 * np.sqrt(max(want * (1 - want), 1e-4) / n_episodes), 0.03)
        hit = abs(rate - want) <= se
        ok &= hit
        print(f"{name:9s}: {rate:6.1%} vs baseline {want:5.1%} "
              f"(+/-{se:.1%})  mean {mean_ticks:5.0f} ticks  "
              f"{'ok' if hit else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()

    results = []
    print("== 1. gymnasium API ==")
    results.append(check_gym_api())
    print("\n== 2. potential-based shaping telescopes ==")
    results.append(check_telescoping(min(args.episodes, 10)))
    print("\n== 3. wrapper does not change the dynamics ==")
    results.append(check_equivalence(min(args.episodes, 10)))
    print(f"\n== 4. outcome rates vs run_episode baselines "
          f"({args.episodes} episodes) ==")
    results.append(check_rates(args.episodes))

    print("\nALL PASSED" if all(results) else "\nSOMETHING FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
