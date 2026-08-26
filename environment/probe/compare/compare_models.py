import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attackers.ppo_policy import make_ppo_policy, resolve_ckpt
from attackers.random_policy import random_actions
from attackers.scripted_policy import make_policy
from environment.lowblock_env import (FAILURE, MAX_TICKS, SUCCESS, TIMEOUT,
                                      LowBlockEnv)
from environment.termination import (ZONE_PC_MIN, ZONE_RADIUS, ZONE_X, ZONE_Y,
                                     make_zone)

OUTCOMES = (SUCCESS, FAILURE, TIMEOUT)
OUTCOME_COLOR = {SUCCESS: "#2ca02c", FAILURE: "#d62728", TIMEOUT: "#8c8c8c"}
BAR_COLOR = "#1f77b4"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "compare_models.png")


def gate_from_ckpt(path):
    ckpt = torch.load(resolve_ckpt(path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    return (float(cfg.get("zone_x", ZONE_X)),
            float(cfg.get("zone_y", ZONE_Y)),
            float(cfg.get("zone_radius", ZONE_RADIUS)),
            float(cfg.get("pc_min", ZONE_PC_MIN)))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(centre - half, 0.0), min(centre + half, 1.0)


def random_episode(rng):
    def start(env):
        def act(env):
            d, s, b = random_actions(env.n_att, rng)
            return np.stack([d, s, b], axis=1)
        return act
    return start


def scripted_episode(env):
    policy = make_policy(env.zone)

    def act(env):
        d, s, b = policy(env.players, env.ball, env.attacker_ids)
        return np.stack([d, s, b], axis=1)
    return act


def model_episode(ppo):
    def start(env):
        ppo.reset()

        def act(env):
            d, s, b = ppo(env.players, env.ball, env.attacker_ids,
                          pc_att=env.pc_att)
            return np.stack([d, s, b], axis=1)
        return act
    return start


def hybrid_episode(ppo):
    def start(env):
        ppo.reset()
        policy = make_policy(env.zone)

        def act(env):
            d, s, _ = policy(env.players, env.ball, env.attacker_ids)
            _, _, b = ppo(env.players, env.ball, env.attacker_ids,
                          pc_att=env.pc_att)
            return np.stack([d, s, b], axis=1)
        return act
    return start


def hybrid_movement_episode(ppo):
    def start(env):
        ppo.reset()
        policy = make_policy(env.zone)

        def act(env):
            _, _, b = policy(env.players, env.ball, env.attacker_ids)
            d, s, _ = ppo(env.players, env.ball, env.attacker_ids,
                          pc_att=env.pc_att)
            return np.stack([d, s, b], axis=1)
        return act
    return start


def evaluate(env, start_episode, n_episodes, base_seed):
    outcomes, ticks = [], []
    for e in range(n_episodes):
        env.reset(seed=base_seed + e)
        act = start_episode(env)
        t = 0
        while True:
            _, _, terminated, truncated, info = env.step(act(env))
            t += 1
            if terminated or truncated:
                break
        outcomes.append(info["outcome"])
        ticks.append(t)
    return outcomes, ticks


def summarise(name, outcomes, ticks):
    n = len(outcomes)
    counts = {o: outcomes.count(o) for o in OUTCOMES}
    lo, hi = wilson(counts[SUCCESS], n)
    return {"name": name, "n": n, "counts": counts,
            "success": counts[SUCCESS] / n, "lo": lo, "hi": hi,
            "mean_ticks": float(np.mean(ticks))}


def print_row(r, prefix=""):
    print("%s%-46s success %5.1f%%  [%4.1f, %4.1f]  failure %5.1f%%  "
          "timeout %5.1f%%  mean %5.0f ticks"
          % (prefix, r["name"].replace("\n", " "),
             100 * r["success"], 100 * r["lo"], 100 * r["hi"],
             100 * r["counts"][FAILURE] / r["n"],
             100 * r["counts"][TIMEOUT] / r["n"], r["mean_ticks"]),
          flush=True)


def plot(results, gate, n_episodes, n_seeds, start_holder, out_path):
    names = [r["name"] for r in results]
    x = np.arange(len(results))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(7.0, 2.4 * len(results)), 8.2),
        gridspec_kw={"height_ratios": [3, 2]})

    rates = np.array([r["success"] for r in results])
    err = np.vstack([rates - [r["lo"] for r in results],
                     [r["hi"] for r in results] - rates])
    ax1.bar(x, rates, color=BAR_COLOR, width=0.6)
    ax1.errorbar(x, rates, yerr=err, fmt="none", ecolor="black",
                 capsize=5, linewidth=1.2)
    for i, r in enumerate(results):
        ax1.text(i, r["hi"] + 0.02, "%.1f%%" % (100 * r["success"]),
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, min(1.08, max(0.35, float(rates.max()) + 0.18)))
    ax1.set_ylabel("success rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.set_title("success rate with 95%% Wilson interval (n=%d each)"
                  % n_episodes, fontsize=10, loc="left")

    bottom = np.zeros(len(results))
    for outcome in OUTCOMES:
        share = np.array([r["counts"][outcome] / r["n"] for r in results])
        ax2.bar(x, share, bottom=bottom, width=0.6,
                color=OUTCOME_COLOR[outcome], label=outcome)
        for i, v in enumerate(share):
            if v > 0.06:
                ax2.text(i, bottom[i] + v / 2, "%.0f%%" % (100 * v),
                         ha="center", va="center", fontsize=8, color="white")
        bottom += share
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("outcome share")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["%s\n%.0f ticks" % (r["name"], r["mean_ticks"])
                         for r in results], fontsize=9)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
               ncol=3, frameon=False, fontsize=9)

    fig.suptitle("gate (%.0f, %.0f) r=%.0f pc_min=%.2f | start_holder %d"
                 " | %d seed%s"
                 % (gate[0], gate[1], gate[2], gate[3], start_holder,
                    n_seeds, "" if n_seeds == 1 else "s"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


def build_runs(args, zone, seed):
    """Policies for one evaluation seed; `seed` drives every stochastic source."""
    runs = [("random", random_episode(np.random.default_rng(seed))),
            ("scripted", scripted_episode)]
    for path in args.models:
        ppo = make_ppo_policy(path, zone=zone, max_ticks=args.max_ticks,
                              deterministic=args.deterministic, seed=seed)
        runs.append((ppo.label(), model_episode(ppo)))
        if args.hybrid:
            runs.append((ppo.label() + "\n+ scripted movement",
                         hybrid_episode(ppo)))
            runs.append((ppo.label() + "\n+ scripted passing",
                         hybrid_movement_episode(ppo)))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--episodes", type=int, default=200,
                    help="episodes per seed")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of evaluation seeds; results are pooled")
    ap.add_argument("--seed-stride", type=int, default=100_000,
                    help="spacing between seeds (must exceed --episodes)")
    ap.add_argument("--start-holder", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--zone-x", type=float, default=None)
    ap.add_argument("--zone-y", type=float, default=None)
    ap.add_argument("--zone-radius", type=float, default=None)
    ap.add_argument("--pc-min", type=float, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.seeds < 1:
        ap.error("--seeds must be at least 1")
    if args.seeds > 1 and args.seed_stride < args.episodes:
        ap.error("--seed-stride (%d) must be >= --episodes (%d) so seeds do "
                 "not replay the same episodes"
                 % (args.seed_stride, args.episodes))

    gate = gate_from_ckpt(args.models[0]) if args.models else (
        ZONE_X, ZONE_Y, ZONE_RADIUS, ZONE_PC_MIN)
    override = (args.zone_x, args.zone_y, args.zone_radius, args.pc_min)
    gate = tuple(g if o is None else o for g, o in zip(gate, override))

    zone = make_zone(gate[0], gate[1], gate[2])
    env = LowBlockEnv(max_tick=args.max_ticks, start_holder=args.start_holder,
                      zone_x=gate[0], zone_y=gate[1], zone_radius=gate[2],
                      pc_min=gate[3])

    seeds = [args.seed + k * args.seed_stride for k in range(args.seeds)]
    total = args.episodes * args.seeds

    print("gate (%.0f, %.0f) r=%.0f pc_min=%.2f | start_holder %d | "
          "%d episodes x %d seed%s = %d | base seeds %s"
          % (gate[0], gate[1], gate[2], gate[3], args.start_holder,
             args.episodes, args.seeds, "" if args.seeds == 1 else "s", total,
             ", ".join(str(s) for s in seeds)))

    pooled = {}
    for seed in seeds:
        if args.seeds > 1:
            print("\nseed %d (episodes %d..%d)"
                  % (seed, seed, seed + args.episodes - 1), flush=True)
        for name, start_episode in build_runs(args, zone, seed):
            outcomes, ticks = evaluate(env, start_episode, args.episodes, seed)
            o, t = pooled.setdefault(name, ([], []))
            o.extend(outcomes)
            t.extend(ticks)
            print_row(summarise(name, outcomes, ticks),
                      prefix="  " if args.seeds > 1 else "")

    results = [summarise(name, o, t) for name, (o, t) in pooled.items()]
    if args.seeds > 1:
        print("\npooled over %d seeds (n=%d each)" % (args.seeds, total))
        for r in results:
            print_row(r)

    print("wrote %s" % plot(results, gate, total, args.seeds,
                            args.start_holder, args.out))


if __name__ == "__main__":
    main()
