"""Three sequential 1M-step runs asking one question.

Did the constrained formulation remove the tuning burden, or only move it from
reward weights to constraint thresholds? README.md has the arms and the answers.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import plots
from config import CurriculumConfig, EnvConfig, LagrangeConfig, PPOConfig
from environment import costs
from ppo import train

FIG_DIR = os.path.join("models", "figs")

# Arm C's thresholds: what someone writes down without measuring the game first.
MISTUNED = {
    "pass_lost": 0.02,       # measured 26% for 5M, 10% for random play
    "cross_field": 0.02,     # measured 47% / 38%
    "hot_potato": 0.05,      # measured 98% / 100%
    "pass_back": 0.05,       # measured 38% / 43%
    "offside": 0.005,        # measured 6.4% / 1.9%
    "far_from_ball": 0.10,   # measured 29% / 27%
    "no_success": 0.30,      # asks for a 70% success rate; random gets 2.6%
}


def mistuned_thresholds():
    return tuple(MISTUNED[name] for name in costs.COST_NAMES)


def arms(steps, seed):
    """(name, PPOConfig, EnvConfig, LagrangeConfig, CurriculumConfig) each.

    Curriculum and seed are identical across arms. Only the way the objective
    is specified varies, since an arm that moved both would confound them.
    """
    ppo = PPOConfig(total_timesteps=steps, seed=seed)
    curr = CurriculumConfig()

    return [
        ("reward_only",
         ppo, EnvConfig(agent_alpha=0.5, offside_in_potential=True),
         LagrangeConfig(enabled=False), curr),

        ("constrained",
         ppo, EnvConfig(), LagrangeConfig(), curr),

        ("constrained_mistuned",
         ppo, EnvConfig(), LagrangeConfig(thresholds=mistuned_thresholds()),
         curr),
    ]


def save(name, result, outdir):
    """Checkpoint, history JSON, and this arm's four figures.

    Written the moment an arm finishes rather than at the end of all three,
    since 1M steps is hours and a crash in arm C must not cost arms A and B.
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    ckpt = {"actor": result.actor.state_dict(),
            "critic": result.critic.state_dict()}
    if result.cost_critics is not None:
        # The multipliers are part of what the run learned, so a resume that
        # restarted them at z_init would throw that away.
        ckpt["cost_critics"] = result.cost_critics.state_dict()
        ckpt["lagrange_z"] = result.lagrange.z
    torch.save(ckpt, os.path.join("models", f"{name}.pt"))

    with open(os.path.join(outdir, f"{name}_history.json"), "w") as f:
        json.dump({"arm": name,
                   "constrained": result.constrained,
                   "thresholds": [float(t) for t in result.thresholds],
                   "cost_names": list(costs.COST_NAMES),
                   "history": result.history}, f, indent=1)

    plots.plot_all(name, result.history, np.asarray(result.thresholds), outdir)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=1_000_000,
                   help="env steps per arm (default 1M)")
    p.add_argument("--arms", nargs="+", default=None,
                   choices=[a[0] for a in arms(1, 0)],
                   help="run a subset, in the order given")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figs", default=FIG_DIR)
    p.add_argument("--sync", action="store_true",
                   help="SyncVectorEnv; slower, but a traceback is readable")
    args = p.parse_args()

    todo = [a for a in arms(args.steps, args.seed)
            if args.arms is None or a[0] in args.arms]
    if args.arms is not None:
        todo.sort(key=lambda a: args.arms.index(a[0]))

    # Measured at ~50 sps on 6 async envs. Say so up front, because three 1M
    # arms is most of a day and hour six is a bad time to find that out.
    est_h = len(todo) * args.steps / 50.0 / 3600.0
    print(f"{len(todo)} arm(s) x {args.steps:,} steps, sequential.")
    print(f"Rough estimate at ~50 sps: {est_h:.1f} h total "
          f"({est_h / max(len(todo), 1):.1f} h per arm).")
    print(f"Figures -> {args.figs}/   checkpoints -> models/\n")

    runs = {}
    t0 = time.time()
    for i, (name, ppo_cfg, env_cfg, lag_cfg, curr_cfg) in enumerate(todo, 1):
        print("=" * 78)
        print(f"ARM {i}/{len(todo)}: {name}")
        if lag_cfg.thresholds is not None:
            print("  thresholds: " + "  ".join(
                f"{n}<={d}" for n, d in zip(costs.COST_NAMES,
                                            lag_cfg.thresholds)))
        print("=" * 78)

        t = time.time()
        result = train(ppo_cfg, env_cfg, asynchronous=not args.sync,
                       lag_cfg=lag_cfg, curr_cfg=curr_cfg)
        print(f"\n  arm {name} finished in {(time.time() - t) / 3600:.2f} h")

        save(name, result, args.figs)
        runs[name] = result.history

    if len(runs) >= 2:
        print("\ncross-arm figures:")
        plots.plot_comparison(runs, args.figs)
        plots.plot_final_table(runs, args.figs)

    print(f"\nall done in {(time.time() - t0) / 3600:.2f} h. "
          f"Figures in {args.figs}/")


if __name__ == "__main__":
    main()
