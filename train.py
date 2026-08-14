"""Three sequential 1M-step runs asking one question.

Did the constrained formulation remove the tuning burden, or only move it from
reward weights to constraint thresholds? README.md has the arms and the answers.
"""
import argparse
import json
import os
import time
from datetime import date

import numpy as np
import torch

import plots
from config import CurriculumConfig, EnvConfig, LagrangeConfig, PPOConfig
from environment import costs
from ppo import train


def fmt_steps(n):
    """500000 -> '500k', 1000000 -> '1M'."""
    for size, suffix in ((1_000_000, "M"), (1_000, "k")):
        if n >= size:
            return f"{round(n / size, 1):g}{suffix}"
    return str(n)


def run_dir(steps, when=None):
    """models/MMDDYY-500k, the folder one run's checkpoints and figures share."""
    stamp = (when or date.today()).strftime("%m%d%y")
    return os.path.join("models", f"{stamp}-{fmt_steps(steps)}")

# Arm C's thresholds: what someone writes down without measuring the game first.
MISTUNED = {
    "pass_lost": 0.02,       # measured 26% for 5M, 10% for random play
    "cross_field": 0.02,     # measured 47% / 38%
    "hot_potato": 0.05,      # measured 98% / 100%
    "pass_back": 0.05,       # measured 38% / 43%
    "offside": 0.005,        # measured 6.4% / 1.9%
    "far_from_ball": 0.10,   # measured 29% / 27%
    "held_too_long": 0.01,   # "never dwell on the ball"
    # The one threshold arm C makes laxer rather than tighter, because that is
    # the mistake section 7 warns about: a lax bootstrap "will simply be used to
    # exit the initialisation point and the other constraints will quickly take
    # over". This was our own default until the fidelity pass, so arm C now
    # reruns that mistake against arm B doing it the paper's way.
    "no_success": 0.60,
}


def mistuned_thresholds():
    missing = [n for n in costs.COST_NAMES if n not in MISTUNED]
    if missing:
        raise KeyError(
            f"arm C has no mistuned threshold for {missing}. Add one to "
            f"MISTUNED, deliberately tighter than the measured rate, or arm C "
            f"stops being the badly-specified arm it is supposed to be.")
    return tuple(MISTUNED[name] for name in costs.COST_NAMES)


def arms(steps, seed):
    """(name, PPOConfig, EnvConfig, LagrangeConfig, CurriculumConfig) each.

    Curriculum and seed are identical across arms. Only the way the objective
    is specified varies, since an arm that moved both would confound them.
    """
    ppo = PPOConfig(total_timesteps=steps, seed=seed)
    curr = CurriculumConfig()
    env = EnvConfig()

    # Every arm gets the SAME EnvConfig. An earlier version also switched off
    # agent_alpha and the offside potential in the constrained arms, which
    # bundled two reward changes into what is supposed to be a one-variable
    # comparison. The constraints are layered on top of the reward now, not
    # swapped in for parts of it.
    return [
        ("reward_only", ppo, env, LagrangeConfig(enabled=False), curr),

        ("constrained", ppo, env, LagrangeConfig(), curr),

        ("constrained_mistuned", ppo, env,
         LagrangeConfig(thresholds=mistuned_thresholds()), curr),
    ]


def save(name, result, ckpt_dir, outdir):
    """Checkpoint, history JSON, and this arm's four figures.

    Written the moment an arm finishes rather than at the end of all three,
    since 1M steps is hours and a crash in arm C must not cost arms A and B.
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {"actor": result.actor.state_dict(),
            "critic": result.critic.state_dict(),
            # The critic emits normalised values, so it is unusable without
            # the statistics that scale them back.
            "ret_norm": result.ret_norm.state()}
    if result.cost_critics is not None:
        # The multipliers are part of what the run learned, so a resume that
        # restarted them at z_init would throw that away.
        ckpt["cost_critics"] = result.cost_critics.state_dict()
        ckpt["cost_norms"] = result.cost_critics.norm_state()
        # A list, not the numpy array, so torch.load can read the file back
        # under weights_only=True. numpy objects are rejected there.
        ckpt["lagrange_z"] = [float(z) for z in result.lagrange.z]
    torch.save(ckpt, os.path.join(ckpt_dir, f"{name}.pt"))

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
    p.add_argument("--run-dir", default=None,
                   help="where checkpoints go (default models/MMDDYY-<steps>)")
    p.add_argument("--figs", default=None,
                   help="where figures go (default <run-dir>/figs)")
    p.add_argument("--sync", action="store_true",
                   help="SyncVectorEnv; slower, but a traceback is readable")
    args = p.parse_args()

    todo = [a for a in arms(args.steps, args.seed)
            if args.arms is None or a[0] in args.arms]
    if args.arms is not None:
        todo.sort(key=lambda a: args.arms.index(a[0]))

    # One folder per run, so two runs of different lengths cannot overwrite
    # each other's checkpoints. Same day and same step count still can.
    ckpt_dir = args.run_dir or run_dir(args.steps)
    fig_dir = args.figs or os.path.join(ckpt_dir, "figs")

    # Measured at ~50 sps on 6 async envs. Say so up front, because three 1M
    # arms is most of a day and hour six is a bad time to find that out.
    est_h = len(todo) * args.steps / 50.0 / 3600.0
    print(f"{len(todo)} arm(s) x {args.steps:,} steps, sequential.")
    print(f"Rough estimate at ~50 sps: {est_h:.1f} h total "
          f"({est_h / max(len(todo), 1):.1f} h per arm).")
    print(f"Checkpoints -> {ckpt_dir}/   figures -> {fig_dir}/\n")

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

        save(name, result, ckpt_dir, fig_dir)
        runs[name] = result.history

    if len(runs) >= 2:
        print("\ncross-arm figures:")
        plots.plot_comparison(runs, fig_dir)
        plots.plot_final_table(runs, fig_dir)

    print(f"\nall done in {(time.time() - t0) / 3600:.2f} h. "
          f"Checkpoints in {ckpt_dir}/, figures in {fig_dir}/")


if __name__ == "__main__":
    main()
