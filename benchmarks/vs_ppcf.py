# Throughput of the CUDA kernel PPCF backend against the numpy implementation.

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gymnasium.vector import AutoresetMode

from attackers.ppo_policy import load_agent, resolve_ckpt
from environment.lowblock_env import MAX_TICKS, make_vector_env
from environment.termination import ZONE_PC_MIN, ZONE_RADIUS, ZONE_X, ZONE_Y

DEFAULT_CKPT = os.path.join(REPO_ROOT, "runs", "ppo_10m_s0_entsplit", "best.pt")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vs_ppcf.png")
COLOR = {"numpy": "#1f77b4", "cuda": "#2ca02c"}
WARMUP = 5


def gate_of(ckpt):
    cfg = ckpt.get("cfg", {})
    return dict(zone_x=float(cfg.get("zone_x", ZONE_X)),
                zone_y=float(cfg.get("zone_y", ZONE_Y)),
                zone_radius=float(cfg.get("zone_radius", ZONE_RADIUS)),
                pc_min=float(cfg.get("pc_min", ZONE_PC_MIN)),
                start_holder=cfg.get("start_holder", 0))


def run(name, ckpt_path, episodes, seed, n_envs, asynchronous, max_ticks):
    # Async workers are spawned fresh and read the variable at import time. A
    # sync vector env runs in this process, where lowblock_env is already
    # imported, so that one needs the module attribute swapped directly.
    os.environ["PPCF_BACKEND"] = name
    if not asynchronous:
        import environment.lowblock_env as lowblock
        if name == "cuda":
            from physics.ppcf_kernel import PPCF_grid as backend
        else:
            from physics.ppcf import PPCF_grid as backend
        lowblock.PPCF_grid = backend

    torch.manual_seed(seed)

    agent, ckpt = load_agent(resolve_ckpt(ckpt_path))
    gate = gate_of(ckpt)
    holder = gate.pop("start_holder")
    venv = make_vector_env(n_envs=n_envs, asynchronous=asynchronous,
                           autoreset_mode=AutoresetMode.SAME_STEP,
                           max_tick=max_ticks,
                           start_holder=(None if holder is None or holder < 0
                                         else holder),
                           **gate)
    obs, _info = venv.reset(seed=seed)

    def act(obs):
        with torch.no_grad():
            a, _logp, _v = agent.act(torch.as_tensor(obs))
        return a.numpy()

    for _ in range(WARMUP):
        obs, _r, _term, _trunc, _info = venv.step(act(obs))

    times, steps = [0.0], [0]
    done = 0
    n = 0
    t0 = time.perf_counter()
    while done < episodes:
        obs, _r, term, trunc, _info = venv.step(act(obs))
        n += n_envs
        done += int(np.count_nonzero(np.asarray(term) | np.asarray(trunc)))
        times.append(time.perf_counter() - t0)
        steps.append(n)
    venv.close()

    elapsed = times[-1]
    return {"name": name, "times": np.array(times), "steps": np.array(steps),
            "total": n, "episodes": done, "elapsed": elapsed,
            "rate": n / elapsed}


def plot(results, episodes, n_envs, asynchronous, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    t_max = max(float(r["times"][-1]) for r in results)
    line = np.array([0.0, t_max])

    for r in results:
        color = COLOR[r["name"]]
        ax.plot(r["times"], r["steps"], color=color, linewidth=1.8,
                label="%s  -  %.0f steps/s" % (r["name"], r["rate"]))
        ax.plot(line, r["rate"] * line, color=color, linestyle=":",
                linewidth=1.3)

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, max(int(r["steps"][-1]) for r in results) * 1.02)
    ax.set_xlabel("wall clock (s)")
    ax.set_ylabel("env steps")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("PPCF backend throughput, %d envs %s, %d episodes each "
                 "(dotted = mean steps/s)"
                 % (n_envs, "async" if asynchronous else "sync", episodes),
                 fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    asynchronous = not args.sync
    print("%d envs, %s, %d episodes per backend"
          % (args.n_envs, "async" if asynchronous else "sync", args.episodes))

    results = []
    for name in ("numpy", "cuda"):
        r = run(name, args.ckpt, args.episodes, args.seed, args.n_envs,
                asynchronous, args.max_ticks)
        results.append(r)
        print("%-6s %6d steps  %3d episodes  %7.2f s  %8.1f steps/s  "
              "%6.3f ms/step"
              % (r["name"], r["total"], r["episodes"], r["elapsed"], r["rate"],
                 1000 * r["elapsed"] / r["total"]), flush=True)

    base = results[0]
    for r in results[1:]:
        print("%s vs %s: %.2fx" % (r["name"], base["name"],
                                   r["rate"] / base["rate"]))

    print("wrote %s" % plot(results, args.episodes, args.n_envs, asynchronous,
                            args.out))


if __name__ == "__main__":
    main()
