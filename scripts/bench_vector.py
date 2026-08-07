# Latency benchmark for the vectorised env: ms per venv.step() call, and the
# per-env ms/step that falls out of it.
#
#   python scripts/bench_vector.py                 # 6 envs, SyncVectorEnv
#   python scripts/bench_vector.py --async         # 6 envs, AsyncVectorEnv
#   python scripts/bench_vector.py --envs 1 2 4 6  # scaling sweep
#
# scripts/bench_parallel.py measures sustained throughput of N independent
# worker processes over minutes. This one measures the latency of a single
# vector call, which is what an on-policy learner actually waits on between
# forward passes.
#
# limit_threads() runs before numpy loads so each env is single-threaded and
# SyncVectorEnv's serial loop isn't quietly competing with itself in BLAS.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.thread_limits import limit_threads  # noqa: E402

limit_threads()

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from environment.lowblock_env import make_vector_env  # noqa: E402

ENV_COUNTS = [6]
WARMUP_CALLS = 50
TIMED_CALLS = 400
MAX_TICKS = 300


def _stats(times_ms, n_envs):
    t = np.asarray(times_ms, dtype=float)
    return {
        "calls": t.size,
        "mean_call": t.mean(),
        "p50_call": np.percentile(t, 50),
        "p90_call": np.percentile(t, 90),
        "p99_call": np.percentile(t, 99),
        "max_call": t.max(),
        "mean_step": t.mean() / n_envs,
        "p50_step": np.percentile(t, 50) / n_envs,
        "steps_per_s": n_envs * 1000.0 / t.mean(),
    }


def bench(n_envs, asynchronous, warmup, timed, max_ticks, seed=0):
    # scripted_attackers=False is what training runs, and it is also the only
    # honest thing to time now: the step does the full policy path -- decode the
    # (n_att, 3) action, integrate, PPCF, reward, then build the per-agent obs,
    # the global state and the action mask.
    venv = make_vector_env(n_envs=n_envs, asynchronous=asynchronous,
                           max_ticks=max_ticks, scripted_attackers=False)
    try:
        # Time an explicit reset separately: it samples both formations and pays
        # one extra PPCF call per env to seed Phi(s0).
        t0 = time.perf_counter()
        venv.reset(seed=seed)
        reset_ms = (time.perf_counter() - t0) * 1000.0

        # Sampled once, not per call, so the sampler isn't in the timing. The
        # ball head is left unmasked here: the env decodes only the holder's row
        # regardless, so an illegal entry costs nothing and changes no timing.
        action = venv.action_space.sample()

        for _ in range(warmup):
            venv.step(action)

        times_ms = np.empty(timed, dtype=float)
        rewards = np.empty((timed, n_envs), dtype=float)
        n_term = n_trunc = 0
        for i in range(timed):
            t0 = time.perf_counter()
            _obs, r, term, trunc, _info = venv.step(action)
            times_ms[i] = (time.perf_counter() - t0) * 1000.0
            rewards[i] = r
            n_term += int(np.sum(term))
            n_trunc += int(np.sum(trunc))
    finally:
        venv.close()

    out = _stats(times_ms, n_envs)
    out.update(n_envs=n_envs, asynchronous=asynchronous, reset_ms=reset_ms,
               mean_reward=float(rewards.mean()), terminated=n_term,
               truncated=n_trunc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=ENV_COUNTS)
    ap.add_argument("--warmup", type=int, default=WARMUP_CALLS)
    ap.add_argument("--timed", type=int, default=TIMED_CALLS)
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--async", dest="asynchronous", action="store_true",
                    help="AsyncVectorEnv (one process per env) instead of Sync")
    ap.add_argument("--both", action="store_true",
                    help="run Sync and Async for each env count")
    args = ap.parse_args()

    modes = [False, True] if args.both else [args.asynchronous]

    print(f"logical cpus: {os.cpu_count()}   OMP_NUM_THREADS="
          f"{os.environ.get('OMP_NUM_THREADS')}")
    print(f"envs={args.envs}  warmup={args.warmup} calls  timed={args.timed} calls"
          f"  max_ticks={args.max_ticks}\n")

    rows = []
    for asynchronous in modes:
        for n_envs in args.envs:
            label = "async" if asynchronous else "sync"
            print(f"--- {label}  n_envs={n_envs} ---", flush=True)
            r = bench(n_envs, asynchronous, args.warmup, args.timed,
                      args.max_ticks)
            rows.append(r)
            print(f"  reset:      {r['reset_ms']:8.2f} ms")
            print(f"  ms/call:    mean {r['mean_call']:7.3f}  p50 {r['p50_call']:7.3f}"
                  f"  p90 {r['p90_call']:7.3f}  p99 {r['p99_call']:7.3f}"
                  f"  max {r['max_call']:7.3f}")
            print(f"  ms/step:    mean {r['mean_step']:7.3f}  p50 {r['p50_step']:7.3f}"
                  f"   ({r['steps_per_s']:.1f} env-steps/sec)")
            print(f"  sanity:     mean reward {r['mean_reward']:+.5f}, "
                  f"{r['terminated']} terminated, {r['truncated']} truncated\n",
                  flush=True)

    # Per-env ms/step against the 1-env figure, when the sweep includes one:
    # for SyncVectorEnv the loop is serial, so ms/step should stay flat and
    # ms/call should grow linearly. Async is where ms/step should actually fall.
    print("=" * 76)
    print(f"{'mode':>6} | {'n_envs':>6} | {'ms/call':>9} | {'ms/step':>9} | "
          f"{'steps/sec':>10} | {'vs 1 env':>9}")
    print("-" * 76)
    base = {}
    for r in rows:
        mode = "async" if r["asynchronous"] else "sync"
        if r["n_envs"] == 1:
            base[mode] = r["mean_step"]
        speedup = f"{base[mode] / r['mean_step']:.2f}x" if mode in base else "-"
        print(f"{mode:>6} | {r['n_envs']:>6} | {r['mean_call']:>9.3f} | "
              f"{r['mean_step']:>9.3f} | {r['steps_per_s']:>10.1f} | {speedup:>9}")
    print("=" * 76)


if __name__ == "__main__":
    main()
