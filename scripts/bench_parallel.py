# Throughput benchmark: how many env steps/sec N parallel workers sustain.
# Run: python scripts/bench_parallel.py
# limit_threads() runs before numpy loads so each worker is single-threaded and
# the only parallelism measured is across processes, not inside BLAS.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.thread_limits import limit_threads  # noqa: E402

limit_threads()

# render.py imports pyplot at module level; workers are headless
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

ENV_COUNTS = [1, 2, 4, 5, 6]
WARMUP_SECONDS = 15.0
TIMED_SECONDS = 180.0
EPISODE_TICKS = 300  # reset cadence, so workers don't sit in a post-turnover state
N_ATT = 10
N_DEF = 10


def _worker(worker_id, seed, barrier, out_q, warmup_s, timed_s):
    limit_threads()

    import numpy as np
    from render import make_initial_world, make_ppcf_grid, step

    rng = np.random.default_rng(seed)
    grid = make_ppcf_grid()

    # step() takes no action argument -- both policies are scripted internally --
    # so the per-episode seed and start_holder are the env's only random inputs.
    def reset():
        ep_seed = int(rng.integers(0, 2**31 - 1))
        holder = int(rng.integers(0, N_ATT))
        players, ball, ids, _rng, dstate = make_initial_world(
            n_att=N_ATT, n_def=N_DEF, seed=ep_seed, start_holder=holder)
        return players, ball, ids, dstate, 0

    state = reset()

    def run_until(deadline):
        nonlocal state
        players, ball, ids, dstate, tick = state
        count = 0
        while time.time() < deadline:
            try:
                players, ball, _pc = step(players, ball, ids, dstate, tick,
                                          ppcf_grid=grid)
            except SystemExit:  # turnover exit path; treated as episode end
                players, ball, ids, dstate, tick = reset()
                count += 1
                continue
            tick += 1
            count += 1
            if tick >= EPISODE_TICKS:
                players, ball, ids, dstate, tick = reset()
        state = (players, ball, ids, dstate, tick)
        return count

    # turnovers print to stdout; keep the workers quiet without touching engine code
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull):
            # one step before the barrier so first-call allocation isn't in the warmup
            run_until(time.time() + 0.001)
            barrier.wait(timeout=600)

            run_until(time.time() + warmup_s)

            t0 = time.perf_counter()
            steps = run_until(time.time() + timed_s)
            elapsed = time.perf_counter() - t0
    finally:
        devnull.close()

    out_q.put({
        "worker_id": worker_id,
        "steps": steps,
        "elapsed": elapsed,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
    })


def run_config(n_envs, warmup_s, timed_s):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n_envs)
    out_q = ctx.Queue()

    procs = [ctx.Process(target=_worker,
                         args=(i, 1000 + i, barrier, out_q, warmup_s, timed_s))
             for i in range(n_envs)]
    for p in procs:
        p.start()

    # drain before join: a queue-writing child can block on a full pipe otherwise
    results = [out_q.get() for _ in range(n_envs)]
    for p in procs:
        p.join()

    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"worker(s) exited non-zero: {failed}")

    results.sort(key=lambda r: r["worker_id"])
    # summing per-worker rates is robust to the small skew in when each one starts
    total_rate = sum(r["steps"] / r["elapsed"] for r in results)
    return results, total_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=float, default=WARMUP_SECONDS)
    ap.add_argument("--timed", type=float, default=TIMED_SECONDS)
    ap.add_argument("--envs", type=int, nargs="+", default=ENV_COUNTS)
    # carry a 1-env rate from an earlier run so efficiency stays comparable when
    # this run doesn't include n_envs=1 itself
    ap.add_argument("--baseline-per-env", type=float, default=None)
    args = ap.parse_args()

    total_min = len(args.envs) * (args.warmup + args.timed) / 60.0
    print(f"logical cpus: {mp.cpu_count()}")
    print(f"parent OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"configs: {args.envs}  warmup={args.warmup:.0f}s  timed={args.timed:.0f}s")
    print(f"estimated runtime: ~{total_min:.1f} min\n")

    summary = []
    for n_envs in args.envs:
        print(f"--- n_envs={n_envs} ---", flush=True)
        t0 = time.time()
        results, total_rate = run_config(n_envs, args.warmup, args.timed)
        for r in results:
            print(f"  worker {r['worker_id']}: {r['steps']:6d} steps in "
                  f"{r['elapsed']:6.1f}s = {r['steps'] / r['elapsed']:6.2f} steps/s  "
                  f"OMP={r['omp']} MKL={r['mkl']} OPENBLAS={r['openblas']}",
                  flush=True)
        total_steps = sum(r["steps"] for r in results)
        print(f"  total: {total_steps} steps, {total_rate:.2f} steps/sec "
              f"({total_rate / n_envs:.2f} per env)  "
              f"[wall {time.time() - t0:.0f}s]\n", flush=True)
        summary.append((n_envs, total_steps, total_rate))

    if args.baseline_per_env:
        base = args.baseline_per_env
        base_note = f"supplied 1-env baseline {base:.2f} steps/sec"
    else:
        base = summary[0][2] / summary[0][0] if summary else 0.0
        base_note = f"n_envs={summary[0][0]} baseline {base:.2f} steps/sec"

    print("=" * 62)
    print(f"{'n_envs':>7} | {'steps/sec':>10} | {'per env':>9} | {'efficiency':>10}")
    print("-" * 62)
    for n_envs, _steps, rate in summary:
        per_env = rate / n_envs
        eff = per_env / base if base else 0.0
        print(f"{n_envs:>7} | {rate:>10.2f} | {per_env:>9.2f} | {eff:>9.1%}")
    print("=" * 62)
    print(f"efficiency = per-env rate vs {base_note}; 100% is linear scaling")


if __name__ == "__main__":
    main()
