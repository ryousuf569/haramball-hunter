import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "model", "ppo_constrained.py")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

THRESHOLDS = (0.85, 0.75, 0.65, 0.55, 0.45)
SEEDS = (1, 2, 3)
STEPS = 2_500_000
SUCC_D = 0.40
COOLDOWN = 150


def run_name(slow_d, seed):
    return "sweep_slow%02d_s%d" % (round(slow_d * 100), seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--succ-d", type=float, default=SUCC_D)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    args = ap.parse_args()

    jobs = [(d, s) for s in args.seeds for d in args.thresholds]
    t0 = time.perf_counter()

    for i, (slow_d, seed) in enumerate(jobs, 1):
        name = run_name(slow_d, seed)
        hours = (time.perf_counter() - t0) / 3600.0

        if os.path.exists(os.path.join(RUNS_DIR, name, "final.pt")):
            print("[%2d/%d] %s done already, skipping" % (i, len(jobs), name),
                  flush=True)
            continue

        print("[%2d/%d] %s | slow_d %.2f seed %d | %.1f h elapsed"
              % (i, len(jobs), name, slow_d, seed, hours), flush=True)

        rc = subprocess.run([sys.executable, SCRIPT,
                             "--total-steps", str(args.steps),
                             "--seed", str(seed),
                             "--slow-d", str(slow_d),
                             "--succ-d", str(args.succ_d),
                             "--run-name", name,
                             "--log-every", "50"],
                            cwd=REPO_ROOT).returncode
        if rc != 0:
            print("[%2d/%d] %s FAILED rc=%d" % (i, len(jobs), name, rc),
                  flush=True)

        if i < len(jobs):
            # let the laptop's CPU/GPU thermals settle so later arms are not throttled
            time.sleep(args.cooldown)

    print("sweep finished in %.1f h" % ((time.perf_counter() - t0) / 3600.0),
          flush=True)


if __name__ == "__main__":
    main()
