import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "model", "ppo_constrained.py")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

SEEDS = (1, 2, 3)
STEPS = 2_500_000
COOLDOWN = 150

# One arm per dial
ARMS = {
    "slow": {"flag": "--slow-d", "prefix": "sweep_slow",
             "thresholds": (0.85, 0.75, 0.65, 0.55, 0.45),
             "held": (("--crowd-d", 0.10), ("--succ-d", 0.40))},
    "crowd": {"flag": "--crowd-d", "prefix": "sweep_crowd",
              # brackets the observed drift (0.211 on c_lagr_075_040) and the
              # near-baseline rate (0.02-0.03 on scripted / vanilla)
              "thresholds": (0.15, 0.10, 0.05),
              "held": (("--slow-d", 0.75), ("--succ-d", 0.40))},}
DEFAULT_ARM = "crowd"


def run_name(arm, threshold, seed, suffix=""):
    return "%s%02d_s%d%s" % (ARMS[arm]["prefix"], round(threshold * 100), seed,
                             suffix)


def held_of(arm, succ_d):
    return [(f, succ_d if f == "--succ-d" else v) for f, v in ARMS[arm]["held"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default=DEFAULT_ARM,
                    help="which constraint threshold to sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="defaults to the arm's own grid")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--succ-d", type=float, default=0.40)
    ap.add_argument("--suffix", default="",
                    help="appended to every run name, so a rerun at a different "
                         "budget does not collide with (and get skipped by) an "
                         "existing sweep")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    args = ap.parse_args()

    arm = ARMS[args.arm]
    thresholds = args.thresholds or list(arm["thresholds"])
    held = held_of(args.arm, args.succ_d)

    jobs = [(d, s) for s in args.seeds for d in thresholds]
    t0 = time.perf_counter()

    print("arm %s: sweeping %s over %s | held %s | seeds %s | %d steps each"
          % (args.arm, arm["flag"],
             " ".join("%.2f" % d for d in thresholds),
             " ".join("%s %.2f" % (f, v) for f, v in held),
             " ".join(str(s) for s in args.seeds), args.steps), flush=True)

    for i, (threshold, seed) in enumerate(jobs, 1):
        name = run_name(args.arm, threshold, seed, args.suffix)
        hours = (time.perf_counter() - t0) / 3600.0

        if os.path.exists(os.path.join(RUNS_DIR, name, "final.pt")):
            print("[%2d/%d] %s done already, skipping" % (i, len(jobs), name),
                  flush=True)
            continue

        print("[%2d/%d] %s | %s %.2f seed %d | %.1f h elapsed"
              % (i, len(jobs), name, arm["flag"], threshold, seed, hours),
              flush=True)

        cmd = [sys.executable, SCRIPT,
               "--total-steps", str(args.steps),
               "--seed", str(seed),
               arm["flag"], str(threshold),
               "--run-name", name,
               "--log-every", "50"]
        for flag, value in held:
            cmd += [flag, str(value)]

        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
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
