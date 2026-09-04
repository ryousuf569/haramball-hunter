import argparse
import os
import shlex
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "model", "ppo_ablation.py")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

SEEDS = (1, 2)
STEPS = 2_500_000
COOLDOWN = 150

SLOW_ARM = {"slow-d": 0.75, "crowd-d": 1.0, "succ-d": 0.40}

ABLATIONS = {
    "control": {"thresholds": SLOW_ARM,
                "why": "unmodified; the matched 2.5M baseline the others move against"},
    "no_bootstrap": {"thresholds": dict(SLOW_ARM, **{"succ-d": 0.0}),
                     "why": "floor disabled, so lam0 has nothing to hand off to"},
    "softplus": {"thresholds": SLOW_ARM,
                 "why": "unnormalised multipliers, no simplex bound on sum(lam)"},
    "team_slow": {"thresholds": SLOW_ARM,
                  "why": "slow read off the team mean, not per attacker"},
}
DEFAULT = ("no_bootstrap", "softplus", "team_slow")
DIALS = ("slow-d", "crowd-d", "succ-d")


def run_name(ablation, seed, suffix):
    return "abl_%s_s%d%s" % (ablation, seed, suffix)


def build_cmd(ablation, name, seed, steps, log_every, extra):
    thresholds = ABLATIONS[ablation]["thresholds"]
    cmd = [sys.executable, SCRIPT,
           "--total-steps", str(steps),
           "--seed", str(seed),
           "--run-name", name,
           "--ablation", ablation,
           "--log-every", str(log_every)]
    for dial in DIALS:
        cmd += ["--" + dial, str(thresholds[dial])]
    return cmd + extra


def main():
    ap = argparse.ArgumentParser(
        description="Run the constrained-PPO ablations against the slow arm at "
                    "0.75. Each config is one deviation from ppo_constrained.py.")
    ap.add_argument("--which", nargs="+", choices=sorted(ABLATIONS),
                    default=list(DEFAULT), help="ablations to run")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--suffix", default="_25m", help="appended to every run name")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--extra", default="",
                    help="extra args passed straight to ppo_ablation.py, quoted")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run and exit")
    args = ap.parse_args()

    extra = shlex.split(args.extra)
    jobs = [(a, s) for a in args.which for s in args.seeds]
    t0 = time.perf_counter()

    print("%d ablations x %d seeds = %d runs | %d steps each"
          % (len(args.which), len(args.seeds), len(jobs), args.steps))
    for a in args.which:
        t = ABLATIONS[a]["thresholds"]
        print("  %-13s %s | %s"
              % (a, " ".join("%s %g" % (d, t[d]) for d in DIALS),
                 ABLATIONS[a]["why"]))
    print(flush=True)

    for i, (ablation, seed) in enumerate(jobs, 1):
        name = run_name(ablation, seed, args.suffix)
        cmd = build_cmd(ablation, name, seed, args.steps, args.log_every, extra)
        hours = (time.perf_counter() - t0) / 3600.0

        if args.dry_run:
            print("[%2d/%d] %s\n        %s"
                  % (i, len(jobs), name, " ".join(cmd[1:])))
            continue

        if os.path.exists(os.path.join(RUNS_DIR, name, "final.pt")):
            print("[%2d/%d] %s done already, skipping" % (i, len(jobs), name),
                  flush=True)
            continue

        print("[%2d/%d] %s | seed %d | %.1f h elapsed"
              % (i, len(jobs), name, seed, hours), flush=True)

        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            print("[%2d/%d] %s FAILED rc=%d" % (i, len(jobs), name, rc),
                  flush=True)

        if i < len(jobs):
            # let the laptop's CPU/GPU thermals settle so later arms are not throttled
            time.sleep(args.cooldown)

    if not args.dry_run:
        print("ablations finished in %.1f h"
              % ((time.perf_counter() - t0) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
