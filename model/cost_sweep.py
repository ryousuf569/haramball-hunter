import argparse
import os
import shlex
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "model", "ppo_constrained.py")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

SEEDS = (1, 2, 3)
STEPS = 2_500_000
COOLDOWN = 150

DIALS = ("slow-d", "crowd-d", "succ-d")
DEFAULTS = {"slow-d": 0.75, "crowd-d": 0.10, "succ-d": 0.40}
OFF = {"slow-d": 1.0, "crowd-d": 1.0, "succ-d": 0.0}

ARMS = {
    "slow": {"sweep": ("slow-d", (0.85, 0.75, 0.65, 0.55, 0.45)),
             "hold": {"crowd-d": 0.10, "succ-d": 0.40},
             "prefix": "sweep_slow"},
    "crowd": {"sweep": ("crowd-d", (0.15, 0.10, 0.05)),
              "hold": {"slow-d": 0.75, "succ-d": 0.40},
              "prefix": "sweep_crowd"},
}


def parse_hold(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--hold takes name=value, got %r" % item)
        name, value = item.split("=", 1)
        name = name.lstrip("-")
        if name not in DIALS:
            raise SystemExit("unknown threshold %r, pick from %s"
                             % (name, ", ".join(DIALS)))
        out[name] = float(value)
    return out


def run_name(prefix, dial, value, seed, suffix):
    # sweep_crowd15_s3_5m for a swept dial, base_slow75_s3_5m without one
    stem = prefix if dial is None else "%s%02d" % (prefix, round(value * 100))
    return "%s_s%d%s" % (stem, seed, suffix)


def thresholds_for(dial, value, held):
    # all three are always passed explicitly, so config.json records the whole
    # triple and a rerun cannot inherit a drifted Config default
    out = dict(DEFAULTS)
    out.update(held)
    if dial is not None:
        out[dial] = value
    return out


def build_cmd(name, seed, steps, thresholds, log_every, extra):
    cmd = [sys.executable, SCRIPT,
           "--total-steps", str(steps),
           "--seed", str(seed),
           "--run-name", name,
           "--log-every", str(log_every)]
    for dial in DIALS:
        cmd += ["--" + dial, str(thresholds[dial])]
    return cmd + extra


def main():
    ap = argparse.ArgumentParser(
        description="Drive ppo_constrained.py over seeds and thresholds. "
                    "Give --sweep to move one dial, omit it for a fixed-config "
                    "run repeated across seeds.")
    ap.add_argument("--arm", choices=sorted(ARMS), default=None,
                    help="preset defaults for --sweep/--hold/--prefix")
    ap.add_argument("--sweep", nargs="+", metavar="DIAL VALUE",
                    help="threshold to move and its grid, e.g. "
                         "--sweep crowd-d 0.05 0.10 0.15")
    ap.add_argument("--hold", nargs="+", metavar="NAME=VALUE", default=[],
                    help="pinned thresholds, e.g. --hold slow-d=1.0 succ-d=0.40; "
                         "1.0 switches an upper bound off, 0.0 the success floor")
    ap.add_argument("--prefix", default=None, help="run-name stem")
    ap.add_argument("--suffix", default="", help="appended to every run name")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--extra", default="",
                    help="extra args passed straight to ppo_constrained.py, "
                         "quoted, e.g. --extra \"--n-envs 8 --lr 1e-4\"")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run and exit")
    args = ap.parse_args()

    preset = ARMS[args.arm] if args.arm else {}
    held = dict(preset.get("hold", {}))
    held.update(parse_hold(args.hold))
    prefix = args.prefix or preset.get("prefix")
    extra = shlex.split(args.extra)

    if args.sweep:
        dial, values = args.sweep[0].lstrip("-"), [float(v) for v in args.sweep[1:]]
        if dial not in DIALS:
            raise SystemExit("unknown threshold %r, pick from %s"
                             % (dial, ", ".join(DIALS)))
        if not values:
            raise SystemExit("--sweep needs at least one value after the dial")
    elif preset.get("sweep"):
        dial, values = preset["sweep"][0], list(preset["sweep"][1])
    else:
        dial, values = None, [None]
    if dial in held:
        raise SystemExit("%s is both swept and held" % dial)
    if prefix is None:
        raise SystemExit("give --prefix (or --arm) to name the runs")

    jobs = [(v, s) for s in args.seeds for v in values]
    t0 = time.perf_counter()

    shown = thresholds_for(dial, values[0], held)
    print("%s | %s | seeds %s | %d steps each | %d runs"
          % ("sweeping --%s over %s" % (dial, " ".join("%g" % v for v in values))
             if dial else "fixed config",
             " ".join("%s %g%s" % (d, shown[d], " (swept)" if d == dial else "")
                      for d in DIALS),
             " ".join(str(s) for s in args.seeds), args.steps, len(jobs)),
          flush=True)

    for i, (value, seed) in enumerate(jobs, 1):
        name = run_name(prefix, dial, value, seed, args.suffix)
        thresholds = thresholds_for(dial, value, held)
        cmd = build_cmd(name, seed, args.steps, thresholds, args.log_every, extra)
        hours = (time.perf_counter() - t0) / 3600.0

        if args.dry_run:
            print("[%2d/%d] %s\n        %s"
                  % (i, len(jobs), name, " ".join(cmd[1:])))
            continue

        if os.path.exists(os.path.join(RUNS_DIR, name, "final.pt")):
            print("[%2d/%d] %s done already, skipping" % (i, len(jobs), name),
                  flush=True)
            continue

        print("[%2d/%d] %s | %s | seed %d | %.1f h elapsed"
              % (i, len(jobs), name,
                 " ".join("%s %g" % (d, thresholds[d]) for d in DIALS),
                 seed, hours), flush=True)

        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            print("[%2d/%d] %s FAILED rc=%d" % (i, len(jobs), name, rc),
                  flush=True)

        if i < len(jobs):
            # let the laptop's CPU/GPU thermals settle so later arms are not throttled
            time.sleep(args.cooldown)

    if not args.dry_run:
        print("sweep finished in %.1f h" % ((time.perf_counter() - t0) / 3600.0),
              flush=True)


if __name__ == "__main__":
    main()
