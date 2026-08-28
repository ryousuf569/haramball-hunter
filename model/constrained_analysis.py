import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.join(REPO_ROOT, "assets")
BASELINE = 0.885
BLUE = "#1f4e9c"
LIGHT = "#8ab4e8"
RED = "#d62728"


def tail_mean(rows, key, frac):
    take = rows[-max(1, int(len(rows) * frac)):]
    v = [float(r[key]) for r in take if r.get(key) not in (None, "")]
    return float(np.mean(v)) if v else float("nan")


def boot_step(rows, min_step=50_000):
    for r in rows:
        if r.get("boot") in (None, ""):
            return float("nan")
        if float(r["boot"]) == 0.0 and float(r["step"]) >= min_step:
            return float(r["step"])
    return float("nan")


def load(pattern, frac):
    out = []
    for d in sorted(glob.glob(os.path.join(REPO_ROOT, "runs", pattern))):
        cfg_path = os.path.join(d, "config.json")
        csv_path = os.path.join(d, "metrics.csv")
        if not (os.path.exists(cfg_path) and os.path.exists(csv_path)):
            continue
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "slow_d" not in cfg:
            continue
        out.append({"run": os.path.basename(d),
                    "seed": int(cfg["seed"]),
                    "requested": float(cfg["slow_d"]),
                    "achieved": tail_mean(rows, "c_slow", frac),
                    "success": tail_mean(rows, "success", frac),
                    "speed": tail_mean(rows, "speed", frac),
                    "ep_len": tail_mean(rows, "len", frac),
                    "lam_slow": tail_mean(rows, "lam_slow", frac),
                    "lam_succ": tail_mean(rows, "lam_succ", frac),
                    "ev_slow": tail_mean(rows, "ev_slow", frac),
                    "boot_step": boot_step(rows),
                    "final_step": float(rows[-1]["step"])})
    return out


def regress(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def per_target(runs):
    out = []
    for d in sorted({r["requested"] for r in runs}):
        g = [r for r in runs if r["requested"] == d]
        row = {"requested": d, "n": len(g)}
        for k in ("achieved", "success", "speed", "ep_len", "lam_slow",
                  "lam_succ", "boot_step"):
            v = np.array([r[k] for r in g], dtype=float)
            v = v[~np.isnan(v)]
            row[k] = float(v.mean()) if v.size else float("nan")
            row[k + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        row["error"] = row["achieved"] - d
        row["feasible"] = float(np.mean([r["achieved"] <= d for r in g]))
        out.append(row)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def plot(runs, targets, out_path):
    x = np.array([r["requested"] for r in runs])
    y = np.array([r["achieved"] for r in runs])
    tx = np.array([t["requested"] for t in targets])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    a, b, c, d = axes.ravel()

    lo, hi = min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05
    a.plot([lo, hi], [lo, hi], "--", color=RED, lw=1.4, label="achieved = target")
    a.scatter(x, y, s=55, color=LIGHT, zorder=3, label="seeds")
    a.errorbar(tx, [t["achieved"] for t in targets],
               yerr=[t["achieved_sd"] for t in targets], fmt="o-",
               color=BLUE, lw=2, capsize=4, zorder=4, label="mean")
    a.axhline(BASELINE, color="grey", lw=1, ls=":")
    a.text(lo + 0.01, BASELINE + 0.005, "unconstrained %.3f" % BASELINE,
           fontsize=8, color="grey")
    a.set_xlabel("requested rate")
    a.set_ylabel("achieved rate")
    a.set_title("calibration", loc="left", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25)

    b.errorbar(tx, [t["success"] for t in targets],
               yerr=[t["success_sd"] for t in targets], fmt="o-",
               color=BLUE, lw=2, capsize=4)
    b.set_xlabel("requested rate")
    b.set_ylabel("success rate")
    b.set_title("price of the constraint", loc="left", fontsize=10)
    b.grid(alpha=0.25)

    c.errorbar(tx, [t["lam_slow"] for t in targets],
               yerr=[t["lam_slow_sd"] for t in targets], fmt="o-",
               color=BLUE, lw=2, capsize=4)
    c.set_xlabel("requested rate")
    c.set_ylabel("lambda_slow at convergence")
    c.set_title("multiplier vs strictness", loc="left", fontsize=10)
    c.grid(alpha=0.25)

    d.errorbar(tx, [t["boot_step"] / 1e6 for t in targets],
               yerr=[t["boot_step_sd"] / 1e6 for t in targets], fmt="o-",
               color=BLUE, lw=2, capsize=4)
    d.set_xlabel("requested rate")
    d.set_ylabel("bootstrap handover (M steps)")
    d.set_title("when lambda_0 overtakes lambda_boot", loc="left", fontsize=10)
    d.grid(alpha=0.25)

    fig.suptitle("Constrained PPO: %d runs, %d targets x %d seeds"
                 % (len(runs), len(targets), len(runs) // max(1, len(targets))))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="sweep_slow*_s*")
    ap.add_argument("--tail", type=float, default=0.20)
    ap.add_argument("--baseline", type=float, default=BASELINE)
    ap.add_argument("--out-dir", default=ASSETS)
    args = ap.parse_args()

    runs = load(args.pattern, args.tail)
    if not runs:
        raise SystemExit("no runs matched %r" % args.pattern)
    targets = per_target(runs)

    x = np.array([r["requested"] for r in runs])
    y = np.array([r["achieved"] for r in runs])
    err = y - x
    slope, intercept, r2 = regress(x, y)
    viol = np.clip(err, 0.0, None)

    os.makedirs(args.out_dir, exist_ok=True)

    print("%d runs | %d targets | tail %.0f%% of each run"
          % (len(runs), len(targets), 100 * args.tail))
    print()
    print("%-22s %5s %9s %9s %8s %8s %8s %7s"
          % ("run", "seed", "requested", "achieved", "error", "success", "lam_slw", "bootM"))
    for r in sorted(runs, key=lambda r: (r["requested"], r["seed"])):
        print("%-22s %5d %9.2f %9.3f %+8.3f %8.3f %8.4f %7.2f"
              % (r["run"], r["seed"], r["requested"], r["achieved"],
                 r["achieved"] - r["requested"], r["success"], r["lam_slow"],
                 r["boot_step"] / 1e6))

    print()
    print("%-10s %3s %9s %7s %8s %9s %8s %8s"
          % ("requested", "n", "achieved", "sd", "error", "feasible", "success", "speed"))
    for t in targets:
        print("%-10.2f %3d %9.3f %7.3f %+8.3f %9.2f %8.3f %8.2f"
              % (t["requested"], t["n"], t["achieved"], t["achieved_sd"],
                 t["error"], t["feasible"], t["success"], t["speed"]))

    print()
    print("tracking")
    print("  MAE                    %.4f  (%.2f pp)" % (np.abs(err).mean(), 100 * np.abs(err).mean()))
    print("  RMSE                   %.4f" % np.sqrt((err ** 2).mean()))
    print("  bias (mean signed)     %+.4f" % err.mean())
    print("  max abs error          %.4f  (%s)"
          % (np.abs(err).max(), runs[int(np.argmax(np.abs(err)))]["run"]))
    print("  slope                  %.4f" % slope)
    print("  intercept              %+.4f" % intercept)
    print("  R2                     %.4f" % r2)
    print()
    print("constraint satisfaction")
    print("  feasible runs          %d / %d  (%.0f%%)"
          % (int((y <= x).sum()), len(runs), 100 * (y <= x).mean()))
    print("  mean violation         %.4f" % viol.mean())
    print("  max violation          %.4f" % viol.max())
    print()
    feas = [r for r in runs if r["achieved"] <= r["requested"]]
    print("  success, all runs      %.3f" % np.mean([r["success"] for r in runs]))
    if feas:
        print("  success, feasible only %.3f" % np.mean([r["success"] for r in feas]))
    print("  seed sd of achieved    %.4f"
          % float(np.mean([t["achieved_sd"] for t in targets])))
    print()
    print("range")
    print("  unconstrained baseline %.3f" % args.baseline)
    print("  achieved span          %.3f -> %.3f  (%.1f pp of dial)"
          % (y.min(), y.max(), 100 * (y.max() - y.min())))

    write_csv(os.path.join(args.out_dir, "constraint_runs.csv"),
              sorted(runs, key=lambda r: (r["requested"], r["seed"])))
    write_csv(os.path.join(args.out_dir, "constraint_targets.csv"), targets)
    path = plot(runs, targets, os.path.join(args.out_dir, "constraint_analysis.png"))
    print()
    print("wrote %s" % os.path.join(args.out_dir, "constraint_runs.csv"))
    print("wrote %s" % os.path.join(args.out_dir, "constraint_targets.csv"))
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
