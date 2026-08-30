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
BLUE = "#1f4e9c"
LIGHT = "#8ab4e8"
RED = "#d62728"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"

# Which dial the sweep moved. baseline is the unconstrained reference rate:
# 0.885 is measured c_slow, 0.029 is crowd_disc for vanilla_10m_cuda_rung2 in
# attackers/probe/behaviour_episodes.csv.
DIALS = {
    "slow": {"cfg": "slow_d", "rate": "c_slow", "lam": "lam_slow",
             "ev": "ev_slow", "pattern": "sweep_slow*_s*", "baseline": 0.885},
    "crowd": {"cfg": "crowd_d", "rate": "c_crowd", "lam": "lam_crowd",
              "ev": "ev_crowd_disc", "pattern": "sweep_crowd*_s*",
              "baseline": 0.029},
}

# every run reports all three, swept or held: (label, metrics.csv key,
# config.json key, is the threshold a lower bound)
RATES = (("slow", "c_slow", "slow_d", False),
         ("crowd", "c_crowd", "crowd_d", False),
         ("success", "success", "succ_d", True))


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


def satisfied(achieved, threshold, lower):
    if np.isnan(achieved) or np.isnan(threshold):
        return float("nan")
    return float(achieved >= threshold if lower else achieved <= threshold)


def load(pattern, frac, dial):
    d_spec = DIALS[dial]
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
        if not rows or d_spec["cfg"] not in cfg:
            continue
        row = {"run": os.path.basename(d),
               "seed": int(cfg["seed"]),
               "requested": float(cfg[d_spec["cfg"]]),
               "achieved": tail_mean(rows, d_spec["rate"], frac),
               "speed": tail_mean(rows, "speed", frac),
               "ep_len": tail_mean(rows, "len", frac),
               "lam_dial": tail_mean(rows, d_spec["lam"], frac),
               "lam_slow": tail_mean(rows, "lam_slow", frac),
               "lam_crowd": tail_mean(rows, "lam_crowd", frac),
               "lam_succ": tail_mean(rows, "lam_succ", frac),
               "ev_dial": tail_mean(rows, d_spec["ev"], frac),
               "boot_step": boot_step(rows),
               "final_step": float(rows[-1]["step"])}
        # all three constraint rates, whichever one the sweep moved
        for label, metric, key, lower in RATES:
            row[label] = tail_mean(rows, metric, frac)
            row["d_" + label] = float(cfg.get(key, float("nan")))
            row["ok_" + label] = satisfied(row[label], row["d_" + label], lower)
        row["ok_all"] = float(np.nanmin([row["ok_" + l] for l, _, _, _ in RATES]))
        out.append(row)
    return out


def regress(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def per_target(runs):
    keys = ("achieved", "speed", "ep_len", "lam_dial", "lam_slow", "lam_crowd",
            "lam_succ", "boot_step") + tuple(l for l, _, _, _ in RATES) \
           + tuple("ok_" + l for l, _, _, _ in RATES) + ("ok_all",)
    out = []
    for d in sorted({r["requested"] for r in runs}):
        g = [r for r in runs if r["requested"] == d]
        row = {"requested": d, "n": len(g)}
        for k in keys:
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


def plot(runs, targets, dial, baseline, out_path):
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
    if not np.isnan(baseline):
        a.axhline(baseline, color="grey", lw=1, ls=":")
        a.text(lo + 0.01, baseline + 0.005, "unconstrained %.3f" % baseline,
               fontsize=8, color="grey")
    a.set_xlabel("requested %s rate" % dial)
    a.set_ylabel("achieved %s rate" % dial)
    a.set_title("calibration", loc="left", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25)

    # all three rates, so a tightening dial that drags the other two off their
    # pinned targets is visible rather than inferred
    for label, colour in (("slow", BLUE), ("crowd", ORANGE), ("success", GREEN)):
        vals = np.array([t[label] for t in targets], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        b.errorbar(tx, vals, yerr=[t[label + "_sd"] for t in targets],
                   fmt="o-", color=colour, lw=2, capsize=4, label=label)
    for label, colour, key in (("slow", BLUE, "d_slow"),
                               ("crowd", ORANGE, "d_crowd"),
                               ("success", GREEN, "d_success")):
        held = {r[key] for r in runs if not np.isnan(r[key])}
        if len(held) == 1 and label != dial:
            b.axhline(held.pop(), color=colour, lw=1, ls=":")
    b.set_xlabel("requested %s rate" % dial)
    b.set_ylabel("achieved rate")
    b.set_ylim(0, 1.0)
    b.set_title("all three constraints (dotted = held target)", loc="left",
                fontsize=10)
    b.legend(fontsize=8, frameon=False)
    b.grid(alpha=0.25)

    c.errorbar(tx, [t["lam_dial"] for t in targets],
               yerr=[t["lam_dial_sd"] for t in targets], fmt="o-",
               color=BLUE, lw=2, capsize=4)
    c.set_xlabel("requested %s rate" % dial)
    c.set_ylabel("lambda_%s at convergence" % dial)
    c.set_title("multiplier vs strictness", loc="left", fontsize=10)
    c.grid(alpha=0.25)

    d.errorbar(tx, [t["boot_step"] / 1e6 for t in targets],
               yerr=[t["boot_step_sd"] / 1e6 for t in targets], fmt="o-",
               color=BLUE, lw=2, capsize=4)
    d.set_xlabel("requested %s rate" % dial)
    d.set_ylabel("bootstrap handover (M steps)")
    d.set_title("when lambda_0 overtakes lambda_boot", loc="left", fontsize=10)
    d.grid(alpha=0.25)

    fig.suptitle("Constrained PPO, %s dial: %d runs, %d targets x %d seeds"
                 % (dial, len(runs), len(targets),
                    len(runs) // max(1, len(targets))))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", choices=sorted(DIALS), default="crowd",
                    help="which threshold the sweep moved")
    ap.add_argument("--pattern", default=None,
                    help="defaults to the dial's run-name pattern")
    ap.add_argument("--tail", type=float, default=0.20)
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--out-dir", default=ASSETS)
    ap.add_argument("--tag", default=None,
                    help="suffix for the output files, defaults to the dial")
    args = ap.parse_args()

    spec = DIALS[args.dial]
    pattern = args.pattern or spec["pattern"]
    baseline = spec["baseline"] if args.baseline is None else args.baseline
    tag = args.tag or args.dial

    runs = load(pattern, args.tail, args.dial)
    if not runs:
        raise SystemExit("no runs matched %r" % pattern)
    targets = per_target(runs)

    x = np.array([r["requested"] for r in runs])
    y = np.array([r["achieved"] for r in runs])
    err = y - x
    slope, intercept, r2 = regress(x, y)
    viol = np.clip(err, 0.0, None)

    os.makedirs(args.out_dir, exist_ok=True)

    print("%d runs | %d targets | dial %s | tail %.0f%% of each run"
          % (len(runs), len(targets), args.dial, 100 * args.tail))
    print()
    print("%-22s %5s %9s %9s %8s | %7s %7s %7s | %4s %7s"
          % ("run", "seed", "requested", "achieved", "error",
             "slow", "crowd", "succ", "all", "bootM"))
    for r in sorted(runs, key=lambda r: (r["requested"], r["seed"])):
        print("%-22s %5d %9.2f %9.3f %+8.3f | %7.3f %7.3f %7.3f | %4s %7.2f"
              % (r["run"], r["seed"], r["requested"], r["achieved"],
                 r["achieved"] - r["requested"],
                 r["slow"], r["crowd"], r["success"],
                 "yes" if r["ok_all"] == 1.0 else "no", r["boot_step"] / 1e6))

    print()
    print("%-10s %3s %9s %7s %8s %9s | %7s %7s %7s"
          % ("requested", "n", "achieved", "sd", "error", "feasible",
             "slow", "crowd", "succ"))
    for t in targets:
        print("%-10.2f %3d %9.3f %7.3f %+8.3f %9.2f | %7.3f %7.3f %7.3f"
              % (t["requested"], t["n"], t["achieved"], t["achieved_sd"],
                 t["error"], t["feasible"], t["slow"], t["crowd"],
                 t["success"]))

    print()
    print("tracking (%s dial)" % args.dial)
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
    print("all three rates (the two held ones should not move with the dial)")
    for label, _, key, lower in RATES:
        vals = np.array([r[label] for r in runs], dtype=float)
        vals = vals[~np.isnan(vals)]
        ok = np.array([r["ok_" + label] for r in runs], dtype=float)
        ok = ok[~np.isnan(ok)]
        pinned = {r["d_" + label] for r in runs if not np.isnan(r["d_" + label])}
        target = "%.2f" % pinned.pop() if len(pinned) == 1 else "swept"
        print("  %-8s %s %-5s %-6s  achieved %5s +- %5s  |  satisfied in %d / %d runs"
              % (label, ">=" if lower else "<=", target,
                 "(dial)" if label == args.dial else "",
                 "%.3f" % vals.mean() if vals.size else "n/a",
                 "%.3f" % vals.std() if vals.size else "n/a",
                 int(ok.sum()), int(ok.size)))
    print("  all three at once                                          "
          "     %d / %d runs"
          % (int(sum(r["ok_all"] == 1.0 for r in runs)), len(runs)))
    print()
    feas = [r for r in runs if r["achieved"] <= r["requested"]]
    print("  success, all runs      %.3f" % np.mean([r["success"] for r in runs]))
    if feas:
        print("  success, feasible only %.3f" % np.mean([r["success"] for r in feas]))
    print("  seed sd of achieved    %.4f"
          % float(np.mean([t["achieved_sd"] for t in targets])))
    print()
    print("range")
    if not np.isnan(baseline):
        print("  unconstrained baseline %.3f" % baseline)
    print("  achieved span          %.3f -> %.3f  (%.1f pp of dial)"
          % (y.min(), y.max(), 100 * (y.max() - y.min())))

    runs_csv = os.path.join(args.out_dir, "constraint_runs_%s.csv" % tag)
    targets_csv = os.path.join(args.out_dir, "constraint_targets_%s.csv" % tag)
    write_csv(runs_csv, sorted(runs, key=lambda r: (r["requested"], r["seed"])))
    write_csv(targets_csv, targets)
    path = plot(runs, targets, args.dial, baseline,
                os.path.join(args.out_dir, "constraint_analysis_%s.png" % tag))
    print()
    print("wrote %s" % runs_csv)
    print("wrote %s" % targets_csv)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
