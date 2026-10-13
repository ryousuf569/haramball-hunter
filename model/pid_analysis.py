import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import csv
import glob
import io
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.constrained_analysis import boot_step, tail_mean, write_csv

ASSETS = os.path.join(REPO_ROOT, "assets")
BLUE = "#1f4e9c"
RED = "#d62728"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GREY = "#777777"

SEED_STYLES = (("-", BLUE), ("--", ORANGE), (":", GREEN), ("-.", RED),
               ("-", GREY))

# the 2x2 of model/ppo_pid.py: (label, run-name glob, what slow counts). The two
# adam arms are the existing abl_* runs, so only the pid arms are new.
ARMS = (
    ("adam x slow", "abl_control_s*_25m", "attacker-tick"),
    ("adam x team_slow", "abl_team_slow_s*_25m", "team-tick"),
    ("pid x slow", "pid_control_s*_25m", "attacker-tick"),
    ("pid x team_slow", "pid_team_slow_s*_25m", "team-tick"),
)
FIG_ARMS = ("adam x team_slow", "pid x team_slow")

# printed and copied to the out dir before any number, every invocation
PREREG_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pid_prereg.md")

def read_rows(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not (os.path.exists(cfg_path) and os.path.exists(csv_path)):
        return None, None
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return cfg, (rows or None)


def column(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if v not in (None, "") else float("nan"))
    return np.array(out)


def lam0_eff_column(rows):
    # the abl_* runs predate the column, so rebuild it from its definition
    if rows[0].get("lam0_eff") not in (None, ""):
        return column(rows, "lam0_eff")
    return np.maximum(column(rows, "lam0"), column(rows, "lam_succ"))


def tail_std(rows, key, frac):
    take = rows[-max(1, int(len(rows) * frac)):]
    v = [float(r[key]) for r in take if r.get(key) not in (None, "")]
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def crossings(rows, key, d):
    # read next to tail_std: a settled controller crosses often with small
    # amplitude, a ringing one rarely with large amplitude
    v = column(rows, key)
    v = v[~np.isnan(v)] - d
    sign = np.sign(v)
    sign = sign[sign != 0]
    return int(np.sum(sign[1:] != sign[:-1]))


def load_probe(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["run"]: {"p_c_slow": float(r["c_slow"]),
                           "p_c_team": float(r["c_team"]),
                           "p_success": float(r["success"])}
                for r in csv.DictReader(fh)}


def load(frac, probe):
    out = []
    for arm, pattern, unit in ARMS:
        for d in sorted(glob.glob(os.path.join(REPO_ROOT, "runs", pattern))):
            cfg, rows = read_rows(d)
            if rows is None:
                continue
            name = os.path.basename(d)
            d_slow = float(cfg.get("slow_d", float("nan")))
            lam_slow = column(rows, "lam_slow")
            eff = lam0_eff_column(rows)
            n_tail = max(1, int(len(rows) * frac))
            row = {
                "run": name, "arm": arm, "unit": unit,
                "seed": int(cfg.get("seed", -1)),
                "d_slow": d_slow,
                "n_points": len(rows),
                "final_step": float(rows[-1]["step"]),
                "rate": tail_mean(rows, "c_slow", frac),
                "rate_sd": tail_std(rows, "c_slow", frac),
                "c_team": tail_mean(rows, "c_team", frac),
                "crossings": crossings(rows, "c_slow", d_slow),
                "max_lam_slow": float(np.nanmax(lam_slow)),
                "n_lam_gt_half": int(np.sum(lam_slow > 0.5)),
                "boot_step": boot_step(rows),
                "lam0": tail_mean(rows, "lam0", frac),
                "lam0_eff": float(np.nanmean(eff[-n_tail:])),
                "lam_succ": tail_mean(rows, "lam_succ", frac),
                "success": tail_mean(rows, "success", frac),
                "cadv_std_slow": tail_mean(rows, "cadv_std_slow", frac),
                "cadv_std_crowd": tail_mean(rows, "cadv_std_crowd", frac),
                "cadv_std_succ": tail_mean(rows, "cadv_std_succ", frac),
                "mult_lr": float(cfg.get("mult_lr", float("nan"))),
            }
            row["abs_err"] = abs(row["rate"] - d_slow)
            for g in ("kp", "ki", "kd", "i_max", "i_floor", "d_ema"):
                row[g] = float(cfg.get(g, float("nan")))
            row.update(probe.get(name, {"p_c_slow": float("nan"),
                                        "p_c_team": float("nan"),
                                        "p_success": float("nan")}))
            out.append(row)
    return out


def score(runs, arm, success_ref, tol=0.05):
    lo, hi = success_ref
    out = []
    for r in sorted((x for x in runs if x["arm"] == arm), key=lambda x: x["seed"]):
        c = {
            "tracks": bool(r["abs_err"] <= tol),
            "no_swing": r["n_lam_gt_half"] == 0,
            "handover": bool(np.isfinite(r["boot_step"])),
            "success_ok": bool(np.isfinite(r["success"])
                               and lo - 1e-9 <= r["success"] <= hi + 1e-9),
        }
        c["all"] = all(c.values())
        out.append((r, c))
    return out


def verdict(scored, n_needed=2):
    n_pass = sum(1 for _, c in scored if c["all"])
    if not scored:
        return "NO RUNS", "pid x team_slow has not been run yet"
    if n_pass >= n_needed:
        return ("UPDATE RULE",
                "%d of %d seeds met all four criteria, so on this evidence the "
                "section 6.4 failure is the multiplier update rule and not the "
                "counting unit" % (n_pass, len(scored)))
    swings = sum(1 for _, c in scored if not c["no_swing"])
    no_hand = sum(1 for _, c in scored if not c["handover"])
    if swings == len(scored) and no_hand == len(scored):
        return ("COUNTING UNIT",
                "lam_slow still swings above 0.5 and the bootstrap never hands "
                "over in all %d seeds, so the counting unit is load-bearing for "
                "this optimizer family" % len(scored))
    return ("MIXED",
            "%d of %d seeds met all four criteria; swing in %d, no handover in "
            "%d. Report as mixed, claim nothing either way"
            % (n_pass, len(scored), swings, no_hand))


def plot(runs, out_path, d_slow=0.75):
    # same layout as paper Fig 5: rate against steps with the target dashed in,
    # one column per arm, seeds overlaid. lam_slow shares the x axis below.
    fig, axes = plt.subplots(2, len(FIG_ARMS), figsize=(11.5, 7.0),
                             sharex=True, sharey="row")
    units = {a: u for a, _, u in ARMS}

    for col, arm in enumerate(FIG_ARMS):
        group = sorted((r for r in runs if r["arm"] == arm),
                       key=lambda r: r["seed"])
        top, bot = axes[0][col], axes[1][col]

        for r, (ls, colour) in zip(group, SEED_STYLES):
            cfg, rows = read_rows(os.path.join(REPO_ROOT, "runs", r["run"]))
            if rows is None:
                continue
            x = column(rows, "step") / 1e6
            top.plot(x, column(rows, "c_slow"), ls, color=colour, lw=1.4,
                     alpha=0.9, label="seed %d" % r["seed"])
            bot.plot(x, column(rows, "lam_slow"), ls, color=colour, lw=1.4,
                     alpha=0.9, label="seed %d" % r["seed"])

        top.axhline(d_slow, color=RED, ls="--", lw=1.4)
        top.text(0.02, d_slow + 0.02, "target %.2f" % d_slow, fontsize=8,
                 color=RED)
        top.set_ylim(0, 1.05)
        top.set_title("%s  (slow counted per %s)" % (arm, units[arm]),
                      loc="left", fontsize=10)
        top.grid(alpha=0.25)
        if group:
            top.legend(fontsize=8, frameon=False, ncol=len(group))

        bot.axhline(0.5, color=RED, ls="--", lw=1.2)
        bot.text(0.02, 0.52, "swing threshold 0.5", fontsize=8, color=RED)
        bot.set_ylim(-0.03, 1.03)
        bot.set_xlabel("steps (M)")
        bot.grid(alpha=0.25)
        if not group:
            for ax in (top, bot):
                ax.text(0.5, 0.5, "no runs matched", ha="center", va="center",
                        transform=ax.transAxes, color=GREY)

    axes[0][0].set_ylabel("achieved slow rate")
    axes[1][0].set_ylabel("lambda_slow")
    fig.suptitle("Section 6.4 follow-up: counting unit held at team-mean, "
                 "multiplier update rule swapped")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


RUN_FMT = ("%-24s %4d %7.3f %7.3f %7.3f %5d %8.3f %5d %7s %8.3f %8.3f"
           " | %7s %7s %7s")
RUN_HEAD = ("%-24s %4s %7s %7s %7s %5s %8s %5s %7s %8s %8s | %7s %7s %7s"
            % ("run", "seed", "rate", "|err|", "tailsd", "xing", "maxlam",
               ">0.5", "bootM", "lam0eff", "succ", "pc_slow", "pc_team",
               "p_succ"))


def fmt(v, spec="%7.3f"):
    return "    nan" if v != v else spec % v


def print_runs(runs, arm):
    group = sorted((r for r in runs if r["arm"] == arm), key=lambda r: r["seed"])
    if not group:
        print("  (no runs matched %s)" % next(p for a, p, _ in ARMS if a == arm))
        return
    for r in group:
        print(RUN_FMT % (r["run"], r["seed"], r["rate"], r["abs_err"],
                         r["rate_sd"], r["crossings"], r["max_lam_slow"],
                         r["n_lam_gt_half"],
                         "   none" if not np.isfinite(r["boot_step"])
                         else "%7.2f" % (r["boot_step"] / 1e6),
                         r["lam0_eff"], r["success"],
                         fmt(r["p_c_slow"]), fmt(r["p_c_team"]),
                         fmt(r["p_success"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=float, default=0.20,
                    help="fraction of logged points in the tail window")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="pre-registered |rate - d| bar")
    ap.add_argument("--probe-csv",
                    default=os.path.join(ASSETS, "pid_speed_probe.csv"),
                    help="frozen-probe output from model/probe/speed_probe.py, "
                         "joined on run name; missing rows come out nan")
    ap.add_argument("--success-ref", choices=("adam x slow", "pid x slow",
                                              "both"),
                    default="adam x slow",
                    help="which control arm's seed range criterion 4 uses")
    ap.add_argument("--out-dir", default=ASSETS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # the reading goes out before a single number, and to disk
    prereg = io.open(PREREG_SRC, encoding="utf-8").read()
    prereg_path = os.path.join(args.out_dir, "pid_prereg.md")
    with io.open(prereg_path, "w", encoding="utf-8") as fh:
        fh.write(prereg)
    print(prereg)
    print("(the above is %s, copied to %s)"
          % (os.path.relpath(PREREG_SRC, REPO_ROOT), prereg_path))
    print()
    print("=" * 118)
    print()

    probe = load_probe(args.probe_csv)
    if not probe:
        print("no frozen probe at %s; the p_* columns will be nan. Fill them "
              "with:" % args.probe_csv)
        print("  python model/probe/speed_probe.py runs/abl_control_s*_25m "
              "runs/abl_team_slow_s*_25m runs/pid_control_s*_25m "
              "runs/pid_team_slow_s*_25m --tag pid_speed_probe")
        print()

    runs = load(args.tail, probe)
    if not runs:
        raise SystemExit("no runs matched any arm pattern under runs/")

    print("tail = last %.0f%% of logged points | rate is c_slow, i.e. whatever "
          "the arm's indicator counts" % (100 * args.tail))
    print("xing = threshold crossings over the whole trace | >0.5 = logged "
          "points with lam_slow > 0.5 | p_* = frozen probe")
    print()

    for arm, pattern, unit in ARMS:
        print("-- %s  (slow counted per %s, runs/%s)" % (arm, unit, pattern))
        print(RUN_HEAD)
        print_runs(runs, arm)
        print()

    refs = {}
    for arm in ("adam x slow", "pid x slow"):
        v = np.array([r["success"] for r in runs if r["arm"] == arm],
                     dtype=float)
        v = v[~np.isnan(v)]
        refs[arm] = (float(v.min()), float(v.max())) if v.size else (np.nan,
                                                                     np.nan)
    print("control tail-success seed ranges")
    for arm, (lo, hi) in refs.items():
        seeds = [(r["seed"], r["success"]) for r in runs if r["arm"] == arm]
        print("  %-16s %s   -> range [%s, %s]"
              % (arm, " ".join("s%d %.3f" % s for s in sorted(seeds)) or "none",
                 fmt(lo, "%.3f"), fmt(hi, "%.3f")))
    if args.success_ref == "both":
        lo = float(np.nanmin([refs[a][0] for a in refs]))
        hi = float(np.nanmax([refs[a][1] for a in refs]))
        ref, ref_name = (lo, hi), "both control arms"
    else:
        ref, ref_name = refs[args.success_ref], args.success_ref
    print("  criterion 4 uses %s: [%s, %s]"
          % (ref_name, fmt(ref[0], "%.3f"), fmt(ref[1], "%.3f")))
    print()

    print("pre-registered criteria, per seed, pid x team_slow")
    print("%-24s %4s %8s %9s %10s %11s %6s"
          % ("run", "seed", "tracks", "no_swing", "handover", "success_ok",
             "all"))
    scored = score(runs, "pid x team_slow", ref, args.tol)
    for r, c in scored:
        print("%-24s %4d %8s %9s %10s %11s %6s"
              % (r["run"], r["seed"],
                 "yes" if c["tracks"] else "no",
                 "yes" if c["no_swing"] else "no",
                 "yes" if c["handover"] else "no",
                 "yes" if c["success_ok"] else "no",
                 "PASS" if c["all"] else "fail"))
    if not scored:
        print("  (none yet)")
    print()

    tag, why = verdict(scored)
    print("VERDICT: %s" % tag)
    print("  %s." % why)
    print("  n = 3. No claim beyond n = 3.")
    print()

    # a near-constant cost channel has a tiny raw std and standardise() divides
    # it straight back out
    have = [r for r in runs if np.isfinite(r["cadv_std_slow"])]
    print("raw (pre-standardise) cost_adv std, tail mean")
    if have:
        print("%-24s %12s %12s %12s" % ("run", "slow", "crowd", "succ"))
        for r in sorted(have, key=lambda r: (r["arm"], r["seed"])):
            print("%-24s %12.3e %12.3e %12.3e"
                  % (r["run"], r["cadv_std_slow"], r["cadv_std_crowd"],
                     r["cadv_std_succ"]))
    else:
        print("  not logged in any matched run; only model/ppo_pid.py writes "
              "these columns, so this fills in once the pid_* arms exist")
    print()

    runs_csv = write_csv(os.path.join(args.out_dir, "pid_runs.csv"),
                         sorted(runs, key=lambda r: (r["arm"], r["seed"])))
    fig = plot(runs, os.path.join(args.out_dir, "pid_tracking.png"))
    print("wrote %s" % prereg_path)
    print("wrote %s" % runs_csv)
    print("wrote %s" % fig)


if __name__ == "__main__":
    main()
