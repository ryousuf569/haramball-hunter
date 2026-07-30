import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import defenders.turnover as turnover
from render import make_initial_world, make_ppcf_grid, step
from engine import DT, ball_action
from attackers.baseline_attacker import compute_attacker_targets

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1.0 is a control: unreachable threshold, so it isolates the duel's contribution
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
SEEDS = [0, 1, 2, 3, 4]
START_HOLDERS = [0, 4, 7, 9]
MAX_TICKS = 1200

def run_episode(seed, start_holder, max_ticks):
    players, ball, attacker_ids, _rng, dstate = make_initial_world(
        seed=seed, start_holder=start_holder)
    ppcf_grid = make_ppcf_grid()
    att_ids = set(int(i) for i in attacker_ids)

    n_passes = 0
    flight_ticks = 0
    min_def_dist = np.inf

    for tick in range(1, max_ticks + 1):
        # peek before stepping so a pass on the final tick still counts
        _tv, ball_idx = compute_attacker_targets(players, ball, tick)
        _h, is_pass, _t = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)
        if is_pass:
            n_passes += 1

        prev_state = ball["state"]
        players, ball, _pc = step(players, ball, attacker_ids, dstate, tick,
                                  ppcf_grid=ppcf_grid, exit_on_turnover=False)

        if ball["state"] == "in_flight":
            flight_ticks += 1
            dmask = players["team"] == "defender"
            d = np.linalg.norm(players["position"][dmask] - ball["position"], axis=1)
            min_def_dist = min(min_def_dist, float(d.min()))

        holder = ball.get("holder_id")
        if holder is not None and int(holder) not in att_ids:
            # airborne when it flipped -> interception; otherwise duel or offside
            cause = "intercept" if prev_state == "in_flight" else "duel_or_offside"
            return {"ended": 1, "end_tick": tick, "cause": cause,
                    "n_passes": n_passes, "flight_ticks": flight_ticks,
                    "min_def_dist_m": min_def_dist}

    return {"ended": 0, "end_tick": max_ticks, "cause": "timeout",
            "n_passes": n_passes, "flight_ticks": flight_ticks,
            "min_def_dist_m": min_def_dist}

def main():
    rows = []
    n_cells = len(THRESHOLDS) * len(SEEDS) * len(START_HOLDERS)
    done = 0

    for thr in THRESHOLDS:
        turnover.INTERCEPT_P_MIN = thr
        for seed in SEEDS:
            for holder in START_HOLDERS:
                r = run_episode(seed, holder, MAX_TICKS)
                r.update(threshold=thr, seed=seed, start_holder=holder)
                rows.append(r)
                done += 1
                d = r["min_def_dist_m"]
                print("[%3d/%3d] thr=%.2f seed=%d holder=%d -> %-18s tick %4d (%d passes, %s)"
                      % (done, n_cells, thr, seed, holder, r["cause"], r["end_tick"],
                         r["n_passes"], "%.2fm" % d if np.isfinite(d) else "no flight"))

    runs_path = os.path.join(OUT_DIR, "intercept_calibration_runs.csv")
    fields = ["threshold", "seed", "start_holder", "cause", "end_tick", "ended",
              "n_passes", "flight_ticks", "min_def_dist_m"]
    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            d = out["min_def_dist_m"]
            out["min_def_dist_m"] = round(d, 4) if np.isfinite(d) else ""
            w.writerow({k: out[k] for k in fields})
    print("\nwrote", runs_path)

    summary = [
        ("n_thresholds", len(THRESHOLDS)),
        ("episodes_per_threshold", len(SEEDS) * len(START_HOLDERS)),
        ("max_ticks", MAX_TICKS),
        ("dt_s", DT),
        ("kickable_area_m", turnover.KICKABLE_AREA),
        ("ball_speed_m_s", turnover.BALL_SPEED),
        ("lam_max_duel", turnover.LAM_MAX),
    ]
    for thr in THRESHOLDS:
        sub = [r for r in rows if r["threshold"] == thr]
        n = len(sub)
        n_int = sum(1 for r in sub if r["cause"] == "intercept")
        passes = sum(r["n_passes"] for r in sub)
        ticks = [r["end_tick"] for r in sub if r["ended"]]
        tag = "thr_%.2f" % thr
        summary += [
            (tag + "_intercept_episodes", n_int),
            (tag + "_duel_or_offside_episodes",
             sum(1 for r in sub if r["cause"] == "duel_or_offside")),
            (tag + "_timeout_episodes", sum(1 for r in sub if r["cause"] == "timeout")),
            (tag + "_intercept_frac_of_episodes", round(n_int / n, 4) if n else ""),
            (tag + "_passes_total", passes),
            (tag + "_intercepts_per_pass", round(n_int / passes, 4) if passes else ""),
            (tag + "_median_end_tick", int(np.median(ticks)) if ticks else ""),
            (tag + "_median_episode_s", round(float(np.median(ticks)) * DT, 2) if ticks else ""),
        ]

    summary_path = os.path.join(OUT_DIR, "intercept_calibration_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary)
    print("wrote", summary_path)

    print("\n%-6s %10s %10s %9s %8s %10s" % (
        "thr", "intercept", "duel/offs", "timeout", "passes", "median_s"))
    for thr in THRESHOLDS:
        sub = [r for r in rows if r["threshold"] == thr]
        ticks = [r["end_tick"] for r in sub if r["ended"]]
        print("%-6.2f %10d %10d %9d %8d %10s" % (
            thr,
            sum(1 for r in sub if r["cause"] == "intercept"),
            sum(1 for r in sub if r["cause"] == "duel_or_offside"),
            sum(1 for r in sub if r["cause"] == "timeout"),
            sum(r["n_passes"] for r in sub),
            ("%.1f" % (float(np.median(ticks)) * DT)) if ticks else "-"))

if __name__ == "__main__":
    main()
