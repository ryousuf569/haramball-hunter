import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import defenders.turnover as turnover
import environment.lowblock_env as lowblock_env
from environment.lowblock_env import make_initial_world, make_ppcf_grid, step
from physics.engine import DT, ball_action
from attackers.baseline_attacker import compute_attacker_targets
from physics.tti import intercept_probability_vec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1.0 is a control: unreachable threshold, so it isolates the duel's contribution
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
# ~50 episodes per threshold: 13 seeds x 4 start_holders = 52. Every seed now
# reaches the turnover RNG (see make_defender_state(seed=...)), so these are 52
# distinct trajectories -- previously the seed was inert and each threshold cell
# was a single trajectory replayed 20 times.
SEEDS = list(range(13))
START_HOLDERS = [0, 4, 7, 9]
MAX_TICKS = 1200

# ---------------------------------------------------------------------------
# Cause attribution. Inferring "intercept" from prev_state == "in_flight" merges
# offsides into the duel bucket and mislabels a duel that lands on a flight tick.
# Wrapping the two model functions instead attributes every flip exactly, and
# lets us log the interception probabilities that were actually reached -- which
# is what justifies a threshold rather than just counting outcomes.
# ---------------------------------------------------------------------------

_duel = turnover.ground_duel
_intercept = turnover.intercept_pass

_ep = {}

def _reset_episode_probe():
    _ep.clear()
    _ep.update(duel_win=0, intercept_win=0, in_range_ticks=0, p_samples=[])

def _duel_probe(players, ball, rng, holder_idx, dt=0.1):
    w = _duel(players, ball, rng, holder_idx, dt)
    if w is not None:
        _ep["duel_win"] += 1
    return w

def _intercept_probe(players, ball, rng, prev_pos, dt=0.1):
    # Recompute the gate's geometry to log p on ticks where a defender is in
    # range. Mirrors turnover.intercept_pass; kept in sync by construction since
    # it reads the same module constants.
    if ball["state"] == "in_flight":
        lob = ball["flight_target"] - ball["flight_start"]
        if np.sqrt(lob @ lob) <= turnover.LOB_DIST:
            dmask = players["team"] == "defender"
            dpos = players["position"][dmask]
            a = np.asarray(prev_pos, dtype=float)
            ab = np.asarray(ball["position"], dtype=float) - a
            L2 = ab @ ab
            if L2 > turnover.EPS:
                t = np.clip(((dpos - a) @ ab) / L2, 0.0, 1.0)
                closest = a + t[:, None] * ab
            else:
                closest = a
            delta = dpos - closest
            sq = np.einsum("ij,ij->i", delta, delta)
            near = sq <= turnover.KICKABLE_AREA * turnover.KICKABLE_AREA
            if near.any():
                _ep["in_range_ticks"] += 1
                to_target = ball["flight_target"] - ball["position"]
                T = np.sqrt(to_target @ to_target) / turnover.BALL_SPEED
                p = intercept_probability_vec(T, players["i_p"][dmask][near])
                _ep["p_samples"].append(float(np.max(p)))

    w = _intercept(players, ball, rng, prev_pos, dt)
    if w is not None:
        _ep["intercept_win"] += 1
    return w

# step() resolved these names at import time, so patch the env module's globals
lowblock_env.ground_duel = _duel_probe
lowblock_env.intercept_pass = _intercept_probe

def run_episode(seed, start_holder, max_ticks):
    players, ball, attacker_ids, _rng, dstate = make_initial_world(
        seed=seed, start_holder=start_holder)
    ppcf_grid = make_ppcf_grid()
    att_ids = set(int(i) for i in attacker_ids)
    _reset_episode_probe()

    n_passes = 0
    flight_ticks = 0
    min_def_dist = np.inf

    for tick in range(1, max_ticks + 1):
        # peek before stepping so a pass on the final tick still counts
        _tv, ball_idx = compute_attacker_targets(players, ball, tick)
        _h, is_pass, _t = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)
        if is_pass:
            n_passes += 1

        players, ball, _pc, _out = step(players, ball, attacker_ids, dstate, tick,
                                        ppcf_grid=ppcf_grid, exit_on_turnover=False)

        if ball["state"] == "in_flight":
            flight_ticks += 1
            dmask = players["team"] == "defender"
            d = np.linalg.norm(players["position"][dmask] - ball["position"], axis=1)
            min_def_dist = min(min_def_dist, float(d.min()))

        holder = ball.get("holder_id")
        if holder is not None and int(holder) not in att_ids:
            if _ep["intercept_win"]:
                cause = "intercept"
            elif _ep["duel_win"]:
                cause = "duel"
            else:
                cause = "offside"
            return _finish(cause, 1, tick, n_passes, flight_ticks, min_def_dist)

    return _finish("timeout", 0, max_ticks, n_passes, flight_ticks, min_def_dist)

def _finish(cause, ended, end_tick, n_passes, flight_ticks, min_def_dist):
    ps = _ep["p_samples"]
    return {"ended": ended, "end_tick": end_tick, "cause": cause,
            "n_passes": n_passes, "flight_ticks": flight_ticks,
            "min_def_dist_m": min_def_dist,
            # interception diagnostics: how often the geometric gate opened, and
            # the best control probability seen while it was open
            "in_range_ticks": _ep["in_range_ticks"],
            "max_p": max(ps) if ps else np.nan,
            "median_p": float(np.median(ps)) if ps else np.nan}

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
              "n_passes", "flight_ticks", "min_def_dist_m",
              "in_range_ticks", "max_p", "median_p"]
    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in ("min_def_dist_m", "max_p", "median_p"):
                v = out[k]
                out[k] = round(v, 4) if np.isfinite(v) else ""
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
            (tag + "_duel_episodes", sum(1 for r in sub if r["cause"] == "duel")),
            (tag + "_offside_episodes", sum(1 for r in sub if r["cause"] == "offside")),
            (tag + "_timeout_episodes", sum(1 for r in sub if r["cause"] == "timeout")),
            (tag + "_intercept_frac_of_episodes", round(n_int / n, 4) if n else ""),
            (tag + "_passes_total", passes),
            (tag + "_intercepts_per_pass", round(n_int / passes, 4) if passes else ""),
            (tag + "_median_end_tick", int(np.median(ticks)) if ticks else ""),
            (tag + "_median_episode_s", round(float(np.median(ticks)) * DT, 2) if ticks else ""),
        ]

    # Reachability. The threshold only accepts or rejects; the probabilities
    # themselves come from the physics and don't move with it. So the set of
    # achievable p bounds which thresholds can EVER fire -- a value above the
    # max is unreachable by construction, no matter what the outcome mix shows.
    # Sampled from the thr=1.0 control, where no interception fires and episodes
    # therefore run their full course instead of being truncated by an early win.
    ctrl = [r for r in rows if r["threshold"] == 1.0]
    ctrl_p = np.array([r["max_p"] for r in ctrl if np.isfinite(r["max_p"])])
    if ctrl_p.size:
        summary += [
            ("ctrl_episodes_with_in_range_tick", int(ctrl_p.size)),
            ("ctrl_episodes_total", len(ctrl)),
            ("ctrl_p_max", round(float(ctrl_p.max()), 4)),
            ("ctrl_p_p90", round(float(np.percentile(ctrl_p, 90)), 4)),
            ("ctrl_p_median", round(float(np.median(ctrl_p)), 4)),
        ]
        for t in THRESHOLDS[:-1]:
            summary.append(("ctrl_episodes_best_p_above_%.2f" % t,
                            int((ctrl_p > t).sum())))

    summary_path = os.path.join(OUT_DIR, "intercept_calibration_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary)
    print("wrote", summary_path)

    print("\n%-6s %10s %6s %8s %8s %8s %9s" % (
        "thr", "intercept", "duel", "offside", "timeout", "passes", "median_s"))
    for thr in THRESHOLDS:
        sub = [r for r in rows if r["threshold"] == thr]
        ticks = [r["end_tick"] for r in sub if r["ended"]]
        print("%-6.2f %10d %6d %8d %8d %8d %9s" % (
            thr,
            sum(1 for r in sub if r["cause"] == "intercept"),
            sum(1 for r in sub if r["cause"] == "duel"),
            sum(1 for r in sub if r["cause"] == "offside"),
            sum(1 for r in sub if r["cause"] == "timeout"),
            sum(r["n_passes"] for r in sub),
            ("%.1f" % (float(np.median(ticks)) * DT)) if ticks else "-"))

    if ctrl_p.size:
        print("\nachievable control probability (thr=1.0 control, %d/%d episodes "
              "had an in-range tick):" % (ctrl_p.size, len(ctrl)))
        print("  median %.4f   p90 %.4f   max %.4f" % (
            float(np.median(ctrl_p)), float(np.percentile(ctrl_p, 90)),
            float(ctrl_p.max())))
        print("  episodes whose best p clears a threshold of:")
        for t in THRESHOLDS[:-1]:
            print("    %.2f -> %3d/%d" % (t, int((ctrl_p > t).sum()), ctrl_p.size))
    else:
        print("\nno in-range interception ticks observed at all")

    make_plots(rows, ctrl_p)


def make_plots(rows, ctrl_p):
    """Three panels: outcome mix vs threshold, interception rate vs threshold,
    and the achievable-p distribution that bounds which thresholds can fire."""
    import matplotlib.pyplot as plt

    n_per = len(SEEDS) * len(START_HOLDERS)
    counts = {c: [] for c in ("intercept", "duel", "offside", "timeout")}
    per_pass = []
    for thr in THRESHOLDS:
        sub = [r for r in rows if r["threshold"] == thr]
        for c in counts:
            counts[c].append(sum(1 for r in sub if r["cause"] == c))
        passes = sum(r["n_passes"] for r in sub)
        n_int = sum(1 for r in sub if r["cause"] == "intercept")
        per_pass.append(n_int / passes if passes else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    x = np.arange(len(THRESHOLDS))
    labels = ["%.2f" % t for t in THRESHOLDS]

    # (1) outcome mix -- how the episode termination splits at each threshold
    ax = axes[0]
    bottom = np.zeros(len(THRESHOLDS))
    for c, color in (("intercept", "#2b7bba"), ("duel", "#d1495b"),
                     ("offside", "#edae49"), ("timeout", "#8d99ae")):
        v = np.array(counts[c], dtype=float)
        ax.bar(x, v, bottom=bottom, label=c, color=color, width=0.68)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("INTERCEPT_P_MIN"); ax.set_ylabel("episodes (n=%d)" % n_per)
    ax.set_title("Episode outcome mix vs threshold")
    ax.legend(fontsize=8, frameon=False)

    # (2) interception rate, normalised per pass rather than per episode
    ax = axes[1]
    ax.plot(x, per_pass, "o-", color="#2b7bba")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("INTERCEPT_P_MIN"); ax.set_ylabel("interceptions per pass")
    ax.set_title("Interception rate vs threshold")
    ax.grid(alpha=0.3)

    # (3) achievable p -- the ceiling on any usable threshold
    ax = axes[2]
    if ctrl_p.size:
        s = np.sort(ctrl_p)
        frac_above = 1.0 - np.arange(s.size) / s.size
        ax.step(s, frac_above, where="post", color="#d1495b")
        ax.axvline(float(s.max()), ls="--", lw=1, color="k",
                   label="max achievable = %.3f" % s.max())
        ax.set_xlabel("best in-range control probability per episode")
        ax.set_ylabel("fraction of episodes above")
        ax.set_title("Achievable p (thr=1.0 control)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no in-range ticks observed",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Achievable p (thr=1.0 control)")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "intercept_calibration.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

if __name__ == "__main__":
    main()
