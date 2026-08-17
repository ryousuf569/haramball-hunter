import argparse
import csv
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from attackers.random_policy import (MAX_TICKS, make_initial_world,  # noqa: E402
                                     step)
from attackers.scripted_policy import make_policy                    # noqa: E402
from environment.grid import make_ppcf_grid                          # noqa: E402
from environment.termination import (ball_in_zone, make_zone,        # noqa: E402
                                     zone_control)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

X_VALUES = (82.0, 84.0, 86.0, 88.0)
Y_VALUES = (34.0,)   # off-centre is a different task, not a harder one
RADII = (6.0, 8.0, 10.0, 12.0)
THETAS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

N_EPISODES = 150
SEED0 = 1000

# The bands we are calibrating into
RANDOM_BAND = (0.05, 0.15)
SCRIPTED_BAND = (0.30, 0.50)

# A threshold zone_control can never reach, so success never terminates an
# episode and every disc can be scored after the fact off the same trajectory.
NEVER = 2.0


def targets_for(x_values, y_values):
    # [((x, y, r), Zone)] for every disc we want to score
    return [((x, y, r), make_zone(x, y, r))
            for x in x_values for y in y_values for r in RADII]


def run_episode(seed, targets, policy_zone, ppcf_grid):
    # max zone_control seen while an attacker held the ball inside each disc
    players, ball, aid, rng, dstate = make_initial_world(seed=seed)
    policy = None if policy_zone is None else make_policy(policy_zone)
    dummy = targets[0][1]
    best = {}

    for tick in range(1, MAX_TICKS + 1):
        actions = None if policy is None else policy(players, ball, aid)
        players, ball, pc_att, outcome = step(
            players, ball, aid, dstate, tick, rng, ppcf_grid=ppcf_grid,
            zone=dummy, max_ticks=MAX_TICKS, pc_min=NEVER, actions=actions)

        holder = ball.get("holder_id")
        held = (ball["state"] == "held" and holder is not None
                and bool(np.any(aid == holder)))
        # pc_att is None on the offside early-return
        if pc_att is not None and held:
            for key, z in targets:
                if ball_in_zone(ball["position"], z):
                    c = zone_control(pc_att, z)
                    if c > best.get(key, -1.0):
                        best[key] = c

        if outcome is not None:
            break
    return best


def sweep(policy_name, policy_zone, targets, n_episodes, ppcf_grid, rows):
    t0 = time.perf_counter()
    for e in range(n_episodes):
        best = run_episode(SEED0 + e, targets, policy_zone, ppcf_grid)
        for key, _z in targets:
            x, y, r = key
            c = best.get(key)
            rows.append({"policy": policy_name, "x": x, "y": y, "radius": r,
                         "episode": e, "arrived": int(c is not None),
                         "max_control": "" if c is None else round(c, 6)})
        if (e + 1) % 25 == 0:
            print(f"  {policy_name} {e + 1}/{n_episodes} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)


def summarise(rows):
    # (policy, x, y, r) -> list of max_control, None for never arrived
    grouped = {}
    for row in rows:
        key = (row["policy"], row["x"], row["y"], row["radius"])
        c = None if row["max_control"] == "" else float(row["max_control"])
        grouped.setdefault(key, []).append(c)

    out = []
    for (policy, x, y, r), vals in grouped.items():
        n = len(vals)
        arrived = [v for v in vals if v is not None]
        for theta in THETAS:
            hits = sum(1 for v in arrived if v >= theta)
            out.append({"policy": policy, "x": x, "y": y, "radius": r,
                        "theta": theta, "n": n,
                        "arrival_rate": round(len(arrived) / n, 4),
                        "success_rate": round(hits / n, 4)})
    return out


def choose(summary):
    rates = {}
    for s in summary:
        key = (s["x"], s["y"], s["radius"], s["theta"])
        rates.setdefault(key, {})[s["policy"]] = s["success_rate"]

    rmid = sum(RANDOM_BAND) / 2
    smid = sum(SCRIPTED_BAND) / 2
    out = []
    for key, by_policy in rates.items():
        rand = by_policy.get("random")
        scr = by_policy.get("scripted")
        if rand is None or scr is None:
            continue
        in_band = (RANDOM_BAND[0] <= rand <= RANDOM_BAND[1]
                   and SCRIPTED_BAND[0] <= scr <= SCRIPTED_BAND[1])
        x, y, r, theta = key
        out.append({"x": x, "y": y, "radius": r, "theta": theta,
                    "random_rate": rand, "scripted_rate": scr,
                    "in_band": int(in_band),
                    "score": round(abs(rand - rmid) + abs(scr - smid), 4)})
    # in-band first, then closest to the middle of both bands
    out.sort(key=lambda d: (-d["in_band"], d["score"]))
    return out


def write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    args = ap.parse_args()

    ppcf_grid = make_ppcf_grid()
    rows = []

    # Random never reads the zone, so one pass scores every disc.
    print("random: 1 pass over all discs")
    sweep("random", None, targets_for(X_VALUES, Y_VALUES), args.episodes,
          ppcf_grid, rows)

    # Scripted aims at zone.centre, so it needs one pass per centre. Radius and
    # theta still come off the same trajectories.
    for x in X_VALUES:
        for y in Y_VALUES:
            print(f"scripted: centre ({x}, {y})")
            sweep("scripted", make_zone(x, y, RADII[0]),
                  targets_for((x,), (y,)), args.episodes, ppcf_grid, rows)

    write_csv("zone_probe_raw.csv", rows)
    summary = summarise(rows)
    write_csv("zone_probe_summary.csv", summary)
    ranked = choose(summary)
    write_csv("zone_probe_choice.csv", ranked)

    print(f"\ntarget: random in {RANDOM_BAND}, scripted in {SCRIPTED_BAND}")
    print(f"{'x':>5} {'y':>5} {'r':>5} {'theta':>6} {'random':>8} "
          f"{'scripted':>9}  in_band")
    for d in ranked[:10]:
        print(f"{d['x']:5.0f} {d['y']:5.0f} {d['radius']:5.0f} "
              f"{d['theta']:6.2f} {d['random_rate']:8.1%} "
              f"{d['scripted_rate']:9.1%}  {'yes' if d['in_band'] else ''}")


if __name__ == "__main__":
    main()
