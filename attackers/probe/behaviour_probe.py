import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attackers.ppo_policy import make_ppo_policy, resolve_ckpt
from attackers.scripted_policy import make_policy as make_scripted_policy
from defenders.turnover import HALFWAY_X, offside_line
from environment.lowblock_env import (FAILURE, MAX_TICKS, SUCCESS, TIMEOUT,
                                      LowBlockEnv)
from environment.termination import (ZONE_PC_MIN, ZONE_RADIUS, ZONE_X, ZONE_Y,
                                     zone_control)
from render import render_frame

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CROWD_RADIUS = 3.0
CROWD_MIN = 4
TIGHT_SPREAD = 6.0
PRESS_RADIUS = 3.0
STALL_LAG = 10
STALL_EPS = 0.5

INDICATORS = ("crowd_disc", "crowd_near", "tight", "zone_dwell", "press",
              "offside", "stall")
BAR_COLOR = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def gate_from_ckpt(path):
    ckpt = torch.load(resolve_ckpt(path), map_location="cpu",
                      weights_only=False)
    cfg = ckpt.get("cfg", {})
    return (float(cfg.get("zone_x", ZONE_X)),
            float(cfg.get("zone_y", ZONE_Y)),
            float(cfg.get("zone_radius", ZONE_RADIUS)),
            float(cfg.get("pc_min", ZONE_PC_MIN)))


def tick_stats(env, prev_ball):
    n = env.n_att
    pos = np.asarray(env.players["position"][:n], dtype=float)
    vel = np.asarray(env.players["velocity"][:n], dtype=float)
    dpos = np.asarray(env.players["position"][n:], dtype=float)
    centre = np.asarray(env.zone.centre, dtype=float)
    radius = float(env.zone.radius)

    spread = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
    pair = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(pair, np.inf)
    nn = pair.min(axis=1)

    d_zone = np.linalg.norm(pos - centre, axis=1)
    ball_pos = np.asarray(env.ball["position"], dtype=float)
    ball_d = float(np.linalg.norm(ball_pos - centre))

    holder = env.ball.get("holder_id")
    carried = holder is not None and bool(np.any(env.attacker_ids == holder))
    if carried:
        row = int(np.flatnonzero(env.attacker_ids == holder)[0])
        carrier_clear = float(np.linalg.norm(dpos - pos[row], axis=1).min())
    else:
        carrier_clear = float("nan")

    line = offside_line(env.players)
    margin = pos[:, 0] - line
    offside = (margin > 0.0) & (pos[:, 0] > HALFWAY_X)

    started = (prev_ball.get("state") == "held"
               and env.ball.get("state") == "in_flight")
    if started:
        pass_len = float(np.linalg.norm(
            np.asarray(env.ball["flight_target"], dtype=float)
            - np.asarray(env.ball["flight_start"], dtype=float)))
    else:
        pass_len = float("nan")

    return {"spread": spread,
            "nn_mean": float(nn.mean()),
            "nn_min": float(nn.min()),
            "n_in_disc": int((d_zone <= radius).sum()),
            "n_near": int((nn < CROWD_RADIUS).sum()),
            "zone_control": float(zone_control(env.pc_att, env.zone)),
            "ball_d_zone": ball_d,
            "ball_in_disc": int(ball_d <= radius),
            "carried": int(carried),
            "in_flight": int(env.ball.get("state") == "in_flight"),
            "carrier_clear": carrier_clear,
            "n_offside": int(offside.sum()),
            "max_offside": float(margin.max()),
            "mean_speed": float(np.linalg.norm(vel, axis=1).mean()),
            "pass_started": int(started),
            "pass_len": pass_len}


def save_frame(env, tick, seed, frame_dir, outcome):
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    tag = "" if outcome is None else "  %s" % outcome
    render_frame(env.players, env.ball, ax=ax, zone=env.zone,
                 pc_att=env.pc_att, title="seed %d  tick %d%s"
                                          % (seed, tick, tag))
    path = os.path.join(frame_dir, "ep%05d_t%03d.png" % (seed, tick))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def run_episode(env, start_episode, seed, frame_dir=None, frame_every=10):
    env.reset(seed=seed)
    act = start_episode(env)
    rows = []
    t = 0
    while True:
        prev_ball = dict(env.ball)
        _, _, term, trunc, info = env.step(act(env))
        t += 1
        row = tick_stats(env, prev_ball)
        row["tick"] = t
        rows.append(row)
        done = bool(term or trunc)
        if frame_dir is not None and (t % frame_every == 0 or done):
            save_frame(env, t, seed, frame_dir,
                       info["outcome"] if done else None)
        if done:
            return info["outcome"], rows


def nanmean(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def episode_summary(outcome, rows, seed):
    col = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    in_disc = col["ball_in_disc"] > 0
    passes = col["pass_started"] > 0
    held = col["carried"] > 0

    d = col["ball_d_zone"]
    if len(d) > STALL_LAG:
        stall = np.zeros(len(d), dtype=bool)
        stall[STALL_LAG:] = (d[:-STALL_LAG] - d[STALL_LAG:]) < STALL_EPS
    else:
        stall = np.zeros(len(d), dtype=bool)

    press = held & (col["carrier_clear"] < PRESS_RADIUS)

    return {"seed": seed,
            "outcome": outcome,
            "ticks": len(rows),
            "passes": int(passes.sum()),
            "passes_in_disc": int((passes & in_disc).sum()),
            "mean_pass_len": nanmean(col["pass_len"]),
            "dwell_ticks": int(in_disc.sum()),
            "spread_mean": float(col["spread"].mean()),
            "spread_final": float(col["spread"][-1]),
            "n_in_disc_mean": float(col["n_in_disc"].mean()),
            "n_in_disc_final": float(col["n_in_disc"][-1]),
            "n_in_disc_max": float(col["n_in_disc"].max()),
            "zc_mean": float(col["zone_control"].mean()),
            "zc_final": float(col["zone_control"][-1]),
            "carrier_clear_mean": nanmean(col["carrier_clear"]),
            "speed_mean": float(col["mean_speed"].mean()),
            "possession_rate": float(held.mean()),
            "crowd_disc": float((col["n_in_disc"] >= CROWD_MIN).mean()),
            "crowd_near": float((col["n_near"] >= CROWD_MIN).mean()),
            "tight": float((col["spread"] < TIGHT_SPREAD).mean()),
            "zone_dwell": float(in_disc.mean()),
            "press": float(press.mean()),
            "offside": float((col["n_offside"] > 0).mean()),
            "stall": float(stall.mean())}


def summarise(name, episodes):
    skip = ("seed", "outcome", "policy")
    keys = [k for k in episodes[0] if k not in skip]
    out = {"name": name, "n": len(episodes)}
    for o in (SUCCESS, FAILURE, TIMEOUT):
        out[o] = sum(e["outcome"] == o for e in episodes) / len(episodes)
    for k in keys:
        out[k] = nanmean([e[k] for e in episodes])
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def plot(results, gate, out_path):
    x = np.arange(len(INDICATORS))
    width = 0.8 / len(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2),
                                   gridspec_kw={"width_ratios": [3, 2]})

    for i, r in enumerate(results):
        ax1.bar(x + i * width - 0.4 + width / 2,
                [r[k] for k in INDICATORS], width=width,
                color=BAR_COLOR[i % len(BAR_COLOR)], label=r["name"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(INDICATORS, fontsize=9, rotation=20)
    ax1.set_ylabel("rate over ticks")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_title("candidate indicator rates", fontsize=10, loc="left")

    shape = ("spread_mean", "n_in_disc_mean", "passes", "passes_in_disc")
    x2 = np.arange(len(shape))
    for i, r in enumerate(results):
        ax2.bar(x2 + i * width - 0.4 + width / 2, [r[k] for k in shape],
                width=width, color=BAR_COLOR[i % len(BAR_COLOR)])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["spread m", "in disc", "passes", "passes\nin disc"],
                        fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.set_title("shape and passing", fontsize=10, loc="left")

    fig.suptitle("gate (%.0f, %.0f) r=%.0f pc_min=%.2f"
                 % (gate[0], gate[1], gate[2], gate[3]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


def ppo_start(ppo):
    def start(env):
        ppo.reset()

        def act(env):
            d, s, b = ppo(env.players, env.ball, env.attacker_ids,
                          pc_att=env.pc_att)
            return np.stack([d, s, b], axis=1)
        return act
    return start


def scripted_start(env):
    policy = make_scripted_policy(env.zone)

    def act(env):
        d, s, b = policy(env.players, env.ball, env.attacker_ids)
        return np.stack([d, s, b], axis=1)
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--start-holder", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--frame-episodes", type=int, default=3)
    ap.add_argument("--frame-every", type=int, default=10)
    ap.add_argument("--zone-x", type=float, default=None)
    ap.add_argument("--zone-y", type=float, default=None)
    ap.add_argument("--zone-radius", type=float, default=None)
    ap.add_argument("--pc-min", type=float, default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    gate = gate_from_ckpt(args.ckpt)
    override = (args.zone_x, args.zone_y, args.zone_radius, args.pc_min)
    gate = tuple(g if o is None else o for g, o in zip(gate, override))

    env = LowBlockEnv(max_tick=args.max_ticks, start_holder=args.start_holder,
                      zone_x=gate[0], zone_y=gate[1], zone_radius=gate[2],
                      pc_min=gate[3])
    ppo = make_ppo_policy(args.ckpt, zone=env.zone, max_ticks=args.max_ticks,
                          deterministic=args.deterministic, seed=args.seed)

    runs = [(ppo.label(), ppo_start(ppo))]
    if args.scripted:
        runs.append(("scripted", scripted_start))

    print("gate (%.0f, %.0f) r=%.0f pc_min=%.2f | start_holder %d | "
          "%d episodes | seeds %d..%d"
          % (gate[0], gate[1], gate[2], gate[3], args.start_holder,
             args.episodes, args.seed, args.seed + args.episodes - 1))

    os.makedirs(args.out_dir, exist_ok=True)
    results, all_eps = [], []
    for name, start_episode in runs:
        slug = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        frame_dir = os.path.join(args.out_dir, "frames", slug)
        if args.frame_episodes > 0:
            os.makedirs(frame_dir, exist_ok=True)
        episodes, ticks = [], []
        for e in range(args.episodes):
            seed = args.seed + e
            fd = frame_dir if e < args.frame_episodes else None
            outcome, rows = run_episode(env, start_episode, seed, fd,
                                        args.frame_every)
            episodes.append(episode_summary(outcome, rows, seed))
            if e < args.frame_episodes:
                for r in rows:
                    r["seed"] = seed
                    r["policy"] = name
                ticks.extend(rows)
        if ticks:
            write_csv(os.path.join(args.out_dir, "ticks_%s.csv" % slug), ticks)
        for ep in episodes:
            ep["policy"] = name
        all_eps.extend(episodes)
        results.append(summarise(name, episodes))

        r = results[-1]
        print()
        print("%s  (n=%d)" % (name.replace("\n", " "), r["n"]))
        print("  outcome      success %5.1f%%  failure %5.1f%%  timeout %5.1f%%"
              % (100 * r[SUCCESS], 100 * r[FAILURE], 100 * r[TIMEOUT]))
        print("  shape        spread %5.1f m  in-disc %4.1f (max %4.1f, final "
              "%4.1f)  nn-near %4.2f"
              % (r["spread_mean"], r["n_in_disc_mean"], r["n_in_disc_max"],
                 r["n_in_disc_final"], r["crowd_near"]))
        print("  ball         passes %4.1f (%4.1f in disc)  len %5.1f m  "
              "possession %5.1f%%  dwell %4.0f ticks"
              % (r["passes"], r["passes_in_disc"], r["mean_pass_len"],
                 100 * r["possession_rate"], r["dwell_ticks"]))
        print("  control      zc mean %5.3f  zc final %5.3f  carrier clear "
              "%5.2f m  speed %4.2f m/s"
              % (r["zc_mean"], r["zc_final"], r["carrier_clear_mean"],
                 r["speed_mean"]))
        print("  indicators   " + "  ".join(
            "%s %5.1f%%" % (k, 100 * r[k]) for k in INDICATORS))

    write_csv(os.path.join(args.out_dir, "behaviour_episodes.csv"), all_eps)
    print()
    print("wrote %s" % plot(results, gate,
                            os.path.join(args.out_dir, "behaviour_probe.png")))
    print("wrote %s" % os.path.join(args.out_dir, "behaviour_episodes.csv"))
    if args.frame_episodes > 0:
        print("frames in %s" % os.path.join(args.out_dir, "frames"))


if __name__ == "__main__":
    main()
