import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse
import csv
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gymnasium.vector import AutoresetMode

from environment.lowblock_env import SUCCESS, make_vector_env, obs_dim
from model.ppo_constrained import ActorCritic, OWN_VEL, SPEED_LIMIT
from physics.engine import V_MAX

ASSETS = os.path.join(REPO_ROOT, "assets")
N_ENVS = 6
TICKS = 2500
BANDS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
GRID = np.linspace(0.0, 5.0, 101)
PCTS = (10, 25, 50, 75, 90)


def resolve(path, ckpt):
    if path.endswith(".pt"):
        return path
    for name in (ckpt + ".pt", "final.pt", "best.pt"):
        cand = os.path.join(path, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(path)


def gate_of(cfg):
    return (float(cfg["zone_x"]), float(cfg["zone_y"]),
            float(cfg["zone_radius"]), float(cfg["pc_min"]),
            int(cfg.get("start_holder", 0)))


def make_env(gate):
    return make_vector_env(n_envs=N_ENVS, asynchronous=True,
                           autoreset_mode=AutoresetMode.SAME_STEP,
                           zone_x=gate[0], zone_y=gate[1], zone_radius=gate[2],
                           pc_min=gate[3],
                           start_holder=None if gate[4] < 0 else gate[4])


def measure(agent, venv, ticks, seed):
    speeds, wins, ends = [], 0, 0
    obs, _ = venv.reset(seed=seed)
    obs = torch.as_tensor(obs)
    with torch.no_grad():
        for _ in range(ticks):
            speeds.append(
                torch.linalg.norm(obs[..., OWN_VEL:OWN_VEL + 2], dim=-1) * V_MAX)
            act = agent.act(obs)[0]
            o, _, term, trunc, info = venv.step(act.numpy())
            done = term | trunc
            if done.any():
                outs = info["final_info"]["outcome"]
                for i in np.flatnonzero(done):
                    ends += 1
                    wins += int(outs[i] == SUCCESS)
            obs = torch.as_tensor(o)

    s = torch.stack(speeds)
    flat = s.numpy().ravel()
    return {"c_slow": float((s < SPEED_LIMIT).float().mean()),
            "c_team": float((s.mean(dim=-1) < SPEED_LIMIT).float().mean()),
            "mean_speed": float(s.mean()),
            "success": wins / ends if ends else float("nan"),
            "episodes": ends,
            "attacker_ticks": int(flat.size),
            "pcts": np.percentile(flat, PCTS),
            "bands": [float((flat < b).mean()) for b in BANDS],
            "cdf": np.array([(flat < g).mean() for g in GRID])}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def plot(rows, out_path):
    have_target = any(r["requested"] == r["requested"] for r in rows)
    targets = sorted({r["requested"] for r in rows
                      if r["requested"] == r["requested"]})
    ncol = 2 if have_target and targets else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5.0), squeeze=False)
    ax = axes[0]

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(targets), 1)))

    if have_target and targets:
        a = ax[0]
        x = np.array([r["requested"] for r in rows])
        y = np.array([r["c_slow"] for r in rows])
        lo, hi = min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05
        a.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=1.4,
               label="achieved = target")
        a.scatter(x, y, s=55, color="#8ab4e8", zorder=3, label="seeds")
        mu = [np.mean([r["c_slow"] for r in rows if r["requested"] == t])
              for t in targets]
        sd = [np.std([r["c_slow"] for r in rows if r["requested"] == t], ddof=0)
              for t in targets]
        a.errorbar(targets, mu, yerr=sd, fmt="o-", color="#1f4e9c", lw=2,
                   capsize=4, zorder=4, label="mean")
        a.set_xlabel("requested rate")
        a.set_ylabel("achieved rate (frozen policy)")
        a.set_title("calibration", loc="left", fontsize=10)
        a.legend(fontsize=8, frameon=False)
        a.grid(alpha=0.25)

    b = ax[-1]
    if targets:
        for c, t in zip(colours, targets):
            curves = [r["cdf"] for r in rows if r["requested"] == t]
            b.plot(GRID, np.mean(curves, axis=0), color=c, lw=2,
                   label="d = %.2f" % t)
    else:
        for r in rows:
            b.plot(GRID, r["cdf"], lw=2, label=r["run"])
    b.axvline(SPEED_LIMIT, color="#d62728", ls="--", lw=1.2)
    b.text(SPEED_LIMIT + 0.05, 0.05, "limit %.1f m/s" % SPEED_LIMIT,
           fontsize=8, color="#d62728")
    b.set_xlabel("attacker speed (m/s)")
    b.set_ylabel("P(speed < x)")
    b.set_ylim(0, 1)
    b.set_title("speed distribution", loc="left", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    b.grid(alpha=0.25)

    fig.suptitle("%d policies | %d attacker-ticks each"
                 % (len(rows), rows[0]["attacker_ticks"]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default=ASSETS)
    ap.add_argument("--tag", default="speed_probe")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        hits = sorted(glob.glob(p))
        paths.extend(hits if hits else [p])

    os.makedirs(args.out_dir, exist_ok=True)
    D = obs_dim(10, 11)
    venv, gate = None, None
    rows = []

    print("%-22s %8s %7s %7s %7s %6s | %5s %5s %5s %5s %5s"
          % ("run", "step", "succ", "c_slow", "c_team", "mean",
             "p10", "p25", "p50", "p75", "p90"))

    for path in paths:
        ck_path = resolve(path, args.ckpt)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg", {})
        g = gate_of(cfg)
        if g != gate:
            if venv is not None:
                venv.close()
            venv, gate = make_env(g), g
            print("gate (%.0f, %.0f) r=%.0f pc_min=%.2f start_holder %d"
                  % g, flush=True)

        agent = ActorCritic(obs_dim=D, hidden=int(cfg.get("hidden", 256)))
        agent.load_state_dict(ck["model"], strict=False)
        agent.eval()

        r = measure(agent, venv, args.ticks, args.seed)
        r["run"] = os.path.basename(os.path.dirname(os.path.abspath(ck_path)))
        r["ckpt"] = os.path.splitext(os.path.basename(ck_path))[0]
        r["step"] = int(ck.get("step", 0))
        r["seed"] = int(cfg.get("seed", -1))
        r["requested"] = float(cfg["slow_d"]) if "slow_d" in cfg else float("nan")
        rows.append(r)

        p = r["pcts"]
        print("%-22s %8d %7.3f %7.3f %7.3f %6.2f | %5.2f %5.2f %5.2f %5.2f %5.2f"
              % (r["run"], r["step"], r["success"], r["c_slow"], r["c_team"],
                 r["mean_speed"], p[0], p[1], p[2], p[3], p[4]), flush=True)

    if venv is not None:
        venv.close()

    flat = []
    for r in rows:
        row = {k: r[k] for k in ("run", "ckpt", "seed", "step", "requested",
                                 "c_slow", "c_team", "mean_speed", "success",
                                 "episodes", "attacker_ticks")}
        for q, v in zip(PCTS, r["pcts"]):
            row["p%d" % q] = float(v)
        for bnd, v in zip(BANDS, r["bands"]):
            row["lt_%.1f" % bnd] = v
        flat.append(row)

    csv_path = write_csv(os.path.join(args.out_dir, args.tag + ".csv"), flat)
    png_path = plot(rows, os.path.join(args.out_dir, args.tag + ".png"))

    have = [r for r in rows if r["requested"] == r["requested"]]
    if have:
        err = np.array([r["c_slow"] - r["requested"] for r in have])
        print()
        print("MAE %.4f (%.2f pp) | bias %+.4f | feasible %d/%d"
              % (np.abs(err).mean(), 100 * np.abs(err).mean(), err.mean(),
                 int((err <= 0).sum()), len(have)))

    print()
    print("wrote %s" % csv_path)
    print("wrote %s" % png_path)


if __name__ == "__main__":
    main()
