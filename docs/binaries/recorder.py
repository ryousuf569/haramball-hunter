import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _pick_backend():
    want = "cuda"
    if "--backend" in sys.argv:
        want = sys.argv[sys.argv.index("--backend") + 1]
    if want == "cuda":
        import importlib.util
        if importlib.util.find_spec("cupy") is None:
            print("cupy not importable, falling back to PPCF_BACKEND=numpy")
            want = "numpy"
    os.environ["PPCF_BACKEND"] = want
    return want


BACKEND = _pick_backend()

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse
import csv
import json

import numpy as np

from attackers.ppo_policy import make_ppo_policy, resolve_ckpt
from environment.grid import PC_CELL_SIZE, PC_EXTENT, PC_NX, PC_NY
from environment.lowblock_env import (MAX_TICKS, LowBlockEnv,
                                      compute_attacker_ppcf)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEED_CSV = os.path.join(REPO_ROOT, "assets", "speed_probe.csv")
NCELL = PC_NX * PC_NY
STRIDE = 90 + NCELL
BALL_STATE = {"held": 0, "in_flight": 1, "loose": 2}

DEFAULT_POLICIES = [
    ("unconstrained", None, "runs/vanilla_10m_cuda_rung2"),
    ("d = 0.85", 0.85, "runs/sweep_slow85_s%d"),
    ("d = 0.75", 0.75, "runs/sweep_slow75_s%d"),
    ("d = 0.65", 0.65, "runs/sweep_slow65_s%d"),
    ("d = 0.55", 0.55, "runs/sweep_slow55_s%d"),
    ("d = 0.45", 0.45, "runs/sweep_slow45_s%d"),
]


def load_probe():
    if not os.path.exists(SPEED_CSV):
        return []
    with open(SPEED_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def choose_run(template, requested, probe, seeds):
    if requested is None:
        return template
    rows = [r for r in probe
            if r.get("requested") not in (None, "")
            and abs(float(r["requested"]) - requested) < 1e-9]
    if rows:
        best = min(rows, key=lambda r: abs(float(r["c_slow"]) - requested))
        return os.path.join("runs", best["run"])
    for s in seeds:
        cand = template % s
        if os.path.exists(os.path.join(REPO_ROOT, cand)):
            return cand
    return template % seeds[0]


def probe_stats(run, probe):
    name = os.path.basename(run)
    for r in probe:
        if r["run"] == name:
            return (float(r["c_slow"]), float(r["success"]),
                    float(r["mean_speed"]))
    return (float("nan"), float("nan"), float("nan"))


def record_episode(env, ppo, seed, max_ticks):
    env.reset(seed=seed)
    ppo.reset()
    pos, ball, state, holder, heat = [], [], [], [], []
    t = 0
    while True:
        d, s, b = ppo(env.players, env.ball, env.attacker_ids,
                      pc_att=env.pc_att)
        _, _, term, trunc, info = env.step(np.stack([d, s, b], axis=1))
        t += 1

        pc = compute_attacker_ppcf(env.players, env.ppcf_grid,
                                   env.ball["position"])
        env.pc_att = pc

        p = np.asarray(env.players["position"], dtype=float)
        pos.append(np.clip(p * 100.0, -32000, 32000).astype("<i2").ravel())
        ball.append(np.clip(np.asarray(env.ball["position"], dtype=float)
                            * 100.0, -32000, 32000).astype("<i2"))
        state.append(BALL_STATE.get(env.ball.get("state"), 3))
        hid = env.ball.get("holder_id")
        rows = np.flatnonzero(env.players["id"] == hid) if hid is not None else []
        holder.append(int(rows[0]) if len(rows) else -1)
        heat.append(np.clip(pc.ravel() * 255.0, 0, 255).astype("u1"))

        if term or trunc or t >= max_ticks:
            return info["outcome"], t, (pos, ball, state, holder, heat)


def pack(blocks):
    pos, ball, state, holder, heat = blocks
    return b"".join([
        np.concatenate(pos).tobytes(),
        np.concatenate(ball).tobytes(),
        np.asarray(state, dtype="u1").tobytes(),
        np.asarray(holder, dtype="i1").tobytes(),
        np.concatenate(heat).tobytes(),
    ])


def slug(name):
    return (name.replace(" ", "").replace("=", "")
                .replace(".", "").replace("/", "_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--start-holder", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "numpy"])
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    probe = load_probe()
    os.makedirs(args.out_dir, exist_ok=True)
    print("PPCF_BACKEND=%s | %d episodes/policy | grid %dx%d | stride %d B"
          % (BACKEND, args.episodes, PC_NX, PC_NY, STRIDE), flush=True)

    manifest = {"grid": {"nx": PC_NX, "ny": PC_NY, "cell": PC_CELL_SIZE,
                         "x0": PC_EXTENT[0], "y0": PC_EXTENT[2]},
                "stride": STRIDE, "n_players": 21, "n_attackers": 10,
                "ball_state": BALL_STATE, "policies": []}
    env = None

    for name, requested, template in DEFAULT_POLICIES:
        run = choose_run(template, requested, probe, args.seeds)
        ckpt = resolve_ckpt(os.path.join(REPO_ROOT, run)
                            if not os.path.isabs(run) else run)
        if args.ckpt != "final":
            alt = os.path.join(os.path.dirname(ckpt), args.ckpt + ".pt")
            ckpt = alt if os.path.exists(alt) else ckpt

        ppo = make_ppo_policy(ckpt, max_ticks=args.max_ticks, seed=args.seed)
        if env is None:
            c = ppo.zone.centre
            env = LowBlockEnv(max_tick=args.max_ticks,
                              start_holder=args.start_holder,
                              zone_x=float(c[0]), zone_y=float(c[1]),
                              zone_radius=float(ppo.zone.radius),
                              pc_min=ppo.pc_min)
            manifest["zone"] = {"x": float(c[0]), "y": float(c[1]),
                                "r": float(ppo.zone.radius),
                                "pc_min": float(ppo.pc_min)}
        ppo.zone = env.zone
        ppo.env.zone = env.zone
        ppo.env.zone_centre = np.asarray(env.zone.centre, dtype="f4")

        achieved, success, mean_speed = probe_stats(run, probe)
        chunks, episodes, offset = [], [], 0
        for e in range(args.episodes):
            outcome, ticks, blocks = record_episode(
                env, ppo, args.seed + e, args.max_ticks)
            buf = pack(blocks)
            assert len(buf) == ticks * STRIDE, (len(buf), ticks * STRIDE)
            chunks.append(buf)
            episodes.append({"offset": offset, "ticks": ticks,
                             "outcome": outcome, "seed": args.seed + e})
            offset += len(buf)

        fname = slug(name) + ".bin"
        with open(os.path.join(args.out_dir, fname), "wb") as fh:
            fh.write(b"".join(chunks))

        manifest["policies"].append({
            "name": name, "file": fname, "run": os.path.basename(run),
            "requested": requested, "achieved": achieved,
            "success": success, "mean_speed": mean_speed,
            "episodes": episodes})

        print("%-14s %-22s req %-5s ach %-6s | %d eps, %d ticks, %.2f MB"
              % (name, os.path.basename(run),
                 "-" if requested is None else "%.2f" % requested,
                 "nan" if achieved != achieved else "%.3f" % achieved,
                 len(episodes), sum(x["ticks"] for x in episodes),
                 offset / 1e6), flush=True)

    path = os.path.join(args.out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    total = sum(os.path.getsize(os.path.join(args.out_dir, p["file"]))
                for p in manifest["policies"])
    print()
    print("wrote %s" % path)
    print("%d binaries, %.2f MB total" % (len(manifest["policies"]), total / 1e6))


if __name__ == "__main__":
    main()
