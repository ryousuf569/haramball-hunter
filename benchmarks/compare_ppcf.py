# Agreement stress test: the CUDA kernel PPCF backend against the numpy one.

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse
import csv

import numpy as np

import environment.lowblock_env as lowblock
from environment.grid import PC_NX, PC_NY, make_ppcf_grid
from environment.lowblock_env import LowBlockEnv, make_initial_world
from environment.termination import make_zone, zone_control
from physics.engine import PITCH_X, PITCH_Y
from physics.ppcf import PPCF_grid as numpy_ppcf
from physics.ppcf_kernel import PPCF_grid as cuda_ppcf
from physics.tti import a_max, v_max
from schema import player_dt

TARGETS = make_ppcf_grid()
N_CELLS = len(TARGETS)
ZONE = make_zone()

RTOL, ATOL = 1e-3, 1e-5
PC_TOL = 1e-3
ZONE_TOL = 1e-4
BALL_TOL = 1e-4
CUTOFF = 0.99
CUTOFF_BAND = 1e-5

TIE_DISTANCE = 0.5 * v_max ** 2 / a_max
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(OUT_DIR, "compare_ppcf.csv")
DEFAULT_TRAJ_OUT = os.path.join(OUT_DIR, "compare_ppcf_trajectories.csv")

FIELDS = ["case", "n_states", "n_players", "verdict", "deterministic",
          "max_abs", "max_rel", "cells_over_tol", "cells_at_cutoff",
          "pc_att_max_abs", "zone_ctrl_max_abs", "ip_max_abs", "status"]
TRAJ_FIELDS = ["episode", "seed", "match", "outcome_numpy", "outcome_cuda",
               "ticks_numpy", "ticks_cuda", "return_numpy", "return_cuda",
               "actions_diverged_at", "max_ball_delta", "max_pc_att_delta"]


def f4(x):
    return np.asarray(np.asarray(x, dtype="f4"), dtype=np.float64)


def make_players(positions, velocities=None, n_att=None):
    positions = np.asarray(positions, dtype="f4").reshape(-1, 2)
    n = len(positions)
    n_att = n // 2 if n_att is None else n_att
    players = np.zeros(n, dtype=player_dt)
    players["id"] = np.arange(n, dtype="int16")
    players["position"] = positions
    if velocities is not None:
        players["velocity"] = np.asarray(velocities, dtype="f4").reshape(-1, 2)
    players["team"][:n_att] = "attacker"
    players["team"][n_att:] = "defender"
    return players


def world(seed=0):
    players, ball, _ids, _rng, _state = make_initial_world(seed=seed)
    return players, f4(ball["position"])


def scatter(n, seed, lo=0.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(lo * PITCH_X, hi * PITCH_X, n),
                     rng.uniform(lo * PITCH_Y, hi * PITCH_Y, n)], axis=1)


def initial_formations():
    return [world(seed) for seed in range(8)]


def zero_velocity():
    players, ball = world(0)
    players["velocity"] = 0.0
    return [(players, ball)]


def tiny_velocity():
    players, ball = world(0)
    players["velocity"] = 1e-8
    return [(players, ball)]


def on_cell_centre():
    players, ball = world(0)
    players["position"][0] = TARGETS[0]
    players["position"][1] = TARGETS[N_CELLS // 2]
    players["position"][-1] = TARGETS[-1]
    return [(players, ball)]


def all_on_cell_centres():
    idx = np.linspace(0, N_CELLS - 1, 21).astype(int)
    players = make_players(TARGETS[idx])
    return [(players, f4(players["position"][0]))]


def ball_on_player():
    players, _ball = world(0)
    return [(players, f4(players["position"][3]))]


def near_zero_distance():
    states = []
    for eps in (0.0, 1e-7, 1e-6, 1e-4, 1e-2):
        players, _ball = world(0)
        players["position"][0] = TARGETS[N_CELLS // 2]
        states.append((players, f4(players["position"][0] + np.array([eps, 0.0]))))
    return states


def clustered_outfield():
    players, ball = world(0)
    players["position"][1:] = players["position"][1]
    return [(players, ball)]


def stacked_on_ball():
    players, ball = world(0)
    players["position"][:] = ball
    return [(players, f4(players["position"][0]))]


def out_of_bounds():
    players, ball = world(0)
    players["position"][0] = [-40.0, -25.0]
    players["position"][1] = [PITCH_X + 60.0, PITCH_Y + 40.0]
    players["position"][2] = [0.0, -900.0]
    players["position"][3] = [5000.0, 5000.0]
    return [(players, ball)]


def moving_away():
    players, ball = world(0)
    away = np.asarray(players["position"], dtype=float) - TARGETS.mean(axis=0)
    players["velocity"] = v_max * away / np.maximum(
        np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
    return [(players, ball)]


def moving_toward():
    players, ball = world(0)
    toward = TARGETS.mean(axis=0) - np.asarray(players["position"], dtype=float)
    players["velocity"] = v_max * toward / np.maximum(
        np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
    return [(players, ball)]


def above_v_max():
    players, ball = world(0)
    rng = np.random.default_rng(3)
    players["velocity"] = rng.normal(0, 4 * v_max, size=(len(players), 2))
    return [(players, ball)]


def tie_distance():
    offsets = [(TIE_DISTANCE, 0.0), (0.0, TIE_DISTANCE), (-TIE_DISTANCE, 0.0),
               (0.0, -TIE_DISTANCE)]
    cells = [0, N_CELLS // 3, 2 * N_CELLS // 3, N_CELLS - 1]
    positions = [TARGETS[c] - np.asarray(o) for c, o in zip(cells, offsets)]
    players = make_players(np.asarray(positions))
    return [(players, f4(players["position"][0]))]


def far_corners():
    positions = [[0.0, 0.0], [PITCH_X, PITCH_Y], [0.0, PITCH_Y], [PITCH_X, 0.0]]
    return [(make_players(positions), f4([0.0, 0.0]))]


def single_player():
    return [(make_players([[50.0, 34.0]], n_att=1), f4([50.0, 34.0]))]


def all_attackers():
    players, ball = world(0)
    players["team"] = "attacker"
    return [(players, ball)]


def all_defenders():
    players, ball = world(0)
    players["team"] = "defender"
    return [(players, ball)]


def duplicate_positions():
    players, ball = world(0)
    players["position"][5] = players["position"][4]
    players["velocity"][5] = players["velocity"][4]
    return [(players, ball)]


def many_players(n=92):
    players = make_players(scatter(n, seed=11))
    players["velocity"] = np.random.default_rng(12).normal(0, v_max, size=(n, 2))
    return [(players, f4([60.0, 34.0]))]


def over_shared_cap(n=128):
    return many_players(n)


def random_rollout(n_states=200, seed=7):
    lowblock.PPCF_grid = numpy_ppcf
    env = LowBlockEnv(start_holder=0)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    states = []
    while len(states) < n_states:
        states.append((env.players.copy(), f4(env.ball["position"])))
        _obs, _r, terminated, truncated, _info = env.step(
            env.action_space.sample())
        if terminated or truncated:
            env.reset(seed=seed + len(states))
    return states


def fuzz_states(n_states, seed):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        n = int(rng.integers(1, 93))
        pos = np.stack([rng.uniform(-30, PITCH_X + 30, n),
                        rng.uniform(-30, PITCH_Y + 30, n)], axis=1)
        snap = rng.random(n) < 0.15
        pos[snap] = TARGETS[rng.integers(0, N_CELLS, int(snap.sum()))]
        vel = rng.normal(0, 2 * v_max, size=(n, 2))
        vel[rng.random(n) < 0.15] = 0.0
        players = make_players(pos, vel, n_att=int(rng.integers(0, n + 1)))
        if rng.random() < 0.2:
            ball = f4(players["position"][rng.integers(0, n)])
        else:
            ball = f4([rng.uniform(0, PITCH_X), rng.uniform(0, PITCH_Y)])
        states.append((players, ball))
    return states


CASES = [
    ("initial_formations", initial_formations, True),
    ("zero_velocity", zero_velocity, True),
    ("tiny_velocity", tiny_velocity, True),
    ("player_on_cell_centre", on_cell_centre, True),
    ("all_on_cell_centres", all_on_cell_centres, True),
    ("ball_on_player", ball_on_player, True),
    ("near_zero_distance", near_zero_distance, True),
    ("clustered_outfield", clustered_outfield, True),
    ("stacked_on_ball", stacked_on_ball, True),
    ("out_of_bounds", out_of_bounds, True),
    ("moving_away", moving_away, True),
    ("moving_toward", moving_toward, True),
    ("above_v_max", above_v_max, True),
    ("tie_distance", tie_distance, True),
    ("far_corners", far_corners, True),
    ("single_player", single_player, True),
    ("all_attackers", all_attackers, True),
    ("all_defenders", all_defenders, True),
    ("duplicate_positions", duplicate_positions, True),
    ("many_players_92", many_players, True),
    ("over_shared_cap_128", over_shared_cap, True),
    ("no_ball_pos", initial_formations, False),
    ("random_rollout", random_rollout, True),
]


def pc_att_of(matrix, players):
    return matrix[:, players["team"] == "attacker"].sum(1).reshape(PC_NX, PC_NY)


def compare(states, with_ball):
    m = {"max_abs": 0.0, "max_rel": 0.0, "cells_over_tol": 0,
         "cells_at_cutoff": 0, "pc_att_max_abs": 0.0,
         "zone_ctrl_max_abs": 0.0, "ip_max_abs": 0.0, "deterministic": True}

    for players, ball_pos in states:
        ref_players, got_players = players.copy(), players.copy()
        bp = ball_pos if with_ball else None

        ref = numpy_ppcf(TARGETS, ref_players, bp)
        got = cuda_ppcf(TARGETS, got_players, bp)
        again = cuda_ppcf(TARGETS, players.copy(), bp)
        m["deterministic"] &= bool(np.array_equal(got, again))

        diff = np.abs(got - ref)
        m["max_abs"] = max(m["max_abs"], float(diff.max()))
        m["max_rel"] = max(m["max_rel"],
                           float((diff / np.maximum(np.abs(ref), 1e-6)).max()))

        over = (diff > ATOL + RTOL * np.abs(ref)).any(axis=1)
        saturated = (np.minimum(ref.sum(1), got.sum(1)) >= CUTOFF - CUTOFF_BAND)
        m["cells_over_tol"] += int(over.sum())
        m["cells_at_cutoff"] += int((over & saturated).sum())

        pc_ref, pc_got = pc_att_of(ref, ref_players), pc_att_of(got, got_players)
        m["pc_att_max_abs"] = max(m["pc_att_max_abs"],
                                  float(np.abs(pc_got - pc_ref).max()))
        m["zone_ctrl_max_abs"] = max(
            m["zone_ctrl_max_abs"],
            abs(float(zone_control(pc_got, ZONE)) - float(zone_control(pc_ref, ZONE))))

        if with_ball:
            m["ip_max_abs"] = max(m["ip_max_abs"], float(
                np.abs(got_players["i_p"] - ref_players["i_p"]).max()))
    return m


def verdict_of(m):
    env_ok = (m["pc_att_max_abs"] <= PC_TOL
              and m["zone_ctrl_max_abs"] <= ZONE_TOL
              and m["ip_max_abs"] <= ATOL + RTOL)
    if not m["deterministic"]:
        return "NONDETERMINISTIC"
    if m["cells_over_tol"] == 0:
        return "ok"
    if m["cells_over_tol"] == m["cells_at_cutoff"] and env_ok:
        return "cutoff_boundary"
    return "DIFFER"


def evaluate(name, builder, with_ball):
    try:
        states = builder()
        m = compare(states, with_ball)
        return {"case": name, "n_states": len(states),
                "n_players": len(states[0][0]), "verdict": verdict_of(m),
                "deterministic": m["deterministic"],
                "max_abs": "%.3e" % m["max_abs"],
                "max_rel": "%.3e" % m["max_rel"],
                "cells_over_tol": m["cells_over_tol"],
                "cells_at_cutoff": m["cells_at_cutoff"],
                "pc_att_max_abs": "%.3e" % m["pc_att_max_abs"],
                "zone_ctrl_max_abs": "%.3e" % m["zone_ctrl_max_abs"],
                "ip_max_abs": "%.3e" % m["ip_max_abs"], "status": "ok"}
    except Exception as exc:
        blank = dict.fromkeys(FIELDS, "")
        blank.update({"case": name, "n_states": 0, "n_players": 0,
                      "verdict": "ERROR", "deterministic": "",
                      "status": "%s: %s" % (type(exc).__name__, exc)})
        return blank


def trajectories(episodes, seed, max_ticks, start_holder, ckpt):
    ppo_np = ppo_cu = None
    kwargs = {}
    if ckpt:
        from attackers.ppo_policy import make_ppo_policy
        ppo_np = make_ppo_policy(ckpt, max_ticks=max_ticks, deterministic=True)
        ppo_cu = make_ppo_policy(ckpt, max_ticks=max_ticks, deterministic=True)
        kwargs = dict(zone_x=float(ppo_np.zone.centre[0]),
                      zone_y=float(ppo_np.zone.centre[1]),
                      zone_radius=float(ppo_np.zone.radius),
                      pc_min=ppo_np.pc_min)

    env_np = LowBlockEnv(max_tick=max_ticks, start_holder=start_holder, **kwargs)
    env_cu = LowBlockEnv(max_tick=max_ticks, start_holder=start_holder, **kwargs)

    rows = []
    for e in range(episodes):
        ep_seed = seed + e
        lowblock.PPCF_grid = numpy_ppcf
        env_np.reset(seed=ep_seed)
        lowblock.PPCF_grid = cuda_ppcf
        env_cu.reset(seed=ep_seed)
        if ckpt:
            ppo_np.reset()
            ppo_cu.reset()
        env_np.action_space.seed(ep_seed)

        state = {"outcome_numpy": "", "outcome_cuda": "", "ticks_numpy": 0,
                 "ticks_cuda": 0, "return_numpy": 0.0, "return_cuda": 0.0,
                 "actions_diverged_at": "", "max_ball_delta": 0.0,
                 "max_pc_att_delta": 0.0}
        done_np = done_cu = False
        tick = 0

        while not (done_np and done_cu):
            tick += 1
            if ckpt:
                lowblock.PPCF_grid = numpy_ppcf
                a_np = np.stack(ppo_np(env_np.players, env_np.ball,
                                       env_np.attacker_ids,
                                       pc_att=env_np.pc_att), axis=1)
                lowblock.PPCF_grid = cuda_ppcf
                a_cu = np.stack(ppo_cu(env_cu.players, env_cu.ball,
                                       env_cu.attacker_ids,
                                       pc_att=env_cu.pc_att), axis=1)
            else:
                a_np = a_cu = env_np.action_space.sample()

            if state["actions_diverged_at"] == "" and not np.array_equal(a_np, a_cu):
                state["actions_diverged_at"] = tick

            if not done_np:
                lowblock.PPCF_grid = numpy_ppcf
                _o, r, term, trunc, info = env_np.step(a_np)
                state["return_numpy"] += float(r)
                if term or trunc:
                    done_np = True
                    state["ticks_numpy"] = tick
                    state["outcome_numpy"] = str(info["outcome"])
            if not done_cu:
                lowblock.PPCF_grid = cuda_ppcf
                _o, r, term, trunc, info = env_cu.step(a_cu)
                state["return_cuda"] += float(r)
                if term or trunc:
                    done_cu = True
                    state["ticks_cuda"] = tick
                    state["outcome_cuda"] = str(info["outcome"])

            if not (done_np or done_cu):
                state["max_ball_delta"] = max(state["max_ball_delta"], float(
                    np.abs(f4(env_np.ball["position"])
                           - f4(env_cu.ball["position"])).max()))
                state["max_pc_att_delta"] = max(state["max_pc_att_delta"], float(
                    np.abs(env_np.pc_att - env_cu.pc_att).max()))

        match = (state["outcome_numpy"] == state["outcome_cuda"]
                 and state["ticks_numpy"] == state["ticks_cuda"])
        rows.append({"episode": e, "seed": ep_seed, "match": match,
                     "outcome_numpy": state["outcome_numpy"],
                     "outcome_cuda": state["outcome_cuda"],
                     "ticks_numpy": state["ticks_numpy"],
                     "ticks_cuda": state["ticks_cuda"],
                     "return_numpy": "%.6f" % state["return_numpy"],
                     "return_cuda": "%.6f" % state["return_cuda"],
                     "actions_diverged_at": state["actions_diverged_at"],
                     "max_ball_delta": "%.3e" % state["max_ball_delta"],
                     "max_pc_att_delta": "%.3e" % state["max_pc_att_delta"]})
        print("  ep %-3d seed %-6d %-5s  %-8s/%-8s  ticks %3d/%-3d  "
              "return %8.4f/%-8.4f  ball %s"
              % (e, ep_seed, "match" if match else "DIFFER",
                 rows[-1]["outcome_numpy"], rows[-1]["outcome_cuda"],
                 rows[-1]["ticks_numpy"], rows[-1]["ticks_cuda"],
                 float(rows[-1]["return_numpy"]), float(rows[-1]["return_cuda"]),
                 rows[-1]["max_ball_delta"]), flush=True)
    return rows


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--traj-out", default=DEFAULT_TRAJ_OUT)
    ap.add_argument("--only", default=None)
    ap.add_argument("--fuzz", type=int, default=500)
    ap.add_argument("--fuzz-seed", type=int, default=2024)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--max-ticks", type=int, default=lowblock.MAX_TICKS)
    ap.add_argument("--start-holder", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cases = list(CASES)
    if args.fuzz:
        cases.append(("fuzz_%d" % args.fuzz,
                      lambda: fuzz_states(args.fuzz, args.fuzz_seed), True))
    if args.only:
        cases = [c for c in cases if args.only in c[0]]

    rows = []
    for name, builder, with_ball in cases:
        row = evaluate(name, builder, with_ball)
        rows.append(row)
        print("%-22s %4s states  n=%-4s %-16s det=%-5s  over_tol %-5s "
              "pc %-10s zone %-10s %s"
              % (row["case"], row["n_states"], row["n_players"], row["verdict"],
                 row["deterministic"], row["cells_over_tol"],
                 row["pc_att_max_abs"], row["zone_ctrl_max_abs"],
                 "" if row["status"] == "ok" else row["status"]), flush=True)

    write_csv(args.out, FIELDS, rows)

    traj = []
    if args.episodes:
        print("\nlockstep episodes (%s):"
              % ("ckpt " + os.path.basename(args.ckpt) if args.ckpt
                 else "random actions"))
        traj = trajectories(args.episodes, args.seed, args.max_ticks,
                            args.start_holder, args.ckpt)
        write_csv(args.traj_out, TRAJ_FIELDS, traj)

    bad = [r for r in rows if r["verdict"] in ("DIFFER", "NONDETERMINISTIC")]
    nondet = [r for r in rows if r["deterministic"] is False]
    print("\nstates:   %d/%d cases ok or cutoff-explained, %d nondeterministic"
          % (len(rows) - len(bad), len(rows), len(nondet)))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    print("          worst pc_att %.3e, worst zone_control %.3e"
          % (max(float(r["pc_att_max_abs"]) for r in ok_rows),
             max(float(r["zone_ctrl_max_abs"]) for r in ok_rows)))
    if traj:
        print("episodes: %d/%d matched outcome and length"
              % (sum(1 for r in traj if r["match"]), len(traj)))
    print("wrote %s" % args.out + ("" if not traj else "\nwrote %s" % args.traj_out))


if __name__ == "__main__":
    main()
