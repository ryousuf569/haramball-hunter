"""What a trained checkpoint actually does, not just whether it wins.

Always run --random alongside it. That is the control, and 5M.pt lost to it.
The gate here is always the calibrated one, never the training curriculum's.
"""
import sys, io, os, argparse
from collections import Counter
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.thread_limits import limit_threads
limit_threads(1, torch_threads=1)

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from environment.lowblock_env import (
    ATTACKER_LABEL, DEFENDER_LABEL, SUCCESS, FAILURE, TIMEOUT,
    build_obs, ball_action_mask, compute_attacker_ppcf, make_initial_world,
    make_ppcf_grid, holder_row_of, offside_line, N_DIRECTIONS, N_SPEEDS,
    obs_dim, ball_actions,
)
from environment.lowblock_env import step as world_step
from environment.reward import build_zone_masks
from physics.engine import action_decoding, direction_lookup, speed_lookup, V_MAX
from ppo import Actor, MultiCategorial
from config import EnvConfig

# Read from the config so the clock feature cannot drift away from training.
MAX_TICKS = EnvConfig.t_max


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    """Checkpoint paths are relative to the repo root, not to the cwd.

    Windows spawns fresh workers, so a relative path that resolved in the
    parent process may not resolve in a child.
    """
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_actor(path, n_att=10, n_def=11):
    # weights_only=False: see the same note in environment/learned_env.py.
    ckpt = torch.load(resolve(path), map_location="cpu", weights_only=False)
    state = ckpt.get("actor", ckpt) if isinstance(ckpt, dict) else ckpt
    a = Actor(obs_dim(n_att + n_def, n_att), N_DIRECTIONS, N_SPEEDS,
              ball_actions(n_att))
    a.load_state_dict(state)
    a.eval()
    return a


def nearest_def_dist(players, pos, n_att):
    d = np.asarray(players["position"][n_att:], dtype=float) - np.asarray(pos, float)
    return float(np.linalg.norm(d, axis=1).min())


def run_episode(actor, seed, start_holder, n_att=10, n_def=11,
                deterministic=True, random_policy=False):
    players, ball, attacker_ids, _rng, dstate = make_initial_world(
        n_att, n_def, seed, start_holder=start_holder)
    grid = make_ppcf_grid()
    zmasks = build_zone_masks(grid)
    pc_att = compute_attacker_ppcf(players, grid, ball["position"])

    rec = dict(
        dir_hist=np.zeros(N_DIRECTIONS, int),
        spd_hist=np.zeros(N_SPEEDS, int),
        holder_dir_hist=np.zeros(N_DIRECTIONS, int),
        holder_spd_hist=np.zeros(N_SPEEDS, int),
        ent_dir=[], ent_spd=[], ent_ball=[], hold_prob=[],
        passes=[],           # dicts
        holder_ticks=0, flight_ticks=0,
        holder_speed=[], offball_speed=[],
        ball_x=[], holder_dx=[],
        legal_targets=[],
        att_x_mean=[], att_x_max=[],
        turnover_kind=None,
        held_ticks=[],        # ticks of possession before each release
        offside_rate=[],      # fraction of attackers offside, per tick
        holder_press=[],      # nearest-defender distance to the carrier
        ball_dist=[],         # every attacker's distance to the ball
        x_spread=[],          # attacker x std, per tick
        f3_count=[],          # attackers in the final third, per tick
    )
    rng = np.random.default_rng(seed + 7777)
    pending = None  # in-flight pass being tracked
    held_since = 0 if start_holder is not None else None

    for tick in range(MAX_TICKS):
        o = build_obs(players, ball, n_att, tick, MAX_TICKS, pc_att, *zmasks)
        m = ball_action_mask(players, ball, attacker_ids, n_att)
        with torch.no_grad():
            ot = torch.as_tensor(o)
            d, s, b = actor(ot)
            dist = MultiCategorial([d, s, b], [None, None, torch.as_tensor(m)])
            if random_policy:
                act = np.stack([
                    rng.integers(0, N_DIRECTIONS, n_att),
                    rng.integers(0, N_SPEEDS, n_att),
                    np.array([rng.choice(np.flatnonzero(m[i])) for i in range(n_att)]),
                ], axis=1)
            else:
                act = (dist.mode() if deterministic else dist.sample()).numpy()
            ents = dist.head_entropies()
            ball_probs = torch.softmax(dist.logits_list[2], dim=-1).numpy()

        row = holder_row_of(ball, attacker_ids)

        # shape / offside / pressure stats on the pre-step state
        apos = np.asarray(players["position"][:n_att], float)
        line = offside_line(players)
        off = (apos[:, 0] > line) & (apos[:, 0] > 52.5) & \
              (apos[:, 0] > float(ball["position"][0]))
        rec["offside_rate"].append(float(off.mean()))
        rec["x_spread"].append(float(apos[:, 0].std()))
        rec["f3_count"].append(int((apos[:, 0] > 70.0).sum()))
        bpos = np.asarray(ball["position"], float)
        rec["ball_dist"].extend(np.linalg.norm(apos - bpos, axis=1).tolist())
        if row is not None:
            rec["holder_press"].append(nearest_def_dist(players, apos[row], n_att))

        rec["dir_hist"] += np.bincount(act[:, 0], minlength=N_DIRECTIONS)
        rec["spd_hist"] += np.bincount(act[:, 1], minlength=N_SPEEDS)
        rec["ent_dir"].append(float(ents[0].mean()))
        rec["ent_spd"].append(float(ents[1].mean()))
        if row is not None:
            rec["holder_ticks"] += 1
            rec["holder_dir_hist"][act[row, 0]] += 1
            rec["holder_spd_hist"][act[row, 1]] += 1
            rec["ent_ball"].append(float(ents[2][row]))
            rec["hold_prob"].append(float(ball_probs[row, 0]))
            rec["legal_targets"].append(int(m[row].sum()) - 1)
        if ball["state"] == "in_flight":
            rec["flight_ticks"] += 1

        vel = action_decoding(act[:, 0], act[:, 1]).astype("f4")
        ball_idx = np.zeros(n_att, int)
        if row is not None:
            ball_idx[row] = int(act[row, 2])

        # pass attempt bookkeeping (before the step overwrites state)
        pass_rec = None
        if row is not None and ball_idx[row] != 0:
            holder_id = int(attacker_ids[row])
            mates = np.sort(attacker_ids[attacker_ids != holder_id])
            tgt_id = int(mates[ball_idx[row] - 1])
            hp = np.asarray(players["position"][row], float)
            tp = np.asarray(players["position"][players["id"] == tgt_id][0], float)
            pass_rec = dict(
                tick=tick, length=float(np.linalg.norm(tp - hp)),
                dx=float(tp[0] - hp[0]), dy=float(abs(tp[1] - hp[1])),
                holder_x=float(hp[0]), target_x=float(tp[0]),
                holder_press=nearest_def_dist(players, hp, n_att),
                target_press=nearest_def_dist(players, tp, n_att),
                held=int(tick - held_since) if held_since is not None else -1,
                outcome="?",
            )
            rec["held_ticks"].append(pass_rec["held"])

        pre_holder_pos = (np.asarray(players["position"][row], float)
                          if row is not None else None)

        buf = io.StringIO()
        with redirect_stdout(buf):
            players, ball, pc_att, _own, outcome = world_step(
                players, ball, attacker_ids, dstate, tick, ppcf_grid=grid,
                attacker_velocities=vel, attacker_ball_idx=ball_idx,
                verbose=True)
        log = buf.getvalue()

        kind = None
        if "TURNOVER" in log:
            kind = log.split("TURNOVER (")[1].split(")")[0]

        if pass_rec is not None:
            pending = pass_rec
            rec["passes"].append(pass_rec)

        # resolve a pending pass
        if pending is not None:
            if kind in ("intercept", "offside", "loose pass"):
                pending["outcome"] = kind
                pending = None
            elif ball["state"] == "held":
                hid = ball.get("holder_id")
                if hid is not None and hid in set(attacker_ids.tolist()):
                    pending["outcome"] = "complete"
                else:
                    pending["outcome"] = "lost"
                pending = None

        # movement stats
        spd = np.linalg.norm(players["velocity"][:n_att], axis=1)
        new_row = holder_row_of(ball, attacker_ids)
        if new_row != row:
            held_since = tick + 1 if new_row is not None else None
        if new_row is not None:
            rec["holder_speed"].append(float(spd[new_row]))
            mask = np.ones(n_att, bool); mask[new_row] = False
            rec["offball_speed"].append(float(spd[mask].mean()))
        else:
            rec["offball_speed"].append(float(spd.mean()))
        if row is not None and pre_holder_pos is not None and new_row == row:
            rec["holder_dx"].append(
                float(players["position"][row][0] - pre_holder_pos[0]))
        rec["ball_x"].append(float(ball["position"][0]))
        rec["att_x_mean"].append(float(players["position"][:n_att, 0].mean()))
        rec["att_x_max"].append(float(players["position"][:n_att, 0].max()))

        if outcome is not None:
            rec["turnover_kind"] = kind
            return outcome, tick + 1, rec
    return TIMEOUT, MAX_TICKS, rec


def worker(args):
    ckpt, seed, holder, det, rand = args
    return run_episode(load_actor(ckpt), seed, holder, deterministic=det,
                       random_policy=rand)


def summarize(results, label):
    n = len(results)
    outs = Counter(o for o, _, _ in results)
    lens = [t for _, t, _ in results]
    kinds = Counter(r["turnover_kind"] for o, _, r in results if o == FAILURE)

    allp = [p for _, _, r in results for p in r["passes"]]
    pouts = Counter(p["outcome"] for p in allp)
    dirh = sum(r["dir_hist"] for _, _, r in results)
    spdh = sum(r["spd_hist"] for _, _, r in results)
    hdirh = sum(r["holder_dir_hist"] for _, _, r in results)
    hspdh = sum(r["holder_spd_hist"] for _, _, r in results)

    def m(key, agg=np.mean):
        vals = [v for _, _, r in results for v in r[key]]
        return float(agg(vals)) if vals else float("nan")

    print(f"\n{'='*78}\n{label}   ({n} episodes)\n{'='*78}")
    for k in (SUCCESS, FAILURE, TIMEOUT):
        print(f"  {k:8s} {outs.get(k,0)/n:6.1%}  ({outs.get(k,0)})")
    print(f"  ep length  mean {np.mean(lens):6.1f}  median {np.median(lens):6.1f} ticks"
          f"  ({np.mean(lens)*0.1:.1f}s)")
    print(f"  turnovers by kind: {dict(kinds)}")

    print(f"\n  PASSES: {len(allp)} total, {len(allp)/n:.2f} per episode")
    if allp:
        print(f"    outcomes: " + "  ".join(
            f"{k} {v/len(allp):.1%}" for k, v in pouts.most_common()))
        L = np.array([p["length"] for p in allp])
        DX = np.array([p["dx"] for p in allp])
        DY = np.array([p["dy"] for p in allp])
        print(f"    length  mean {L.mean():5.1f}m  median {np.median(L):5.1f}  "
              f"p90 {np.percentile(L,90):5.1f}  max {L.max():5.1f}")
        print(f"    dx      mean {DX.mean():+5.1f}m  forward {np.mean(DX>2):.1%}  "
              f"sideways {np.mean(np.abs(DX)<=2):.1%}  backward {np.mean(DX<-2):.1%}")
        print(f"    lateral |dy| mean {DY.mean():5.1f}m  "
              f"cross-field(|dy|>20) {np.mean(DY>20):.1%}")
        print(f"    presser: holder {np.mean([p['holder_press'] for p in allp]):5.1f}m  "
              f"target {np.mean([p['target_press'] for p in allp]):5.1f}m")
        for kk in ("complete", "intercept", "loose pass", "lost"):
            sub = [p for p in allp if p["outcome"] == kk]
            if sub:
                print(f"      {kk:11s} n={len(sub):4d}  len {np.mean([p['length'] for p in sub]):5.1f}m"
                      f"  dx {np.mean([p['dx'] for p in sub]):+5.1f}m"
                      f"  tgt_press {np.mean([p['target_press'] for p in sub]):5.1f}m")

    tot_holder = sum(r["holder_ticks"] for _, _, r in results)
    print(f"\n  ON-BALL: holder ticks {tot_holder} ({tot_holder/sum(lens):.1%} of ticks), "
          f"flight {sum(r['flight_ticks'] for _,_,r in results)/sum(lens):.1%}")
    print(f"    hold prob (softmax P[HOLD] at holder)  mean {m('hold_prob'):.3f}")
    print(f"    legal pass targets available  mean {m('legal_targets'):.2f} / 9")
    print(f"    holder speed {m('holder_speed'):.2f} m/s   "
          f"off-ball speed {m('offball_speed'):.2f} m/s   (V_MAX {V_MAX})")
    hdx = [v for _, _, r in results for v in r["holder_dx"]]
    if hdx:
        print(f"    holder dx/tick {np.mean(hdx):+.3f} m  -> "
              f"{np.mean(hdx)/0.1:+.2f} m/s net upfield while carrying")

    print(f"\n  ACTION HEADS (all agent-ticks)")
    names = [f"({d[0]:+.1f},{d[1]:+.1f})" for d in direction_lookup]
    order = np.argsort(-dirh)
    print("    direction top5: " + "  ".join(
        f"{names[i]}={dirh[i]/dirh.sum():.1%}" for i in order[:5]))
    print(f"    speed: " + "  ".join(
        f"{speed_lookup[i]:.1f}={spdh[i]/spdh.sum():.1%}" for i in range(N_SPEEDS)))
    if hdirh.sum():
        ho = np.argsort(-hdirh)
        print("    HOLDER direction top5: " + "  ".join(
            f"{names[i]}={hdirh[i]/hdirh.sum():.1%}" for i in ho[:5]))
        print(f"    HOLDER speed: " + "  ".join(
            f"{speed_lookup[i]:.1f}={hspdh[i]/max(hspdh.sum(),1):.1%}"
            for i in range(N_SPEEDS)))
    print(f"    entropy  dir {m('ent_dir'):.3f}/{np.log(N_DIRECTIONS):.3f}"
          f"   spd {m('ent_spd'):.3f}/{np.log(N_SPEEDS):.3f}"
          f"   ball {m('ent_ball'):.3f}")

    bx = [r["ball_x"] for _, _, r in results]
    print(f"\n  FIELD PROGRESS")
    print(f"    ball x: start {np.mean([b[0] for b in bx]):5.1f}  "
          f"end {np.mean([b[-1] for b in bx]):5.1f}  "
          f"max {np.mean([max(b) for b in bx]):5.1f}  (final third = 70)")
    print(f"    reached final third: "
          f"{np.mean([max(b) > 70 for b in bx]):.1%} of episodes")
    print(f"    mean attacker x: start {np.mean([r['att_x_mean'][0] for _,_,r in results]):5.1f}"
          f"  end {np.mean([r['att_x_mean'][-1] for _,_,r in results]):5.1f}")

    ht = [v for _, _, r in results for v in r["held_ticks"] if v >= 0]
    bd = [v for _, _, r in results for v in r["ball_dist"]]
    hp = [v for _, _, r in results for v in r["holder_press"]]
    print(f"\n  CANDIDATE CONSTRAINT RATES (measured -- use to set thresholds d_k)")
    if allp:
        bad = pouts.get("intercept", 0) + pouts.get("loose pass", 0) + pouts.get("lost", 0)
        print(f"    pass lost                   {bad/len(allp):6.1%} of passes")
        print(f"    pass cross-field |dy|>20m   {np.mean(DY>20):6.1%} of passes")
        print(f"    pass backward dx<-2m        {np.mean(DX<-2):6.1%} of passes")
        print(f"    pass long >25m              {np.mean(L>25):6.1%} of passes")
    if ht:
        ht = np.array(ht)
        print(f"    release within <5 ticks     {np.mean(ht<5):6.1%} of passes"
              f"   (held: mean {ht.mean():.1f} median {np.median(ht):.0f} ticks)")
        print(f"    release within <10 ticks    {np.mean(ht<10):6.1%} of passes")
    if hp:
        hp = np.array(hp)
        print(f"    carrier pressed <3m         {np.mean(hp<3):6.1%} of carry-ticks")
    print(f"    attacker offside            {m('offside_rate'):6.1%} of attacker-ticks")
    print(f"    attacker >30m from ball     {np.mean(np.array(bd)>30):6.1%} of attacker-ticks"
          f"   (mean dist {np.mean(bd):.1f}m)")
    print(f"    attackers in final third    {m('f3_count'):.2f} / 10 per tick")
    print(f"    attacker x spread (std)     {m('x_spread'):.1f} m")
    return dict(outcomes=dict(outs), n=n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="5M.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    # Fixed seeds and every start_holder in rotation, so two checkpoints get
    # compared on the same episodes rather than on two different samples.
    seeds = [(args.ckpt, 1000 + i, i % 10, not args.stochastic, args.random)
             for i in range(args.n)]

    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, seeds)

    label = args.label or (
        f"{args.ckpt} {'sampled' if args.stochastic else 'argmax'}"
        + (" [RANDOM POLICY]" if args.random else ""))
    summarize(results, label)
