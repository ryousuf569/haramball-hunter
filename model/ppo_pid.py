import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# limit_threads(1) runs on import of ppo_ablation, before torch is touched
import argparse, json, time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from gymnasium.vector import AutoresetMode

from environment.lowblock_env import make_vector_env, obs_dim
from environment.termination import make_zone
from model.ppo_ablation import (ABLATIONS, CONSTRAINTS, EpisodeTracker,
                                MULTIPLIERS, N_COSTS, SLOW_INDICATORS,
                                ActorCritic, Multipliers, RolloutBuffer,
                                append_csv, collect_rollout,
                                combine_advantages, compute_gae,
                                constraint_metrics, constraints_from, cost_gae,
                                explained_variance, init_state, ppo_update,
                                probe_baselines, save, set_seed,
                                slow_indicator)
from model.ppo_ablation import Config as AblationConfig

# Paper 6.4 follow-up: is the team-mean-indicator failure the counting unit or
# the multiplier update rule? A 2x2 over {adam, pid} x {slow, team_slow}, where
# adam x slow == abl_control and adam x team_slow == abl_team_slow, so only the
# two pid cells are new. Everything else is imported from ppo_ablation.

# Adam at mult_lr 0.03 steps z by ~0.03 per update whatever the violation, so
# ki * 0.25 = 0.03 matches that ramp at a persistent violation of 0.25. kp gives
# an immediate 0.10 offset there. i_max 50 lets ki*I reach +6, past the z ~ +5
# Adam reaches on the bootstrap; i_floor 25 reaches -3, past the z ~ -1.7 Adam
# reaches when a constraint is satisfied.
DEFAULT_GAINS = {"kp": 0.4, "ki": 0.12, "kd": 1.0,
                 "i_max": 50.0, "i_floor": 25.0, "d_ema": 0.9}

MULTIPLIER_MODES = ("adam", "pid")
INDICATOR_MODES = ("slow", "team_slow")


@dataclass
class Config(AblationConfig):
    multiplier: str = "adam"
    indicator: str = "slow"
    kp: float = DEFAULT_GAINS["kp"]
    ki: float = DEFAULT_GAINS["ki"]
    kd: float = DEFAULT_GAINS["kd"]
    i_max: float = DEFAULT_GAINS["i_max"]
    i_floor: float = DEFAULT_GAINS["i_floor"]
    d_ema: float = DEFAULT_GAINS["d_ema"]
    d_on: str = "j" # or "violation"; see PIDMultipliers.update


class PIDMultipliers(Multipliers):
    """Positional PID on the violation, writing the same softmax logits z that
    Adam writes in the baseline. forward, violation, d and lower are inherited,
    so the simplex and the max(lam0, lam_succ) bootstrap are untouched."""

    def __init__(self, constraints, z_init=0.02, kp=DEFAULT_GAINS["kp"],
                 ki=DEFAULT_GAINS["ki"], kd=DEFAULT_GAINS["kd"],
                 i_max=DEFAULT_GAINS["i_max"], i_floor=DEFAULT_GAINS["i_floor"],
                 d_ema=DEFAULT_GAINS["d_ema"], d_on="j"):
        super().__init__(constraints, z_init)
        if d_on not in ("j", "violation"):
            raise ValueError("d_on must be 'j' or 'violation', got %r" % (d_on,))
        K = len(constraints)
        self.z.requires_grad_(False) # driven by update(), never by autograd
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_max, self.i_floor = float(i_max), float(i_floor)
        self.d_ema, self.d_on = float(d_ema), d_on
        self.z_init = float(z_init)
        self.register_buffer("I", torch.zeros(K))
        self.register_buffer("ema", torch.zeros(K))
        self.register_buffer("prev_ema", torch.zeros(K))
        self.register_buffer("seen", torch.zeros(K)) # 0 until the EMA is primed
        # logged per update, so a saturated integral is visible in metrics.csv
        self.register_buffer("last_e", torch.full((K,), float("nan")))
        self.register_buffer("last_d", torch.full((K,), float("nan")))

    @torch.no_grad()
    def update(self, J):
        """z = z_init + kp*e + ki*I + kd*D, from the same J the Adam baseline
        feeds to loss(). The negative half of the I clamp is what lets a
        satisfied constraint drive its lam to 0 the way it does under Adam.
        d_on "j" tracks the measured rate as specified, which on the lower-bound
        bootstrap means rising success raises lam_succ; "violation" tracks the
        violation instead, identical on slow and crowd."""
        J = torch.as_tensor(J, dtype=self.z.dtype).reshape(-1)
        e = self.violation(J) # > 0 exactly when violated, either direction
        self.I.add_(e).clamp_(-self.i_floor, self.i_max)

        signal = J if self.d_on == "j" else e
        primed = self.seen > 0
        self.prev_ema.copy_(torch.where(primed, self.ema, signal))
        self.ema.copy_(torch.where(primed,
                                   self.d_ema * self.ema
                                   + (1.0 - self.d_ema) * signal,
                                   signal))
        self.seen.fill_(1.0)
        D = (self.ema - self.prev_ema).clamp(min=0.0) # rectified, only damps

        self.z.copy_(self.z_init + self.kp * e + self.ki * self.I + self.kd * D)
        self.last_e.copy_(e)
        self.last_d.copy_(D)
        return e, self.I.clone(), D

    def state(self):
        return {"e": self.last_e.tolist(), "I": self.I.tolist(),
                "D": self.last_d.tolist()}


def make_multipliers(cfg):
    constraints = constraints_from(cfg)
    if cfg.multiplier == "pid":
        return PIDMultipliers(constraints, cfg.z_init, kp=cfg.kp, ki=cfg.ki,
                              kd=cfg.kd, i_max=cfg.i_max, i_floor=cfg.i_floor,
                              d_ema=cfg.d_ema, d_on=cfg.d_on), None
    mult = MULTIPLIERS.get(cfg.ablation, Multipliers)(constraints, cfg.z_init)
    return mult, torch.optim.Adam(mult.parameters(), lr=cfg.mult_lr)


NAN = float("nan")
# as ppo_ablation.HEADER, plus lam0e: lam0_eff = max(lam0, lam_succ) is the
# weight the reward actually carries, and in the team_slow runs it is not lam0
HEADER = (" upd     step |    succ   len   eps | c_slow c_crwd c_team |"
          "   lam0  lam0e lam_slw lam_crd lam_suc boot |"
          " cadvS cadvC cadvU |      kl   clip    ent |   sps")


def log(update, global_step, metrics, stats, t0):
    if log.n % 20 == 0:
        print(HEADER)
    log.n += 1
    now = time.perf_counter()
    if log.prev_t is None:
        log.prev_t = t0
    sps = (global_step - log.prev_step) / max(now - log.prev_t, 1e-9)
    log.prev_step, log.prev_t = global_step, now
    g = metrics.get
    print("%4d %8d | %6.1f%% %5.0f %5d | %6.3f %6.3f %6.3f |"
          " %6.3f %6.3f %7.3f %7.3f %7.3f %4.0f |"
          " %5.3f %5.3f %5.3f | %7.4f %6.3f %6.3f | %5.0f"
          % (update, global_step,
             100.0 * stats["success"], stats["len"], stats["n_eps"],
             g("c_slow", NAN), g("c_crowd", NAN), g("c_team", NAN),
             g("lam0", NAN), g("lam0_eff", NAN), g("lam_slow", NAN),
             g("lam_crowd", NAN), g("lam_succ", NAN), g("boot", NAN),
             g("cadv_std_slow", NAN), g("cadv_std_crowd", NAN),
             g("cadv_std_succ", NAN),
             g("approx_kl", NAN), g("clipfrac", NAN), g("entropy", NAN),
             sps), flush=True)
    return sps


log.n = 0
log.prev_step = 0
log.prev_t = None


def banner(cfg, n_updates, obs_d):
    steps_per_update = cfg.rollout * cfg.n_envs
    print("obs %d | act [9, 3, 10] | %d async envs x %d ticks = %d steps/update"
          " | %d rows/update" % (obs_d, cfg.n_envs, cfg.rollout,
                                 steps_per_update, cfg.batch))
    print("%d updates -> %d steps | lr %.1e (%s) | seed %d | logging every %d"
          % (n_updates, n_updates * steps_per_update, cfg.lr,
             "annealed" if cfg.anneal_lr else "constant", cfg.seed,
             cfg.log_every))
    print("gate (%.0f, %.0f) r=%.0f pc_min=%.2f | gamma %.3f | start_holder %s"
          % (cfg.zone_x, cfg.zone_y, cfg.zone_radius, cfg.pc_min, cfg.gamma,
             "random" if cfg.start_holder < 0 else cfg.start_holder))
    base = probe_baselines(cfg.zone_x, cfg.zone_y, cfg.zone_radius, cfg.pc_min)
    if base is None:
        print("probe baselines: this gate is not in "
              "environment/probe/zone_probe_summary.csv")
    else:
        print("probe baselines here (start_holder=0): random floor %.1f%% | "
              "scripted %.1f%%" % (100 * base.get("random", NAN),
                                   100 * base.get("scripted", NAN)))
    spec = " | ".join("%s %s %.2f" % (c.name, ">=" if c.lower else "<=",
                                      c.threshold)
                      for c in constraints_from(cfg))
    print("constraints: %s | cost_gamma %.3f" % (spec, cfg.cost_gamma))
    print("CELL multiplier=%s x indicator=%s" % (cfg.multiplier, cfg.indicator))
    if cfg.indicator == "team_slow":
        print("  slow counts one bit per tick off the team-mean speed, "
              "broadcast over the squad")
    else:
        print("  slow counts per attacker-tick, own speed < 2.0 m/s (paper 4.1)")
    if cfg.multiplier == "pid":
        print("  z = z_init + kp*e + ki*I + kd*D | kp %.4g ki %.4g kd %.4g | "
              "I in [%.4g, %.4g] | d_ema %.3g | D on %s"
              % (cfg.kp, cfg.ki, cfg.kd, -cfg.i_floor, cfg.i_max, cfg.d_ema,
                 cfg.d_on))
        print("  ramps z by ki*e = %.4g per update at a persistent violation "
              "of 0.25 (Adam at mult_lr %.3g moves ~%.3g)"
              % (cfg.ki * 0.25, cfg.mult_lr, cfg.mult_lr))
    else:
        print("  z driven by Adam (lr %.3g) on the softmax logits, i.e. the "
              "repo baseline, not plain gradient ascent" % cfg.mult_lr)
    print("  simplex and max(lam0, lam_succ) bootstrap identical in all cells; "
          "ablation tag %r (%s)"
          % (cfg.ablation, ABLATIONS.get(cfg.ablation, "pid variant")))
    print("spawning workers, first line after update 1 ...", flush=True)


def train(cfg):
    run_name = cfg.run_name or time.strftime("ppo_pid_%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, "runs", run_name)
    if os.path.exists(run_dir):
        raise SystemExit("runs/%s already exists; pick another --run-name "
                         "(this script never overwrites a run)" % run_name)

    set_seed(cfg.seed)
    A, D = 10, obs_dim(10, 11)
    venv = make_vector_env(n_envs=cfg.n_envs, asynchronous=True,
                           autoreset_mode=AutoresetMode.SAME_STEP,
                           zone_x=cfg.zone_x, zone_y=cfg.zone_y,
                           zone_radius=cfg.zone_radius, pc_min=cfg.pc_min,
                           start_holder=(None if cfg.start_holder < 0
                                         else cfg.start_holder))
    agent = ActorCritic(obs_dim=D, hidden=cfg.hidden)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    mult, mopt = make_multipliers(cfg)
    slow_fn = SLOW_INDICATORS.get(cfg.indicator, slow_indicator)

    buf = RolloutBuffer(cfg.rollout, cfg.n_envs, A, D)
    state = init_state(venv, cfg.seed, cfg.n_envs)
    tracker = EpisodeTracker(maxlen=100)

    n_updates = cfg.total_steps // (cfg.rollout * cfg.n_envs)
    zone_centre = torch.as_tensor(
        make_zone(cfg.zone_x, cfg.zone_y, cfg.zone_radius).centre,
        dtype=torch.float32)

    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    csv_path = os.path.join(run_dir, "metrics.csv")
    best_success = -1.0

    banner(cfg, n_updates, D)
    print("run dir: %s" % run_dir, flush=True)
    t0 = time.perf_counter()

    for update in range(1, n_updates + 1):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)

        state = collect_rollout(agent, venv, buf, state, tracker,
                                zone_centre, cfg.zone_radius, slow_fn=slow_fn)

        with torch.no_grad():
            next_value, next_cost_value = agent.value_of(state.next_obs)
        adv, ret = compute_gae(buf.rew, buf.done, buf.val, next_value,
                               state.next_done, cfg.gamma, cfg.gae_lambda)
        cost_adv, cost_ret = cost_gae(buf, next_cost_value, state.next_done, cfg)

        # read before combine_advantages standardises it: a near-constant cost
        # channel has a tiny std here and standardise() divides it back out
        cadv_std = [float(cost_adv[..., k].std()) for k in range(N_COSTS)]

        lam0, lam = mult()
        lam0, lam = lam0.detach(), lam.detach() # frozen to the policy step
        adv_total = combine_advantages(adv, cost_adv, lam0, lam)

        metrics = ppo_update(agent, opt, buf.flat(adv_total, ret, cost_ret), cfg)

        global_step = update * cfg.rollout * cfg.n_envs
        stats = tracker.stats()

        if not np.isnan(stats["success"]):
            J = torch.tensor([float(buf.cost[..., 0].mean()),
                              float(buf.cost[..., 1].mean()),
                              float(stats["success"])])
            if cfg.multiplier == "pid":
                mult.update(J)
            else:
                mopt.zero_grad()
                mult.loss(J).backward()
                mopt.step()

        metrics["ev"] = explained_variance(buf.val.reshape(-1), ret.reshape(-1))
        for k, c in enumerate(CONSTRAINTS):
            metrics["ev_" + c.name] = explained_variance(
                buf.cost_val[..., k].reshape(-1), cost_ret[..., k].reshape(-1))
        metrics.update(constraint_metrics(buf))
        for k, name in enumerate(("slow", "crowd", "succ")):
            metrics["cadv_std_" + name] = cadv_std[k]
        with torch.no_grad():
            l0, lk = mult()
            metrics["lam0"] = float(l0)
            metrics["lam_slow"] = float(lk[0])
            metrics["lam_crowd"] = float(lk[1])
            metrics["lam_succ"] = float(lk[2])
            metrics["lam0_eff"] = float(torch.max(l0, lk[2]))
            metrics["boot"] = float(lk[2] > l0) # 1 while the bootstrap carries R
        # nan in adam mode, so both modes share one csv schema
        pid = (mult.state() if cfg.multiplier == "pid"
               else {"e": [NAN] * N_COSTS, "I": [NAN] * N_COSTS,
                     "D": [NAN] * N_COSTS})
        for k, name in enumerate(("slow", "crowd", "succ")):
            for field in ("e", "I", "D"):
                metrics["pid_%s_%s" % (field, name)] = float(pid[field][k])

        lr = opt.param_groups[0]["lr"]
        if update == 1 or update % cfg.log_every == 0:
            sps = log(update, global_step, metrics, stats, t0)
            append_csv(csv_path, dict(update=update, step=global_step,
                                      lr=lr, sps=round(sps, 1),
                                      **{k: round(v, 6)
                                         for k, v in stats.items()},
                                      **{k: round(v, 6)
                                         for k, v in metrics.items()}))

        if update % cfg.save_every == 0:
            save(run_dir, "ckpt_%d" % global_step, agent, cfg, global_step, stats)
        if stats["n_eps"] >= 100 and stats["success"] > best_success:
            best_success = stats["success"]
            save(run_dir, "best", agent, cfg, global_step, stats)

    save(run_dir, "final", agent, cfg, global_step, stats)
    venv.close()
    print("saved final.pt at step %d | best success %.1f%% -> best.pt"
          % (global_step, 100 * best_success), flush=True)
    return agent


def build_config(args):
    # ablation tag so config.json stays self-describing next to the abl_* runs
    ablation = ("control" if (args.multiplier, args.indicator) == ("adam", "slow")
                else "team_slow" if args.multiplier == "adam"
                else "pid_%s" % args.indicator)
    return Config(n_envs=args.n_envs, rollout=args.rollout, lr=args.lr,
                  total_steps=args.total_steps, seed=args.seed,
                  log_every=args.log_every, save_every=args.save_every,
                  run_name=args.run_name,
                  zone_x=args.zone_x, zone_y=args.zone_y,
                  zone_radius=args.zone_radius, pc_min=args.pc_min,
                  start_holder=args.start_holder,
                  slow_d=args.slow_d, crowd_d=args.crowd_d, succ_d=args.succ_d,
                  mult_lr=args.mult_lr, z_init=args.z_init,
                  ablation=ablation,
                  multiplier=args.multiplier, indicator=args.indicator,
                  kp=args.kp, ki=args.ki, kd=args.kd,
                  i_max=args.i_max, i_floor=args.i_floor, d_ema=args.d_ema,
                  d_on=args.d_on,
                  anneal_lr=not args.no_anneal)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="2x2 over {adam, pid} x {slow, team_slow}; "
                    "adam x slow and adam x team_slow reproduce the abl_* runs")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-anneal", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--zone-x", type=float, default=Config.zone_x)
    ap.add_argument("--zone-y", type=float, default=Config.zone_y)
    ap.add_argument("--zone-radius", type=float, default=Config.zone_radius)
    ap.add_argument("--pc-min", type=float, default=Config.pc_min)
    ap.add_argument("--slow-d", type=float, default=Config.slow_d)
    ap.add_argument("--crowd-d", type=float, default=Config.crowd_d)
    ap.add_argument("--succ-d", type=float, default=Config.succ_d)
    ap.add_argument("--start-holder", type=int, default=Config.start_holder,
                    help="attacker row that kicks off with the ball; "
                         "-1 draws a new one each episode")
    ap.add_argument("--multiplier", choices=MULTIPLIER_MODES, default="adam",
                    help="adam: the repo baseline, Adam on the softmax logits. "
                         "pid: positional PID on the violation")
    ap.add_argument("--indicator", choices=INDICATOR_MODES, default="slow",
                    help="slow: per attacker-tick. team_slow: one bit per tick "
                         "off the team mean")
    ap.add_argument("--mult-lr", type=float, default=Config.mult_lr,
                    help="Adam lr on the logits; ignored under --multiplier pid")
    ap.add_argument("--z-init", type=float, default=Config.z_init)
    ap.add_argument("--kp", type=float, default=DEFAULT_GAINS["kp"])
    ap.add_argument("--ki", type=float, default=DEFAULT_GAINS["ki"])
    ap.add_argument("--kd", type=float, default=DEFAULT_GAINS["kd"])
    ap.add_argument("--i-max", type=float, default=DEFAULT_GAINS["i_max"],
                    help="upper anti-windup clamp on the integral")
    ap.add_argument("--i-floor", type=float, default=DEFAULT_GAINS["i_floor"],
                    help="magnitude of the lower clamp; the integral may reach "
                         "-i_floor so a satisfied constraint can still drive "
                         "its lam to 0")
    ap.add_argument("--d-ema", type=float, default=DEFAULT_GAINS["d_ema"])
    ap.add_argument("--d-on", choices=("j", "violation"), default="j",
                    help="signal the derivative term watches; see "
                         "PIDMultipliers.update")
    args = ap.parse_args()
    train(build_config(args))
