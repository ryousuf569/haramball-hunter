import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.thread_limits import limit_threads
limit_threads(1)

import argparse, csv, json, time, io
from collections import deque, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.vector import AutoresetMode
from physics.engine import V_MAX
from environment.lowblock_env import (GAMMA, SUCCESS, IS_CARRIER, OWN_POS,
                                      REL_BALL, W, REL_SCALE, POS_SCALE,
                                      make_vector_env, obs_dim)
from environment.termination import make_zone

@dataclass
class Config:
    n_envs: int = 6
    rollout: int = 128 
    total_steps: int = 1_000_000
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = GAMMA
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    zone_x: float = 86.0
    zone_y: float = 34.0
    zone_radius: float = 8.0
    pc_min: float = 0.40
    start_holder: int = 0
    seed: int = 1
    log_every: int = 10
    save_every: int = 100
    run_name: str = ""
    cost_gamma: float = 0.99 # paper §4.1: gamma_k < 1, never the reward's gamma (yours is 1.0)
    mult_lr: float = 0.03 # paper Table 1, Arena
    z_init: float = 0.02 # paper Table 1

    @property
    def batch(self):
        return self.rollout * self.n_envs * 10

@dataclass
class Constraint:
    name: str
    threshold: float # d̃, a probability in [0,1] - paper §4.1
    lower: bool = False # True: J >= d̃ (the bootstrap). False: J <= d̃.

OWN_VEL = 2
SPEED_LIMIT = 2.0
CONSTRAINTS = [Constraint("slow",0.75), Constraint("success", 0.60, lower=True)]
N_COSTS = len(CONSTRAINTS)

def slow_indicator(obs):
    speed = torch.linalg.norm(obs[..., OWN_VEL:OWN_VEL + 2], dim=-1) * V_MAX
    return (speed < SPEED_LIMIT).float()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MultiCategorical:
    def __init__(self, logits, nvec, ball_mask=None):
        chunks = list(torch.split(logits, list(nvec), dim=-1))
        if ball_mask is not None:
            chunks[2] = chunks[2].masked_fill(~ball_mask, -1e8)
        self.dists = [Categorical(logits=c) for c in chunks]

    def sample(self):
        return torch.stack([c.sample() for c in self.dists], dim=-1)

    def mode(self):
        return torch.stack([c.logits.argmax(dim=-1) for c in self.dists], dim=-1)

    def log_prob(self, actions):
        return sum(c.log_prob(actions[:, i]) for i, c in enumerate(self.dists))

    def entropy(self):
        return sum(self.head_entropies())

    def head_entropies(self):
        return [c.entropy() for c in self.dists]

def ball_mask_from_obs(obs, n_att=10):
    """obs (B, 120) -> bool (B, n_att). Column 0 = HOLD, always legal."""
    carrier = obs[:, IS_CARRIER] > 0.5
    mask = torch.zeros(obs.shape[0], n_att, dtype=torch.bool, device=obs.device)
    mask[:, 0] = True
    mask |= carrier[:, None]
    return mask

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=120, nvec=(9, 3, 10), hidden=256):
        super().__init__()
        self.nvec = list(nvec)
        self.n_att = nvec[2]
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.head = layer_init(nn.Linear(hidden, sum(nvec)), std=0.01)
        self.value = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.cost_value = layer_init(nn.Linear(hidden, N_COSTS), std=1.0)

    def _dist_and_value(self, obs_flat):
        h = self.trunk(obs_flat)
        dist = MultiCategorical(self.head(h), self.nvec, ball_mask_from_obs(obs_flat, self.n_att))
        return dist, self.value(h).squeeze(-1), self.cost_value(h)

    def act(self, obs):
        n, a, d = obs.shape
        dist, v, cv = self._dist_and_value(obs.reshape(n * a, d))
        act = dist.sample()
        return (act.reshape(n, a, 3), dist.log_prob(act).reshape(n, a), v.reshape(n, a), cv.reshape(n, a, N_COSTS))

    def evaluate(self, obs_flat, act_flat):
        dist, v, cv = self._dist_and_value(obs_flat)
        return dist.log_prob(act_flat), dist.entropy(), v, cv

    def value_of(self, obs):
        h = self.trunk(obs)
        return self.value(h).squeeze(-1), self.cost_value(h) 

class RolloutBuffer:
    def __init__(self, T, N, A, D):
        self.T, self.N, self.A, self.D = T, N, A, D
        self.obs = torch.zeros(T, N, A, D)
        self.act = torch.zeros(T, N, A, 3, dtype=torch.long)
        self.logp = torch.zeros(T, N, A)
        self.val = torch.zeros(T, N, A)
        self.rew = torch.zeros(T, N)
        self.done = torch.zeros(T, N)
        self.cost = torch.zeros(T, N, A, N_COSTS)
        self.cost_val = torch.zeros(T, N, A, N_COSTS)

    def flat(self, adv, ret, cost_ret):
        B = self.T * self.N * self.A
        return {
            "obs": self.obs.reshape(B, self.D),
            "act": self.act.reshape(B, 3),
            "logp": self.logp.reshape(B),
            "adv": adv.reshape(B),
            "ret": ret.reshape(B),
            "cost_ret": cost_ret.reshape(B, N_COSTS),
        }


@dataclass
class LoopState:
    next_obs: torch.Tensor
    next_done: torch.Tensor
    ep_ret: np.ndarray
    ep_len: np.ndarray

def init_state(venv, seed, n_envs):
    obs, _ = venv.reset(seed=seed)
    return LoopState(torch.as_tensor(obs), torch.zeros(n_envs),
                     np.zeros(n_envs), np.zeros(n_envs, dtype=np.int64))

class EpisodeTracker:
    def __init__(self, maxlen=100):
        self.success = deque(maxlen=maxlen)
        self.ret = deque(maxlen=maxlen)
        self.len = deque(maxlen=maxlen)
        self.n_eps = 0

    def record(self, done_np, info, ep_ret, ep_len):

        flags = np.zeros(len(done_np), dtype="f4")
        if not done_np.any():
            return flags
        outcomes = info["final_info"]["outcome"]
        for i in np.flatnonzero(done_np):
            won = outcomes[i] == SUCCESS
            flags[i] = float(won)
            self.success.append(won)
            self.ret.append(float(ep_ret[i]))
            self.len.append(int(ep_len[i]))
            self.n_eps += 1
        ep_ret[done_np] = 0.0 # in place; these are LoopState's own arrays
        ep_len[done_np] = 0
        return flags

    def stats(self):
        if not self.success:
            return {"success": float("nan"), "ret": float("nan"),
                    "len": float("nan"), "n_eps": 0}
        return {"success": float(np.mean(self.success)),
                "ret": float(np.mean(self.ret)),
                "len": float(np.mean(self.len)),
                "n_eps": self.n_eps}


@torch.no_grad()
def collect_rollout(agent, venv, buf, state, tracker):
    for t in range(buf.T):
        buf.obs[t] = state.next_obs
        buf.done[t] = state.next_done

        act, logp, val, cval = agent.act(state.next_obs)
        buf.act[t], buf.logp[t], buf.val[t], buf.cost_val[t] = act, logp, val, cval

        obs, rew, term, trunc, info = venv.step(act.numpy())
        buf.rew[t] = torch.as_tensor(rew, dtype=torch.float32)

        state.ep_ret += rew
        state.ep_len += 1
        done_np = term | trunc
        flags = tracker.record(done_np, info, state.ep_ret, state.ep_len)
        buf.cost[t, :, :, 0] = slow_indicator(state.next_obs)
        buf.cost[t, :, :, 1] = torch.as_tensor(flags)[:, None] # (N,) broadcast over agents

        state.next_obs = torch.as_tensor(obs)
        state.next_done = torch.as_tensor(done_np, dtype=torch.float32)
    return state

_POS_SCALE = torch.as_tensor(POS_SCALE)


def phi_from_obs(obs, zone_centre):
    own = obs[..., OWN_POS:OWN_POS + 2] * _POS_SCALE
    ball = own + obs[..., REL_BALL:REL_BALL + 2] * REL_SCALE
    return -W * torch.linalg.norm(ball - zone_centre, dim=-1)


def phi_r2(phi, returns):
    """r^2 of the best linear predictor of the return from Phi alone"""
    p, r = phi.reshape(-1), returns.reshape(-1)
    if p.std() == 0 or r.std() == 0:
        return float("nan")
    p = p - p.mean()
    r = r - r.mean()
    return float((p @ r) ** 2 / ((p @ p) * (r @ r)))

@torch.no_grad()
def head_entropy_metrics(agent, obs, obs_dim):
    flat = obs.reshape(-1, obs_dim)
    h = agent.trunk(flat)
    dist = MultiCategorical(agent.head(h), agent.nvec,
                            ball_mask_from_obs(flat, agent.n_att))
    heads = torch.stack(dist.head_entropies(), dim=1)
    carrier = flat[:, IS_CARRIER] > 0.5
    out = {}
    for i, name in enumerate(("dir", "spd", "ball")):
        for tag, rows in (("car", carrier), ("off", ~carrier)):
            out["h_%s_%s" % (name, tag)] = (
                float(heads[rows, i].mean()) if bool(rows.any())
                else float("nan"))
    return out


def constraint_metrics(buf):
    speed = torch.linalg.norm(buf.obs[..., OWN_VEL:OWN_VEL + 2], dim=-1) * V_MAX
    return {
        "c_slow": float(buf.cost[..., 0].mean()),
        "c_team": float((speed.mean(dim=-1) < SPEED_LIMIT).float().mean()),
        "speed": float(speed.mean()),
        # cost channel 1 is broadcast across agents, so undo that to get episodes
        "n_succ": float(buf.cost[..., 1].sum()) / buf.A,
    }


def explained_variance(values, returns):
    var_returns = returns.var()
    if var_returns == 0:
        return float("nan")
    return float(1 - (returns - values).var() / var_returns)

def compute_gae(rew, done, val, next_value, next_done, gamma, lam):
    T, N, A = val.shape
    adv = torch.zeros_like(val)
    lastgaelam = torch.zeros(N, A)
    per_agent = rew.dim() == 3
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterm, nextvalues = 1.0 - next_done, next_value
        else:
            nextnonterm, nextvalues = 1.0 - done[t + 1], val[t + 1]
        nextnonterm = nextnonterm[:, None] # (N,) -> (N,1)
        r = rew[t] if per_agent else rew[t][:, None]
        delta = r + gamma * nextvalues * nextnonterm - val[t]
        lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + val

def standardise(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def combine_advantages(adv, cost_adv, lam0, lam):
    """Algorithm 1's policy objective, expressed at the advantage level:

        max(lam0, lam_boot) * A_R  +  lam_boot * A_boot  -  sum_k lam_k * A_k
    """
    lam0_tilde = torch.max(lam0, lam[-1]) # paper 4.3; bootstrap is last
    total = lam0_tilde * standardise(adv)
    for k, c in enumerate(CONSTRAINTS):
        sign = 1.0 if c.lower else -1.0
        total = total + sign * lam[k] * standardise(cost_adv[..., k])
    return total


def cost_gae(buf, next_cost_value, next_done, cfg):
    """GAE per cost channel, at cost_gamma rather than the reward's gamma."""
    scale = 1.0 - cfg.cost_gamma
    cost_adv, cost_ret = torch.zeros_like(buf.cost), torch.zeros_like(buf.cost)
    for k in range(N_COSTS):
        cost_adv[..., k], cost_ret[..., k] = compute_gae(
            scale * buf.cost[..., k], buf.done, buf.cost_val[..., k],
            next_cost_value[..., k], next_done, cfg.cost_gamma, cfg.gae_lambda)
    return cost_adv, cost_ret


def ppo_update(agent, opt, flat, cfg):
    B = flat["obs"].shape[0]
    metrics = defaultdict(list)
    for epoch in range(cfg.epochs):
        idx = torch.randperm(B)
        for start in range(0, B, cfg.minibatch):
            mb = idx[start:start + cfg.minibatch]
            newlogp, entropy, newval, newcval = agent.evaluate(flat["obs"][mb],
                                                               flat["act"][mb])

            logratio = newlogp - flat["logp"][mb]
            ratio = logratio.exp()

            mb_adv = flat["adv"][mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg_loss = torch.max(-mb_adv * ratio,
                                -mb_adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
            v_loss  = 0.5 * ((newval - flat["ret"][mb]) ** 2).mean()
            c_loss = 0.5 * ((newcval - flat["cost_ret"][mb]) ** 2).mean(0).sum()
            loss = (pg_loss - cfg.ent_coef * entropy.mean()
                    + cfg.vf_coef * (v_loss + c_loss))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                metrics["pg_loss"].append(float(pg_loss))
                metrics["v_loss"].append(float(v_loss))
                metrics["c_loss"].append(float(c_loss))
                metrics["entropy"].append(float(entropy.mean()))
                metrics["approx_kl"].append(float(((ratio - 1) - logratio).mean()))
                metrics["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))

    return {k: float(np.mean(v)) for k, v in metrics.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

HEADER = (" upd     step |    succ   len   eps | c_slow   spd |"
           "     ev ev_slw ev_suc |      kl   clip    ent |"
           "  v_loss  c_loss |   sps")


def save(run_dir, name, agent, cfg, step, stats):
    path = os.path.join(run_dir, name + ".pt")
    torch.save({"model": agent.state_dict(), "cfg": asdict(cfg),
                "step": step, "stats": stats}, path)
    return path


def append_csv(path, row):
    new = not os.path.exists(path)
    with io.open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


PROBE_SUMMARY = os.path.join(REPO_ROOT, "environment", "probe",
                             "zone_probe_summary.csv")

def probe_baselines(x, y, radius, pc_min):
    try:
        with io.open(PROBE_SUMMARY, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    key = (float(x), float(y), float(radius), float(pc_min))
    hit = {r["policy"]: float(r["success_rate"]) for r in rows
           if (float(r["x"]), float(r["y"]), float(r["radius"]),
               float(r["theta"])) == key}
    return hit or None


def banner(cfg, n_updates, obs_d):
    steps_per_update = cfg.rollout * cfg.n_envs
    print("obs %d | act [9, 3, 10] | %d async envs x %d ticks = %d steps/update"
          " | %d rows/update" % (obs_d, cfg.n_envs, cfg.rollout,
                                 steps_per_update, cfg.batch))
    print("%d updates -> %d steps | lr %.1e (%s) | seed %d | logging every %d"
          % (n_updates, n_updates * steps_per_update, cfg.lr,
             "annealed" if cfg.anneal_lr else "constant", cfg.seed, cfg.log_every))
    print("gate (%.0f, %.0f) r=%.0f pc_min=%.2f | gamma %.3f | start_holder %s"
          % (cfg.zone_x, cfg.zone_y, cfg.zone_radius, cfg.pc_min, cfg.gamma,
             "random" if cfg.start_holder < 0 else cfg.start_holder))
    base = probe_baselines(cfg.zone_x, cfg.zone_y, cfg.zone_radius, cfg.pc_min)
    if base is None:
        print("probe baselines: this gate is not in "
              "environment/probe/zone_probe_summary.csv")
    else:
        print("probe baselines here (start_holder=0): random floor %.1f%% | "
              "scripted %.1f%%" % (100 * base.get("random", float("nan")),
                                   100 * base.get("scripted", float("nan"))))
    spec = " | ".join("%s %s %.2f" % (c.name, ">=" if c.lower else "<=", c.threshold)
                      for c in CONSTRAINTS)
    print("constraints (indicators only, multipliers OFF): %s" % spec)
    print("slow fires when own speed < %.1f m/s, per attacker-tick "
          "(speed_lookup = %.1f / %.1f / %.1f m/s)"
          % (SPEED_LIMIT, 0.3 * V_MAX, 0.6 * V_MAX, V_MAX))
    print("spawning workers, first line after update 1 ...", flush=True)


def log(update, global_step, metrics, stats, t0, lr=float("nan")):
    # One fixed-width line per call so the run is greppable; header every 20.
    if log.n % 20 == 0:
        print(HEADER)
    log.n += 1
    nan = float("nan")
    now = time.perf_counter()
    if log.prev_t is None:
        log.prev_t = t0
    sps = (global_step - log.prev_step) / max(now - log.prev_t, 1e-9)
    log.prev_step, log.prev_t = global_step, now
    g = metrics.get
    print("%4d %8d | %6.1f%% %5.0f %5d | %6.3f %5.2f |"
          " %6.3f %6.3f %6.3f | %7.4f %6.3f %6.3f |"
          " %7.4f %7.4f | %5.0f"
          % (update, global_step,
             100.0 * stats["success"], stats["len"], stats["n_eps"],
             g("c_slow", nan), g("speed", nan),
             g("ev", nan), g("ev_slow", nan), g("ev_success", nan),
             g("approx_kl", nan), g("clipfrac", nan), g("entropy", nan),
             g("v_loss", nan), g("c_loss", nan),
             sps), flush=True)
    return sps

log.n = 0
log.prev_step = 0
log.prev_t = None

def train(cfg):
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

    buf = RolloutBuffer(cfg.rollout, cfg.n_envs, A, D)
    state = init_state(venv, cfg.seed, cfg.n_envs)
    tracker = EpisodeTracker(maxlen=100)

    n_updates = cfg.total_steps // (cfg.rollout * cfg.n_envs)
    zone_centre = torch.as_tensor(
        make_zone(cfg.zone_x, cfg.zone_y, cfg.zone_radius).centre,
        dtype=torch.float32)

    run_name = cfg.run_name or time.strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with io.open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    csv_path = os.path.join(run_dir, "metrics.csv")
    best_success = -1.0

    banner(cfg, n_updates, D)
    print("run dir: %s" % run_dir, flush=True)
    t0 = time.perf_counter()

    for update in range(1, n_updates + 1):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)

        state = collect_rollout(agent, venv, buf, state, tracker)

        with torch.no_grad():
            next_value, next_cost_value = agent.value_of(state.next_obs)
        adv, ret = compute_gae(buf.rew, buf.done, buf.val, next_value,
                               state.next_done, cfg.gamma, cfg.gae_lambda)
        cost_adv, cost_ret = cost_gae(buf, next_cost_value, state.next_done, cfg)
        lam0 = torch.tensor(1.0)
        lam = torch.zeros(N_COSTS)
        adv_total = combine_advantages(adv, cost_adv, lam0, lam)

        metrics = ppo_update(agent, opt, buf.flat(adv_total, ret, cost_ret), cfg)
        metrics["ev"] = explained_variance(buf.val.reshape(-1), ret.reshape(-1))
        for k, c in enumerate(CONSTRAINTS):
            metrics["ev_" + c.name] = explained_variance(
                buf.cost_val[..., k].reshape(-1), cost_ret[..., k].reshape(-1))
        metrics.update(constraint_metrics(buf))

        global_step = update * cfg.rollout * cfg.n_envs
        stats = tracker.stats()
        lr = opt.param_groups[0]["lr"]
        if update == 1 or update % cfg.log_every == 0:
            sps = log(update, global_step, metrics, stats, t0, lr)
            append_csv(csv_path, dict(update=update, step=global_step,
                                      lr=lr, sps=round(sps, 1),
                                      **{k: round(v, 6) for k, v in stats.items()},
                                      **{k: round(v, 6) for k, v in metrics.items()}))

        if update % cfg.save_every == 0:
            save(run_dir, "ckpt_%d" % global_step, agent, cfg, global_step, stats)
            # best policy tracking
        if stats["n_eps"] >= 100 and stats["success"] > best_success:
            best_success = stats["success"]
            save(run_dir, "best", agent, cfg, global_step, stats)

    save(run_dir, "final", agent, cfg, global_step, stats)
    venv.close()
    print("saved final.pt at step %d | best success %.1f%% -> best.pt"
          % (global_step, 100 * best_success), flush=True)
    return agent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--start-holder", type=int, default=Config.start_holder,
                    help="attacker row that kicks off with the ball; "
                         "-1 draws a new one each episode")
    args = ap.parse_args()
    train(Config(n_envs=args.n_envs, rollout=args.rollout, lr=args.lr,
                 total_steps=args.total_steps, seed=args.seed,
                 log_every=args.log_every, save_every=args.save_every,
                 run_name=args.run_name,
                 zone_x=args.zone_x, zone_y=args.zone_y,
                 zone_radius=args.zone_radius, pc_min=args.pc_min,
                 start_holder=args.start_holder,
                 anneal_lr=not args.no_anneal))
