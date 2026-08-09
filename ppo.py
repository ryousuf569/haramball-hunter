from environment.thread_limits import limit_threads
limit_threads(1, torch_threads=1)

import os, random, time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from gymnasium.vector import AutoresetMode

from environment.lowblock_env import (
    LowBlockEnv,
    make_vector_env,
    ACTION_HEADS,
    SUCCESS,
    FAILURE,
    TIMEOUT,
    ball_actions as ball_actions_fn,
    obs_dim as obs_dim_fn,
    state_dim as state_dim_fn,)

from environment.reward import RewardConfig
from config import EnvConfig, PPOConfig

N_DIR = 9
N_SPEED = 3
N_ATT = 10
N_DEF = 11
DIR, SPEED, BALL = 0, 1, 2
BALL_ACTIONS = N_ATT

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):

    def __init__(self, obs_dim, n1, n2, n3, hold_bias=5.0):

        super().__init__()

        # 3 layer network
        # obs_dim -> 256 -> 256
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh())

        self.head1 = layer_init(nn.Linear(256, n1), std=0.01)
        self.head2 = layer_init(nn.Linear(256, n2), std=0.01)
        self.head3 = layer_init(nn.Linear(256, n3), std=0.01)
        with torch.no_grad():
            self.head3.bias[0] = hold_bias

    def forward(self, obs):
        h = self.trunk(obs)
        return self.head1(h), self.head2(h), self.head3(h)

class Critic(nn.Module):

    def __init__(self, state, n1):

            super().__init__()

            # obs_dim -> 256 -> 256 -> 256. The two-layer version explained 5%
            # of the return variance on-policy, which made the advantages mostly
            # value error.
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(state, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh())

            self.val = layer_init(nn.Linear(256, n1), std=1.0)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.val(h)


class RunningNorm:
    """Running mean/std of the value targets. The critic regresses normalised
    returns and its output is scaled back for GAE, so a reward scale that drifts
    over training does not keep re-scaling the value loss."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = float(x.mean()), float(x.var()), x.numel()
        total = self.count + bc
        d = bm - self.mean
        self.mean += d * bc / total
        self.var = ((self.var * self.count + bv * bc
                     + d * d * self.count * bc / total) / total)
        self.count = total

    @property
    def std(self):
        return max(self.var ** 0.5, 1e-6)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

class MultiCategorial:
    def __init__(self, logits_list, masks_list=None):

        self.logits_list = list(logits_list)
        self.masks_list = None if masks_list is None else list(masks_list)

        if self.masks_list is not None:
            assert len(self.masks_list) == len(self.logits_list)
            masked = []
            for logits, mask in zip(self.logits_list, self.masks_list):
                if mask is None:
                    masked.append(logits)
                    continue
                assert mask.any(dim=-1).all(), "row with zero legal actions"
                masked.append(torch.where(
                    mask, logits, torch.tensor(
                        torch.finfo(logits.dtype).min,
                        dtype=logits.dtype, device=logits.device)))
            self.logits_list = masked
        self.cats = [Categorical(logits=logits) for logits in self.logits_list]

    def sample(self):
        return torch.stack([c.sample() for c in self.cats], dim=-1)

    def mode(self):
        return torch.stack([c.logits.argmax(dim=-1) for c in self.cats], dim=-1)

    def log_prob(self, actions):
        return sum(c.log_prob(actions[:, i]) for i, c in enumerate(self.cats))

    def entropy(self):
        return sum(self.head_entropies())

    def head_entropies(self):
        return [c.entropy() for c in self.cats]

    def masked_head_entropies(self):
        """One scalar per head, averaged over rows that have a real choice.

        A row masked down to a single legal action has entropy exactly 0 and no
        gradient. Averaging those in -- which entropy().mean() does -- scaled the
        ball head's regulariser by the fraction of rows holding the ball, about
        0.4%, while the two movement heads took the full coefficient on every
        row. That asymmetry is why the ball head collapsed and the movement heads
        never left uniform."""
        out = []
        masks = self.masks_list or [None] * len(self.cats)
        for c, mask in zip(self.cats, masks):
            e = c.entropy()
            if mask is None:
                out.append(e.mean())
                continue
            live = mask.sum(-1) > 1
            out.append(e[live].mean() if live.any() else e.sum() * 0.0)
        return out



def env_kwargs(cfg: EnvConfig):
    reward_cfg = RewardConfig(
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        terminal_bonus=cfg.terminal_bonus,
        turnover_penalty=cfg.turnover_penalty,
        timeout_penalty=cfg.timeout_penalty,
        agent_alpha=cfg.agent_alpha,
        use_gamma=cfg.use_gamma_in_shaping,
        zero_terminal_potential=cfg.zero_terminal_potential)
    return dict(
        n_att=cfg.n_att,
        n_def=cfg.n_def,
        max_ticks=cfg.t_max,
        cfg=reward_cfg,
        scripted_attackers=False)


def make_env(cfg: EnvConfig):
    return LowBlockEnv(**env_kwargs(cfg))


def make_venv(cfg: EnvConfig, n_envs, seed=None, asynchronous=True):
    return make_vector_env(n_envs=n_envs, asynchronous=asynchronous, seed=seed,
                           autoreset_mode=AutoresetMode.SAME_STEP,
                           **env_kwargs(cfg))


# global_state()'s tail is [..., pc_f3, pc_hs, remaining_time], so Phi is
# recoverable from the rollout with no second pitch-control call.
STATE_PC_F3, STATE_PC_HS = -3, -2


def episode_outcomes(info):
    """(outcomes, ticks) for episodes that ended on this vector step"""
    final = info.get("final_info")
    if not final or "outcome" not in final:
        return [], []
    outcomes, ticks = final["outcome"], final.get("tick")
    live = [i for i, o in enumerate(outcomes) if o is not None]
    return ([outcomes[i] for i in live],
            [] if ticks is None else [float(ticks[i]) for i in live])


def agent_rewards(info, done, n_envs, n_att):
    """(n_envs, n_att) per-attacker shaping for this vector step.

    Under SAME_STEP autoreset a terminated env's info is the one reset() built,
    so the terminal step's per-agent reward is in final_info instead -- taking
    the top-level array there would silently pair the team's terminal reward
    with the next episode's shaping."""
    out = np.asarray(info["agent_reward"], dtype=np.float32).reshape(n_envs, n_att)
    final = info.get("final_info")
    if final is not None and "agent_reward" in final:
        have = np.asarray(final["_agent_reward"], dtype=bool)
        fin = np.asarray(final["agent_reward"], dtype=np.float32).reshape(n_envs, n_att)
        out = np.where((have & done)[:, None], fin, out)
    return out


def phi_r2(phi, returns):
    """r^2 of the best linear predictor of the return from Phi alone"""
    p = phi.reshape(-1)
    r = returns.reshape(-1)
    if p.std() == 0 or r.std() == 0:
        return float("nan")
    p = p - p.mean()
    r = r - r.mean()
    return float((p @ r) ** 2 / ((p @ p) * (r @ r)))


def explained_variance(values, returns):
    var_returns = returns.var()
    if var_returns == 0:
        return float("nan")
    return float(1 - (returns - values).var() / var_returns)


class RolloutBuffer:

    def __init__(self, n_steps, n_envs, n_att, obs_dim, state_dim, ball_actions, device="cpu"):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs = torch.zeros(n_steps, n_envs, n_att, obs_dim, device=device)
        self.masks = torch.zeros(n_steps, n_envs, n_att, ball_actions, dtype=torch.bool, device=device)
        self.actions = torch.zeros(n_steps, n_envs, n_att, ACTION_HEADS, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(n_steps, n_envs, n_att, device=device)
        self.states = torch.zeros(n_steps, n_envs, state_dim, device=device)
        # Per-agent now: one value head and one reward channel per attacker.
        self.values = torch.zeros(n_steps, n_envs, n_att, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, n_att, device=device)
        self.dones = torch.zeros(n_steps, n_envs, 1, device=device)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, mask, state, action, logprob, value, reward, done):
        t = self.ptr
        assert t < self.n_steps, "buffer full, call reset()"
        self.obs[t] = obs
        self.masks[t] = mask
        self.states[t] = state
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.ptr += 1

    def flat(self):
        """Collapse (T, E) -> B, leaving the agent axis intact."""
        return dict(
            obs=self.obs.flatten(0, 1), # (B, n_att, obs_dim)
            masks=self.masks.flatten(0, 1), # (B, n_att, ball_actions)
            actions=self.actions.flatten(0, 1), # (B, n_att, 3)
            logprobs=self.logprobs.flatten(0, 1), # (B, n_att)
            states=self.states.flatten(0, 1), # (B, state_dim)
            values=self.values.flatten(0, 1)) # (B, n_att)

def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values

def ppo_losses(newlogprob, ent_terms, newvalue, old_logprob, advantages, returns, cfg: PPOConfig):
    # advantages, returns, newvalue and newlogprob are all (mb, n_att): the
    # advantage is the agent's own now, not one team scalar broadcast ten ways.
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    logratio = newlogprob - old_logprob
    ratio = logratio.exp()
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),).mean()

    v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    coefs = (cfg.ent_coef_dir, cfg.ent_coef_speed, cfg.ent_coef_ball)
    entropy_loss = sum(c * e for c, e in zip(coefs, ent_terms))

    loss = pg_loss - entropy_loss + cfg.vf_coef * v_loss
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1).abs() > cfg.clip_coef).float().mean()
    return loss, pg_loss, v_loss, sum(ent_terms), approx_kl, clipfrac

def policy_dist(actor, obs, mask):
    """obs (N, n_att, obs_dim), mask (N, n_att, ball_actions) -> one
    MultiCategorial over N*n_att flattened agent rows."""
    flat_obs = obs.flatten(0, 1)
    flat_mask = mask.flatten(0, 1)
    d, s, b = actor(flat_obs)
    return MultiCategorial([d, s, b], [None, None, flat_mask])


def train(cfg: PPOConfig = None, env_cfg: EnvConfig = None, asynchronous=True,
          torch_threads=None):
    cfg = cfg if cfg is not None else PPOConfig()
    env_cfg = env_cfg if env_cfg is not None else EnvConfig()
    if torch_threads is None:
        torch_threads = max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(torch_threads)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")

    n_att, n_def = env_cfg.n_att, env_cfg.n_def
    n_players = n_att + n_def
    obs_dim = obs_dim_fn(n_players, n_att)
    state_dim = state_dim_fn(n_players, n_att)
    n_ball = ball_actions_fn(n_att)

    venv = make_venv(env_cfg, cfg.n_envs, seed=cfg.seed,
                     asynchronous=asynchronous)

    actor = Actor(obs_dim, N_DIR, N_SPEED, n_ball).to(device)
    critic = Critic(state_dim, n_att).to(device)
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                     lr=cfg.lr, eps=1e-5)
    ret_norm = RunningNorm()

    buf = RolloutBuffer(cfg.n_steps, cfg.n_envs, n_att, obs_dim, state_dim,
                        n_ball, device)

    # batch_size counts env-steps, not agent-steps: it is the unit the critic,
    # the reward and the advantage all live on.
    batch_size = cfg.n_envs * cfg.n_steps
    minibatch_size = batch_size // cfg.n_minibatches
    n_updates = cfg.total_timesteps // batch_size

    # gamma and t_max are coupled (see EnvConfig.gamma). Printed so a horizon
    # far shorter than the episode, which silently kills the terminal bonus,
    # is visible at startup rather than inferred from a flat success rate.
    horizon = 1.0 / (1.0 - env_cfg.gamma)
    discount_at_T = env_cfg.gamma ** env_cfg.t_max
    credit = 1.0 / (1.0 - env_cfg.gamma * cfg.gae_lambda)
    print(f"obs {obs_dim} state {state_dim} | horizon {horizon:.0f} ticks "
          f"(gamma {env_cfg.gamma}) vs episode {env_cfg.t_max} ticks")
    print(f"  GAE credit window 1/(1-gamma*lambda) = {credit:.0f} ticks "
          f"(lambda {cfg.gae_lambda})")
    print(f"  terminals from kickoff (x gamma^T = {discount_at_T:.3f}):  "
          f"success {discount_at_T * env_cfg.terminal_bonus:+.3f}  "
          f"turnover {discount_at_T * env_cfg.turnover_penalty:+.3f}  "
          f"timeout {discount_at_T * env_cfg.timeout_penalty:+.3f}")
    if horizon < env_cfg.t_max / 4:
        print("  WARNING: discount horizon << episode length, the terminal "
              "bonus is effectively invisible from the start of an episode")

    def to_t(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    obs_np, info = venv.reset(seed=cfg.seed)
    next_obs = to_t(obs_np)
    next_mask = to_t(info["action_mask"], torch.bool)
    next_state = to_t(info["state"])
    next_done = torch.zeros(cfg.n_envs, 1, device=device)

    global_step = 0
    start = time.time()
    ep_return = np.zeros(cfg.n_envs) # undiscounted, per env
    ep_disc = np.zeros(cfg.n_envs) # discounted from each episode's own t=0
    ep_t = np.zeros(cfg.n_envs, dtype=np.int64)
    recent = deque(maxlen=100) # (outcome, ticks, undiscounted return)

    for update in range(1, n_updates + 1):
        buf.reset()
        outcomes, ep_lens, ep_returns = [], [], []
        t_update = time.time()

        for _ in range(cfg.n_steps):
            with torch.no_grad():
                dist = policy_dist(actor, next_obs, next_mask)
                action = dist.sample() # (E*n_att, 3)
                logprob = dist.log_prob(action) # (E*n_att,)
                value = ret_norm.denormalize(critic(next_state)) # (E, n_att)
            action = action.view(cfg.n_envs, n_att, ACTION_HEADS)

            obs_np, reward, term, trunc, info = venv.step(action.cpu().numpy())

            done = np.logical_or(term, trunc)
            # Each attacker is paid the team reward plus its own local shaping.
            # Both are potential-based, so neither moves the optimum; the second
            # is the only part of the signal an individual attacker controls.
            r = np.asarray(reward)
            agent_r = (r[:, None]
                       + agent_rewards(info, done, cfg.n_envs, n_att))

            buf.add(next_obs, next_mask, next_state, action,
                    logprob.view(cfg.n_envs, n_att), value,
                    to_t(agent_r), next_done)

            ep_return += r
            ep_disc += (env_cfg.gamma ** ep_t) * r
            ep_t += 1
            if done.any():
                outs, ticks = episode_outcomes(info)
                rets = ep_return[done].tolist()
                discs = ep_disc[done].tolist()
                outcomes += outs
                ep_lens += ticks
                ep_returns += rets
                recent.extend(zip(outs, ticks, rets, discs))
                ep_return[done] = 0.0
                ep_disc[done] = 0.0
                ep_t[done] = 0

            next_obs = to_t(obs_np)
            next_mask = to_t(info["action_mask"], torch.bool)
            next_state = to_t(info["state"])
            next_done = to_t(done).unsqueeze(-1)
            global_step += cfg.n_envs

        with torch.no_grad():
            next_value = ret_norm.denormalize(critic(next_state))
            advantages, returns = compute_gae(
                buf.rewards, buf.values, buf.dones, next_value, next_done,
                env_cfg.gamma, cfg.gae_lambda)

        b = buf.flat()
        b_adv = advantages.flatten(0, 1)
        b_ret = returns.flatten(0, 1)
        # Critic regresses normalised targets; the stats come from this batch
        # before it is used, so the value it produced above is not rescaled
        # halfway through an update.
        ret_norm.update(b_ret)
        b_ret_n = ret_norm.normalize(b_ret)

        idx = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch_size, minibatch_size):
                mb = idx[s:s + minibatch_size]

                dist = policy_dist(actor, b["obs"][mb], b["masks"][mb])
                mb_actions = b["actions"][mb].flatten(0, 1)   # (mb*n_att, 3)
                newlogprob = dist.log_prob(mb_actions).view(len(mb), n_att)
                ent_terms = dist.masked_head_entropies()
                newvalue = critic(b["states"][mb])

                loss, pg_loss, v_loss, ent, approx_kl, clipfrac = ppo_losses(
                    newlogprob, ent_terms, newvalue, b["logprobs"][mb],
                    b_adv[mb], b_ret_n[mb], cfg)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    cfg.max_grad_norm)
                opt.step()

        ev = explained_variance(b["values"], b_ret)
        phi = (env_cfg.alpha * buf.states[..., STATE_PC_F3]
               + env_cfg.beta * buf.states[..., STATE_PC_HS])
        phi = phi.unsqueeze(-1).expand_as(buf.rewards)

        w_out = [o for o, _, _, _ in recent]
        w_len = [t for _, t, _, _ in recent]
        w_ret = [r for _, _, r, _ in recent]
        w_disc = [d for _, _, _, d in recent]
        rate = lambda k: (w_out.count(k) / len(w_out) if w_out
                          else float("nan"))
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")

        dt = time.time() - t_update
        sps = int(batch_size / dt)
        avg_sps = int(global_step / (time.time() - start))
        print(f"upd {update}/{n_updates} | step {global_step} | "
              f"{sps} sps (avg {avg_sps})")
        print(f"  loss   pg {pg_loss.item():+.4f}  v {v_loss.item():.4f}  "
              f"ent {ent.item():.3f} "
              f"(dir {ent_terms[0].item():.2f} spd {ent_terms[1].item():.2f} "
              f"ball {ent_terms[2].item():.2f})  kl {approx_kl.item():.4f}  "
              f"clipfrac {clipfrac.item():.3f}")
        print(f"  value  ev {ev:+.3f} (phi_r2 {phi_r2(phi, b_ret):.3f})  "
              f"ret_std {b_ret.std().item():.4f}  "
              f"ret_mean {b_ret.mean().item():+.4f}  "
              f"adv_std {b_adv.std().item():.4f}")
        print(f"  eps    +{len(outcomes):2d} (win {len(recent):3d})  "
              f"success {rate(SUCCESS):.3f}  failure {rate(FAILURE):.3f}  "
              f"timeout {rate(TIMEOUT):.3f}  "
              f"len {mean(w_len):5.1f}  ret {mean(w_ret):+.3f} "
              f"(disc {mean(w_disc):+.3f})")
        print(f"  phi    min {phi.min().item():.4f}  "
              f"mean {phi.mean().item():.4f}  max {phi.max().item():.4f}  "
              f"std {phi.std().item():.4f}")

    venv.close()
    return actor, critic


if __name__ == "__main__":
    train()