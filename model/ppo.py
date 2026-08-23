from environment.thread_limits import limit_threads
limit_threads(1)

import argparse, time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.vector import AutoresetMode

from environment.lowblock_env import GAMMA, SUCCESS, IS_CARRIER, make_vector_env, obs_dim

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
    seed: int = 1

    @property
    def batch(self):
        return self.rollout * self.n_envs * 10

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

    def _dist_and_value(self, obs_flat):
        h = self.trunk(obs_flat)
        dist = MultiCategorical(self.head(h), self.nvec, ball_mask_from_obs(obs_flat, self.n_att))
        return dist, self.value(h).squeeze(-1)

    def act(self, obs): 
        n, a, d = obs.shape
        dist, v = self._dist_and_value(obs.reshape(n * a, d))
        act = dist.sample()
        return (act.reshape(n, a, 3), dist.log_prob(act).reshape(n, a), v.reshape(n, a))

    def evaluate(self, obs_flat, act_flat):
        dist, v = self._dist_and_value(obs_flat)
        return dist.log_prob(act_flat), dist.entropy(), v

    def value_of(self, obs):
        features = self.trunk(obs) 
        values = self.value(features)
        return values.squeeze(-1)   

class RolloutBuffer:
    def __init__(self, T, N, A, D):
        self.T, self.N, self.A, self.D = T, N, A, D
        self.obs = torch.zeros(T, N, A, D)
        self.act = torch.zeros(T, N, A, 3, dtype=torch.long)
        self.logp = torch.zeros(T, N, A)
        self.val = torch.zeros(T, N, A)
        self.rew = torch.zeros(T, N)
        self.done = torch.zeros(T, N)

    def flat(self, adv, ret):
        B = self.T * self.N * self.A
        return {
            "obs": self.obs.reshape(B, self.D),
            "act": self.act.reshape(B, 3),
            "logp": self.logp.reshape(B),
            "adv": adv.reshape(B),
            "ret": ret.reshape(B),
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
        if not done_np.any():
            return
        outcomes = info["final_info"]["outcome"]
        for i in np.flatnonzero(done_np):
            self.success.append(outcomes[i] == SUCCESS)
            self.ret.append(float(ep_ret[i]))
            self.len.append(int(ep_len[i]))
            self.n_eps += 1
        ep_ret[done_np] = 0.0 # in place; these are LoopState's own arrays
        ep_len[done_np] = 0

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

        act, logp, val = agent.act(state.next_obs)
        buf.act[t], buf.logp[t], buf.val[t] = act, logp, val

        obs, rew, term, trunc, info = venv.step(act.numpy())
        buf.rew[t] = torch.as_tensor(rew, dtype=torch.float32)

        state.ep_ret += rew
        state.ep_len += 1
        done_np = term | trunc
        tracker.record(done_np, info, state.ep_ret, state.ep_len)

        state.next_obs = torch.as_tensor(obs)
        state.next_done = torch.as_tensor(done_np, dtype=torch.float32)
    return state


def phi_r2(phi, returns):
    """r^2 of the best linear predictor of the return from Phi alone"""
    p = phi.reshape(-1)
    if p.std() == 0 or returns.std() == 0:
        return float("nan")
    p = p - p.mean()
    r = returns - returns.mean()
    return float((p @ r) ** 2 / ((p @ p) * (r @ r)))

def explained_variance(values, returns):
    var_returns = returns.var()
    if var_returns == 0:
        return float("nan")
    return float(1 - (returns - values).var() / var_returns)