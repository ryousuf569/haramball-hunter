from environment.thread_limits import limit_threads
limit_threads()

import os, random, time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from environment.lowblock_env import LowBlockEnv
from environment.reward import RewardConfig
from config import EnvConfig, PPOConfig

N_DIR = 9
N_SPEED = 3 
N_ATT = 10
N_DEF = 11

BALL_ACTIONS = N_ATT
OBS_DIM = 6 + 4*(N_ATT - 1) + 3*N_DEF + 27
STATE_DIM = 4*(N_ATT + N_DEF) + 2 + 2 

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):

    def __init__(self, obs_dim, n1, n2, n3):

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

    def forward(self, obs):
        h = self.trunk(obs)
        return self.head1(h), self.head2(h), self.head3(h)

class Critic(nn.Module):

    def __init__(self, state, n1):
    
            super().__init__()
    
            # 3 layer network
            # obs_dim -> 256 -> 256
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(state, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh())
    
            self.val = layer_init(nn.Linear(256, n1), std=1.0)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.val(h)


def make_env(cfg: EnvConfig):
    reward_cfg = RewardConfig(
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        terminal_bonus=cfg.terminal_bonus,
        use_gamma=cfg.use_gamma_in_shaping,
        zero_terminal_potential=cfg.zero_terminal_potential)
    return LowBlockEnv(
        n_att=cfg.n_att,
        n_def=cfg.n_def,
        max_ticks=cfg.t_max,
        cfg=reward_cfg,
        scripted_attackers=False)


env = make_env(EnvConfig())
a = Actor(env.obs_dim, N_DIR, N_SPEED, env.ball_actions)
c = Critic(env.state_dim, 1)

d, s, b = a(torch.zeros(4, env.obs_dim))
assert d.shape == (4, N_DIR) and s.shape == (4, N_SPEED) and b.shape == (4, env.ball_actions)
assert c(torch.zeros(4, env.state_dim)).shape == (4, 1)
assert d.abs().max() < 0.1, "policy head init too large"

# critic head should NOT be tiny, catches the std=0.01 bug
v = c(torch.randn(64, env.state_dim))
assert v.std() > 0.05, "value head init too small, should be std=1.0, not 0.01"
print('all tests pass')