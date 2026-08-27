import os

import numpy as np
import torch

from physics.engine import direction_lookup, speed_lookup
from environment.lowblock_env import (LowBlockEnv, MAX_TICKS,
                                      compute_attacker_ppcf, obs_dim)
from environment.termination import (ZONE_PC_MIN, ZONE_RADIUS, ZONE_X, ZONE_Y,
                                     make_zone)
from model.ppo import ActorCritic


def resolve_ckpt(path):
    if not os.path.isdir(path):
        return path
    for name in ("final.pt", "best.pt"):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    ckpts = [f for f in os.listdir(path)
             if f.startswith("ckpt_") and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"no .pt checkpoint in {path!r}")
    newest = max(ckpts, key=lambda f: int(f[len("ckpt_"):-len(".pt")]))
    return os.path.join(path, newest)


def load_agent(ckpt_path, n_att=10, n_def=11):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("cfg", {}).get("hidden", 256))
    nvec = (len(direction_lookup), len(speed_lookup), n_att)
    agent = ActorCritic(obs_dim=obs_dim(n_att, n_def), nvec=nvec, hidden=hidden)
    # Constrained checkpoints carry cost_value.* heads the plain critic has no
    # use for at inference; anything else missing or extra is a real mismatch.
    missing, unexpected = agent.load_state_dict(ckpt["model"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("cost_value.")]
    if missing or unexpected:
        raise RuntimeError("checkpoint does not match ActorCritic: "
                           "missing %s, unexpected %s" % (missing, unexpected))
    agent.eval()
    return agent, ckpt


class PPOPolicy:
    # Callable wrapper around a trained checkpoint.
    def __init__(self, ckpt_path, zone=None, n_att=10, n_def=11,
                 max_ticks=MAX_TICKS, deterministic=False, seed=None):
        self.ckpt_path = ckpt_path = resolve_ckpt(ckpt_path)
        self.deterministic = deterministic
        self.n_att = n_att
        self.agent, self.ckpt = load_agent(ckpt_path, n_att, n_def)

        cfg = self.ckpt.get("cfg", {})
        self.pc_min = float(cfg.get("pc_min", ZONE_PC_MIN))
        self.train_start_holder = cfg.get("start_holder")
        if zone is None:
            zone = make_zone(cfg.get("zone_x", ZONE_X),
                             cfg.get("zone_y", ZONE_Y),
                             cfg.get("zone_radius", ZONE_RADIUS))
        self.zone = zone

        # Observation shell (see module docstring). max_tick matters: the clock
        # feature is tick / max_ticks, so a mismatch shifts every observation.
        self.env = LowBlockEnv(n_att, n_def, max_tick=max_ticks)
        self.env.zone = zone
        self.env.zone_centre = np.asarray(zone.centre, dtype="f4")
        self.env.pc_min = self.pc_min

        if seed is not None:
            torch.manual_seed(seed)
        self.tick = 0

    @property
    def step(self):
        """Training step the checkpoint was saved at."""
        return self.ckpt.get("step")

    @property
    def stats(self):
        """Rolling stats recorded alongside the checkpoint (success, ret, ...)."""
        return self.ckpt.get("stats", {})

    def gate(self):
        c = self.zone.centre
        holder = self.train_start_holder
        trained_on = "" if holder is None else (
            f" | trained on start_holder "
            f"{'random' if holder < 0 else holder}")
        return (f"({c[0]:.0f}, {c[1]:.0f}) r={self.zone.radius:.0f} "
                f"pc_min={self.pc_min:.2f}{trained_on}")

    def label(self):
        run = os.path.basename(os.path.dirname(os.path.abspath(self.ckpt_path)))
        name = os.path.splitext(os.path.basename(self.ckpt_path))[0]
        mode = " (greedy)" if self.deterministic else ""
        return f"{run}/{name}{mode}"

    def reset(self):
        """Zero the clock feature at the start of a new episode."""
        self.tick = 0

    def observe(self, players, ball, pc_att=None):
        """The (n_att, obs_dim) observation for the current world state."""
        env = self.env
        env.players, env.ball, env.tick = players, ball, self.tick
        # zone_control() needs a field; the driver has one on every tick but the
        # first of an episode, where we pay for it once.
        env.pc_att = (compute_attacker_ppcf(players, env.ppcf_grid,
                                            ball["position"])
                      if pc_att is None else pc_att)
        return env.obs()

    def __call__(self, players, ball, attacker_ids, pc_att=None):
        obs = torch.as_tensor(self.observe(players, ball, pc_att))
        with torch.no_grad():
            if self.deterministic:
                # index rather than unpack: the constrained critic returns an
                # extra cost-value tensor
                act = self.agent._dist_and_value(obs)[0].mode()
            else:
                # act() wants a batch of envs: (1, n_att, obs_dim)
                act = self.agent.act(obs[None])[0][0]
        self.tick += 1
        a = act.numpy()
        return a[:, 0], a[:, 1], a[:, 2]


def make_ppo_policy(ckpt_path, zone=None, n_att=10, n_def=11,
                    max_ticks=MAX_TICKS, deterministic=False, seed=None):
    """make_policy-shaped constructor, mirroring scripted_policy.make_policy."""
    return PPOPolicy(ckpt_path, zone=zone, n_att=n_att, n_def=n_def,
                     max_ticks=max_ticks, deterministic=deterministic,
                     seed=seed)
