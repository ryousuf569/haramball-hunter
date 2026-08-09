"""Run the low-block simulation with a trained attacker policy in the loop.

LowBlockEnv is the training-side wrapper: it owns the world, the reward and the
Gymnasium contract. Nothing here needs the reward or the vector API -- this is
the inference path, so it is plain functions over the same world tuple
(players, ball, attacker_ids, defender_state) that lowblock_env.step already
takes, plus a checkpoint to drive the attackers with.

The observation is no longer cloned: obs() below forwards to
lowblock_env.build_obs, the same function LowBlockEnv.obs calls, because a
policy fed a differently-ordered observation than it was trained on does not
error -- it just plays badly. The ball mask and action decoding are still
small clones and MUST stay byte-identical to LowBlockEnv's.

The defenders stay scripted, exactly as in training.
"""

import numpy as np
import torch

from ppo import Actor, MultiCategorial
from environment.lowblock_env import (
    ACTION_HEADS,
    N_DIRECTIONS,
    N_SPEEDS,
    TIMEOUT,
    ball_action_mask,
    ball_actions,
    build_obs,
    compute_attacker_ppcf,
    make_initial_world,
    make_ppcf_grid,
    obs_dim,
    remaining_time,
)
from environment.lowblock_env import step as world_step
from environment.reward import build_zone_masks
from physics.engine import action_decoding


def load_actor(path, n_att=10, n_def=11, device="cpu"):
    """Rebuild the training Actor and load weights from train.py's checkpoint.

    Accepts either the {"actor": ..., "critic": ...} dict train.py writes or a
    bare actor state_dict. The widths are derived from the roster rather than
    stored, so loading a checkpoint trained on a different n_att/n_def raises
    here instead of silently mis-slicing the observation.
    """
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("actor", ckpt) if isinstance(ckpt, dict) else ckpt

    actor = Actor(obs_dim(n_att + n_def, n_att), N_DIRECTIONS, N_SPEEDS,
                  ball_actions(n_att)).to(device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def holder_row(ball, attacker_ids):
    """Attacker row holding the ball, or None if in flight / lost."""
    holder_id = ball.get("holder_id")
    if holder_id is None:
        return None
    rows = np.flatnonzero(attacker_ids == holder_id)
    return int(rows[0]) if rows.size else None


def obs(players, ball, n_att, tick, max_ticks, pc_att, f3_mask, hs_mask,
        normalization="mean"):
    """(n_att, obs_dim) float32 -- forwards to the one canonical encoder."""
    return build_obs(players, ball, n_att, tick, max_ticks, pc_att,
                     f3_mask, hs_mask, normalization)


def action_mask(players, ball, attacker_ids, n_att):
    """(n_att, ball_actions) bool -- forwards to the one canonical mask.

    This used to be a hand-kept clone of LowBlockEnv.action_mask. It stopped
    being safe as one the moment the mask started filtering offside targets:
    a policy handed a laxer mask at inference than it trained on does not
    error, it just plays passes it was never allowed to consider.
    """
    return ball_action_mask(players, ball, attacker_ids, n_att)


def decode_action(action, ball, attacker_ids, n_att):
    """(n_att, 3) -> (velocities, ball_idx) -- clone of decode_action."""
    a = np.asarray(action, dtype=np.int64).reshape(n_att, ACTION_HEADS)
    vel = action_decoding(a[:, 0], a[:, 1]).astype("f4")
    ball_idx = np.zeros(n_att, dtype=int)
    row = holder_row(ball, attacker_ids)
    if row is not None:
        ball_idx[row] = int(a[row, 2])
    return vel, ball_idx


@torch.no_grad()
def select_action(actor, obs_np, mask_np, deterministic=True, device="cpu"):
    """(n_att, 3) int actions. deterministic takes the argmax of each head;
    False samples, which is what the rollout that produced the weights did."""
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
    d, s, b = actor(obs_t)
    dist = MultiCategorial([d, s, b], [None, None, mask_t])
    a = dist.mode() if deterministic else dist.sample()
    return a.cpu().numpy()


def learned_step(players, ball, attacker_ids, defender_state, tick, actor,
                 ppcf_grid=None, max_ticks=None, deterministic=True,
                 device="cpu", verbose=True, pc_att=None, zone_masks=None):
    """One tick with the policy driving the attackers.

    Drop-in for lowblock_env.step: same returns (players, ball, pc_att,
    outcome), with the timeout imposed here the way LowBlockEnv.step does it
    (world_step has no horizon of its own).

    tick is ticks already elapsed -- the same value LowBlockEnv passes -- since
    it feeds the observation's clock feature as well as the scripted defenders.

    pc_att is the surface for the state as it stands NOW, before this tick runs
    -- i.e. the one the previous call returned. Threading it through is what
    keeps this to one pitch-control call per tick; pass None and it recomputes,
    which is correct but doubles the cost of the step. max_ticks defaults to
    EnvConfig.t_max so the clock feature cannot silently disagree with training.
    """
    if max_ticks is None:
        from config import EnvConfig
        max_ticks = EnvConfig.t_max
    if ppcf_grid is None:
        raise ValueError("learned_step needs a ppcf_grid: the observation now "
                         "carries pitch-control features")
    if zone_masks is None:
        zone_masks = build_zone_masks(ppcf_grid)
    if pc_att is None:
        pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])

    n_att = len(attacker_ids)
    o = obs(players, ball, n_att, tick, max_ticks, pc_att, *zone_masks)
    m = action_mask(players, ball, attacker_ids, n_att)
    action = select_action(actor, o, m, deterministic, device)
    avel, ball_idx = decode_action(action, ball, attacker_ids, n_att)

    players, ball, pc_att, outcome = world_step(
        players, ball, attacker_ids, defender_state, tick,
        ppcf_grid=ppcf_grid, attacker_velocities=avel,
        attacker_ball_idx=ball_idx, verbose=verbose)

    if outcome is None and tick + 1 >= max_ticks:
        outcome = TIMEOUT
    return players, ball, pc_att, outcome


def run_episode(actor, n_att=10, n_def=11, seed=54, max_ticks=None,
                start_holder=1, deterministic=True, device="cpu",
                verbose=False):
    """Headless episode. Returns (outcome, ticks) -- SUCCESS / FAILURE / TIMEOUT."""
    if max_ticks is None:
        from config import EnvConfig
        max_ticks = EnvConfig.t_max

    players, ball, attacker_ids, _rng, defender_state = make_initial_world(
        n_att, n_def, seed, start_holder=start_holder)
    ppcf_grid = make_ppcf_grid()
    zone_masks = build_zone_masks(ppcf_grid)

    # Seeded once here, then carried tick to tick: learned_step wants the
    # surface for the state it is about to act in, which is the one the
    # previous tick returned.
    pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])

    for tick in range(max_ticks):
        players, ball, pc_att, outcome = learned_step(
            players, ball, attacker_ids, defender_state, tick, actor,
            ppcf_grid=ppcf_grid, max_ticks=max_ticks,
            deterministic=deterministic, device=device, verbose=verbose,
            pc_att=pc_att, zone_masks=zone_masks)
        if outcome is not None:
            return outcome, tick + 1
    return TIMEOUT, max_ticks
