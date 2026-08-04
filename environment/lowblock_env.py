import sys
from functools import partial

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from physics.engine import (
    DT,
    V_MAX,
    ball_action,
    ball_mechanics,
    kinematics_integrator,
    local_to_global,
)
from schema import player_dt
from physics.ppcf import PPCF_grid
from defenders.defenders import (
    make_defender_state,
    compute_defender_targets,
    gk_positioning,
)
from defenders.turnover import (
    ground_duel,
    intercept_pass,
    check_offside,
    nearest_defender_to,
    apply_turnover,
)
from environment.grid import PC_NX, PC_NY, PC_CELL_SIZE
from environment.termination import check_shot_opening
from environment.reward import (
    RewardConfig,
    build_zone_masks,
    make_pcf_state,
    reset_potential_from_pc_att,
    step_reward_from_pc_att,
)
from attackers.baseline_attacker import compute_attacker_targets
from attackers.calibrate_attacker_formation import sample_attacker_formation
from defenders.calibrate_defender_formation import sample_defender_formation
ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

# Episode outcomes, step()'s 4th return. None means the episode is still live:
# the defenders have neither conceded the shot nor won the ball back yet.
SUCCESS = "success"   # attackers worked a shot opening
FAILURE = "failure"   # defenders forced a turnover
# step() itself has no horizon, so it never returns TIMEOUT -- that is imposed
# by whatever drives it. LowBlockEnv below raises it at max_ticks; the constant
# lives here so the three outcome labels read from one place.
TIMEOUT = "timeout"

PITCH_CENTER_Y = 34.0


# Row 0 keeper, 1-5 backline, 6-9 midfield, 10 forward: the row layout
# compute_defender_targets indexes with GK_INDEX/BACKLINE_INDICES/etc. The 10
# outfielders are sampled from real low-block frames; those frames never
# labelled a keeper, so row 0 just starts on the resting target the policy
# would hold it at for a centred ball, rather than on a made-up sampled slot.
def _defender_formation_gk_5_4_1(rng, n_def=11):
    outfield = sample_defender_formation(rng, n_def=n_def - 1)
    gk = gk_positioning(PITCH_CENTER_Y, PITCH_CENTER_Y, 0.0001)
    return np.vstack([gk, outfield]).astype("f4")


# Rows deepest-first (2 backline, 5 midfield, 3 forward), sampled the same way
def _attacker_formation_2_5_3(rng, n_att=10):
    return sample_attacker_formation(rng, n_att=n_att).astype("f4")


# pitch-control grid. PC_NX/PC_NY/PC_CELL_SIZE are imported from environment.grid
# so termination.py can bin a position onto these cells without importing this
# module back; they stay re-exported here for the callers that already read them.
def make_ppcf_grid():
    i = np.arange(PC_NX)
    j = np.arange(PC_NY)
    ii, jj = np.meshgrid((i + 0.5) * PC_CELL_SIZE, (j + 0.5) * PC_CELL_SIZE,
                         indexing="ij")
    grid_local = np.stack([ii, jj], axis=-1).reshape(-1, 2)  # (1054, 2)
    return local_to_global(grid_local)


# Global-frame extent of the grid, for imshow/pcolormesh overlays.
_pc_corners = local_to_global(np.array([[0.0, 0.0],
                                        [PC_NX * PC_CELL_SIZE,
                                         PC_NY * PC_CELL_SIZE]]))
PC_EXTENT = (_pc_corners[0, 0], _pc_corners[1, 0],
             _pc_corners[0, 1], _pc_corners[1, 1])


def compute_attacker_ppcf(players, ppcf_grid, ball_pos):
    result = PPCF_grid(ppcf_grid, players, ball_pos)  # (n_cells, n_players)
    is_att = players["team"] == ATTACKER_LABEL
    return result[:, is_att].sum(axis=1).reshape(PC_NX, PC_NY)


def make_initial_world(n_att=10, n_def=11, seed=11, start_holder=0):
    rng = np.random.default_rng(seed)

    players = np.zeros(n_att + n_def, dtype=player_dt)
    players["id"] = np.arange(1, n_att + n_def + 1)  # ids are 1-based, HOLD==0
    players["team"][:n_att] = ATTACKER_LABEL
    players["team"][n_att:] = DEFENDER_LABEL

    # Attackers (10: a 2-5-3) start in a shape sampled from real low-block
    # freeze frames, so `seed` now actually changes the initial state instead of
    # replaying one hand-placed formation. Defenders are 11: a keeper on its
    # line plus a resting low-block 5-4-1 near the x=105 goal they defend. Row
    # layout matches defenders.py: row 0 keeper, rows 1-5 backline, rows 6-9
    # midfield, row 10 forward.
    players["position"][:n_att] = _attacker_formation_2_5_3(rng, n_att)
    players["position"][n_att:] = _defender_formation_gk_5_4_1(rng, n_def)
    players["velocity"][:] = 0.0

    attacker_ids = players["id"][:n_att]
    # Pick the starting holder by attacker row index (clipped into range).
    holder_row = int(np.clip(start_holder, 0, n_att - 1))
    holder_id = int(attacker_ids[holder_row])

    ball = {
        "state": "held",
        "holder_id": holder_id,
        "position": players["position"][players["id"] == holder_id][0].copy(),
        "target_id": None,
        "flight_start": np.zeros(2, dtype="f4"),
        "flight_target": np.zeros(2, dtype="f4"),
    }

    # Persistent state for the scripted defenders (holds a short ball_x history
    # deque used to lag the block's depth reference). Created once and threaded
    # through every step(). The seed flows into the turnover RNG, so distinct
    # seeds give distinct duel/interception rolls rather than identical replays.
    defender_state = make_defender_state(seed=seed)
    return players, ball, attacker_ids, rng, defender_state


def step(players, ball, attacker_ids, defender_state, tick_count,
         ppcf_grid=None, exit_on_turnover=False, attacker_velocities=None,
         attacker_ball_idx=None, verbose=True):
    """Advance the world one tick.

    attacker_velocities: optional (n_att, 2) target-velocity array, and
    attacker_ball_idx: optional (n_att,) pass-choice array, which together
        replace compute_attacker_targets for this tick. This is the seam the
        learned attacker policy drives: supply both and baseline_attacker.py is
        not consulted at all, so the attackers move straight off the physics
        with no hand-tuned model in the loop. Pass neither to keep the baseline
        (it is a test harness for the defenders, not a policy -- see that
        module's docstring).

        The defenders are scripted by design and have no such seam: the low
        block is the fixed opponent the attackers are learning against.
    verbose: turnover/shot events print to stdout. Vectorised rollouts run many
        envs at once, so LowBlockEnv turns this off.
    """
    att_mask = players["team"] == ATTACKER_LABEL
    def_mask = players["team"] == DEFENDER_LABEL

    # Roster-wide target-velocity array, filled per team below. Rows are laid
    # out attackers-first (rows :n_att) then defenders (rows n_att:), matching
    # both policies' row ordering.
    target_velocities = np.zeros((len(players), 2), dtype="f4")

    # 1) attackers: the policy's target velocities + pass choice when supplied,
    #    otherwise the throwaway baseline (tick_count drives its fixed passing
    #    cadence). The integrator clamps to A_MAX/V_MAX, so a supplied array
    #    only needs to be the right shape.
    #    Both or neither: supplying one alone would silently fall back to the
    #    baseline and discard it.
    if (attacker_velocities is None) != (attacker_ball_idx is None):
        raise ValueError("pass attacker_velocities and attacker_ball_idx "
                         "together, or neither")
    if attacker_velocities is None:
        attacker_velocities, attacker_ball_idx = compute_attacker_targets(
            players, ball, tick_count)
    ball_idx = attacker_ball_idx
    target_velocities[att_mask] = attacker_velocities

    # 2) defenders run their script -> target velocities for the block.
    #    compute_defender_targets returns velocities in defender-row order,
    #    which matches how the roster is laid out (defenders occupy rows n_att:).
    defender_velocities = compute_defender_targets(players, ball, defender_state)
    target_velocities[def_mask] = defender_velocities

    # 3) integrate kinematics for everyone with the combined targets.
    players = kinematics_integrator(players, target_velocities)

    # 4) resolve the ball: pass decision, then flight/hold mechanics. Remember
    #    where the ball started the tick so intercept_pass can test the segment it
    #    swept rather than just where it ended up.
    prev_ball_pos = np.asarray(ball["position"], dtype=float).copy()
    pass_decision = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)

    # 4a) offside, checked once at release (as RoboCup does) on the positions the
    #     pass was played from. Must come before ball_mechanics, which overwrites
    #     ball['position'] and flips the state to in_flight. An offside pass never
    #     leaves the holder's feet: the nearest defender to the intended receiver
    #     gets it, and the flight is skipped entirely. The holder is passed in
    #     because the ball's release point is their feet, and ball['position'] is
    #     still a tick stale until ball_mechanics resyncs it below.
    holder_id, is_pass, target_id = pass_decision
    if is_pass and check_offside(players, holder_id, target_id):
        target_pos = players["position"][players["id"] == target_id][0]
        winner = nearest_defender_to(players, target_pos)
        ball = apply_turnover(ball, winner)
        if verbose:
            print(f"    TURNOVER (offside) tick {tick_count}: pass to {target_id} flagged, "
                  f"defender {winner} gets it")
        if exit_on_turnover:
            sys.exit()
        pc_att = None
        if ppcf_grid is not None:
            pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])
        return players, ball, pc_att, FAILURE

    ball = ball_mechanics(ball, players, pass_decision)

    # 5) attacker pitch control on the post-integration positions/velocities.
    #    Runs before the turnover checks because it also caches each player's TTI
    #    to the ball in players['i_p'], which intercept_pass reads below.
    pc_att = None
    if ppcf_grid is not None:
        pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])

    # 6) turnovers. Both run after the integration and the ball resolution so they
    #    see the positions and ball state the tick actually ended on -- a defender
    #    who closes into range this tick can win it this tick. The two are mutually
    #    exclusive on ball state (held vs in_flight), so at most one can fire.
    winner = ground_duel(players, ball, defender_state["rng"], None, dt=DT)
    if winner is not None:
        ball = apply_turnover(ball, winner)
        if verbose:
            print(f"    TURNOVER (duel) tick {tick_count}: defender {winner} won the ball")
        if exit_on_turnover:
            sys.exit()
    else:
        winner = intercept_pass(players, ball, defender_state["rng"],
                                prev_ball_pos, dt=DT)
        if winner is not None:
            ball = apply_turnover(ball, winner)
            if verbose:
                print(f"    TURNOVER (intercept) tick {tick_count}: defender {winner} cut out the pass")
            if exit_on_turnover:
                sys.exit()

    # 7) terminate. A turnover -- from either check above, or from the offside
    #    branch that already returned -- ends the episode in failure. Otherwise
    #    an attacker on the ball, in space, with a real chance ends it in
    #    success. The shot test reads the pitch-control surface, so a run
    #    without a ppcf_grid can only ever end on a turnover.
    outcome = None
    if winner is not None:
        outcome = FAILURE
    elif pc_att is not None and check_shot_opening(players, ball, pc_att):
        outcome = SUCCESS
        if verbose:
            print(f"    SHOT OPENING tick {tick_count}: attacker {ball['holder_id']} "
                  f"is clear to shoot")

    return players, ball, pc_att, outcome


# Bound to the module-level step() so LowBlockEnv.step can call it without the
# method name shadowing it at the call site.
world_step = step


class LowBlockEnv(gym.Env):
    """Gymnasium wrapper: one episode = one low-block possession.

    The reward is environment.reward's potential-based shaping. The PPCF surface
    step() already builds for the shot test is fed straight to the reward, so
    wiring it in costs no extra pitch-control call -- see phi_from_pc_att.

    The attackers are the learners; the low block is scripted and stays that
    way, so it is the fixed opponent they are learning against.

    Observation (float32, 4*n_players + 3): every player's position and velocity
    in roster order (attackers rows :n_att, then defenders), then ball position
    and an in-flight flag. Global state, not per-agent -- enough to benchmark on
    and to swap out once the policy's input is settled.

    Action, a Dict, because an attacker does two things a tick:
      "velocity" (n_att, 2) float32 -- target velocities, handed straight to the
          kinematics integrator. No baseline model in between.
      "pass" Discrete(n_att) -- 0 holds; k passes to the k-th id in the sorted
          teammate-id array, the encoding engine.ball_action decodes. Ignored on
          ticks where an attacker is not holding the ball.

    baseline_attacker.py is a defender test harness, not a policy. With
    scripted_attackers=False -- how training runs -- it is bypassed entirely.
    The default True keeps it, for benchmarking and as a scripted reference arm.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_att=10, n_def=11, max_ticks=300, cfg=None,
                 scripted_attackers=True):
        super().__init__()
        self.n_att = n_att
        self.n_def = n_def
        self.max_ticks = max_ticks
        self.cfg = cfg if cfg is not None else RewardConfig()
        self.scripted_attackers = scripted_attackers

        # Both built once per env, not per episode: the grid is fixed geometry
        # and the masks are a pure function of it.
        self.ppcf_grid = make_ppcf_grid()
        self.f3_mask, self.hs_mask = build_zone_masks(self.ppcf_grid)
        self.pcf_state = make_pcf_state()

        n_players = n_att + n_def
        pos_lo = np.tile([0.0, 0.0], n_players)
        pos_hi = np.tile([105.0, 68.0], n_players)
        vel_lo = np.full(2 * n_players, -V_MAX)
        vel_hi = np.full(2 * n_players, V_MAX)
        obs_lo = np.concatenate([pos_lo, vel_lo, [0.0, 0.0], [0.0]])
        obs_hi = np.concatenate([pos_hi, vel_hi, [105.0, 68.0], [1.0]])
        self.observation_space = gym.spaces.Box(
            low=obs_lo.astype("f4"), high=obs_hi.astype("f4"), dtype=np.float32)
        # "pass" is Discrete(n_att): 0 holds, and 1..n_att-1 index the n_att-1
        # sorted teammate ids, so the top choice is always a legal teammate.
        self.action_space = gym.spaces.Dict({
            "velocity": gym.spaces.Box(low=-V_MAX, high=V_MAX,
                                       shape=(n_att, 2), dtype=np.float32),
            "pass": gym.spaces.Discrete(n_att),
        })

        self.players = None
        self.ball = None
        self.attacker_ids = None
        self.defender_state = None
        self.tick = 0

    def obs(self):
        return np.concatenate([
            np.asarray(self.players["position"], dtype="f4").reshape(-1),
            np.asarray(self.players["velocity"], dtype="f4").reshape(-1),
            np.asarray(self.ball["position"], dtype="f4").reshape(-1),
            np.array([self.ball["state"] == "in_flight"], dtype="f4"),
        ]).astype("f4")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # make_initial_world seeds the formation sampling and the turnover RNG
        # off one integer, so drawing it from self.np_random is what makes
        # reset(seed=...) reproducible and successive episodes distinct.
        ep_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        holder = int(self.np_random.integers(0, self.n_att))
        (self.players, self.ball, self.attacker_ids,
         _rng, self.defender_state) = make_initial_world(
            n_att=self.n_att, n_def=self.n_def, seed=ep_seed,
            start_holder=holder)
        self.tick = 0

        # Phi(s0) has to be seeded from the initial state, which no step() has
        # touched yet -- hence the one extra PPCF call per episode. Skipping it
        # would cost the first transition's shaping and break the telescoping
        # identity that environment/reward_readme.md rests on.
        pc_att = compute_attacker_ppcf(self.players, self.ppcf_grid,
                                       self.ball["position"])
        phi_0, parts = reset_potential_from_pc_att(
            pc_att, self.f3_mask, self.hs_mask, self.pcf_state, self.cfg)

        return self.obs(), {"phi_0": phi_0, "pc_f3": parts["pc_f3"],
                             "pc_hs": parts["pc_hs"], "tick": self.tick}

    def decode_action(self, action):
        vel = np.asarray(action["velocity"], dtype="f4").reshape(self.n_att, 2)

        # ball_action reads only the holder's slot and re-masks anything else,
        # so a tick where a defender has the ball decodes to HOLD on its own.
        ball_idx = np.zeros(self.n_att, dtype=int)
        holder_id = self.ball.get("holder_id")
        if holder_id is not None:
            ball_idx[self.attacker_ids == holder_id] = int(action["pass"])
        return vel, ball_idx

    def step(self, action):
        avel, ball_idx = (None, None)
        if not self.scripted_attackers:
            avel, ball_idx = self.decode_action(action)

        self.players, self.ball, pc_att, outcome = world_step(
            self.players, self.ball, self.attacker_ids, self.defender_state,
            self.tick, ppcf_grid=self.ppcf_grid, attacker_velocities=avel,
            attacker_ball_idx=ball_idx, verbose=False)
        self.tick += 1

        # step() has no horizon of its own; the timeout is imposed here.
        if outcome is None and self.tick >= self.max_ticks:
            outcome = TIMEOUT

        reward, parts = step_reward_from_pc_att(
            pc_att, self.f3_mask, self.hs_mask, self.pcf_state, self.cfg,
            outcome=outcome)

        # All three outcomes terminate, timeout included. reward.py counts
        # TIMEOUT in TERMINAL_OUTCOMES and zeroes Phi(terminal) for it, so the
        # shaping already treats it as absorbing; reporting it as truncated
        # would invite a learner to bootstrap V(s_T) on top and break the exact
        # telescoping identity reward_readme.md rests on. truncated stays False.
        terminated = outcome is not None
        truncated = False

        parts["tick"] = self.tick
        return self.obs(), float(reward), terminated, truncated, parts


def make_vector_env(n_envs=6, asynchronous=False, seed=None,
                    autoreset_mode=None, **env_kwargs):
    """SyncVectorEnv (serial, one process) or AsyncVectorEnv (one process each).

    partial rather than a closure so the factories pickle for the Async spawn.

    autoreset_mode: Gymnasium defaults to AutoresetMode.NEXT_STEP, where the
        call after a terminal one is a reset that ignores its action and pays
        reward 0 -- a junk transition a learner has to mask out. Pass
        AutoresetMode.SAME_STEP to reset on the terminal call instead, with the
        real terminal reward and the final state in info["final_obs"], so every
        call is a genuine transition. Training should prefer SAME_STEP.
    """
    fns = [partial(LowBlockEnv, **env_kwargs) for _ in range(n_envs)]
    cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    kwargs = {} if autoreset_mode is None else {"autoreset_mode": autoreset_mode}
    venv = cls(fns, **kwargs)
    if seed is not None:
        venv.reset(seed=seed)
    return venv
