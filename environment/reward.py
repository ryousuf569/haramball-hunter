import numpy as np
from collections import deque
from dataclasses import dataclass
from environment.grid import PC_CELL_SIZE, PC_NX, PC_NY
from environment.termination import pcf_in_area

CELL_AREA = PC_CELL_SIZE ** 2
ATTACKER_LABEL = "attacker"
FINAL_THIRD_X = 70.0
HALF_SPACE_CENTERS = (20.5, 47.5)
HALF_SPACE_HALF_WIDTH = 6.5
HALF_SPACE_MAX_X = None

# Episode outcome labels. Duplicated from lowblock_env rather than imported
SUCCESS = "success"
FAILURE = "failure"
TIMEOUT = "timeout"
TERMINAL_OUTCOMES = (SUCCESS, FAILURE, TIMEOUT)

NORMALIZATIONS = ("mean", "area")


@dataclass(frozen=True)
class RewardConfig:
    alpha: float = 1.0 # weight on final-third control
    beta: float = 1.0 # extra weight on the half-space corridors
    gamma: float = 0.99 # must match the learner's discount
    terminal_bonus: float = 5.0 # B, paid on SUCCESS
    turnover_penalty: float = -1.0 # FAILURE
    timeout_penalty: float = -0.5 # TIMEOUT, and never harsher than a turnover
    normalization: str = "mean" # see zone_value: "mean" is O(1), "area" is m^2
    agent_alpha: float = 0.5 # weight on the per-attacker local shaping term
    use_gamma: bool = True # flag 1: gamma in the shaping term
    zero_terminal_potential: bool = True # flag 2: Phi(terminal) := 0

    def __post_init__(self):
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"normalization={self.normalization!r} not in {NORMALIZATIONS}")


def make_pcf_state():
    return {
        "prev_pcf": deque(maxlen=2),
        "prev_agent": None,
    }


def build_zone_masks(ppcf_grid):
    """Call ONCE at env construction"""
    grid = np.asarray(ppcf_grid, dtype=float)
    rx, ry = grid[:, 0], grid[:, 1]

    f3 = rx > FINAL_THIRD_X

    in_corridor = np.zeros_like(f3)
    for cy in HALF_SPACE_CENTERS:
        in_corridor |= np.abs(ry - cy) < HALF_SPACE_HALF_WIDTH
    if HALF_SPACE_MAX_X is not None:
        in_corridor &= rx <= HALF_SPACE_MAX_X

    hs = f3 & in_corridor

    if not f3.any() or not hs.any():
        raise ValueError(
            f"Empty zone mask (f3={f3.sum()}, hs={hs.sum()}). "
            "Grid is probably in local coords -- apply local_to_global first."
        )
    return f3, hs


def zone_mean(pc_att, mask):
    # per-cell mean == (sum PC * cell_area) / zone_area with cell_area cancelled.
    # Bounded in [0, 1] and resolution-independent by construction.
    return float(pc_att[mask].mean())


def zone_area(pc_att, mask):
    # Riemann sum: square metres of attacker-controlled space in the zone.
    # Also resolution-independent, but O(1000) -- with this mode the living cost
    # below dwarfs terminal_bonus unless B is rescaled to match.
    return float(pc_att[mask].sum()) * CELL_AREA


def zone_value(pc_att, mask, normalization):
    if normalization == "mean":
        return zone_mean(pc_att, mask)
    if normalization == "area":
        return zone_area(pc_att, mask)
    raise ValueError(f"unknown normalization {normalization!r}")


def attacker_control(players, ppcf_result):
    is_att = np.asarray(players["team"]) == ATTACKER_LABEL
    return np.asarray(ppcf_result)[:, is_att].sum(axis=1)


def phi_from_pc_att(pc_att, f3_mask, hs_mask, cfg):
    pc_att = np.asarray(pc_att, dtype=float).reshape(-1)

    pc_f3 = zone_value(pc_att, f3_mask, cfg.normalization)
    pc_hs = zone_value(pc_att, hs_mask, cfg.normalization)

    # HS is a SUBSET of F3, so corridor cells carry effective weight alpha+beta.
    phi_s = cfg.alpha * pc_f3 + cfg.beta * pc_hs
    return phi_s, {"pc_f3": pc_f3, "pc_hs": pc_hs, "phi": phi_s}


def phi(players, ppcf_result, f3_mask, hs_mask, cfg):
    return phi_from_pc_att(attacker_control(players, ppcf_result),
                           f3_mask, hs_mask, cfg)


# just a simple cache update
def push_phi(pcf_state, phi_s):
    pcf_state["prev_pcf"].append(phi_s)


def delta(pcf_state, cfg, terminal=False):
    # F = gamma * Phi(s') - Phi(s), with Phi(terminal) forced to 0 so the
    # discounted sum over an episode telescopes to exactly -Phi(s0) for ANY policy
    history = pcf_state["prev_pcf"]

    if len(history) < 2:
        return 0.0

    phi_prev, phi_curr = history[0], history[1]

    arriving = 0.0 if (terminal and cfg.zero_terminal_potential) else phi_curr
    g = cfg.gamma if cfg.use_gamma else 1.0

    return g * arriving - phi_prev


def is_terminal(outcome):
    return outcome in TERMINAL_OUTCOMES


def terminal_reward(outcome, cfg):
    # Signed, and no longer a "bonus": B on success, and a penalty on each of
    # the two ways of not scoring. Non-terminal steps pay 0.
    if outcome == SUCCESS:
        return cfg.terminal_bonus
    if outcome == FAILURE:
        return cfg.turnover_penalty
    if outcome == TIMEOUT:
        return cfg.timeout_penalty
    return 0.0


def reset_potential_from_pc_att(pc_att, f3_mask, hs_mask, pcf_state, cfg):
    """Seed the cache with Phi(s0) at episode start. Skipping this costs the
    first transition's shaping and breaks the telescoping identity."""
    pcf_state["prev_pcf"].clear()
    phi_0, parts = phi_from_pc_att(pc_att, f3_mask, hs_mask, cfg)
    push_phi(pcf_state, phi_0)
    return phi_0, parts


def reset_potential(players, ppcf_result, f3_mask, hs_mask, pcf_state, cfg):
    return reset_potential_from_pc_att(attacker_control(players, ppcf_result),
                                       f3_mask, hs_mask, pcf_state, cfg)


def step_reward_from_pc_att(pc_att, f3_mask, hs_mask, pcf_state, cfg, outcome=None):
    history = pcf_state["prev_pcf"]
    phi_prev = history[-1] if history else None

    phi_s, parts = phi_from_pc_att(pc_att, f3_mask, hs_mask, cfg)
    push_phi(pcf_state, phi_s)

    terminal = is_terminal(outcome)
    shaping = delta(pcf_state, cfg, terminal=terminal)
    bonus = terminal_reward(outcome, cfg)
    reward = shaping + bonus

    components = {
        "reward": reward,
        "shaping": shaping,
        "terminal_reward": bonus,
        "phi_prev": phi_prev,
        "phi_next": 0.0 if (terminal and cfg.zero_terminal_potential) else phi_s,
        "pc_f3": parts["pc_f3"],
        "pc_hs": parts["pc_hs"],
        "terminal": terminal,
        "outcome": outcome,
    }
    return reward, components

def agent_potential(pc_att, att_pos):
    grid = np.asarray(pc_att, dtype=float).reshape(PC_NX, PC_NY)
    return pcf_in_area(np.asarray(att_pos, dtype=float), grid)


def reset_agent_potential(pcf_state, pc_att, att_pos):
    phi_i = agent_potential(pc_att, att_pos)
    pcf_state["prev_agent"] = phi_i
    return phi_i


def agent_shaping(pcf_state, pc_att, att_pos, cfg, terminal=False):
    phi_i = agent_potential(pc_att, att_pos)
    prev = pcf_state.get("prev_agent")
    arriving = (np.zeros_like(phi_i)
                if (terminal and cfg.zero_terminal_potential) else phi_i)
    g = cfg.gamma if cfg.use_gamma else 1.0

    out = np.zeros_like(phi_i) if prev is None else g * arriving - prev
    pcf_state["prev_agent"] = phi_i
    return cfg.agent_alpha * out


def step_reward(players, ppcf_result, f3_mask, hs_mask, pcf_state, cfg, outcome=None):
    return step_reward_from_pc_att(attacker_control(players, ppcf_result), f3_mask, hs_mask, pcf_state, cfg, outcome=outcome)
