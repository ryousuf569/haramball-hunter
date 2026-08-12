from dataclasses import dataclass

from environment.termination import SHOT_P_MIN

@dataclass
class EnvConfig:
    n_att: int = 10
    n_def: int = 11
    alpha: float = 1.0 # PC_F3 weight in Phi
    beta: float = 2.0 # PC_HS weight in Phi
    terminal_bonus: float = 5.0
    turnover_penalty: float = -1.0
    # Chosen so attacking dominates both degenerate endings. Playing on is worth
    # 5p - (1 - p) = 6p - 1, which is >= -1 for every p, so it beats conceding
    # (-1) for any p > 0 and beats stalling (-1.5) unconditionally. -2.0 left
    # conceding attractive whenever stalling was the alternative; -0.5 made
    # stalling safe below p = 1/12. See environment/reward_readme.md.
    timeout_penalty: float = -1.5
    # 0.0, because ten attackers each chasing their own final-third pitch
    # control is ten attackers sprinting past the ball. Costs cover this now.
    agent_alpha: float = 0.0
    offside_in_potential: bool = False  # costs.py owns offside now
    gamma: float = 0.999
    use_gamma_in_shaping: bool = True
    zero_terminal_potential: bool = True
    pc_backend: str = "spearman" # or "knn"
    t_max: int = 500
    shot_p_min: float = SHOT_P_MIN  # the target gate CurriculumConfig anneals to

@dataclass
class PPOConfig:
    n_envs: int = 6
    n_steps: int = 128
    lr: float = 3e-4
    gae_lambda: float = 0.99
    clip_coef: float = 0.2
    ent_coef_dir: float = 0.004
    ent_coef_speed: float = 0.004
    ent_coef_ball: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    n_minibatches: int = 4
    total_timesteps: int = 5_000_000
    seed: int = 0
    # One critic per cost, as the paper specifies. Measured at 50 sps with no
    # constraints and 54 with seven critics, so the faithful option is free.
    shared_cost_critic: bool = False


@dataclass
class LagrangeConfig:
    """Multipliers for the behaviour constraints in environment/costs.py.

    lambda_k = exp(z_k) / (exp(a0) + sum_j exp(z_j)),  z_k += lr * (rate_k - d_k)
    See README.md, "The multipliers", for why the softmax keeps this stable.
    """
    enabled: bool = True
    lr: float = 0.05
    a0: float = 0.0
    z_init: float = 0.0
    # A tuple of K floats replacing costs.COST_THRESHOLDS. This is the knob the
    # experiment in train.py sweeps. None uses the measured defaults.
    thresholds: tuple = None
    # The multipliers have to move slower than the policy or the two oscillate,
    # and the warmup lets the cost critics see data before lambda moves at all.
    warmup_updates: int = 5


@dataclass
class CurriculumConfig:
    """Anneal the success gate and the start state onto the real task.

    Success fires on only 0.7% of random-play episodes, so training needs an
    easier gate first. Nothing here changes what a checkpoint is measured on.
    """
    enabled: bool = True
    # Where random play succeeds 22% of the time. 30m gives it 51%, and the
    # calibrated gate is 15.7m.
    radius_start: float = 25.0
    x_shift_start: float = 4.0    # metres the attacking shape starts further on
    advance_at: float = 0.35      # above the 22% floor random play clears
    steps: int = 8                # tightening steps to the real task
    min_episodes: int = 50        # don't advance on a window this thin
    hold_updates: int = 20        # or faster than the value function can follow