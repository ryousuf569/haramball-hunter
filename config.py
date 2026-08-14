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
    # Both stay ON with the constraints layered on top. Switching them off was
    # an overcorrection: they are potential-based, so they cannot bias the
    # optimum, and they were only ever making the problem learnable. Removing
    # them left an off-ball attacker with no per-agent gradient at all and
    # traded the offside term's gradient-toward-the-line for a binary cliff.
    # Measured: direction-head entropy ended at 0.47 of uniform without the
    # per-agent term against 0.36 with it.
    agent_alpha: float = 0.5
    offside_in_potential: bool = True
    gamma: float = 0.999
    # The cost critics get their own discount. The paper trains every Q^(k) at
    # gamma_k < 1 "for numerical stability", 0.9 in Arena and 0.99 in OpenWorld,
    # and never at the reward's gamma. Ours were on 0.999, a 1000-tick horizon
    # for a 500-tick episode, to estimate what are really behaviour frequencies.
    cost_gamma: float = 0.99
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
    # Periodic evaluation against the calibrated gate, which is the only
    # success number comparable across arms. 0 disables it. At 100 updates and
    # 20 episodes this costs a few percent of wall clock.
    eval_every: int = 100
    eval_episodes: int = 20
    eval_envs: int = 6


@dataclass
class LagrangeConfig:
    """Multipliers for the behaviour constraints in environment/costs.py.

    lambda_k = exp(z_k) / (exp(a0) + sum_j exp(z_j)),  z_k += lr * (rate_k - d_k)
    See README.md, "The multipliers", for why the softmax keeps this stable.
    """
    enabled: bool = True
    lr: float = 0.05
    # The dummy logit, held at z_init so the reward starts on 1/(K+1) like every
    # constraint. It was 2.08 = log(K) to hand the reward half the simplex up
    # front, which is not in the paper: the reward's weight is meant to come
    # from the bootstrap constraint, not from a head start.
    a0: float = 0.02
    z_init: float = 0.0
    # Both of these were ours, not the paper's, and both are now off. The clip
    # capped z at 3, which is the one thing the bootstrap needs to be able to
    # do: lambda_no_success has to run away for lambda_0 := max(lambda_0,
    # lambda_no_success) to hand the task its weight back. The floor then
    # patched the symptom the clip caused. A strenuous no_success threshold is
    # the paper's answer to both. 0 disables each.
    z_clip: float = 0.0
    reward_floor: float = 0.0
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
    steps: int = 8                # tightening steps to the real task
    # Fraction of training by which the ramp finishes, so the last 30% of every
    # run is on the real task. It used to advance on success rate, which had
    # two problems: a better policy tightened its own gate and its success rate
    # then described a harder task, and at 500k no arm got past level 4 of 8,
    # so nothing was ever trained on the task it was measured against.
    finish_frac: float = 0.70