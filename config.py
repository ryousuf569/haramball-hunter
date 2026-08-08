from dataclasses import dataclass

@dataclass
class EnvConfig:
    n_att: int = 10
    n_def: int = 11
    alpha: float = 1.0 # PC_F3 weight in Phi
    beta: float = 2.0 # PC_HS weight in Phi
    terminal_bonus: float = 5.0
    turnover_penalty: float = -1.0
    # Was -2.0. A turnover has to be at least as expensive as running the clock
    # out, or conceding possession is the cheap way out of a losing episode.
    timeout_penalty: float = -0.5
    # Weight on the per-attacker local shaping term (reward.agent_shaping).
    agent_alpha: float = 0.5
    # Discount, shared by the learner and the shaping term, potential-based
    # shaping is only policy-invariant when the two match, so this is one knob
    gamma: float = 0.999
    use_gamma_in_shaping: bool = True
    zero_terminal_potential: bool = True
    pc_backend: str = "spearman" # or "knn"
    t_max: int = 500

@dataclass
class PPOConfig:
    n_envs: int = 6
    n_steps: int = 128
    lr: float = 3e-4
    # gamma * lam sets the credit window: 0.95 gave 20 ticks against 300-tick
    # episodes, which is shorter than anything the attackers do off the ball.
    gae_lambda: float = 0.99
    clip_coef: float = 0.2
    # One coefficient per head. The ball head is only unmasked when an attacker
    # actually holds the ball, so a single shared coefficient regularises it
    # hundreds of times more weakly than the two movement heads.
    ent_coef_dir: float = 0.004
    ent_coef_speed: float = 0.004
    ent_coef_ball: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    n_minibatches: int = 4
    total_timesteps: int = 5_000_000
    seed: int = 0