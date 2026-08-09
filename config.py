from dataclasses import dataclass

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
    agent_alpha: float = 0.5
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