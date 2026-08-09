# Tests for the learner-side machinery: per-head entropy, per-agent advantages,
# and the value-target normaliser.
#
# What these guard against is not a crash -- every one of these bugs trains
# perfectly happily and produces a policy that does nothing. The 5M checkpoint
# had direction entropy at 93% of uniform and a ball head collapsed onto three
# actions, because the entropy bonus was being applied to rows that had no
# choice to make and the advantage was one team scalar broadcast ten ways.
# Run: python tests/test_ppo.py   (also works under pytest)
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EnvConfig, PPOConfig  # noqa: E402
from environment.lowblock_env import (  # noqa: E402
    LowBlockEnv,
    obs_dim,
    state_dim,
)
from ppo import (  # noqa: E402
    Actor,
    Critic,
    MultiCategorial,
    RunningNorm,
    agent_rewards,
    compute_gae,
    make_venv,
)

N_ATT, N_DEF = 10, 11
TOL = 1e-6


def _dist(n_rows=8, n_ball=N_ATT, holder=3):
    """Uniform logits with one row unmasked, as an env step actually looks."""
    logits = [torch.zeros(n_rows, 9), torch.zeros(n_rows, 3),
              torch.zeros(n_rows, n_ball)]
    mask = torch.zeros(n_rows, n_ball, dtype=torch.bool)
    mask[:, 0] = True
    if holder is not None:
        mask[holder, :] = True
    return MultiCategorial(logits, [None, None, mask])


def test_masked_entropy_ignores_rows_with_one_legal_action():
    # The ball head is unmasked on the holder's row only. Averaging the other
    # nine zeros in scaled its regulariser by 1/10 on a tick where an attacker
    # holds the ball, and by 0 on the 95% of ticks where none does.
    d = _dist()
    ent = d.masked_head_entropies()

    assert abs(ent[0].item() - np.log(9)) < TOL, ent[0]
    assert abs(ent[1].item() - np.log(3)) < TOL, ent[1]
    assert abs(ent[2].item() - np.log(N_ATT)) < TOL, (
        f"ball-head entropy {ent[2].item():.4f} diluted by masked rows")

    naive = d.entropy().mean().item()
    assert naive < sum(e.item() for e in ent), (
        "the unmasked average should be the smaller one; is the mask wired up?")


def test_masked_entropy_is_finite_when_nobody_holds_the_ball():
    # 95% of ticks. An empty selection must not produce a nan that silently
    # poisons the loss.
    ent = _dist(holder=None).masked_head_entropies()
    for e in ent:
        assert torch.isfinite(e), ent


def test_per_agent_gae_keeps_agents_separate():
    # One agent gets all the reward. With a per-agent advantage only that agent
    # sees it; the old team scalar gave all ten the identical number.
    T, E, n = 4, 2, 3
    rewards = torch.zeros(T, E, n)
    rewards[:, :, 1] = 1.0
    values = torch.zeros(T, E, n)
    dones = torch.zeros(T, E, 1)
    adv, ret = compute_gae(rewards, values, dones, torch.zeros(E, n),
                           torch.zeros(E, 1), gamma=0.99, lam=0.95)

    assert adv.shape == (T, E, n), adv.shape
    assert (adv[:, :, 1] > 0).all(), "the paid agent got no advantage"
    assert torch.allclose(adv[:, :, 0], torch.zeros(T, E)), (
        "an unpaid agent picked up its team-mate's advantage")
    assert torch.allclose(adv[:, :, 2], torch.zeros(T, E))


def test_gae_credit_window_matches_the_configured_lambda():
    # gamma * lam is the whole credit horizon. At 0.95 it was 20 ticks against
    # 300-tick episodes, which is shorter than any off-the-ball movement.
    cfg, env_cfg = PPOConfig(), EnvConfig()
    window = 1.0 / (1.0 - env_cfg.gamma * cfg.gae_lambda)
    assert window > env_cfg.t_max / 10, (
        f"credit window {window:.0f} ticks against {env_cfg.t_max}-tick episodes")


def test_running_norm_tracks_mean_and_std():
    rn = RunningNorm()
    x = torch.randn(4000) * 3.0 + 7.0
    for chunk in x.chunk(8):
        rn.update(chunk)

    assert abs(rn.mean - 7.0) < 0.3, rn.mean
    assert abs(rn.std - 3.0) < 0.3, rn.std
    assert torch.allclose(rn.denormalize(rn.normalize(x)), x, atol=1e-3)


def test_critic_has_one_head_per_attacker():
    critic = Critic(state_dim(N_ATT + N_DEF, N_ATT), N_ATT)
    out = critic(torch.zeros(5, state_dim(N_ATT + N_DEF, N_ATT)))
    assert out.shape == (5, N_ATT), out.shape


def test_hold_bias_makes_carrying_the_default():
    # Carrying into the scoring radius is 40-60 consecutive HOLDs. From a
    # uniform ball head that is 0.1^50, which is why no run has ever observed a
    # success. The head has to start on HOLD for the terminal to be reachable.
    actor = Actor(64, 9, 3, N_ATT)
    _d, _s, b = actor(torch.zeros(1, 64))
    p_hold = torch.softmax(b, dim=-1)[0, 0].item()
    assert p_hold > 0.9, f"P(HOLD) at init is {p_hold:.3f}"

    # Expected carry length, and the odds of a carry long enough to matter.
    assert 1.0 / (1.0 - p_hold) > 10, "carries are too short to cross the block"
    assert p_hold ** 50 > 1e-3, f"a 50-tick carry has probability {p_hold ** 50:.2e}"

    uniform = Actor(64, 9, 3, N_ATT, hold_bias=0.0)
    _d, _s, b = uniform(torch.zeros(1, 64))
    assert abs(torch.softmax(b, dim=-1)[0, 0].item() - 1 / N_ATT) < 0.02, (
        "hold_bias=0 should leave the head uniform")


def test_actor_input_width_matches_the_env():
    env = LowBlockEnv(max_ticks=20, scripted_attackers=False)
    obs, _info = env.reset(seed=0)
    actor = Actor(obs_dim(env.n_players, env.n_att), 9, 3, env.ball_actions)
    d, s, b = actor(torch.as_tensor(obs))
    assert d.shape == (env.n_att, 9) and s.shape == (env.n_att, 3)
    assert b.shape == (env.n_att, env.ball_actions)


def test_agent_reward_comes_from_final_info_on_a_terminal_step():
    # Under SAME_STEP autoreset a done env's top-level info is the one reset()
    # built, so the terminal per-agent reward only exists in final_info. Reading
    # the top level there would pair the team's terminal reward with the next
    # episode's shaping.
    n_envs = 3
    venv = make_venv(EnvConfig(t_max=12), n_envs, seed=0, asynchronous=False)
    try:
        venv.reset(seed=0)
        for _ in range(40):
            _obs, _r, term, trunc, info = venv.step(venv.action_space.sample())
            done = np.logical_or(term, trunc)
            ar = agent_rewards(info, done, n_envs, N_ATT)
            assert ar.shape == (n_envs, N_ATT), ar.shape
            assert np.isfinite(ar).all()
            if done.any():
                final = np.asarray(info["final_info"]["agent_reward"])
                assert np.allclose(ar[done], final[done]), (
                    "terminal step took the reset env's per-agent reward")
                return
        raise AssertionError("no episode terminated in 40 steps")
    finally:
        venv.close()


def test_agent_reward_is_the_top_level_one_on_a_live_step():
    n_envs = 2
    venv = make_venv(EnvConfig(t_max=60), n_envs, seed=0, asynchronous=False)
    try:
        venv.reset(seed=0)
        _obs, _r, term, trunc, info = venv.step(venv.action_space.sample())
        done = np.logical_or(term, trunc)
        assert not done.any(), "this test wants a step where nothing terminated"
        assert np.allclose(agent_rewards(info, done, n_envs, N_ATT),
                           np.asarray(info["agent_reward"]))
    finally:
        venv.close()


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
