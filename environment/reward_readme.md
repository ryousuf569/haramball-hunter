# What the reward is, and what the ablation can claim

Every shaping term in `reward.py` is potential-based: the per-step reward is
`F = gamma * Phi(s') - Phi(s)`, with `Phi(terminal)` forced to zero. The
non-shaping term is `terminal_reward`, which now pays three different values --
`+terminal_bonus` on success, `turnover_penalty` on a turnover,
`timeout_penalty` on running the clock out. That form is the whole
point. By Ng et al. (1999), potential-based shaping leaves the optimal policy
unchanged, and the zeroed terminal potential makes the discounted sum over an
episode telescope to exactly `-Phi(s0)` for *any* policy -- a constant that
cancels out of every comparison between policies. `tests/test_reward.py` asserts
that identity across several different random trajectories from a shared initial
state; it is exact, so a sign error, a misplaced `gamma`, or a mishandled
terminal all fail it in one assertion.

The consequence is that every arm of the ablation -- full, no-half-space
(`beta=0`), pure-F3, total pitch control, KNN, sparse-only (`alpha=beta=0`) --
shares the same optimal policy. In the limit they are the same problem. So the
honest claim is not "reward term X caused behaviour Y." It is: all of these
shaping functions provably preserve the optimum, and what differs between them
is whether they make the problem *learnable within a fixed compute budget*.
Final-third-restricted pitch control does; whole-pitch control does not;
sparse-only does not. Because the arms differ only in reachability and never in
what is being optimised, the ablation is evidence about which spatial signals
carry usable gradient, not about which reward encodes the right football.

That is a stronger result than a biased-reward story, not a weaker one: the
theoretical guarantee says the solution was never tilted, only made easier or
harder to reach. It also sharpens the critique of Gu et al. -- whole-pitch
control is not merely a suboptimal objective, it is a shaping function that
supplies almost no gradient in this regime, because control far from the goal
barely moves while the attackers work the block. Two caveats the write-up should
keep: the terminal penalties on turnover and timeout are **not** potential-based
and **do** genuinely change the optimum -- that is what they are for, and the
next section spells out what it costs the claim; and the
`use_gamma` / `zero_terminal_potential` flags in `RewardConfig` exist to
*demonstrate* the guarantee breaking when either is switched off, not as tuning
knobs.

## The terminal penalties, and what they cost the claim

`terminal_reward` used to pay `B` on success and nothing on the other two
endings. That made a turnover and running the clock out worth exactly the same:
both terminate, both zero `Phi`, both pay zero. Nothing in the objective
distinguished losing the ball from a stalemate, and with success rare the
measured advantage was almost entirely estimation noise.

The two penalties fix that, at a stated price:

- **The shaping is untouched.** `F` is still potential-based and
  `sum gamma^t F_t == -Phi(s0)` still holds exactly, for any policy, on all
  three endings. `tests/test_reward.py::test_telescoping_identity` and
  `tests/test_env_reward.py::test_env_telescoping_identity` both sum the
  `shaping` component alone and are unaffected by this change.
- **The base reward is a different objective.** It is now
  `{+B, turnover_penalty, timeout_penalty}` rather than `{+B, 0, 0}`. The Ng et
  al. guarantee says the *shaping* never tilted the solution; it says nothing
  about this. So the ablation may still claim "the shaping did not change the
  optimum", and may **not** claim "the objective is just whether a shot opening
  was reached". Any no-shaping baseline has to run the same base reward, or the
  comparison is between two different problems.
- **A turnover is now at least as expensive as a timeout** (`-1` vs `-0.5`).
  The original ordering was the other way round -- timeout `-2`, turnover `-1`
  -- on the argument that timeout is the ending a policy can always guarantee by
  recycling possession, so it had to cost more than trying and failing. That
  argument has a price the write-up recorded and the 5M run then paid: from a
  state with success probability `p`, playing on to a timeout is worth
  `5p - 2(1 - p)` and conceding now is worth `-1`, so conceding wins whenever
  `p < 1/7`. The measured success rate was 15-20%, i.e. right on that boundary,
  and the trained policy released the ball on the first tick of every
  possession. Reversing the two removes the incentive to concede; the stalemate
  ending is still the worst non-scoring outcome available on any state the
  attackers can actually convert.

`RewardConfig.turnover_penalty` / `timeout_penalty` are ordinary tuning knobs,
unlike the two flags above.

## The per-agent term

`agent_shaping` pays attacker `i` a second shaping reward built on its own
potential: the pitch control it holds within `termination.AREA_RADIUS` of
itself. It is potential-based on exactly the same construction as `Phi`, so
`sum gamma^t F_i,t == -phi_i(s0)` per agent and the optimum is untouched -- the
same claim the team term gets, made ten times over.

It exists because the team term is one scalar shared by ten agents. Measured on
the 5M checkpoint, one attacker's best-versus-worst single-tick direction choice
moved `Phi` by 0.007, against a return standard deviation of 1.58; after the
advantage is normalised that is about 0.5% of the gradient, against an entropy
coefficient of 0.01. The movement heads were being regularised toward uniform
roughly twice as hard as they were being trained, and after 5M steps they had
93% of the entropy of a uniform policy. `agent_alpha` weights this term; at 0
the learner is back to a single team advantage broadcast to every agent.
