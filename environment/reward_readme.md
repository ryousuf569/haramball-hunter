# What the reward is, and what the ablation can claim

Every shaping term in `reward.py` is potential-based: the per-step reward is
`F = gamma * Phi(s') - Phi(s)`, with `Phi(terminal)` forced to zero, and the only
non-shaping term is a terminal bonus `B` paid on success. That form is the whole
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
keep: a terminal *penalty* on turnover would not be potential-based and would
genuinely change the optimum, which is why `terminal_reward` pays on success
only and any penalty belongs in its own explicitly-labelled arm; and the
`use_gamma` / `zero_terminal_potential` flags in `RewardConfig` exist to
*demonstrate* the guarantee breaking when either is switched off, not as tuning
knobs.
