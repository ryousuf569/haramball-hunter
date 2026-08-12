# haramball-hunter

## Setup (Windows / PowerShell)

From the `haramball-hunter` folder that contains this README:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are the ones this was developed against on Python 3.13.
Once the venv is activated, plain `python` refers to it, so the commands below need no version flag.

If PowerShell blocks the activate script, allow it for the current session first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run

```powershell
python render.py                          # watch a possession, 5M.pt driving the attackers
python train.py                           # the three-arm experiment, 1M steps each
python train.py --steps 200000            # a short rehearsal first
python plots.py models/figs               # redraw figures from saved history JSON
python scripts/diag_policy.py --ckpt models/constrained.pt
python scripts/diag_policy.py --random    # the control -- run this too
```

The attackers are trained with PPO under behaviour **constraints** rather than
a hand-weighted reward — see [Behaviour constraints](#behaviour-constraints).
`train.py --no-constraints` falls back to plain PPO on the reward alone for an
A/B.

Deactivate the venv when done:

```powershell
deactivate
```

## Pitch control: why the integration loop compacts

`PPCF_grid` integrates control forward in 0.08 s steps until every cell on the 31 x 34 grid
reaches 0.99, and cells converge at very different rates. Profiling the production
configuration (20 players, 1054 cells) showed the loop running 71-93 iterations, set entirely
by the slowest cells, while the median cell was finished in about 35 and the fastest in 8.
The slow cells are not scattered noise -- they are the grid corners, far from every player, so
no one has a short time-to-intercept there and control accrues slowly. The original loop
computed the full (1054, 20) intercept-probability array every iteration and multiplied the
already-converged rows out to zero with `np.where`, which meant that by the back half of the
integration most of the array was being recomputed only to be discarded. Across a full call
that was 37,088 of 74,834 cell-iterations, almost exactly half the arithmetic, spent on cells
whose values could no longer change.

The loop now carries a boolean `active` mask and runs each iteration on just the still-integrating
rows. This is safe because control is monotonically non-decreasing -- every increment is
non-negative -- so a cell that crosses 0.99 can never fall back below it, and the active set only
ever shrinks. The rewrite is a pure performance change and is held to that standard: the golden
fixture in `tests/` was generated from the previous implementation, and the current code
reproduces it bit-for-bit, so the threshold, the timestep, the horizon and the intercept physics
are all untouched. Measured over 30 calls after warmup, the same configuration went from
31.79 ms to 24.51 ms per call, a 1.30x speedup. The gain is smaller than the halved arithmetic
suggests because the up-front `TTI_vec` call is a fixed 5.60 ms that compaction cannot touch,
and because gathering and scattering the active rows costs roughly 5.7 ms of the theoretical
saving. This matters because the call runs every tick regardless of whether the heatmap is
drawn -- it also populates `players['i_p']`, which the interception check depends on.

Verify the numbers are unchanged with:

```powershell
python tests/test_ppcf_golden.py
```

## The vectorised env, and why 6 workers means Async

`environment/lowblock_env.py` exposes `LowBlockEnv`, a Gymnasium env wrapping one
low-block possession, with `environment/reward.py`'s potential-based shaping
wired in: `reset()` seeds `Phi(s0)`, `step()` pays `F = gamma * Phi(s') - Phi(s)`
plus the success bonus, and `max_ticks` raises the `TIMEOUT` the reward already
knows about. Wiring it in is free -- `step()` already builds the attacker
pitch-control surface for the shot test, and `reward.phi_from_pc_att` consumes
that same array rather than paying for a second `PPCF_grid` call, which at
~27 ms a tick would have roughly doubled the cost of a step.

**The attackers are the learners.** The low block is scripted and stays that
way: it is the fixed opponent being learned against, and `compute_defender_targets`
has no policy seam. All ten attackers share one set of weights, so the whole
interface is per-agent and batched on a leading `n_att` axis: the action is a
`MultiDiscrete` of shape `(n_att, 3)` -- `(direction, speed, ball)` per attacker,
where the first two index `physics/engine.py`'s lookup tables and the third is
0 for HOLD or `k` for a pass to the `k`-th sorted teammate id. With
`scripted_attackers=False` (how training runs) `attackers/baseline_attacker.py`
is bypassed entirely and the attackers move straight off the physics with no
hand-tuned model in the loop; that module is a test harness for the defenders,
not a policy, and the env only imports it lazily, on the `scripted_attackers=True`
path, so deleting it does not make the env unimportable.

### What the actor sees, what the critic sees

`obs()` returns `(n_att, 92)` float32. Row `i` is attacker `i`'s view: its own
position and velocity, then the shared world -- all 21 players in fixed roster
order, ball position, an in-flight flag, and remaining time. The ego prefix is
the only thing distinguishing the rows, and it is what lets one network act as
ten agents.

`info["state"]` is the critic's 99-dim global vector: the same 21 players and
ball, a holder one-hot over the attackers, the reward's two zone-control values
(free -- they are already computed for the shaping), and remaining time. Both
`reset()` and `step()` return it, along with `info["action_mask"]`, because a
rollout needs `V(s_0)` and a legal first action before it has taken a step.

The action mask is `(n_att, BALL_ACTIONS)` bool over the ball head only;
direction and speed are always fully legal. The holder's row is fully legal, and
every other row is masked down to index 0, HOLD, which is the no-op there
anyway. Exactly one legal action rather than none is deliberate: an all-False
row sends a masked softmax to NaN, whereas a one-hot row normalises to
probability 1 and so contributes log-prob 0 and entropy 0 with no special-casing
in the learner.

Remaining time is `(max_ticks - tick) / max_ticks`, and it is in both vectors on
purpose. Because timeout is a *terminal* here rather than a truncation, the same
board position is worth very different amounts at tick 10 and at tick 299;
without the clock the two are the same input, the env stops being Markov in the
observation, and the value function is fitting an average over a hidden
variable (Pardo et al., *Time Limits in Reinforcement Learning*, 2018). It looks
redundant. It is not.

Positions and velocities go through `norm_pos` and `norm_vel` -- the unit box,
not pitch metres -- and `obs()` and `global_state()` both call those same two
helpers. If they ever diverge the critic fits different units to the same world,
and it presents as a value loss that plateaus for no visible reason.

All three outcomes report `terminated=True`, timeout included, and `truncated`
is always `False`. `reward.py` counts `TIMEOUT` in `TERMINAL_OUTCOMES` and zeroes
`Phi(terminal)` for it, so the shaping already treats a timeout as absorbing;
reporting it as truncated would invite a learner to bootstrap `V(s_T)` on top and
break the exact telescoping identity `environment/reward_readme.md` rests on.

`make_vector_env(n_envs=6)` returns a `SyncVectorEnv`; `asynchronous=True` gives
one process per env. Benchmark both with:

```powershell
python scripts/bench_vector.py --envs 1 2 4 6 --both
```

On a 12-logical-CPU machine, 400 timed calls per config, BLAS pinned to one
thread, `scripted_attackers=False` so the timing covers the whole path a
training step actually pays for -- decode the `(n_att, 3)` action, integrate,
PPCF, reward, then build the per-agent obs, the global state and the action mask:

| mode  | n_envs | ms/call | ms/step | env-steps/sec |
|-------|--------|---------|---------|---------------|
| sync  | 1      |  20.7   |  20.7   |   48.2        |
| sync  | 2      |  40.5   |  20.3   |   49.3        |
| sync  | 4      |  80.6   |  20.1   |   49.7        |
| sync  | 6      | 122.7   |  20.4   |   48.9        |
| async | 1      |  21.7   |  21.7   |   46.2        |
| async | 2      |  25.0   |  12.5   |   80.1        |
| async | 4      |  44.1   |  11.0   |   90.7        |
| async | 6      |  45.6   |   7.6   |  131.6        |

These supersede an earlier set measured before the reward, the global state and
the mask were wired in. Counter-intuitively the numbers went **up**, not down:
those three cost very little -- the reward reuses the pitch-control surface
`step()` already builds, and the state reuses the two zone values the reward
already computed -- while the scripted `compute_attacker_targets` they replaced
was itself doing real work every tick. Async at 6 workers went 102.7 ->
~132 env-steps/sec, confirmed across two runs (131.6, 134.1). Schedule long
campaigns against this figure, not the old one.

`SyncVectorEnv` steps its envs in a serial loop, so `ms/call` grows roughly
linearly with `n_envs` while throughput stays flat at ~48-50 env-steps/sec no
matter how many envs you add. That is the design, not a defect, and it is the
whole reason "6 workers" has to mean Async if the point is collection.

Sync's per-env `ms/step` is essentially flat -- the vector wrapper itself costs
little. Do not read much into the wobble across the async rows either: a repeat
run put `n_envs=4` at 8.7 ms/step (115.1 steps/sec) against the 11.0 above,
while `n_envs=6` reproduced to within 2%. Compare modes, not adjacent rows.

`AsyncVectorEnv` at 6 workers is ~2.9x the per-env speed and ~2.8x the
throughput. Use Sync for debugging and determinism, Async to collect.

Note that `scripts/bench_parallel.py` measures a different thing: sustained
throughput of N independent worker processes over minutes. `bench_vector.py`
measures the latency of a single vector call, which is what an on-policy
learner waits on between forward passes.

## Behaviour constraints

The attacker policy is trained as a **constrained** MDP: maximise the task
reward subject to `J_C_k(pi) <= d_k` for a set of behaviours, with the weights
found by Lagrange multipliers instead of by hand. The formulation is Roy et al.,
*Direct Behavior Specification via Constrained Reinforcement Learning*
(arXiv:2112.12228). Code: `environment/costs.py` (the specification),
`ppo.Lagrange` (the multipliers), `ppo.CostCritics` (one value function per
constraint), `config.LagrangeConfig`.

### Why the behaviour left the reward

Measured over 24 episodes of the 5M checkpoint against a uniform-random
control (`scripts/diag_policy.py --random`):

| | 5M.pt | random |
|---|---|---|
| success / failure / timeout | 0% / 100% / 0% | 0% / 92% / 8% |
| pass completion | 73.9% | 89.2% |
| episode length | 78 ticks | 166 ticks |
| passes per episode | 3.67 | 8.88 |
| mean pass length | 28.1 m | 25.2 m |
| passes >20m laterally | 46.6% | 38.0% |
| mean forward progress per pass | **-0.8 m** | +0.2 m |
| passes released <5 ticks after receiving | 97.7% | 100% |
| attackers in the final third, per tick | 7.37 | 3.14 |
| mean attacker x, start -> end | 64.6 -> 77.1 | 64.6 -> 64.3 |

The policy was worse than chance at every part of the task except getting
upfield, which is the one thing `Phi = alpha*PC_F3 + beta*PC_HS` pays for
directly. The ball was in flight 91% of every episode; the carrier held it for
a median of **0** ticks. That is not a weight that needs another tuning pass.
`alpha`, `beta`, the three terminals, `agent_alpha`, `OFFSIDE_W` and three
entropy coefficients are eight numbers that between them had to encode "attack,
but keep the ball, but don't drift offside, but don't stall", and the paper's
result is that this search does not scale: 0 of 343 weight combinations were
feasible once three behaviours had to hold simultaneously.

Note that a constraint changes the *feasible set*, not the shaping, so the
policy-invariance claim in `environment/reward_readme.md` is unaffected and the
telescoping identity `tests/test_reward.py` asserts still holds exactly.

### The cost is an indicator, not a magnitude

`C_k(s, a)` is in `{0, 1}`. That makes `J_C_k` a behaviour **frequency**, so
`d_k` reads as "at most this fraction of passes" — a number you can defend from
a match, needing no calibration run to interpret. A cost of "how many metres
past the offside line" would need one.

**Numerator and denominator.** Each cost carries an `attempt` alongside it: the
events it is a rate over. A pass-conditioned cost fires on well under 1% of
agent-steps, so a per-step threshold is both unreadable ("at most 0.2% of steps
may be a cross-field pass") and starves its critic. The multiplier update
therefore reads `rate_k = sum(cost_k) / sum(attempt_k)`. This is a deliberate
deviation from the paper, which averages the indicator over the whole batch;
the *advantage* still uses the raw per-step indicator, so only the multiplier
update sees the ratio.

**Who pays.** Costs are per-agent, shaped `(n_att, K)`, on the same axis as the
per-agent reward and advantage. A pass cost lands on the **passer's** row only
— the other nine attackers did not choose it and must not be taught that they
did. Costs that are properties of a player rather than a decision (offside,
distance to the ball) land on that player's own row, and every row is an
attempt.

**When a pass cost fires.** `pass_lost` is only known 1–25 ticks after the ball
leaves the passer's feet, so its attempt is booked at release and its cost at
the tick the ball is actually lost, still on the passer's row. GAE carries it
back across that gap; booking only passes whose outcome resolves in the same
tick would never fire at all.

### The constraint set

Every threshold is a measured rate, not a guess — left column is the 5M
checkpoint, middle is the uniform-random control. `d_k` sits below both where
the policy is worse than random, and inside the current rate otherwise, so no
constraint starts already satisfied (one that does contributes nothing and only
costs a critic).

| constraint | fires on | 5M | random | `d_k` |
|---|---|---|---|---|
| `pass_lost` | pass attempt | 26.1% | 9.9% | 0.08 |
| `cross_field` (\|dy\| > 20m) | pass attempt | 46.6% | 38.0% | 0.10 |
| `hot_potato` (released <5 ticks) | pass attempt | 97.7% | 100% | 0.25 |
| `pass_back` (dx < -2m) | pass attempt | 37.5% | 42.7% | 0.25 |
| `offside` | attacker-tick | 6.4% | 1.9% | 0.02 |
| `far_from_ball` (>30m) | attacker-tick | 29.4% | 26.7% | 0.20 |
| `no_success` | episode | 100% | 99.3% | 0.60 |

`hot_potato` is the one that buys dribbling. `DRIBBLE_V_MAX` already makes
carrying cost speed, so carrying is otherwise strictly dominated and nothing in
the reward makes it preferable. `far_from_ball` is the shape constraint: it is
what stops ten attackers sprinting into the final third to inflate `PC_F3` and
leaving every pass a 30m diagonal. `offside` replaces `OFFSIDE_W`, and
`config.EnvConfig` now ships `agent_alpha = 0.0` and
`offside_in_potential = False` — the code and its tests stay, and one flag each
puts them back.

### The multipliers

```
lambda_k = exp(z_k) / (exp(a0) + sum_j exp(z_j))
z_k     += lr * (rate_k - d_k)
A        = lambda_0 * A_R  -  sum_k lambda_k * A_C_k
```

The minus sign is because the constraints are on costs to be kept *below* a
threshold: an action that raises the expected count of intercepted passes has a
positive cost advantage and must come out of the objective.

The **softmax over `[a0, z]`** is the paper's normalisation and it is why this
is stable rather than divergent. The reward's own weight `lambda_0` and every
constraint's weight live on one simplex, so a constraint that stays violated
for a long stretch — exactly what happens early, when the policy is bad at
everything — cannot drive the gradient scale to infinity. It can only take
weight from the others. An unnormalised multiplier diverges here. It also means
no `1 / (1 + sum lambda)` normaliser is needed on the combined advantage, and
that the combined advantage is normalised **once**, after the sum: normalising
each channel first would rescale every constraint to unit variance and throw
away exactly what the multipliers encode.

`a0` is fixed, not learned. It sets the floor on how much of the simplex the
task keeps: at `a0 = 0` with seven constraints all at `z = 0`, the reward holds
1/8. Multipliers are held still for `warmup_updates` so the cost critics see
data first, and move once per PPO update — a saddle-point problem needs the
multipliers slower than the policy or the two oscillate.

**The bootstrap constraint.** `no_success` is the main task restated as "fail
no more often than `d`", and the reward's weight is
`lambda_0 := max(lambda_0, lambda_no_success)`. Without it, the policy that
satisfies "don't lose the ball", "don't pass across the field" and "don't pass
backwards" for the least effort is the one that **never passes at all** —
constraint satisfaction with the task abandoned, which is the collapse the
paper reports for multi-constraint Lagrangian RL. Tying the reward's weight to
the multiplier of the failure constraint means the worse the policy is at the
task, the more weight the task gets.

It is the one threshold not set from a measured rate. `d = 0.60` asks for a 40%
success rate against 2.6% for random play at the calibrated gate, and that is
the point: it has to stay violated for its multiplier to keep the reward's
weight up. A first pass used `0.85`, which random play nearly satisfied at
curriculum level 0 — `lambda_no_success` sat *below* `lambda_0` for a whole run
and the bootstrap never engaged.

**One critic per cost.** The paper specifies independent critics; a shared
trunk lets a cost whose scale happens to be large dominate the representation
every other head reads. Measured here, 6 async envs: 50 sps with no constraints
at all, 54 with seven independent critics, 59 with seven heads on one trunk —
all inside the run-to-run noise, because the step cost is the PPCF integration
and not the networks. 1.1M extra critic parameters buy nothing back on a step
that spends ~54ms in `physics/ppcf.py`. So the faithful option is free and is
the default; `PPOConfig.shared_cost_critic` exists for a future where the env
gets cheap or the constraint count grows.

### The success gate was the real ceiling

Constrained RL replaces weight tuning. It does not create exploration, and the
bootstrap constraint assumes success is at least occasionally observed — so the
gate had to be fixed too.

`check_shot_opening` needs `scoring_probability >= 0.74`, which is a **15.7m**
radius around the goal, plus `>= 0.30` own pitch control in a 3m disc, plus no
defender within 3m. Measured over 137 random-play episodes: the pitch-control
condition is reached in 96% of episodes and the defender condition in 92%, but
the scoring-probability condition in **6.6%**, and full success fires in
**0.7%**. A 0.7% terminal against a 99.3% penalty is not a signal a policy can
climb; it is noise with a mean. `LowBlockEnv` now reports
`info["shot_gate"]` — the episode's closest approach to each of the three
conditions — so a run can see *which* condition binds rather than only that the
success rate is zero.

`ppo.Curriculum` anneals the gate, on the goal **radius** and not on the
probability. `S(r)` is nonlinear enough that `p = 0.74` is 15.7m but `p = 0.60`
is 50m, over half the pitch; a p-linear schedule spends most of its length
rewarding a shot from the halfway line, hits 93% success in one update, and
teaches a habit the policy has to unlearn when the gate closes. Level 0 is a
25m gate with the attacking shape started 4m further on, which **uniform-random
play converts 22% of the time** — dense enough that the terminal is a signal,
far enough from free that there is something to learn. (30m gives random play
51%, 35m gives 56%; the real gate at `x_shift = 0` gives 2.6%.) `advance_at`
is 0.35, above that 22% floor, so a level is earned by beating chance rather
than by arriving. Level 8 is the calibrated gate with the fitted formation, and
the level never goes back down — oscillating the task definition underneath a
value function is its own failure mode.

Nothing outside training ever sees the loosened gate: `scripts/diag_policy.py`,
`render.py` and every env built by a test use `termination.SHOT_P_MIN`.

### Reading a run

```
lambda r0 0.180 | pass_lost 0.261/0.08 l0.147  cross_field 0.466/0.10 l0.160 ...
gate   pcf 0.824/0.30  scoring_p 0.653/0.690  clear 0.958  | all three reachable 0.31
```

`rate/threshold` and the multiplier, per constraint. This is the line to read:
a rate that will not come down while its `lambda` climbs is a constraint the
policy *cannot* satisfy — a threshold set wrong, not a run that needs longer.
The `gate` line is why success is or is not happening, condition by condition.

**Always run `scripts/diag_policy.py --random` alongside any checkpoint.** It
is the control and it is not a formality: the 5M checkpoint lost to it on pass
completion and episode length. A checkpoint that loses to uniform-random play
is not one that needs more steps.

### The experiment: did the tuning burden go, or move?

`train.py` runs three arms sequentially, 1M steps each, and the question is the
obvious objection to the whole approach — **a threshold is arguably just a
weight wearing a different hat.**

| arm | objective | thresholds |
|---|---|---|
| A `reward_only` | plain PPO, the hand-weighted reward as it stood (`agent_alpha` 0.5, offside inside the potential) | — |
| B `constrained` | the seven constraints | **measured** off 5M.pt and the random control |
| C `constrained_mistuned` | the seven constraints | **guessed** — "no bad passes at all", tighter than the physics allows |

Arm C is the real test. It is not a random perturbation: it is what the
constraints look like written down by someone who has not measured the game
first (`pass_lost ≤ 0.02` against a measured 26%, `cross_field ≤ 0.02` against
47%, `no_success ≤ 0.30` asking for a 70% success rate where random play gets
2.6%).

- **B ≫ A** → the constraints buy behaviour the weights could not, at equal compute.
- **C ≈ B** → the burden really did go. The softmax bounds a multiplier that can never be satisfied, so a badly-set threshold is absorbed rather than fatal.
- **C ≈ A, or worse** → the burden **relocated**, and the paper's claim does not survive contact with this problem.

A second claim is separately testable and does not depend on C's score: whatever
happens, the per-constraint rate/λ trace should say **which** threshold is
impossible. A reward-weight failure never says that. That is
`constrained_mistuned_multipliers.png` — a constraint pinned above zero
violation for a whole run while its λ climbs is infeasible, and you can read it
off the figure in seconds.

The curriculum and the seed are **identical across arms**; only the objective
specification varies. Every arm measures all seven constraint rates, including
A, which never optimises them — that is what makes the comparison meaningful.

Figures land in `models/figs/`, four per arm plus two cross-arm:

| file | what it shows |
|---|---|
| `<arm>_task.png` | outcomes, episode length, return, curriculum level |
| `<arm>_constraints.png` | each rate against its threshold, small multiples, satisfied/VIOLATED per panel |
| `<arm>_multipliers.png` | λ per constraint, and the violation that drives it |
| `<arm>_learning.png` | critic fit, entropy, value losses, KL — the "is this a bug or a finding" panel |
| `comparison.png` | all three arms on success, length, return, and the three headline behaviours |
| `summary_table.png` | final 5% of updates, green where arm B's threshold is met |

Each arm writes its checkpoint, a `_history.json` and its figures the moment it
finishes, so a crash in arm C costs nothing from A and B. `python plots.py
models/figs` redraws everything from the JSON without re-running.

**Budget ~17 h** for the full three-arm 1M run at the measured ~50 sps.
`--steps 200000` is a ~3.5 h rehearsal; `--arms constrained` runs one.

## References

Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* MIT Sloan Sports Analytics Conference.
 
Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.

Roy, J., Girgis, R., Romoff, J., Bacon, P.-L., & Pal, C. (2021). *Direct Behavior Specification via Constrained Reinforcement Learning.* arXiv:2112.12228. — the constrained formulation in `environment/costs.py` and `ppo.Lagrange`.

https://rcsoccersim.readthedocs.io/en/latest/soccerserver.html