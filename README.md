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
python render.py
```

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

## References

Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* MIT Sloan Sports Analytics Conference.
 
Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.

https://rcsoccersim.readthedocs.io/en/latest/soccerserver.html