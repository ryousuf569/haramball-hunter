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
has no policy seam. The action is therefore a `Dict` -- `"velocity"`, an
`(n_att, 2)` array of target velocities handed straight to the kinematics
integrator, and `"pass"`, a `Discrete(n_att)` where 0 holds and `k` passes to
the `k`-th sorted teammate id. With `scripted_attackers=False` (how training
runs) `attackers/baseline_attacker.py` is bypassed entirely and the attackers
move straight off the physics with no hand-tuned model in the loop; that module
is a test harness for the defenders, not a policy. `scripted_attackers=True`,
the default, keeps it for benchmarking and as a scripted reference arm.

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

On a 12-logical-CPU machine, 250 timed calls per config, BLAS pinned to one
thread:

| mode  | n_envs | ms/call | ms/step | env-steps/sec |
|-------|--------|---------|---------|---------------|
| sync  | 1      |  25.5   |  25.5   |   39.3        |
| sync  | 2      |  57.6   |  28.8   |   34.7        |
| sync  | 4      | 127.3   |  31.8   |   31.4        |
| sync  | 6      | 169.9   |  28.3   |   35.3        |
| async | 1      |  25.8   |  25.8   |   38.7        |
| async | 2      |  28.8   |  14.4   |   69.5        |
| async | 4      |  40.9   |  10.2   |   97.7        |
| async | 6      |  58.4   |   9.7   |  102.7        |

`SyncVectorEnv` steps its envs in a serial loop, so `ms/call` grows roughly
linearly with `n_envs` while throughput stays flat at ~31-39 env-steps/sec no
matter how many envs you add. That is the design, not a defect, and it is the
whole reason "6 workers" has to mean Async if the point is collection.

Sync's per-env `ms/step` is essentially flat -- the vector wrapper itself costs
little. Do not read much into the wobble across the sync rows: repeated runs put
`n_envs=1` anywhere in 25.3-29.5 ms and `n_envs=6` anywhere in 27.8-34.8 ms, so
the spread between configs is within run-to-run noise on this machine. Compare
modes, not adjacent sync rows.

`AsyncVectorEnv` at 6 workers is ~2.6x the per-env speed and ~2.6x the
throughput. It is already flattening by 4 envs (10.2 -> 9.7 ms/step going
4 -> 6), so the sim saturates this machine's physical cores at roughly 4-6
workers and more processes would buy little. Use Sync for debugging and
determinism, Async to collect.

Note that `scripts/bench_parallel.py` measures a different thing: sustained
throughput of N independent worker processes over minutes. `bench_vector.py`
measures the latency of a single vector call, which is what an on-policy
learner waits on between forward passes.

## References

Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* MIT Sloan Sports Analytics Conference.
 
Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.

https://rcsoccersim.readthedocs.io/en/latest/soccerserver.html