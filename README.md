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

## References

Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* MIT Sloan Sports Analytics Conference.
 
Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.

https://rcsoccersim.readthedocs.io/en/latest/soccerserver.html