# runs/

Each directory is one training run: `config.json` plus `metrics.csv`.

`sweep_crowd{05,10,15}_s{1,2,3}` and `sweep_crowd{05,10,15}_s{1,2,3}_5m` are the
same nine seed/threshold configs at two step budgets, 2.5M and 5M. The `_5m`
runs supersede the un-suffixed ones for headline numbers (`docs/assets/constraint_*_crowd_5m.csv`,
`docs/results.html`). The 2.5M runs are kept because they're the budget the
original `sweep_slow*` sweep used, so they're the like-for-like comparison for
anything about the cost of training longer, not a stale duplicate to delete.

`pid_{control,team_slow}_s{1,2,3}_25m` are the two new cells of the 2x2 in
`model/ppo_pid.py`, which asks whether the section 6.4 team-mean-indicator
failure is the counting unit or the multiplier update rule. The other two cells
are `abl_control_*` (adam x slow) and `abl_team_slow_*` (adam x team_slow), so
`pid_control_*` is the like-for-like control for the PID arm and the four are
scored together by `model/pid_analysis.py` against `assets/pid_prereg.md`.
