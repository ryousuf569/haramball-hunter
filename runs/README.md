# runs/

Each directory is one training run: `config.json` plus `metrics.csv`.

`sweep_crowd{05,10,15}_s{1,2,3}` and `sweep_crowd{05,10,15}_s{1,2,3}_5m` are the
same nine seed/threshold configs at two step budgets, 2.5M and 5M. The `_5m`
runs supersede the un-suffixed ones for headline numbers (`docs/assets/constraint_*_crowd_5m.csv`,
`docs/results.html`). The 2.5M runs are kept because they're the budget the
original `sweep_slow*` sweep used, so they're the like-for-like comparison for
anything about the cost of training longer, not a stale duplicate to delete.
