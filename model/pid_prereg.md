# Pre-registered reading, section 6.4 follow-up

Fixed before any evaluated PID run. n = 3 seeds per arm; claim nothing beyond
n = 3.

The failure under test, from runs/abl_team_slow_s*_25m: `lam_slow` swings
between ~0.001 and >0.9, and the bootstrap never hands over. `boot == 1` on 100%
of logged rows, so `lam0_eff = max(lam0, lam_succ)` sits at ~0.87-0.89 and the
task reward keeps riding on the success constraint rather than on `lam0`. The
effective reward weight is therefore not "<1%"; `lam0` alone is, but `lam0`
alone is not what multiplies the reward advantage.

A team-mean target of 0.75 is feasible, so infeasibility is not the
explanation: assets/speed_probe.csv has frozen per-agent policies spanning
c_team 0.65-0.87 at success 0.68-0.82.

The only difference between the adam and pid cells is the multiplier update
rule. The softmax simplex, the `max(lam0, lam_succ)` bootstrap, advantage
standardisation, the cost GAE and the policy update are shared code.

## PID fixes it -> the failure is the UPDATE RULE, not the counting unit

All four, in at least 2 of 3 seeds of `pid x team_slow`:

1. tail |rate - 0.75| <= 0.05, tail being the last 20% of logged points
2. no logged point with lam_slow > 0.5
3. the bootstrap hands over, i.e. boot_step is finite
4. tail success inside the control arm's seed range

## PID does not fix it -> the COUNTING UNIT is load-bearing here

`lam_slow` still swings above 0.5 and there is no handover.

## Anything else -> mixed

Report it as mixed. No claim either way.
