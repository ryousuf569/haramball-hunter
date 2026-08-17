import argparse
import csv
import os
import sys

import numpy as np

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CAL_DIR))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MPLBACKEND", "Agg")

# (label, DUEL_A, DUEL_B, DUEL_A_BACK, LAM_MAX, ADV_FLOOR, ADV_EXP). The stages of
# section 7, so the sweep separates what each change did. The multiplier travels
# with each arm: without it the historical rows would run the fitted engagement
# term and stop being faithful reproductions of what they are labelled.
GATES = [
    ("robocup",    1.20, 0.50, 0.40, 2.10, 0.50, 1),  # transferred, what 500k trained on
    ("rate_only",  1.20, 0.50, 0.40, 0.27, 0.50, 1),  # fitted rate, RoboCup geometry
    ("scale_only", 2.40, 1.00, 0.80, 0.27, 0.50, 1),  # + fitted reach, RoboCup ratios
    ("shape_only", 2.38, 2.38, 1.64, 0.24, 0.50, 1),  # + fitted axes, old multiplier
    ("calibrated", 2.38, 2.38, 1.64, 0.57, 0.14, 3),  # + engagement: turnover.py as it is
]

# The 0/150 arm. reward_only.pt is the 4% one and is not the case in question.
CKPT = os.path.join("models", "081226-500k", "constrained.pt")
POLICIES = ["random", "constrained_500k"]

BASE_SEED = 1000          # diag_policy.py's seed base, so the episodes line up
CAUSES = ["duel", "intercept", "loose pass", "offside"]


def worker(args):
    """One episode, at one gate. Module level so it pickles for the spawn pool.

    The gate is set on defenders.turnover inside the child: Windows spawns a
    fresh interpreter per worker, so a patch applied in the parent would not
    survive. ground_duel reads these as module globals at call time, which is
    why rebinding them works even though lowblock_env imported the function by
    name -- the same mechanism intercept_calibration.py relies on.
    """
    gate, policy, seed, holder = args
    _label, a, b, back, lam, floor, aexp = gate

    from environment.thread_limits import limit_threads
    limit_threads(1, torch_threads=1)

    import defenders.turnover as turnover
    turnover.DUEL_A, turnover.DUEL_B = a, b
    turnover.DUEL_A_BACK, turnover.LAM_MAX = back, lam
    turnover.ADV_FLOOR, turnover.ADV_EXP = floor, aexp
    turnover.R_MAX = max(a, b, back)

    import torch
    from diag_policy import load_actor, run_episode

    # run_episode draws actions with dist.sample(), which uses torch's GLOBAL
    # RNG. That is process state, not episode state, so without this line an
    # episode's actions depend on how many episodes its pool worker happened to
    # run first -- and the arms stop being matched, which is the only thing that
    # makes this sweep mean anything. Seed per episode, not per worker.
    torch.manual_seed(seed)

    actor = load_actor(CKPT)     # loaded even for the random arm: run_episode
                                 # still builds the distribution it samples the
                                 # action mask from
    outcome, ticks, rec = run_episode(
        actor, seed, holder, deterministic=False,
        random_policy=(policy == "random"))
    return dict(arm=_label, policy=policy, seed=seed, start_holder=holder,
                duel_a=a, lam_max=lam, outcome=outcome, end_tick=ticks,
                cause=rec["turnover_kind"] or "",
                holder_ticks=rec["holder_ticks"],
                min_press_m=(float(np.min(rec["holder_press"]))
                             if rec["holder_press"] else np.nan),
                ball_x_max=float(np.max(rec["ball_x"])) if rec["ball_x"] else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="episodes per cell")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    jobs = [(gate, policy, BASE_SEED + i, i % 10)
            for gate in GATES for policy in POLICIES for i in range(args.n)]
    print(f"{len(jobs)} episodes: {len(GATES)} gates x {len(POLICIES)} policies "
          f"x {args.n}, on {args.workers} workers")

    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        rows = pool.map(worker, jobs)

    runs_path = os.path.join(CAL_DIR, "duel_gate_recheck_runs.csv")
    fields = ["arm", "policy", "duel_a", "lam_max", "seed", "start_holder",
              "outcome", "cause", "end_tick", "holder_ticks", "min_press_m",
              "ball_x_max"]
    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in ("min_press_m", "ball_x_max"):
                out[k] = round(out[k], 3) if np.isfinite(out[k]) else ""
            w.writerow({k: out[k] for k in fields})

    summary = [("episodes_per_cell", args.n), ("checkpoint", CKPT),
               ("base_seed", BASE_SEED)]
    print("\n" + "=" * 96)
    print("BASE SUCCESS RATE AND TURNOVER MIX BY DUEL GATE")
    print("=" * 96)
    print("%-17s %-6s %-6s %8s %8s %8s %10s %8s %8s %9s" % (
        "policy", "A", "LAM", "success", "timeout", "duel", "intercept",
        "loose", "offside", "med_tick"))
    for policy in POLICIES:
        for label, a, _b, _back, lam, _f, _e in GATES:
            sub = [r for r in rows if r["arm"] == label and r["policy"] == policy]
            n = len(sub)
            succ = sum(1 for r in sub if r["outcome"] == "success")
            tmo = sum(1 for r in sub if r["outcome"] == "timeout")
            c = {k: sum(1 for r in sub if r["cause"] == k) for k in CAUSES}
            med = int(np.median([r["end_tick"] for r in sub]))
            print("%-17s %-6.2f %-6.2f %7.1f%% %8d %8d %10d %8d %8d %9d" % (
                policy, a, lam, 100 * succ / n, tmo, c["duel"], c["intercept"],
                c["loose pass"], c["offside"], med))
            tag = f"{policy}_{label}"
            summary += [
                (f"{tag}_success", succ),
                (f"{tag}_success_rate", round(succ / n, 4)),
                (f"{tag}_timeout", tmo),
                (f"{tag}_median_end_tick", med),
            ] + [(f"{tag}_{k.replace(' ', '_')}", c[k]) for k in CAUSES] + [
                (f"{tag}_duel_rate", round(c["duel"] / n, 4)),
            ]
        print()

    summary_path = os.path.join(CAL_DIR, "duel_gate_recheck_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary)

    print(f"wrote {os.path.basename(runs_path)} and "
          f"{os.path.basename(summary_path)} into {CAL_DIR}")


if __name__ == "__main__":
    main()
