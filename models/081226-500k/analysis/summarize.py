"""Numbers for the write-up, read off the three history JSONs in ../figs.

Prints a phase-by-phase table per arm and the cross-arm comparison, so claims
in findings.md can be checked: python models/081126-500k/analysis/summarize.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
sys.path.insert(0, ROOT)

from environment import costs  # noqa: E402

ARMS = ["reward_only", "constrained", "constrained_mistuned"]
LABEL = {"reward_only": "A reward_only",
         "constrained": "B constrained",
         "constrained_mistuned": "C mistuned"}


def load(arm):
    with open(os.path.join(FIGS, f"{arm}_history.json")) as f:
        return json.load(f)


def col(h, k):
    return np.array([r[k] for r in h], dtype=float)


def mat(h, k):
    return np.array([r[k] for r in h], dtype=float)


def tail(h, frac=0.10):
    """The last `frac` of updates, which is what an end-of-run number means."""
    return h[-max(1, int(len(h) * frac)):]


def agg(rows, key):
    return float(np.mean([r[key] for r in rows]))


def agg_vec(rows, key):
    return np.array([r[key] for r in rows], dtype=float).mean(axis=0)


def headline():
    print("=" * 96)
    print("END OF RUN  (mean over the final 10% of updates)")
    print("=" * 96)
    hdr = f"{'arm':16s} {'success':>8s} {'ep_len':>7s} {'return':>8s} {'curr':>5s} {'ev':>7s}"
    print(hdr)
    for arm in ARMS:
        blob = load(arm)
        t = tail(blob["history"])
        print(f"{LABEL[arm]:16s} {agg(t,'success'):8.3f} {agg(t,'ep_len'):7.1f} "
              f"{agg(t,'ep_return'):+8.2f} "
              f"{agg(t,'curriculum_level'):5.1f} {agg(t,'explained_var'):7.3f}")


def constraint_table():
    print()
    print("=" * 96)
    print("CONSTRAINT RATES, END OF RUN.  * = above arm B's threshold")
    print("=" * 96)
    print(f"{'constraint':15s} {'d_k':>6s} " +
          " ".join(f"{LABEL[a][:12]:>14s}" for a in ARMS))
    ends = {a: agg_vec(tail(load(a)["history"]), "cost_rate") for a in ARMS}
    for k, name in enumerate(costs.COST_NAMES):
        d = costs.COST_THRESHOLDS[k]
        cells = []
        for a in ARMS:
            v = ends[a][k]
            cells.append(f"{v:13.3f}{'*' if v > d else ' '}")
        print(f"{name:15s} {d:6.2f} " + " ".join(cells))
    return ends


def mistuned_feasibility():
    """Which of arm C's thresholds were impossible, and could you tell?"""
    print()
    print("=" * 96)
    print("ARM C: was the infeasible threshold identifiable from the trace?")
    print("=" * 96)
    blob = load("constrained_mistuned")
    h = blob["history"]
    thr = np.asarray(blob["thresholds"], dtype=float)
    rates = mat(h, "cost_rate")
    lam = mat(h, "lam")
    viol = rates - thr

    print(f"{'constraint':15s} {'d_k':>7s} {'end rate':>9s} {'end viol':>9s} "
          f"{'frac>0':>7s} {'lam 1st':>8s} {'lam end':>8s} {'verdict':>12s}")
    for k, name in enumerate(costs.COST_NAMES):
        frac_violating = float((viol[:, k] > 0).mean())
        l0, l1 = lam[0, k], lam[-1, k]
        if frac_violating > 0.95 and l1 > l0:
            verdict = "INFEASIBLE"
        elif frac_violating > 0.5:
            verdict = "struggling"
        else:
            verdict = "ok"
        print(f"{name:15s} {thr[k]:7.3f} {rates[-1, k]:9.3f} {viol[-1, k]:+9.3f} "
              f"{frac_violating:7.2f} {l0:8.3f} {l1:8.3f} {verdict:>12s}")

    print()
    print("Reading: a constraint violated on ~100% of updates whose multiplier")
    print("still climbed is one the policy could not satisfy. That is the")
    print("diagnostic a reward-weight failure never gives you.")


def multiplier_story():
    print()
    print("=" * 96)
    print("MULTIPLIERS: start -> end, and whether the bootstrap held the floor")
    print("=" * 96)
    for arm in ("constrained", "constrained_mistuned"):
        blob = load(arm)
        h = blob["history"]
        lam = mat(h, "lam")
        lam0 = col(h, "lam0")
        boot = lam[:, costs.BOOTSTRAP_IDX]
        # lam0 == lam_no_success means the bootstrap floor is what is setting
        # the reward's weight, which is the mechanism doing its job.
        binding = float(np.isclose(lam0, boot, atol=1e-9).mean())
        print(f"\n{LABEL[arm]}")
        print(f"  lambda_0 {lam0[0]:.3f} -> {lam0[-1]:.3f}   "
              f"bootstrap sets it on {binding:.0%} of updates")
        order = np.argsort(-lam[-1])
        for k in order:
            print(f"    {costs.COST_NAMES[k]:15s} "
                  f"{lam[0, k]:.3f} -> {lam[-1, k]:.3f}  "
                  f"({'gained' if lam[-1, k] > lam[0, k] else 'lost':>6s})")


def behaviour_change():
    print()
    print("=" * 96)
    print("BEHAVIOUR: first 10% vs final 10% of updates, per arm")
    print("=" * 96)
    keys = ["pass_lost", "cross_field", "hot_potato", "pass_back",
            "far_from_ball"]
    for arm in ARMS:
        h = load(arm)["history"]
        first = agg_vec(h[:max(1, len(h) // 10)], "cost_rate")
        last = agg_vec(tail(h), "cost_rate")
        print(f"\n{LABEL[arm]}   episodes {sum(r['n_episodes'] for r in h):,} "
              f"updates {len(h)}")
        for name in keys:
            k = costs.IDX[name]
            arrow = "down" if last[k] < first[k] else "up"
            print(f"    {name:15s} {first[k]:.3f} -> {last[k]:.3f}  {arrow}")


def gate_story():
    print()
    print("=" * 96)
    print("SUCCESS GATE: did the curriculum move, and what was binding")
    print("=" * 96)
    for arm in ARMS:
        h = load(arm)["history"]
        t = tail(h)
        print(f"{LABEL[arm]:16s} level {col(h,'curriculum_level')[0]:.0f} -> "
              f"{col(h,'curriculum_level')[-1]:.0f}   "
              f"p_min {col(h,'shot_p_min')[-1]:.3f}   "
              f"gate pcf {agg(t,'gate_pcf'):.3f}/0.30  "
              f"scoring_p {agg(t,'gate_p'):.3f}/{col(h,'shot_p_min')[-1]:.3f}  "
              f"reachable {agg(t,'gate_reachable'):.3f}")


def verdict():
    print()
    print("=" * 96)
    print("THE QUESTION: removed the tuning burden, or relocated it?")
    print("=" * 96)
    end = {a: tail(load(a)["history"]) for a in ARMS}
    s = {a: agg(end[a], "success") for a in ARMS}
    L = {a: agg(end[a], "ep_len") for a in ARMS}
    r = {a: agg_vec(end[a], "cost_rate") for a in ARMS}

    print(f"  success   A {s['reward_only']:.3f}  B {s['constrained']:.3f}  "
          f"C {s['constrained_mistuned']:.3f}")
    print(f"  ep_len    A {L['reward_only']:.1f}  B {L['constrained']:.1f}  "
          f"C {L['constrained_mistuned']:.1f}")
    met = {a: int(sum(r[a][k] <= costs.COST_THRESHOLDS[k]
                      for k in range(costs.N_COSTS))) for a in ARMS}
    print(f"  arm B's thresholds met, out of {costs.N_COSTS}:  "
          f"A {met['reward_only']}  B {met['constrained']}  "
          f"C {met['constrained_mistuned']}")

    gap_ba = s["constrained"] - s["reward_only"]
    gap_cb = s["constrained_mistuned"] - s["constrained"]
    print(f"\n  B - A success gap: {gap_ba:+.3f}")
    print(f"  C - B success gap: {gap_cb:+.3f}")


if __name__ == "__main__":
    headline()
    constraint_table()
    behaviour_change()
    multiplier_story()
    mistuned_feasibility()
    gate_story()
    verdict()
