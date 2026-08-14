"""Follow-ups the summary raised: multiplier saturation, critic fit, curriculum.

Run: python models/081126-500k/analysis/probe.py
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


def load(a):
    with open(os.path.join(FIGS, f"{a}_history.json")) as f:
        return json.load(f)


def mat(h, k):
    return np.array([r[k] for r in h], dtype=float)


def col(h, k):
    return np.array([r[k] for r in h], dtype=float)


def multiplier_saturation():
    """Does one constraint end up owning the simplex, and when?"""
    print("=" * 92)
    print("MULTIPLIER SATURATION: max lambda over training")
    print("=" * 92)
    for arm in ("constrained", "constrained_mistuned"):
        h = load(arm)["history"]
        lam = mat(h, "lam")
        step = col(h, "step")
        print(f"\n{arm}")
        print(f"{'step':>8s} {'max lam':>8s} {'owner':>15s} "
              f"{'2nd':>8s} {'sum of rest':>12s}")
        for i in np.linspace(0, len(h) - 1, 9).astype(int):
            row = lam[i]
            order = np.argsort(-row)
            rest = row.sum() - row[order[0]]
            print(f"{step[i]:8.0f} {row[order[0]]:8.3f} "
                  f"{costs.COST_NAMES[order[0]]:>15s} {row[order[1]]:8.3f} "
                  f"{rest:12.3f}")


def hot_potato_story():
    """Arm B's hot_potato took 69% of the simplex. Was that the loop working?"""
    print()
    print("=" * 92)
    print("ARM B, hot_potato: rate vs threshold vs multiplier over the run")
    print("=" * 92)
    h = load("constrained")["history"]
    k = costs.IDX["hot_potato"]
    rate = mat(h, "cost_rate")[:, k]
    lam = mat(h, "lam")[:, k]
    step = col(h, "step")
    d = costs.COST_THRESHOLDS[k]
    print(f"{'step':>8s} {'rate':>7s} {'d':>6s} {'violated':>9s} {'lambda':>8s}")
    for i in np.linspace(0, len(h) - 1, 12).astype(int):
        print(f"{step[i]:8.0f} {rate[i]:7.3f} {d:6.2f} "
              f"{'yes' if rate[i] > d else 'no':>9s} {lam[i]:8.3f}")
    w = 40
    print(f"\n  final {w}-update mean rate {rate[-w:].mean():.3f} against d = {d:.2f}")
    print(f"  fraction of the run in violation: {(rate > d).mean():.2f}")


def critic_fit():
    """explained_variance near zero means the advantages are mostly noise."""
    print()
    print("=" * 92)
    print("CRITIC FIT: explained variance over the run (1.0 = perfect)")
    print("=" * 92)
    print(f"{'arm':22s} " + " ".join(f"{p:>8s}" for p in
                                     ["0%", "20%", "40%", "60%", "80%", "100%"]))
    for arm in ARMS:
        h = load(arm)["history"]
        ev = col(h, "explained_var")
        idx = np.linspace(0, len(h) - 1, 6).astype(int)
        sm = [ev[max(0, i - 20):i + 20].mean() for i in idx]
        print(f"{arm:22s} " + " ".join(f"{v:8.3f}" for v in sm))
    print("\n  Near zero throughout means GAE is differencing a value function")
    print("  that predicts nothing, so the advantage is close to a raw return.")


def curriculum_confound():
    """Success rates are not comparable if the arms sat at different gates."""
    print()
    print("=" * 92)
    print("CURRICULUM CONFOUND: each arm scored itself against its own gate")
    print("=" * 92)
    print(f"{'arm':22s} {'end level':>10s} {'end p_min':>10s} "
          f"{'end radius':>11s} {'end success':>12s}")
    for arm in ARMS:
        h = load(arm)["history"]
        from environment.termination import radius_for_p
        p = col(h, "shot_p_min")[-1]
        t = h[-max(1, len(h) // 10):]
        print(f"{arm:22s} {col(h,'curriculum_level')[-1]:10.0f} {p:10.3f} "
              f"{radius_for_p(p):10.1f}m {np.mean([r['success'] for r in t]):12.3f}")
    print("\n  A higher level is a HARDER gate. Any arm that advanced further was")
    print("  scored on a smaller target, so these success rates cannot be")
    print("  compared directly. scripts/diag_policy.py is the fix: it holds every")
    print("  checkpoint to the calibrated gate.")


def when_did_advances_happen():
    print()
    print("=" * 92)
    print("CURRICULUM ADVANCES (step at which each level was reached)")
    print("=" * 92)
    for arm in ARMS:
        h = load(arm)["history"]
        lvl = col(h, "curriculum_level")
        step = col(h, "step")
        jumps = [(int(step[i]), int(lvl[i]))
                 for i in range(1, len(lvl)) if lvl[i] != lvl[i - 1]]
        print(f"{arm:22s} " + ("  ".join(f"{s:,}->L{l}" for s, l in jumps)
                               if jumps else "never advanced"))


def episode_length_vs_success():
    """Longer possessions are the behaviour change; did they buy anything?"""
    print()
    print("=" * 92)
    print("POSSESSION LENGTH over the run")
    print("=" * 92)
    print(f"{'arm':22s} " + " ".join(f"{p:>8s}" for p in
                                     ["0%", "20%", "40%", "60%", "80%", "100%"]))
    for arm in ARMS:
        h = load(arm)["history"]
        L = col(h, "ep_len")
        idx = np.linspace(0, len(h) - 1, 6).astype(int)
        print(f"{arm:22s} " +
              " ".join(f"{L[max(0,i-20):i+20].mean():8.1f}" for i in idx))


if __name__ == "__main__":
    multiplier_saturation()
    hot_potato_story()
    critic_fit()
    curriculum_confound()
    when_did_advances_happen()
    episode_length_vs_success()
