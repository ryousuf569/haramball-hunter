"""Figures for the three-arm constrained-RL experiment in train.py.

Everything redraws from the saved history JSON, so a figure can be restyled
for a write-up without re-running: python plots.py models/figs
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment import costs

# One colour per arm, held fixed across every figure so a line means the same
# thing everywhere.
ARM_COLORS = {
    "reward_only": "#2a78d6",
    "constrained": "#eb6834",
    "constrained_mistuned": "#1baf7a",
}
ARM_LABELS = {
    "reward_only": "A: reward only",
    "constrained": "B: constrained",
    "constrained_mistuned": "C: constrained, mistuned",
}
COST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7"]


def _col(history, key):
    return np.array([h[key] for h in history], dtype=float)


def _mat(history, key):
    return np.array([h[key] for h in history], dtype=float)


def _smooth(y, w=9):
    """Centred moving average. Per-update rates off ~10 episodes are noisy
    enough that the raw line hides the trend; the raw line is still drawn
    underneath at low alpha so nothing is hidden by the smoothing."""
    y = np.asarray(y, dtype=float)
    if len(y) < w or w < 2:
        return y
    k = np.ones(w) / w
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(padded, k, mode="valid")[:len(y)]


def _d(v):
    """Format a threshold without rounding it away. A flat .2f turned
    offside's 0.005 into 0.01, which is a different constraint."""
    return f"{v:.3f}".rstrip("0").rstrip(".") if v < 0.1 else f"{v:.2f}"


def _line(ax, x, y, color, label=None, smooth=True):
    if smooth:
        ax.plot(x, y, color=color, alpha=0.18, lw=1.0)
        y = _smooth(y)
    ax.plot(x, y, color=color, lw=1.8, label=label)


def _finish(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --- per-arm figures -------------------------------------------------------

def plot_task(history, arm, outdir):
    """Did it learn the task at all."""
    x = _col(history, "step")
    c = ARM_COLORS.get(arm, "#2a78d6")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    for key, colr, lbl in [("success", "#008300", "success"),
                           ("failure", "#e34948", "failure"),
                           ("timeout", "#898781", "timeout")]:
        _line(ax, x, _col(history, key), colr, lbl)
    # The training success rate is against whatever gate the curriculum has
    # reached, so it is not comparable across arms. This one always is.
    if "eval_success" in history[0]:
        ev = _col(history, "eval_success")
        ok = np.isfinite(ev)
        if ok.any():
            ax.plot(x[ok], ev[ok], color="#2a78d6", lw=2.0, ls="--",
                    marker="o", ms=3, label="success, calibrated gate")
    ax.set_ylabel("rate (100-episode window)")
    ax.set_title("Episode outcomes")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    _line(ax, x, _col(history, "ep_len"), c)
    ax.set_ylabel("ticks")
    ax.set_title("Episode length -- how long possession survives")

    ax = axes[1, 0]
    _line(ax, x, _col(history, "ep_return"), c)
    ax.axhline(0.0, color="#c3c2b7", lw=1.0, ls="--")
    ax.set_ylabel("undiscounted return")
    ax.set_title("Episode return")

    # Curriculum: the gate the run is actually being scored against. Two
    # quantities on one axes only because both are a distance in metres.
    ax = axes[1, 1]
    from environment.termination import radius_for_p
    radius = np.array([radius_for_p(p) for p in _col(history, "shot_p_min")])
    ax.plot(x, radius, color="#4a3aa7", lw=1.8, label="shot gate radius (m)")
    ax.plot(x, _col(history, "x_shift"), color="#eda100", lw=1.8,
            label="start x_shift (m)")
    ax.set_ylabel("metres")
    ax.set_title(f"Curriculum (level "
                 f"{int(_col(history, 'curriculum_level')[-1])}/8 at end)")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"{ARM_LABELS.get(arm, arm)} -- task performance", y=1.0)
    return _finish(fig, os.path.join(outdir, f"{arm}_task.png"))


def plot_constraints(history, arm, thresholds, outdir):
    """Each constraint's rate against its threshold. Small multiples, because
    seven series on one axes is unreadable and they have different units."""
    x = _col(history, "step")
    rates = _mat(history, "cost_rate")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)

    for k, ax in enumerate(axes.ravel()):
        if k >= costs.N_COSTS:
            ax.axis("off")
            continue
        _line(ax, x, rates[:, k], COST_COLORS[k % len(COST_COLORS)])
        ax.axhline(thresholds[k], color="#0b0b0b", lw=1.2, ls="--")
        ax.annotate(f"d = {_d(thresholds[k])}", xy=(0.98, thresholds[k]),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color="#0b0b0b")
        end = _smooth(rates[:, k])[-1]
        ok = end <= thresholds[k]
        ax.set_title(f"{costs.COST_NAMES[k]}  ({costs.COST_UNITS[k]})\n"
                     f"end {end:.3f} -- {'satisfied' if ok else 'VIOLATED'}",
                     fontsize=9, color="#006300" if ok else "#d03b3b")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlabel("env steps")

    fig.suptitle(f"{ARM_LABELS.get(arm, arm)} -- constraint rates vs thresholds",
                 y=1.0)
    return _finish(fig, os.path.join(outdir, f"{arm}_constraints.png"))


def plot_multipliers(history, arm, thresholds, outdir):
    """Where the Lagrangian put the weight, and how it got there."""
    x = _col(history, "step")
    lam = _mat(history, "lam")
    lam0 = _col(history, "lam0")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for k in range(lam.shape[1]):
        ax.plot(x, lam[:, k], color=COST_COLORS[k % len(COST_COLORS)], lw=1.8,
                label=costs.COST_NAMES[k])
    # Dashed and on top, since a solid line would hide no_success underneath it.
    ax.plot(x, lam0, color="#0b0b0b", lw=2.0, ls="--", zorder=5,
            label="reward (lambda_0)")
    ax.set_title("Multipliers -- one simplex shared with the reward\n"
                 "(lambda_0 tracks no_success while the bootstrap binds)",
                 fontsize=10)
    ax.set_ylabel("lambda")
    ax.legend(fontsize=7, ncol=2)

    # Violation is what drives the multiplier, so plotting it beside lambda is
    # the check that the mechanism is doing what it claims.
    ax = axes[1]
    rates = _mat(history, "cost_rate")
    thr = np.asarray(thresholds, dtype=float)
    for k in range(lam.shape[1]):
        _line(ax, x, rates[:, k] - thr[k], COST_COLORS[k % len(COST_COLORS)])
    ax.axhline(0.0, color="#0b0b0b", lw=1.2, ls="--")
    ax.set_title("Violation (rate - threshold); above 0 the multiplier climbs\n"
                 "a constraint pinned above 0 for the whole run is infeasible",
                 fontsize=10)
    ax.set_ylabel("rate - d")

    for ax in axes:
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"{ARM_LABELS.get(arm, arm)} -- Lagrange multipliers", y=1.0)
    return _finish(fig, os.path.join(outdir, f"{arm}_multipliers.png"))


def plot_learning(history, arm, outdir):
    """Learner health, the panel that says whether a bad result is a bug."""
    x = _col(history, "step")
    c = ARM_COLORS.get(arm, "#2a78d6")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    _line(ax, x, _col(history, "explained_var"), c)
    ax.axhline(0.0, color="#c3c2b7", lw=1.0, ls="--")
    ax.set_ylabel("explained variance")
    ax.set_title("Critic fit (0 = the value head explains nothing)")
    ax.set_ylim(-1, 1)

    ax = axes[0, 1]
    for key, colr, lbl, cap in [("ent_dir", "#2a78d6", "direction", np.log(9)),
                                ("ent_speed", "#eb6834", "speed", np.log(3)),
                                ("ent_ball", "#1baf7a", "ball", np.log(10))]:
        _line(ax, x, _col(history, key) / cap, colr, lbl)
    ax.set_ylabel("entropy / uniform")
    ax.set_title("Policy entropy, as a fraction of uniform")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    _line(ax, x, _col(history, "v_loss"), "#2a78d6", "reward critic")
    _line(ax, x, _col(history, "c_loss"), "#eb6834", "cost critics (sum)")
    ax.set_ylabel("value loss")
    ax.set_title("Value losses")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    _line(ax, x, _col(history, "approx_kl"), c)
    ax.set_ylabel("approx KL")
    ax.set_title("Policy step size")

    for ax in axes.ravel():
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"{ARM_LABELS.get(arm, arm)} -- learner diagnostics", y=1.0)
    return _finish(fig, os.path.join(outdir, f"{arm}_learning.png"))


# --- cross-arm figures -----------------------------------------------------

def plot_comparison(runs, outdir):
    """The headline figure: did the constraints buy the behaviour, and did
    mistuning them cost as much as mistuning a reward weight. runs is
    {arm: history}."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # The first panel is the calibrated-gate eval where it exists, because the
    # training success rate describes a different task in every arm.
    have_eval = all("eval_success" in h[0] and
                    np.isfinite(_col(h, "eval_success")).any()
                    for h in runs.values())
    first = ("eval_success", "success rate, calibrated gate") if have_eval \
        else ("success", "success rate (per-arm gate, NOT comparable)")

    for ax, (key, ylabel) in zip(axes[0], [first,
                                           ("ep_len", "episode length (ticks)"),
                                           ("ep_return", "undiscounted return")]):
        for arm, hist in runs.items():
            y = _col(hist, key)
            ok = np.isfinite(y)
            if key == "eval_success":
                ax.plot(_col(hist, "step")[ok], y[ok],
                        color=ARM_COLORS.get(arm, "#898781"), lw=1.8,
                        marker="o", ms=3, label=ARM_LABELS.get(arm, arm))
            else:
                _line(ax, _col(hist, "step"), y,
                      ARM_COLORS.get(arm, "#898781"), ARM_LABELS.get(arm, arm))
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    # The three behaviours the write-up is about. Every arm measures them,
    # including A, so A's line is what no constraint looks like.
    for ax, name in zip(axes[1], ("pass_lost", "cross_field", "hot_potato")):
        k = costs.IDX[name]
        for arm, hist in runs.items():
            _line(ax, _col(hist, "step"), _mat(hist, "cost_rate")[:, k],
                  ARM_COLORS.get(arm, "#898781"), ARM_LABELS.get(arm, arm))
        ax.axhline(costs.COST_THRESHOLDS[k], color="#0b0b0b", lw=1.2, ls="--")
        ax.annotate(f"d = {_d(costs.COST_THRESHOLDS[k])}",
                    xy=(0.98, costs.COST_THRESHOLDS[k]),
                    xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=8)
        ax.set_ylabel("rate over passes")
        ax.set_title(f"{name} (dashed = arm B's d = {_d(costs.COST_THRESHOLDS[k])})")
        ax.set_ylim(bottom=0)

    for ax in axes.ravel():
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.25, lw=0.6)

    # One legend for all six panels. Laid out first, so _finish must skip
    # tight_layout or it would undo the strip reserved here.
    fig.suptitle("Three arms: reward tuning vs constraint specification",
                 y=0.985, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.945))
    return _finish(fig, os.path.join(outdir, "comparison.png"), tight=False)


def plot_final_table(runs, outdir):
    """End-of-run summary as a figure, so the write-up can paste one image."""
    names = ["success", "ep_len", "ep_return"] + list(costs.COST_NAMES)
    rows = []
    for arm, hist in runs.items():
        tail = hist[-max(1, len(hist) // 20):]        # last 5% of updates
        rates = np.array([h["cost_rate"] for h in tail]).mean(axis=0)
        rows.append([ARM_LABELS.get(arm, arm),
                     f"{np.mean([h['success'] for h in tail]):.3f}",
                     f"{np.mean([h['ep_len'] for h in tail]):.1f}",
                     f"{np.mean([h['ep_return'] for h in tail]):+.2f}"]
                    + [f"{r:.3f}" for r in rates])

    fig, ax = plt.subplots(figsize=(2.0 + 1.15 * len(names), 1.2 + 0.5 * len(rows)))
    ax.axis("off")
    header = ["arm"] + names
    # The arm column needs double, or matplotlib clips the longest label.
    widths = [2.0] + [1.0] * len(names)
    widths = [w / sum(widths) for w in widths]
    tbl = ax.table(cellText=rows, colLabels=header, loc="center",
                   cellLoc="center", colWidths=widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    # Colour the constraint cells by whether they ended satisfied.
    for i, (arm, _hist) in enumerate(runs.items()):
        for j, name in enumerate(costs.COST_NAMES):
            cell = tbl[i + 1, 4 + j]
            val = float(rows[i][4 + j])
            cell.set_facecolor("#e6f4e6" if val <= costs.COST_THRESHOLDS[j]
                               else "#fbe9e9")
    for j, name in enumerate(costs.COST_NAMES):
        tbl[0, 4 + j].set_text_props(
            text=f"{name}\n<={_d(costs.COST_THRESHOLDS[j])}")

    ax.set_title("Final 5% of updates. Green = arm B's threshold satisfied.",
                 fontsize=10)
    return _finish(fig, os.path.join(outdir, "summary_table.png"))


def plot_all(arm, history, thresholds, outdir):
    os.makedirs(outdir, exist_ok=True)
    return [plot_task(history, arm, outdir),
            plot_constraints(history, arm, thresholds, outdir),
            plot_multipliers(history, arm, thresholds, outdir),
            plot_learning(history, arm, outdir)]


def load(outdir, arm):
    with open(os.path.join(outdir, f"{arm}_history.json")) as f:
        return json.load(f)


def redraw(outdir):
    """Rebuild every figure from the JSONs already on disk."""
    runs = {}
    for arm in ARM_LABELS:
        path = os.path.join(outdir, f"{arm}_history.json")
        if not os.path.exists(path):
            continue
        blob = load(outdir, arm)
        runs[arm] = blob["history"]
        plot_all(arm, blob["history"], np.asarray(blob["thresholds"]), outdir)
    if len(runs) >= 2:
        plot_comparison(runs, outdir)
        plot_final_table(runs, outdir)
    if not runs:
        print(f"no *_history.json in {outdir}")
    return runs


def newest_figs():
    """The figs folder of the most recent run under models/."""
    import glob
    dirs = sorted(glob.glob(os.path.join("models", "*", "figs")),
                  key=os.path.getmtime)
    return dirs[-1] if dirs else os.path.join("models", "figs")


if __name__ == "__main__":
    import sys
    redraw(sys.argv[1] if len(sys.argv) > 1 else newest_figs())
