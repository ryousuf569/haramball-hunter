import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(VALIDATION_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from physics.tti import reaction_time  # noqa: E402

DATA_DIR = os.path.join(VALIDATION_DIR, "data", "Sample_Game_1")
OUT_DIR = os.path.join(VALIDATION_DIR, "results")
EVENTS = os.path.join(DATA_DIR, "Sample_Game_1_RawEventsData.csv")

FIELD = (106.0, 68.0)   # Metrica's pitch, metres
FPS = 25

# Filters. Metrica's event coordinates are normalised to [0,1] and its pass rows
# occasionally carry a same-frame start and end, which would divide by zero.
MIN_LEN_M = 2.0
MAX_SPEED_M_S = 40.0    # above this is a coordinate glitch, not a pass
LEN_BINS = [0, 5, 8, 12, 18, 25, 40, 120]

OLD_BALL_SPEED = 15.0   # what engine.py used before this study


def load_passes():
    e = pd.read_csv(EVENTS)
    p = e[(e["Type"] == "PASS") & e["Start X"].notna() & e["End X"].notna()].copy()
    p["len_m"] = np.hypot((p["End X"] - p["Start X"]) * FIELD[0],
                          (p["End Y"] - p["Start Y"]) * FIELD[1])
    p["dur_s"] = (p["End Frame"] - p["Start Frame"]) / FPS
    p = p[(p["dur_s"] > 0) & (p["len_m"] >= MIN_LEN_M)]
    p["speed"] = p["len_m"] / p["dur_s"]
    return p[p["speed"] < MAX_SPEED_M_S].reset_index(drop=True)


def fit_power_law(length, speed):
    """log(speed) = log(A) + B*log(length). Returns (A, B)."""
    X = np.column_stack([np.ones(len(length)), np.log(length)])
    coef, *_ = np.linalg.lstsq(X, np.log(speed), rcond=None)
    return float(np.exp(coef[0])), float(coef[1])


def predict(length, A, B, cap):
    return np.minimum(A * np.asarray(length, dtype=float) ** B, cap)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_and_validate(p):
    """Fit on the first half, score on the second, then refit on everything.

    Splitting by period rather than at random: passes inside one possession are
    near-duplicates, so a random split would put the same phase of play on both
    sides and report an optimistically low error.
    """
    train = p[p["Period"] == 1]
    test = p[p["Period"] == 2]

    A_tr, B_tr = fit_power_law(train["len_m"], train["speed"])
    cap_tr = float(train.loc[train["len_m"] > 25, "speed"].median())

    held = rmse(test["speed"], predict(test["len_m"], A_tr, B_tr, cap_tr))
    baseline = rmse(test["speed"], np.full(len(test), train["speed"].median()))
    flat15 = rmse(test["speed"], np.full(len(test), OLD_BALL_SPEED))

    A, B = fit_power_law(p["len_m"], p["speed"])
    cap = float(p.loc[p["len_m"] > 25, "speed"].median())
    return {
        "A": A, "B": B, "cap": cap,
        "rmse_holdout": held,
        "rmse_holdout_baseline": baseline,
        "rmse_holdout_flat15": flat15,
        "skill_vs_baseline": 1.0 - (held / baseline) ** 2,
        "rmse_in_sample": rmse(p["speed"], predict(p["len_m"], A, B, cap)),
        "n_train": len(train), "n_test": len(test),
    }


def fig_fit(p, f, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    grid = np.linspace(MIN_LEN_M, 70, 200)
    fitted = predict(grid, f["A"], f["B"], f["cap"])

    axes[0].scatter(p["len_m"], p["speed"], s=7, alpha=0.25, color="tab:blue",
                    label=f"{len(p):,} Metrica passes")
    b = pd.cut(p["len_m"], LEN_BINS)
    g = p.groupby(b, observed=True)
    axes[0].plot([iv.mid for iv in g.groups.keys()], g["speed"].median().values,
                 "o", color="k", ms=7, label="binned median")
    axes[0].plot(grid, fitted, "-", color="tab:red", lw=2,
                 label=f"fit: {f['A']:.2f}*len^{f['B']:.2f}, cap {f['cap']:.1f}")
    axes[0].axhline(OLD_BALL_SPEED, ls="--", c="tab:green", lw=1.5,
                    label=f"old flat {OLD_BALL_SPEED:.0f} m/s")
    axes[0].set_xlabel("pass length (m)")
    axes[0].set_ylabel("ball speed (m/s)")
    axes[0].set_title("Short passes are not struck at long-pass speed")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(p["len_m"], p["dur_s"], s=7, alpha=0.25, color="tab:blue")
    axes[1].plot(grid, grid / fitted, "-", color="tab:red", lw=2, label="fitted")
    axes[1].plot(grid, grid / OLD_BALL_SPEED, "--", color="tab:green", lw=1.5,
                 label=f"old flat {OLD_BALL_SPEED:.0f} m/s")
    axes[1].axhline(reaction_time, ls=":", c="k", lw=1.5,
                    label=f"reaction_time {reaction_time}s")
    axes[1].set_xlabel("pass length (m)")
    axes[1].set_ylabel("flight time (s)")
    axes[1].set_ylim(0, 4)
    axes[1].set_title("Flight time is what interception depends on")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # The consequence: how much of a pass is over before anyone can react.
    edges = np.array(LEN_BINS[:-1]) + np.diff(LEN_BINS) / 2
    real = [(p.loc[pd.cut(p["len_m"], LEN_BINS, labels=False) == i, "dur_s"]
             < reaction_time).mean() for i in range(len(LEN_BINS) - 1)]
    sim_old = [((p.loc[pd.cut(p["len_m"], LEN_BINS, labels=False) == i, "len_m"]
                 / OLD_BALL_SPEED) < reaction_time).mean()
               for i in range(len(LEN_BINS) - 1)]
    sim_new = [((p.loc[pd.cut(p["len_m"], LEN_BINS, labels=False) == i, "len_m"]
                 / predict(p.loc[pd.cut(p["len_m"], LEN_BINS, labels=False) == i, "len_m"],
                           f["A"], f["B"], f["cap"])) < reaction_time).mean()
               for i in range(len(LEN_BINS) - 1)]
    w = 0.28
    axes[2].bar(np.arange(len(edges)) - w, 100 * np.array(real), w, label="real", color="k")
    axes[2].bar(np.arange(len(edges)), 100 * np.array(sim_old), w,
                label=f"sim, flat {OLD_BALL_SPEED:.0f}", color="tab:green")
    axes[2].bar(np.arange(len(edges)) + w, 100 * np.array(sim_new), w,
                label="sim, fitted", color="tab:red")
    axes[2].set_xticks(np.arange(len(edges)))
    axes[2].set_xticklabels([f"{LEN_BINS[i]}-{LEN_BINS[i+1]}" for i in range(len(edges))],
                            fontsize=8)
    axes[2].set_xlabel("pass length (m)")
    axes[2].set_ylabel("% of passes landing inside the reaction time")
    axes[2].set_title("Passes no defender can react to")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("Pass speed against pass length, Metrica sample game 1", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p = load_passes()
    f = fit_and_validate(p)

    b = pd.cut(p["len_m"], LEN_BINS)
    table = p.groupby(b, observed=True).agg(
        n=("speed", "size"),
        speed_median=("speed", "median"),
        dur_median=("dur_s", "median"))
    table["fitted_speed"] = [predict(iv.mid, f["A"], f["B"], f["cap"])
                             for iv in table.index]
    table = table.round(2)

    fig_fit(p, f, os.path.join(OUT_DIR, "pass_speed_fit.png"))
    table.to_csv(os.path.join(OUT_DIR, "pass_speed_by_length.csv"))

    dur_old = p["len_m"] / OLD_BALL_SPEED
    dur_new = p["len_m"] / predict(p["len_m"], f["A"], f["B"], f["cap"])

    # Per-pass RMSE is dominated by scatter the model is not trying to explain
    # (see the caveat in the module docstring), so also score the thing the sim
    # actually consumes: the conditional median speed at a given length.
    trend_err = rmse(table["speed_median"], table["fitted_speed"])
    flat_trend_err = rmse(table["speed_median"], np.full(len(table), OLD_BALL_SPEED))

    params = pd.DataFrame([
        ("n_passes", len(p), "Metrica passes after filters"),
        ("PASS_SPEED_A", round(f["A"], 4), "speed = A * length**B"),
        ("PASS_SPEED_B", round(f["B"], 4), "power-law exponent"),
        ("PASS_SPEED_MAX", round(f["cap"], 2), "cap, m/s -- median speed of passes over 25m"),
        ("rmse_in_sample", round(f["rmse_in_sample"], 3), "m/s"),
        ("rmse_holdout", round(f["rmse_holdout"], 3), "fitted on period 1, scored on period 2"),
        ("rmse_holdout_baseline", round(f["rmse_holdout_baseline"], 3), "predicting the training median speed"),
        ("rmse_holdout_flat15", round(f["rmse_holdout_flat15"], 3), "predicting the old flat 15.0 m/s"),
        ("skill_vs_baseline", round(f["skill_vs_baseline"], 4), "1 - (rmse/baseline)^2, per pass -- low by design, see docstring"),
        ("trend_rmse_fitted", round(trend_err, 3), "fit vs binned median speed, m/s -- the quantity the sim consumes"),
        ("trend_rmse_flat15", round(flat_trend_err, 3), "old flat speed vs the same binned medians"),
        ("frac_under_reaction_real", round(float((p["dur_s"] < reaction_time).mean()), 4), f"real passes airborne < reaction_time ({reaction_time}s)"),
        ("frac_under_reaction_flat15", round(float((dur_old < reaction_time).mean()), 4), "same passes at the old flat speed"),
        ("frac_under_reaction_fitted", round(float((dur_new < reaction_time).mean()), 4), "same passes at the fitted speed"),
    ], columns=["parameter", "value", "meaning"])
    params.to_csv(os.path.join(OUT_DIR, "pass_speed_params.csv"), index=False)

    print(f"{len(p):,} passes, {f['n_train']} in period 1 / {f['n_test']} in period 2\n")
    print(f"  speed = {f['A']:.4f} * length ** {f['B']:.4f},  capped at {f['cap']:.2f} m/s\n")
    print(table.to_string())
    print()
    print(params.to_string(index=False))
    print(f"\nwrote results/pass_speed_fit.png, results/pass_speed_by_length.csv,")
    print(f"      results/pass_speed_params.csv")


if __name__ == "__main__":
    main()
