import os
import numpy as np

PITCH_X = 105.0
PITCH_Y = 68.0

# 2-5-3, numbered back-to-front: 2 deepest, then 5, then the 3 most advanced
LINE_SIZES = (2, 5, 3)
LINE_NAMES = ("back", "mid", "fwd")
N_SLOTS = sum(LINE_SIZES)

# Keep sampled players on the pitch, a little inside the touchlines
X_BOUNDS = (2.0, PITCH_X - 2.0)
Y_BOUNDS = (2.0, PITCH_Y - 2.0)

CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
SLOT_CSV = os.path.join(CAL_DIR, "attacker_slot_params.csv")
FRAME_CSV = os.path.join(CAL_DIR, "attacker_frame_params.csv")

_cached_params = None


# Read the fitted CSVs once and cache them; returns (slot_table, frame_dict)
def load_params():
    global _cached_params
    if _cached_params is None:
        slots = np.genfromtxt(SLOT_CSV, delimiter=",", names=True, dtype=None,
                              encoding="utf-8")
        frame_rows = np.genfromtxt(FRAME_CSV, delimiter=",", names=True,
                                   dtype=None, encoding="utf-8")
        frame = {str(r["param"]): r for r in frame_rows}
        _cached_params = (slots, frame)
    return _cached_params


# One clipped Gaussian draw from a fitted frame-parameter row
def _draw(rng, row):
    return float(np.clip(rng.normal(row["mean"], row["sd"]), row["lo"], row["hi"]))


# Sample one 2-5-3 attacker shape in sim coordinates, (n_att, 2), deepest slot first
#
# x_shift pushes the whole sampled shape that many metres goalward. It is a
# curriculum knob rather than part of the fit, and training anneals it to 0 so
# a finished policy is measured on the fitted distribution.
def sample_attacker_formation(rng, n_att=N_SLOTS, anchor_x=None, x_shift=0.0):
    slots, frame = load_params()

    cy = _draw(rng, frame["centroid_y"])
    if anchor_x is None:
        cx = _draw(rng, frame["centroid_x"])  # absolute depth: where real attackers line up
    else:
        cx = float(anchor_x) - _draw(rng, frame["backline_gap"])  # or the fitted gap behind a given backline x
    cx += float(x_shift)

    # One shared factor, so a compressed shape compresses in both directions at once
    rho = float(frame["spread_corr"]["mean"])
    z1 = rng.normal()
    z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rng.normal()
    sx_row, sy_row = frame["spread_x"], frame["spread_y"]
    sx = float(np.clip(sx_row["mean"] + sx_row["sd"] * z1, sx_row["lo"], sx_row["hi"]))
    sy = float(np.clip(sy_row["mean"] + sy_row["sd"] * z2, sy_row["lo"], sy_row["hi"]))

    u = rng.normal(slots["mu_u"], slots["sd_u"])
    v = rng.normal(slots["mu_v"], slots["sd_v"])

    positions = np.empty((N_SLOTS, 2))
    positions[:, 0] = np.clip(cx + sx * u, *X_BOUNDS)
    positions[:, 1] = np.clip(cy + sy * v, *Y_BOUNDS)

    # The fit sorted each line by y, so restore it: an unsorted draw can cross two slots over
    start = 0
    for size in LINE_SIZES:
        line = positions[start:start + size]
        line[:] = line[np.argsort(line[:, 1])]
        start += size

    return positions[:n_att]


# Everything below only runs when fitting; the env imports the sampler above
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "defenders", "calibration")
    os.makedirs(CAL_DIR, exist_ok=True)

    frames = pd.read_csv(os.path.join(SRC_DIR, "low_block_frames.csv")).set_index("event_id")
    players = pd.read_csv(os.path.join(SRC_DIR, "low_block_defenders.csv"))

    att = players[players["line"] == "att"].copy()

    # Rotate the calibration frame into sim coordinates: attackers now attack +x
    att["x"] = PITCH_X - att["x"]
    att["y"] = PITCH_Y - att["y"]
    frames["back_line_x"] = PITCH_X - frames["back_line_x"]

    # Keep frames with a full outfield ten in view, then take the 10 most advanced
    full = frames.index[frames["n_att"] >= N_SLOTS]
    att = att[att["event_id"].isin(full)]
    att = att.sort_values(["event_id", "x"], ascending=[True, False])
    att = att.groupby("event_id").head(N_SLOTS)

    # Slot = rank on x, deepest first, then sorted by y within each line
    att = att.sort_values(["event_id", "x"])
    att["rank"] = att.groupby("event_id").cumcount()
    att["line_id"] = np.searchsorted(np.cumsum(LINE_SIZES), att["rank"], side="right")
    att = att.sort_values(["event_id", "line_id", "y"])
    att["slot"] = att.groupby("event_id").cumcount()

    g = att.groupby("event_id")
    shape = pd.DataFrame({
        "cx": g["x"].mean(), "cy": g["y"].mean(),
        "sx": g["x"].std(), "sy": g["y"].std(),
        "front_x": g["x"].max(), "width": g["y"].max() - g["y"].min(),
    }).join(frames[["back_line_x", "ball_x", "ball_y"]])
    shape["gap"] = shape["back_line_x"] - shape["cx"]

    att = att.join(shape[["cx", "cy", "sx", "sy"]], on="event_id")
    att["u"] = (att["x"] - att["cx"]) / att["sx"]
    att["v"] = (att["y"] - att["cy"]) / att["sy"]

    print(f"frames used: {len(shape)}   attacker rows: {len(att)}")
    print()
    print("--- frame shape (sim coordinates, attackers -> x=105) ---")
    print(shape[["cx", "cy", "sx", "sy", "gap", "front_x", "width"]]
          .describe().T[["mean", "std", "min", "50%", "max"]])
    print()
    print(f"corr(sx, sy) = {shape['sx'].corr(shape['sy']):.3f}   "
          f"corr(cx, back_line_x) = {shape['cx'].corr(shape['back_line_x']):.3f}")
    print(f"gap to backline: mean={shape['gap'].mean():.2f}m  sd={shape['gap'].std():.2f}m")

    # Fit: Gaussian marginals clipped to the 5/95 range
    rows = []
    for name, col in [("centroid_x", "cx"), ("centroid_y", "cy"),
                      ("spread_x", "sx"), ("spread_y", "sy"),
                      ("backline_gap", "gap")]:
        s = shape[col]
        rows.append({"param": name, "mean": s.mean(), "sd": s.std(),
                     "lo": s.quantile(0.05), "hi": s.quantile(0.95)})
    rho = shape["sx"].corr(shape["sy"])
    rows.append({"param": "spread_corr", "mean": rho, "sd": 0.0, "lo": -1.0, "hi": 1.0})
    frame_params = pd.DataFrame(rows)

    slot_params = att.groupby("slot").agg(
        mu_u=("u", "mean"), sd_u=("u", "std"),
        mu_v=("v", "mean"), sd_v=("v", "std"), n=("u", "size")).reset_index()
    slot_params["line"] = [LINE_NAMES[i] for i, size in enumerate(LINE_SIZES)
                           for _ in range(size)]

    print()
    print("--- frame parameters ---")
    print(frame_params.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print()
    print("--- per-slot standardised offsets (u = depth, v = width) ---")
    print(slot_params.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    frame_params.to_csv(FRAME_CSV, index=False)
    slot_params[["slot", "line", "mu_u", "sd_u", "mu_v", "sd_v", "n"]].to_csv(
        SLOT_CSV, index=False)

    # Validate: sample from the fit we just wrote and compare against the data
    _cached_params = None  # force the sampler to read the fresh CSVs
    rng = np.random.default_rng(0)
    sampled = np.array([sample_attacker_formation(rng) for _ in range(len(shape))])

    sim = pd.DataFrame({
        "cx": sampled[:, :, 0].mean(axis=1), "cy": sampled[:, :, 1].mean(axis=1),
        "front_x": sampled[:, :, 0].max(axis=1),
        "width": sampled[:, :, 1].max(axis=1) - sampled[:, :, 1].min(axis=1),
    })
    print()
    print("--- real vs sampled ---")
    for col in ["cx", "cy", "front_x", "width"]:
        print(f"  {col:8s} real {shape[col].mean():6.2f} +/- {shape[col].std():5.2f}"
              f"    sampled {sim[col].mean():6.2f} +/- {sim[col].std():5.2f}")

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(shape["cx"], bins=40, color="firebrick")
    axes[0, 0].axvline(64.0, color="black", ls="--", label="old fixed ref_x=64")
    axes[0, 0].set_xlabel("centroid x (sim frame)")
    axes[0, 0].set_title(f"Attacking centroid depth (n={len(shape)})")
    axes[0, 0].legend()

    axes[0, 1].hist(shape["gap"], bins=40, color="steelblue")
    axes[0, 1].set_xlabel("back_line_x - centroid_x  (m behind the block)")
    axes[0, 1].set_title(f"Gap to the defensive backline "
                         f"(mean {shape['gap'].mean():.1f}m)")

    axes[1, 0].scatter(shape["sx"], shape["sy"], s=5, alpha=0.25, color="seagreen")
    axes[1, 0].set_xlabel("depth spread sx (m)")
    axes[1, 0].set_ylabel("width spread sy (m)")
    axes[1, 0].set_title(f"Shape stretches together (r={rho:.2f})")

    axes[1, 1].hist(shape["cy"], bins=40, color="darkorange")
    axes[1, 1].axvline(PITCH_Y / 2, color="black", ls="--", label="pitch centre")
    axes[1, 1].set_xlabel("centroid y (m)")
    axes[1, 1].set_title("Lateral position of the shape")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CAL_DIR, "attacker_shape_fit.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"back": "tab:blue", "mid": "tab:green", "fwd": "tab:red"}
    for slot, sub in att.groupby("slot"):
        line = slot_params.loc[slot, "line"]
        ax.scatter(sub["u"], sub["v"], s=3, alpha=0.12, color=colors[line])
        mu_u, mu_v = slot_params.loc[slot, "mu_u"], slot_params.loc[slot, "mu_v"]
        ax.errorbar(mu_u, mu_v, xerr=slot_params.loc[slot, "sd_u"],
                    yerr=slot_params.loc[slot, "sd_v"], fmt="o", color="black",
                    ms=5, lw=1.5, capsize=3)
        ax.annotate(f"{slot}", (mu_u, mu_v), xytext=(4, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("u = (x - centroid_x) / sx      (deeper <-- --> goalward)")
    ax.set_ylabel("v = (y - centroid_y) / sy")
    ax.set_title("Per-slot standardised offsets: real frames vs fitted mean +/- 1 sd")
    plt.tight_layout()
    plt.savefig(os.path.join(CAL_DIR, "attacker_slot_scatter.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rng = np.random.default_rng(7)
    for k, ax in enumerate(axes.ravel()):
        pos = sample_attacker_formation(rng)
        start = 0
        for size, name in zip(LINE_SIZES, LINE_NAMES):
            block = pos[start:start + size]
            ax.scatter(block[:, 0], block[:, 1], s=90, color=colors[name], label=name)
            start += size
        ax.set_xlim(0, PITCH_X)
        ax.set_ylim(0, PITCH_Y)
        ax.axvline(PITCH_X / 2, color="grey", lw=0.7)
        ax.set_title(f"sample {k}   centroid x={pos[:, 0].mean():.1f}")
        if k == 0:
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Sampled 2-5-3 formations (attacking toward x=105)")
    plt.tight_layout()
    plt.savefig(os.path.join(CAL_DIR, "attacker_samples.png"), dpi=120)
    plt.close(fig)

    print()
    print(f"wrote {SLOT_CSV}")
    print(f"wrote {FRAME_CSV}")
    print(f"wrote 3 figures to {CAL_DIR}")
