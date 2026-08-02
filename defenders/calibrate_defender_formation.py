import os
import numpy as np

PITCH_X = 105.0
PITCH_Y = 68.0

# 5-4-1, numbered back-to-front: 5 deepest, then 4, then the lone forward
LINE_SIZES = (5, 4, 1)
LINE_NAMES = ("back", "mid", "fwd")
N_SLOTS = sum(LINE_SIZES)

# The freeze-frame data only labelled the 9 deepest defenders, so slot 9 is not fitted
N_FITTED = LINE_SIZES[0] + LINE_SIZES[1]

# Resting x for the lone forward, matching attacker_positioning() in defenders.py, which never takes the haram offset
FORWARD_X = 52.0

# defenders.py parks the block this much deeper than the calibration says (HARAM_DEPTH_OFFSET)
HARAM_DEPTH_OFFSET = 15.0

# Keep sampled players on the pitch, a little inside the touchlines
X_BOUNDS = (2.0, PITCH_X - 2.0)
Y_BOUNDS = (2.0, PITCH_Y - 2.0)

CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
SLOT_CSV = os.path.join(CAL_DIR, "defender_slot_params.csv")
FRAME_CSV = os.path.join(CAL_DIR, "defender_frame_params.csv")

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


# Sample one 5-4-1 defender shape in sim coordinates, (n_def, 2), deepest slot first
def sample_defender_formation(rng, n_def=N_SLOTS, anchor_x=None,
                              depth_offset=HARAM_DEPTH_OFFSET):
    slots, frame = load_params()

    cy = _draw(rng, frame["centroid_y"])
    if anchor_x is None:
        cx = _draw(rng, frame["centroid_x"])  # absolute depth: where real low blocks sit
    else:
        cx = float(anchor_x) + _draw(rng, frame["attline_gap"])  # or the fitted gap behind a given attacking line
    cx += depth_offset  # start where defenders.py will hold the block, not 15m short of it

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

    # Slot 9 has no data behind it: it gets defenders.py's resting target plus jitter, no haram offset
    positions[N_FITTED, 0] = np.clip(FORWARD_X + sx * u[N_FITTED], *X_BOUNDS)

    # The fit sorted each line by y, so restore it: an unsorted draw can cross two slots over
    start = 0
    for size in LINE_SIZES:
        line = positions[start:start + size]
        line[:] = line[np.argsort(line[:, 1])]
        start += size

    return positions[:n_def]


# Everything below only runs when fitting; the env imports the sampler above
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    SRC_DIR = CAL_DIR
    os.makedirs(CAL_DIR, exist_ok=True)

    frames = pd.read_csv(os.path.join(SRC_DIR, "low_block_frames.csv")).set_index("event_id")
    players = pd.read_csv(os.path.join(SRC_DIR, "low_block_defenders.csv"))

    dfn = players[players["line"].isin(["back", "mid"])].copy()

    # Rotate the calibration frame into sim coordinates: defenders now defend x=105
    dfn["x"] = PITCH_X - dfn["x"]
    dfn["y"] = PITCH_Y - dfn["y"]
    frames["att_line_x"] = PITCH_X - frames["att_line_x"]

    # Keep frames that recorded a full back five and midfield four
    counts = dfn.groupby("event_id")["x"].size()
    dfn = dfn[dfn["event_id"].isin(counts.index[counts == N_FITTED])]

    # Slot = rank on x, deepest first, then sorted by y within each line
    dfn = dfn.sort_values(["event_id", "x"], ascending=[True, False])
    dfn["rank"] = dfn.groupby("event_id").cumcount()
    dfn["line_id"] = np.searchsorted(np.cumsum(LINE_SIZES), dfn["rank"], side="right")
    dfn = dfn.sort_values(["event_id", "line_id", "y"])
    dfn["slot"] = dfn.groupby("event_id").cumcount()

    g = dfn.groupby("event_id")
    shape = pd.DataFrame({
        "cx": g["x"].mean(), "cy": g["y"].mean(),
        "sx": g["x"].std(), "sy": g["y"].std(),
        "deepest_x": g["x"].max(), "width": g["y"].max() - g["y"].min(),
    }).join(frames[["att_line_x", "ball_x", "ball_y"]])
    shape["gap"] = shape["cx"] - shape["att_line_x"]

    dfn = dfn.join(shape[["cx", "cy", "sx", "sy"]], on="event_id")
    dfn["u"] = (dfn["x"] - dfn["cx"]) / dfn["sx"]
    dfn["v"] = (dfn["y"] - dfn["cy"]) / dfn["sy"]

    print(f"frames used: {len(shape)}   defender rows: {len(dfn)}")
    print()
    print("--- frame shape (sim coordinates, defenders -> x=105) ---")
    print(shape[["cx", "cy", "sx", "sy", "gap", "deepest_x", "width"]]
          .describe().T[["mean", "std", "min", "50%", "max"]])
    print()
    print(f"corr(sx, sy) = {shape['sx'].corr(shape['sy']):.3f}   "
          f"corr(cx, att_line_x) = {shape['cx'].corr(shape['att_line_x']):.3f}")
    print(f"gap to attacking line: mean={shape['gap'].mean():.2f}m  sd={shape['gap'].std():.2f}m")

    # Fit: Gaussian marginals clipped to the 5/95 range
    rows = []
    for name, col in [("centroid_x", "cx"), ("centroid_y", "cy"),
                      ("spread_x", "sx"), ("spread_y", "sy"),
                      ("attline_gap", "gap")]:
        s = shape[col]
        rows.append({"param": name, "mean": s.mean(), "sd": s.std(),
                     "lo": s.quantile(0.05), "hi": s.quantile(0.95)})
    rho = shape["sx"].corr(shape["sy"])
    rows.append({"param": "spread_corr", "mean": rho, "sd": 0.0, "lo": -1.0, "hi": 1.0})
    frame_params = pd.DataFrame(rows)

    slot_params = dfn.groupby("slot").agg(
        mu_u=("u", "mean"), sd_u=("u", "std"),
        mu_v=("v", "mean"), sd_v=("v", "std"), n=("u", "size")).reset_index()

    # Slot 9 is the lone forward: no freeze-frame data, so it borrows the midfield spread
    mid = slot_params.iloc[LINE_SIZES[0]:]
    slot_params.loc[N_FITTED] = {"slot": N_FITTED, "mu_u": 0.0, "sd_u": mid["sd_u"].median(),
                                 "mu_v": 0.0, "sd_v": mid["sd_v"].median(), "n": 0}
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
    sampled = np.array([sample_defender_formation(rng, depth_offset=0.0)
                        for _ in range(len(shape))])[:, :N_FITTED]

    sim = pd.DataFrame({
        "cx": sampled[:, :, 0].mean(axis=1), "cy": sampled[:, :, 1].mean(axis=1),
        "deepest_x": sampled[:, :, 0].max(axis=1),
        "width": sampled[:, :, 1].max(axis=1) - sampled[:, :, 1].min(axis=1),
    })
    print()
    print("--- real vs sampled (fitted 9 only, haram offset off) ---")
    for col in ["cx", "cy", "deepest_x", "width"]:
        print(f"  {col:9s} real {shape[col].mean():6.2f} +/- {shape[col].std():5.2f}"
              f"    sampled {sim[col].mean():6.2f} +/- {sim[col].std():5.2f}")

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(shape["cx"], bins=40, color="firebrick")
    axes[0, 0].axvline(92.0, color="black", ls="--", label="old fixed pose (centroid 92)")
    axes[0, 0].axvline(shape["cx"].mean() + HARAM_DEPTH_OFFSET, color="blue", ls=":",
                       label="fitted mean + haram 15m")
    axes[0, 0].set_xlabel("centroid x (sim frame)")
    axes[0, 0].set_title(f"Block depth (n={len(shape)})")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(shape["gap"], bins=40, color="steelblue")
    axes[0, 1].set_xlabel("centroid_x - att_line_x  (negative: front line sits goalward of the block centroid)")
    axes[0, 1].set_title(f"Gap to the attacking line "
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
    plt.savefig(os.path.join(CAL_DIR, "defender_shape_fit.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"back": "tab:blue", "mid": "tab:green", "fwd": "tab:red"}
    for slot, sub in dfn.groupby("slot"):
        line = slot_params.loc[slot, "line"]
        ax.scatter(sub["u"], sub["v"], s=3, alpha=0.12, color=colors[line])
        mu_u, mu_v = slot_params.loc[slot, "mu_u"], slot_params.loc[slot, "mu_v"]
        ax.errorbar(mu_u, mu_v, xerr=slot_params.loc[slot, "sd_u"],
                    yerr=slot_params.loc[slot, "sd_v"], fmt="o", color="black",
                    ms=5, lw=1.5, capsize=3)
        ax.annotate(f"{slot}", (mu_u, mu_v), xytext=(4, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("u = (x - centroid_x) / sx      (upfield <-- --> own goal)")
    ax.set_ylabel("v = (y - centroid_y) / sy")
    ax.set_title("Per-slot standardised offsets: real frames vs fitted mean +/- 1 sd")
    plt.tight_layout()
    plt.savefig(os.path.join(CAL_DIR, "defender_slot_scatter.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rng = np.random.default_rng(7)
    for k, ax in enumerate(axes.ravel()):
        pos = sample_defender_formation(rng)
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
    fig.suptitle("Sampled 5-4-1 low blocks (defending x=105, haram offset on)")
    plt.tight_layout()
    plt.savefig(os.path.join(CAL_DIR, "defender_samples.png"), dpi=120)
    plt.close(fig)

    print()
    print(f"wrote {SLOT_CSV}")
    print(f"wrote {FRAME_CSV}")
    print(f"wrote 3 figures to {CAL_DIR}")
