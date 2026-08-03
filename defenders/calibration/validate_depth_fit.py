import os
import numpy as np
import pandas as pd

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_CSV = os.path.join(CAL_DIR, "low_block_frames.csv")
OUT_CSV = os.path.join(CAL_DIR, "depth_fit_holdout.csv")

TARGETS = ("back_line_x", "mid_line_x")

# Same two models calibration_graphing.py fits, as {name: predictor columns}
MODELS = {
    "ball_only": ["ball_x"],
    "ball+att": ["ball_x", "att_line_x"],
}

# Matches held out in the fixed split; the rest are training
N_TEST_MATCHES = 2


# Design matrix with an intercept column appended
def design(df, cols):
    return np.column_stack([df[c].values for c in cols] + [np.ones(len(df))])


# Fit on train, predict on test; returns (test residuals, baseline residuals).
# The baseline predicts the *training* mean, so it is held out on the same terms.
def fit_predict(train, test, cols, target):
    coef = np.linalg.lstsq(design(train, cols), train[target].values, rcond=None)[0]
    actual = test[target].values
    return actual - design(test, cols) @ coef, actual - train[target].mean()


def rmse(residuals):
    return float(np.sqrt(np.mean(residuals ** 2)))


if __name__ == "__main__":
    df = pd.read_csv(FRAMES_CSV)
    match_ids = sorted(int(m) for m in df["match_id"].unique())

    print(f"frames: {len(df)}   matches: {len(match_ids)}")
    print(df.groupby("match_id").size().to_string())
    print()

    rows = []

    # --- fixed split: last N matches held out entirely ---
    test_ids = match_ids[-N_TEST_MATCHES:]
    train = df[~df["match_id"].isin(test_ids)]
    test = df[df["match_id"].isin(test_ids)]

    print(f"--- fixed split: {len(match_ids) - N_TEST_MATCHES} train matches "
          f"({len(train)} frames) / {N_TEST_MATCHES} test matches ({len(test)} frames) ---")
    print(f"test matches: {test_ids}")
    print()
    print(f"{'target':<12} {'model':<10} {'RMSE':>7} {'baseline':>9} {'skill':>7}")
    for target in TARGETS:
        for name, cols in MODELS.items():
            res, base = fit_predict(train, test, cols, target)
            r, b = rmse(res), rmse(base)
            skill = 1.0 - (r / b) ** 2
            print(f"{target:<12} {name:<10} {r:7.3f} {b:9.3f} {skill:7.3f}")
            rows.append({"split": "fixed", "target": target, "model": name,
                         "test_matches": len(test_ids), "n_test": len(test),
                         "rmse": r, "baseline_rmse": b, "skill": skill})
    print()

    # --- leave-one-match-out: every match gets held out once ---
    # With only 8 matches the fixed split above rests on 2 of them, so this is the
    # more stable number. Residuals are pooled across folds before scoring.
    print("--- leave-one-match-out CV (pooled over all folds) ---")
    print(f"{'target':<12} {'model':<10} {'RMSE':>7} {'baseline':>9} {'skill':>7}")
    per_fold = []
    for target in TARGETS:
        for name, cols in MODELS.items():
            res, base = [], []
            for mid in match_ids:
                tr, te = df[df["match_id"] != mid], df[df["match_id"] == mid]
                fold_res, fold_base = fit_predict(tr, te, cols, target)
                res.append(fold_res)
                base.append(fold_base)
                per_fold.append({"target": target, "model": name, "match_id": mid,
                                 "n_test": len(te), "rmse": rmse(fold_res),
                                 "baseline_rmse": rmse(fold_base)})
            r, b = rmse(np.concatenate(res)), rmse(np.concatenate(base))
            skill = 1.0 - (r / b) ** 2
            print(f"{target:<12} {name:<10} {r:7.3f} {b:9.3f} {skill:7.3f}")
            rows.append({"split": "lomo", "target": target, "model": name,
                         "test_matches": len(match_ids), "n_test": len(df),
                         "rmse": r, "baseline_rmse": b, "skill": skill})
    print()

    # Per-fold spread matters: a good pooled RMSE can hide one match the fit misses
    fold_df = pd.DataFrame(per_fold)
    print("--- per-fold held-out RMSE (ball+att) ---")
    print(fold_df[fold_df["model"] == "ball+att"]
          .pivot(index="match_id", columns="target", values="rmse")
          .to_string(float_format=lambda v: f"{v:6.2f}"))
    print()

    # In-sample RMSE for comparison: the gap to the held-out number is the overfit
    print("--- in-sample vs held-out (ball+att) ---")
    for target in TARGETS:
        res_in, _ = fit_predict(df, df, MODELS["ball+att"], target)
        held = [r for r in rows if r["split"] == "lomo" and r["target"] == target
                and r["model"] == "ball+att"][0]
        print(f"  {target:<12} in-sample {rmse(res_in):5.3f}m   held-out {held['rmse']:5.3f}m")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print()
    print(f"wrote {OUT_CSV}")
