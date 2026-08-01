# Regenerates tests/golden_ppcf.npz from the current physics/ppcf.py.
# Run: python tests/generate_golden.py
# The fixture was first built from the pre-optimization loop (commit 0edf140).
# Do NOT rerun this to fix a failing test -- that hides the regression.
# Only rerun when the physics deliberately changes (rates, DT, threshold).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.ppcf import PPCF_grid  # noqa: E402
from scenarios import build_scenarios  # noqa: E402


def main():
    out = {}
    for name, sc in build_scenarios().items():
        players = sc["players"]
        result = PPCF_grid(sc["targets"], players, sc["ball_pos"])

        out[f"{name}/positions"] = np.asarray(players["position"])
        out[f"{name}/velocities"] = np.asarray(players["velocity"])
        out[f"{name}/team"] = np.asarray(players["team"])
        out[f"{name}/targets"] = sc["targets"]
        out[f"{name}/result"] = result
        # stored on both branches so the test can check ball_pos=None leaves it alone
        out[f"{name}/i_p"] = np.asarray(players["i_p"])
        if sc["ball_pos"] is None:
            out[f"{name}/ball_pos"] = np.array([])
        else:
            out[f"{name}/ball_pos"] = np.asarray(sc["ball_pos"], dtype=float)

        print(f"{name:14s} cells={sc['targets'].shape[0]:5d} "
              f"players={len(players):3d} "
              f"total_control={result.sum():.6f}")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden_ppcf.npz")
    np.savez_compressed(path, **out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
