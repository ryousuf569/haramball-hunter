import os

import numpy as np
import cupy as cp

BLOCK = 128
ATTACKER_CONTROL_RATE = 4.30
DEFENDER_CONTROL_RATE = ATTACKER_CONTROL_RATE * 1.72

with open(os.path.join(os.path.dirname(__file__), "ppcf.cu")) as f:
    _SRC = f.read()

ppcf_grid = cp.RawKernel(_SRC, "ppcf_grid")
ppcf_ball_tti = cp.RawKernel(_SRC, "ppcf_ball_tti")


def PPCF_grid(targets, players, ball_pos=None):
    n_cells = len(targets)
    n = len(players)

    d_targets = cp.asarray(np.ascontiguousarray(targets, dtype="f4"))
    d_pos = cp.asarray(np.ascontiguousarray(players["position"], dtype="f4"))
    d_vel = cp.asarray(np.ascontiguousarray(players["velocity"], dtype="f4"))


    lam = np.where(players["team"] == "attacker", ATTACKER_CONTROL_RATE, DEFENDER_CONTROL_RATE).astype("f4")
    d_lam = cp.asarray(lam)
    d_out = cp.empty((n_cells, n), dtype="f4")

    ppcf_grid(((n_cells + BLOCK - 1) // BLOCK,), (BLOCK,),
               (d_targets, d_pos, d_vel, d_lam, d_out,
                np.int32(n_cells), np.int32(n)),
               shared_mem=(5 + BLOCK) * n * 4)

    if ball_pos is not None:
        d_ball = cp.asarray(np.ascontiguousarray(ball_pos, dtype="f4").reshape(2))
        d_ip = cp.empty(n, dtype="f4")
        ppcf_ball_tti(((n + BLOCK - 1) // BLOCK,), (BLOCK,),
                       (d_pos, d_vel, d_ball, d_ip, np.int32(n)))
        players["i_p"] = cp.asnumpy(d_ip)

    return cp.asnumpy(d_out)
