"""Does pitch control at the RECEIVER predict whether a pass is lost?

After a 5M training run the attackers pass into pressure a lot: the ball head
plays into a defender's feet as readily as into space. Two things in the
environment explain why it could not have learned otherwise, and both are
structural rather than a matter of training longer.

  1. The reward cannot see a pass. `physics/ppcf.py` is a pure function of
     player positions and velocities -- `ball_pos` is used only to cache TTI
     into `players['i_p']` -- so `Phi = alpha*pc_f3 + beta*pc_hs` is blind to
     the ball, and so is the per-agent `phi_i`. Releasing the ball moves no
     player, so it moves no potential. Held constant, the reward on the release
     tick is bit-identical for HOLD and for all nine pass targets. The only pass
     content in the whole signal is the terminal `turnover_penalty`, which is
     the team scalar every attacker is paid equally, arriving ~20 ticks later,
     and identical to the one a dribble dispossession pays.

  2. The observation cannot see the receiver's pressure. An attacker gets
     `nearest_defender_feats` about ITSELF and the `K_DEF = 5` defenders nearest
     ITSELF; the teammate block is relative position and velocity only. Measured
     over 40 episodes, the defender who would actually contest a given receiver
     is inside the holder's 5-defender window just 54.3% of the time, so for
     nearly half the targets the threat is not in the input at all.

So we are running this test to establish the missing quantity before adding it:
whether attacker pitch control in the disc around the RECEIVER, at the moment of
release, actually predicts that the pass is lost in this sim. If it does, it is
the feature to put in the teammate block, one column per pass slot.

The variable that decides this is the RADIUS, not any threshold on it. Read at
`termination.AREA_RADIUS = 3.0` -- the +/-2m block the shot gate uses -- the
statistic is saturated at a median of 0.97 and separates almost nothing, because
every player controls the grass under his own feet. That is a statement about
the radius, not about pitch control. See RECEIVER_RADII.

Nearest-defender distance to the receiver is measured alongside as the cheap
alternative, so "use PPCF" ends up a finding rather than an assumption, and
every predictor is scored by AUC rather than by eye: bucket tables show shape
but hide how much of the sample is piled into one bucket.

Writes pass_pressure_samples.csv, pass_pressure_sweep.csv and
pass_pressure_calibration.png next to this file.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import physics.ppcf as ppcf                                    # noqa: E402
from environment.grid import PC_CELL_SIZE, PC_NX, PC_NY        # noqa: E402
from environment.lowblock_env import LowBlockEnv, PC_EXTENT    # noqa: E402
from environment.termination import (AREA_DI, AREA_DJ,         # noqa: E402
                                     AREA_RADIUS, pcf_in_area)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# The cells pcf_in_area averages, as metre offsets from the player. Recentred on
# an arbitrary point this is the same statistic computed off its own PPCF call
# instead of read out of the fixed grid -- see ON_GRID_MIN_X.
AREA_OFFSETS = np.stack([AREA_DI * PC_CELL_SIZE, AREA_DJ * PC_CELL_SIZE], axis=1)

# THE RADIUS IS THE VARIABLE, not the threshold. termination.AREA_RADIUS = 3.0
# against 2m cells is a 3x3 block spanning +/-2m: the grass under the player's
# own feet. It was calibrated for the shot gate, where the question is whether a
# shooter owns his own body space, and at that scale everyone owns it -- the sim
# median is 0.97, so the statistic is saturated and cannot separate anything.
#
# The pass question is a different one at a different scale: how much of the
# space a receiver must move into, take a touch in and turn out of does his team
# control. That is tens of metres, not two. So the receiver statistic is swept
# over radius here, and a single PPCF call over the largest disc serves all of
# them -- the smaller radii are subsets of the same cells.
# 20m was the top of the first sweep and came out best, which means the sweep
# never found a turning point -- "best" only meant "largest tried". Extended
# past it. There is a confound at this end and report() checks it: a 30m disc
# around a receiver in empty midfield is mostly grass nobody contests, so a large
# radius can drift from measuring pressure into measuring field position.
RECEIVER_RADII = (3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0)
PITCH_MAX = np.array([105.0, 68.0])


def _disc_offsets(radius, cell=PC_CELL_SIZE):
    """Metre offsets of every cell centre within `radius`, same construction as
    termination.AREA_DI/AREA_DJ."""
    r = int(radius // cell)
    di, dj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1),
                         indexing="ij")
    within = (di ** 2 + dj ** 2) * cell ** 2 <= radius ** 2
    return np.stack([di[within] * cell, dj[within] * cell], axis=1)


DISC_OFFSETS = _disc_offsets(max(RECEIVER_RADII))
DISC_RADIUS = np.linalg.norm(DISC_OFFSETS, axis=1)
RADIUS_MASKS = {r: DISC_RADIUS <= r for r in RECEIVER_RADII}

# The PPCF grid spans x in [43, 105]. pcf_in_area clips cell indices into that
# box and _masked_mean counts the out-of-range cells as zero while still
# dividing by AREA_CELLS, so a player near or behind the edge reads LOW control
# because the grid does not cover him, not because he is marked. Any receiver
# whose whole disc is not inside the grid is off-grid for this purpose.
ON_GRID_MIN_X = PC_EXTENT[0] + AREA_RADIUS

N_EPISODES = 220
MAX_TICKS = 300
PASS_RATE = 0.15        # per holder-tick, so passes are sampled across the state
                        # distribution rather than only where a policy would play
SETTLE_TICKS = 10       # ticks after arrival within which a loss still counts as
                        # the pass's fault -- receiving into pressure and being
                        # dispossessed immediately IS passing into pressure
SEED = 1

# Buckets for the reported curve. Uneven on purpose: the interesting structure is
# at the bottom, where the gate would sit.
EDGES = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def receiver_pcf(env, target_row):
    """Attacker pitch control in the disc around the intended receiver, read off
    the surface for the state the pass is being decided in (env.pc_att is the
    one from the step that produced the current state, which is what the policy
    sees). This is the cheap version: free off a call step() already makes, but
    only meaningful inside PC_EXTENT."""
    pos = np.asarray(env.players["position"][target_row], dtype=float)[None, :]
    surface = np.asarray(env.pc_att, dtype=float).reshape(PC_NX, PC_NY)
    return float(pcf_in_area(pos, surface)[0])


def receiver_pcf_radii(env, target_row):
    """Attacker pitch control around the receiver at every radius in
    RECEIVER_RADII, from ONE PPCF call over the largest disc.

    Centred on the receiver rather than read off the fixed grid, so it is
    defined everywhere on the pitch (see ON_GRID_MIN_X). Cells that fall off the
    pitch are dropped rather than averaged in: for a receiver near a touchline a
    large disc hangs well over the line, and whoever happens to be nearest that
    dead space would otherwise be credited with controlling it.

    ball_pos is left None on purpose -- it only drives the players['i_p'] TTI
    cache, which the env owns and overwrites on its own step.
    """
    pos = np.asarray(env.players["position"][target_row], dtype=float)
    pts = pos + DISC_OFFSETS
    on_pitch = ((pts >= 0.0) & (pts <= PITCH_MAX)).all(axis=1)

    per_player = ppcf.PPCF_grid(pts, env.players)
    att = per_player[:, env.players["team"] == "attacker"].sum(axis=1)
    return {r: float(att[m & on_pitch].mean())
            for r, m in RADIUS_MASKS.items()}


def receiver_defender_distance(env, target_row):
    pos = np.asarray(env.players["position"][target_row], dtype=float)
    dpos = np.asarray(env.players["position"][env.n_att:], dtype=float)
    return float(np.linalg.norm(dpos - pos, axis=1).min())


def target_row_of(env, slot):
    """Ball action `slot` -> attacker row, mirroring engine.ball_action's
    sorted-teammate-id indexing."""
    holder_id = int(env.attacker_ids[env.holder_row()])
    mates = np.sort(env.attacker_ids[env.attacker_ids != holder_id])
    return int(np.flatnonzero(env.attacker_ids == int(mates[slot - 1]))[0])


def random_movement(rng, n_att):
    a = np.zeros((n_att, 3), dtype=np.int64)
    a[:, 0] = rng.integers(0, 9, n_att)
    a[:, 1] = rng.integers(0, 3, n_att)
    return a


def collect():
    """One row per attempted pass. The attackers move randomly and pass at
    random to a random legal target: this is a state-distribution sampler, not a
    policy, and the caveat that it is not the trained policy's distribution is
    recorded in the README rather than papered over."""
    rng = np.random.default_rng(SEED)
    env = LowBlockEnv(max_ticks=MAX_TICKS, scripted_attackers=False)

    rows = []
    for _ in range(N_EPISODES):
        env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        pending = None   # a pass in the air, or just landed and still settling

        for t in range(MAX_TICKS):
            action = random_movement(rng, env.n_att)
            row = env.holder_row()

            if row is not None and pending is None:
                legal = np.flatnonzero(env.action_mask()[row])
                legal = legal[legal != 0]        # HOLD is not a pass
                if legal.size and rng.random() < PASS_RATE:
                    slot = int(rng.choice(legal))
                    trow = target_row_of(env, slot)
                    action[row, 2] = slot
                    pending = {
                        "pcf": receiver_pcf(env, trow),
                        "radii": receiver_pcf_radii(env, trow),
                        "x": float(env.players["position"][trow][0]),
                        "def_dist": receiver_defender_distance(env, trow),
                        "length": float(np.linalg.norm(
                            env.players["position"][trow]
                            - env.players["position"][row])),
                        "target_row": trow,
                        "released": t,
                        "arrived": None,
                        "in_flight_loss": False,
                    }

            _obs, _r, term, _trunc, info = env.step(action)

            if pending is not None:
                lost_now = info["outcome"] == "failure"
                if pending["arrived"] is None:
                    # still in the air, until the ball is held again
                    if lost_now:
                        pending["in_flight_loss"] = True
                        rows.append(_finish(pending, True, True))
                        pending = None
                    elif env.ball["state"] == "held":
                        holder = env.holder_row()
                        if holder is None:
                            # a defender collected it -- ball_mechanics gives it
                            # to whoever is nearest the landing spot
                            rows.append(_finish(pending, True, True))
                            pending = None
                        else:
                            pending["arrived"] = t
                elif lost_now or t - pending["arrived"] >= SETTLE_TICKS:
                    # completed; did the receiver survive the settling window?
                    rows.append(_finish(pending, False, bool(lost_now)))
                    pending = None

            if term:
                if pending is not None:
                    rows.append(_finish(pending, pending["arrived"] is None,
                                        info["outcome"] == "failure"))
                    pending = None
                break

    return rows


def _finish(pending, lost_in_transit, lost_soon):
    row = {"pcf_r{:g}".format(r): v for r, v in pending["radii"].items()}
    row.update({
        "receiver_pcf": pending["pcf"],
        "receiver_x": pending["x"],
        "on_grid": int(pending["x"] >= ON_GRID_MIN_X),
        "receiver_def_dist": pending["def_dist"],
        "pass_length": pending["length"],
        # the pass never reached its man
        "lost_in_transit": int(lost_in_transit),
        # ... or it did, and was gone inside SETTLE_TICKS
        "lost_within_settle": int(lost_soon),
    })
    return row


def auc(score, y):
    """P(a lost pass scores higher than a kept one), ties counted as half.

    Mann-Whitney U. Bucket tables show shape but hide how much of the sample
    sits in one bucket; this is the single number that says whether a predictor
    separates the two classes at all. 0.5 is a coin flip. `score` must be
    oriented so that higher means more dangerous.
    """
    a, b = score[y], score[~y]
    if not len(a) or not len(b):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([a, b])))[:len(a)] + 1
    u = ranks.sum() - len(a) * (len(a) + 1) / 2
    return float(u / (len(a) * len(b)))


def logistic_auc(scores, y, iters=300, lr=0.5):
    """AUC of the best linear combination of `scores`, each standardised. If a
    predictor is redundant given the others its fitted weight collapses toward
    zero and the combined AUC does not beat the best single one."""
    X = np.stack(scores, axis=1)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        w += lr * X.T @ (y.astype(float) - p) / len(X)
    return auc(X @ w, y), w[:-1]


def curve(x, lost, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        out.append((lo, hi, int(m.sum()),
                    float(lost[m].mean()) if m.any() else float("nan")))
    return out


def report(rows):
    pcf = np.array([r["receiver_pcf"] for r in rows])
    radii = {r: np.array([row["pcf_r{:g}".format(r)] for row in rows])
             for r in RECEIVER_RADII}
    dist = np.array([r["receiver_def_dist"] for r in rows])
    on = np.array([r["on_grid"] for r in rows], dtype=bool)
    transit = np.array([r["lost_in_transit"] for r in rows], dtype=float)
    settle = np.array([r["lost_within_settle"] for r in rows], dtype=float)
    lost = np.maximum(transit, settle)

    print(f"passes {len(rows)}  lost in transit {transit.mean():.1%}  "
          f"lost within {SETTLE_TICKS} ticks of arrival {settle.mean():.1%}  "
          f"either {lost.mean():.1%}")
    print(f"receivers off the PPCF grid (x < {ON_GRID_MIN_X:.0f}): "
          f"{(~on).sum()} of {len(rows)} = {(~on).mean():.1%}, "
          f"loss rate {lost[~on].mean():.1%} against {lost[on].mean():.1%} on it")
    print()

    # The grid-read column is only interpretable on-grid, so it is reported both
    # ways: the contaminated curve is what you get if you forget, and the
    # difference between them IS the finding.
    for label, mask in (("ALL passes (contaminated)", np.ones_like(on)),
                        ("on-grid receivers only", on)):
        print(f"receiver pcf_in_area, grid-read -- {label}")
        print("  bucket              n    P(lost)")
        for lo, hi, n, p in curve(pcf[mask], lost[mask], EDGES):
            print(f"  [{lo:.2f}, {hi:.2f})     {n:5d}   {p:.1%}")
        print()

    y = lost.astype(bool)

    # THE HEADLINE. Radius, not threshold, is what decides whether PPCF can see
    # pressure at all. `spread` is the interquartile range: a statistic pinned
    # against 1.0 for everybody has nothing left to discriminate with, so it is
    # reported next to the AUC rather than inferred from it.
    print("receiver PPCF by radius (own centred call):")
    print("  radius   median   IQR     AUC     vs distance, best linear combo")
    for r in RECEIVER_RADII:
        x = radii[r]
        iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
        combo, w = logistic_auc([-x, -dist], y)
        print(f"  {r:5.0f}m   {np.median(x):.3f}   {iqr:.3f}   "
              f"{auc(-x, y):.3f}   {combo:.3f}  "
              f"(pcf {w[0]:+.2f}, dist {w[1]:+.2f})")
    print()

    best_r = max(RECEIVER_RADII, key=lambda r: auc(-radii[r], y))
    best = radii[best_r]
    print(f"best radius {best_r:.0f}m -- P(lost) by bucket")
    print("  bucket              n    P(lost)")
    for lo, hi, n, p in curve(best, lost, EDGES):
        print(f"  [{lo:.2f}, {hi:.2f})     {n:5d}   {p:.1%}")
    print()

    print("receiver nearest defender  n    P(lost)     (the cheap alternative)")
    for lo, hi, n, p in curve(dist, lost, (0.0, 2.0, 4.0, 6.0, 9.0, 99.0)):
        print(f"  [{lo:.0f}m, {hi:.0f}m)        {n:5d}   {p:.1%}")
    print()

    print("predictor                                  AUC")
    for name, score, m in (
            ("receiver pcf_in_area, grid-read (all)", -pcf, np.ones_like(on)),
            ("receiver pcf_in_area, grid-read (on-grid)", -pcf, on),
            (f"receiver PPCF at {best_r:.0f}m, centred", -best, np.ones_like(on)),
            ("receiver nearest-defender distance", -dist, np.ones_like(on)),
            ("pass length", np.array([r["pass_length"] for r in rows]),
             np.ones_like(on))):
        print(f"  {name:<42} {auc(score[m], y[m]):.3f}   (n={int(m.sum())})")
    print()

    print(f"gate on the {best_r:.0f}m PPCF, all passes:")
    for thr in THRESHOLDS:
        below = best < thr
        if not below.any():
            continue
        print(f"  < {thr:.2f}: masks {below.mean():5.1%} of passes, "
              f"carrying {lost[below].mean():5.1%} loss "
              f"vs {lost[~below].mean():5.1%} for the rest")
    print()

    # THE CONFOUND. A big disc around a receiver standing in empty midfield is
    # mostly uncontested grass, and deep receivers are safe for reasons that have
    # nothing to do with pressure. If a large radius is really just reading field
    # position, then receiver x explains it and PPCF's weight collapses once x is
    # in the model -- the same collapse that exposed the 3m column as redundant
    # against distance. Reported for every radius, because where it sets in is
    # exactly where the radius stops being a pressure measurement.
    x = np.array([r["receiver_x"] for r in rows])
    print(f"field-position control (receiver x alone, AUC {auc(-x, y):.3f}):")
    print("  radius   pcf alone   +x      weights (pcf, x)")
    for r in RECEIVER_RADII:
        with_x, w = logistic_auc([-radii[r], -x], y)
        print(f"  {r:5.0f}m   {auc(-radii[r], y):.3f}       {with_x:.3f}   "
              f"({w[0]:+.2f}, {w[1]:+.2f})")
    print()
    all_three, w3 = logistic_auc([-radii[best_r], -dist, -x], y)
    print(f"  all three at {best_r:.0f}m: AUC {all_three:.3f}  "
          f"(pcf {w3[0]:+.2f}, dist {w3[1]:+.2f}, x {w3[2]:+.2f})")

    with open(os.path.join(OUT_DIR, "pass_pressure_samples.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(OUT_DIR, "pass_pressure_sweep.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["radius", "median", "iqr", "auc", "auc_with_distance",
                    "weight_pcf", "weight_distance"])
        for r in RECEIVER_RADII:
            x = radii[r]
            combo, wt = logistic_auc([-x, -dist], y)
            w.writerow([r, float(np.median(x)),
                        float(np.percentile(x, 75) - np.percentile(x, 25)),
                        auc(-x, y), combo, float(wt[0]), float(wt[1])])

    plot(radii, best_r, dist, lost, y)
    return radii, best_r, dist, lost


def plot(radii, best_r, dist, lost, y):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    best = radii[best_r]
    ax[0].hist([best[lost == 0], best[lost == 1]], bins=20, stacked=True,
               label=["kept", "lost"], color=["#4c9f70", "#c1584a"])
    ax[0].set_xlabel(f"attacker pitch control at the receiver, {best_r:.0f}m")
    ax[0].set_ylabel("passes")
    ax[0].set_title("Receiver PPCF at release")
    ax[0].legend()

    # Every radius on one axis: the 3m curve is the flat one, and the spread
    # between them is the finding.
    for r in RECEIVER_RADII:
        pts = curve(radii[r], lost, EDGES)
        ax[1].plot([0.5 * (lo + hi) for lo, hi, _n, _p in pts],
                   [p for *_x, p in pts], "o-", alpha=0.85,
                   label=f"{r:.0f}m (AUC {auc(-radii[r], y):.3f})")

    ax[1].axhline(lost.mean(), ls="--", lw=1, color="0.5")
    ax[1].set_xlabel("attacker pitch control at the receiver")
    ax[1].set_ylabel("P(pass lost)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("P(lost) by PPCF, per radius")
    ax[1].legend(fontsize=8)

    pts = curve(dist, lost, (0.0, 2.0, 4.0, 6.0, 9.0, 99.0))
    centres = [0.5 * (lo + min(hi, dist.max())) for lo, hi, _n, _p in pts]
    ax[2].plot(centres, [p for *_x, p in pts], "o-", color="#c1584a")
    for c, (_lo, _hi, n, p) in zip(centres, pts):
        if n:
            ax[2].annotate(f"n={n}", (c, p), textcoords="offset points",
                           xytext=(0, 7), ha="center", fontsize=8)
    ax[2].axhline(lost.mean(), ls="--", lw=1, color="0.5")
    ax[2].set_xlabel("receiver nearest defender (m)")
    ax[2].set_ylabel("P(pass lost)")
    ax[2].set_ylim(0, 1)
    ax[2].set_title("P(lost) against nearest defender")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pass_pressure_calibration.png"), dpi=130)


def load():
    """Re-report from the committed samples CSV. Collection is ~13 minutes of
    PPCF; reworking the tables should not cost that again."""
    with open(os.path.join(OUT_DIR, "pass_pressure_samples.csv")) as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


if __name__ == "__main__":
    report(load() if "--reuse" in sys.argv else collect())
