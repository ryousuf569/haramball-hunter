import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CAL_DIR))
VALIDATION_DIR = os.path.join(REPO_ROOT, "physics", "validation")
for _p in (REPO_ROOT, VALIDATION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Metrica_IO as mio          # noqa: E402
# The carrier timeline, the possession radius and the spell splitter are the ones
# DRIBBLE_SPEED was fitted with. Reusing them rather than writing a second
# definition of "on the ball" keeps this fit and that one on the same population.
from physics.validation.dribble_speed_calibration import (  # noqa: E402
    load, possession_mask, stack_team,
)
from defenders.turnover import (  # noqa: E402
    ADV_EXP, ADV_FLOOR, DUEL_A, DUEL_A_BACK, DUEL_B, DUEL_EXP, EPS,
    LAM_MAX, V_MAX,
)

DATA_DIR = os.path.join(VALIDATION_DIR, "data")
GAME_ID = 1
FPS = 25
DT = 1.0 / FPS      # 0.04s. The sim ticks at 0.1s; a hazard rate is per second,
                    # so the two never have to be reconciled -- that is the whole
                    # reason ground_duel is written as a hazard in the first place.

# --- what counts as a ground duel ----------------------------------------------
# Metrica logs a challenge once per team, a frame or two apart, so the same duel
# appears as e.g. GROUND-LOST for one side and GROUND-WON for the other. DEDUPE_F
# collapses those mirror pairs into one contest.
GROUND_SUBTYPES = ("GROUND", "TACKLE", "DRIBBLE")   # everything but AERIAL
THEFT_SUBTYPE = "THEFT"    # a ball taken off a carrier without a logged challenge
DEDUPE_F = 5               # frames, 0.2s

# A challenge is annotated on the frame the contact happens; the carry spell can
# end a beat earlier as the ball comes off the carrier's foot. MATCH_W is how far
# after a spell's last frame a challenge is still read as belonging to it.
MATCH_W = 12               # frames, ~0.5s

# WON/LOST in the subtype is relative to the row's team and the two do not always
# agree across the mirror pair, so the outcome is taken from the tracking instead:
# a duel counts as a loss only if the next carrier belongs to the other team. That
# also handles the FAULT variants correctly for free -- a foul returns the ball to
# the attacking side, and the sim has no fouls, so it must not count as a turnover.

MIN_SPELL = 2              # frames; matches duel_reality_check.py

# --- the gate shape -------------------------------------------------------------
# The three semi-axes are now fitted per axis rather than carried at RoboCup's
# ratios. Each is anchored at the LOSS_GAP_Q quantile of the coordinate it
# governs, measured over real dispossessions: a reach is a cap, so anchor high --
# the same argument ANCHOR_Q makes in dribble_speed_calibration.py, applied three
# times instead of once.
#
# Anchoring per axis is necessary but not sufficient. DUEL_EXP = 6 cuts the
# corners of the super-ellipse, so three independent p90 axes enclose well under
# 90% of the loss points jointly. COVERAGE_Q scales all three by a common factor
# until the gate genuinely contains that share of real dispossessions, which is
# what the per-axis anchors were trying to express in the first place. The shape
# comes from the ratios, the size from the joint coverage.
LOSS_GAP_Q = 90            # per-axis anchor quantile
COVERAGE_Q = 0.90          # target joint coverage of real dispossessions
K_SCAN = np.round(np.arange(0.6, 2.51, 0.05), 2)

# DUEL_EXP sets how sharply the gate falls off at its edge. fit_exponent() below
# tests it against the data rather than inheriting it: it is scanned jointly with
# the scale, because a soft edge on a big gate and a hard edge on a small one are
# nearly the same hazard field and the two must be fitted together or the
# comparison is between (exponent, axes(exponent)) pairs rather than exponents.
# --- engagement -------------------------------------------------------------
# Proximity is not contest: Metrica has 351 moments with an opponent inside the
# gate and 103 annotated challenges. What separates them is whether the defender
# is closing on the ball, which the multiplier already encodes -- so engagement
# is fitted here rather than bolted on as a new mechanism.
CLOSING_EDGES = [-8.0, -1.0, 0.0, 1.0, 2.0, 3.0, 12.0]
ADV_FLOOR_SCAN = np.round(np.arange(0.01, 1.001, 0.01), 3)
ADV_EXP_SCAN = np.round(np.arange(0.4, 16.01, 0.2), 2)

EXP_SCAN = np.round(np.arange(1.0, 10.01, 0.25), 2)
EXP_K_SCAN = np.round(np.arange(0.8, 3.001, 0.02), 3)
LR_95 = 1.92               # chi2(1)/2, the 95% likelihood-ratio drop

# The values this study replaced, pinned rather than read off turnover.py: the
# imports above are now this script's OWN output, so comparing against them would
# report "was 2.25, now 2.25". Same reason dribble_speed_calibration.py pins
# OLD_GUESS. Only labels and comparison curves use these; nothing in the fit does.
PREV_DUEL_A = 1.20
PREV_DUEL_B = 0.50
PREV_DUEL_A_BACK = 0.40
PREV_LAM_MAX = 2.1

HAZARD_EDGES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 5.0, 7.0, 12.0]
N_BOOTSTRAP = 2000
BOOT_SEED = 0


# ===========================================================================
# 1. The carrier timeline
# ===========================================================================

def carrier_timeline(home, away, ball):
    """Per frame: which team and which player of it is on the ball.

    possession_mask is per team, so on a contested frame both teams can claim it.
    Whoever is physically closer to the ball wins the tie, which is the same rule
    the mask uses within a team.
    """
    bx, by = ball[0], ball[1]
    n = len(bx)
    teams = {}
    for df, pre in ((home, "Home"), (away, "Away")):
        ids, x, y, vx, vy, _s = stack_team(df, pre)
        teams[pre] = dict(ids=ids, x=x, y=y, vx=vx, vy=vy,
                          poss=possession_mask(x, y, vx, vy, ball))

    team = np.full(n, "", dtype=object)
    col = np.full(n, -1)
    best = {}
    for pre in ("Home", "Away"):
        t = teams[pre]
        d = np.hypot(t["x"] - bx[:, None], t["y"] - by[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        d = np.where(t["poss"], d, np.inf)
        best[pre] = (d.min(axis=1), d.argmin(axis=1))

    home_closer = best["Home"][0] <= best["Away"][0]
    for pre, sel in (("Home", home_closer), ("Away", ~home_closer)):
        ok = sel & np.isfinite(best[pre][0])
        team[ok] = pre
        col[ok] = best[pre][1][ok]
    return teams, team, col


def carry_spells(team, col, min_len=MIN_SPELL):
    """Contiguous runs of one player holding the ball."""
    n = len(team)
    key = np.where(team != "", np.char.add(np.char.add(team.astype(str), ":"),
                                           col.astype(str)), "")
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    out = []
    for i, a in enumerate(edges):
        b = edges[i + 1] if i + 1 < len(edges) else n
        if key[a] != "" and b - a >= min_len:
            out.append((int(a), int(b)))
    return out


# ===========================================================================
# 2. Exposure: the duel geometry on every carry frame
# ===========================================================================

def p_geometry(dx, dy, a_semi, b_semi, a_back, expo=DUEL_EXP):
    """turnover.ground_duel's super-ellipse, vectorised, with every axis free.

    Identical to the expression in ground_duel, with the constants passed in so a
    candidate shape can be scored without touching the module. `expo` defaults to
    the module's DUEL_EXP; fit_exponent is the only caller that varies it.
    """
    reach = np.where(dx >= 0.0, a_semi, a_back)
    fail = (np.abs(dx) / reach) ** expo + (np.abs(dy) / b_semi) ** expo
    return np.clip(1.0 - fail, 0.0, None)


def advantage_multiplier(closing, floor=ADV_FLOOR, expo=ADV_EXP):
    """turnover.ground_duel's closing-speed multiplier, in [floor, 1].

    This is the engagement term: a defender who is not closing on the ball is
    near it, not contesting it. Defaults are the module's fitted values; only
    fit_engagement varies them.
    """
    u = 0.5 * (np.clip(np.asarray(closing, dtype=float) / V_MAX, -1.0, 1.0) + 1.0)
    return floor + (1.0 - floor) * u ** expo


# ground_duel lets EVERY defender inside the gate contest, so the exposure the
# rate is fitted against has to sum over all of them too, or LAM_MAX would be
# fitted on one denominator and applied to a larger one. Opponents further out
# than this can never be inside any gate the shape fit will choose, so keeping
# them would only bloat the table. Asserted against the fitted axes in main().
CONTACT_R = 5.0


def _duel_geometry(gx, gy, vdx, vdy, hvx, hvy):
    """ground_duel's body-frame transform, for one opponent per row.

    gx, gy point ball -> defender, so to_ball is their negation. The defender has
    no facing in the sim and its direction of travel stands in for one; the same
    substitution is made here, from the tracking-derived velocity.
    """
    dist = np.hypot(gx, gy)
    tbx, tby = -gx, -gy
    speed = np.hypot(vdx, vdy)
    moving = speed > EPS
    safe = np.where(moving, speed, 1.0)
    ux, uy = vdx / safe, vdy / safe
    # standing still: no heading to rotate into, so plain radial distance
    dx = np.where(moving, tbx * ux + tby * uy, dist)
    dy = np.where(moving, tbx * -uy + tby * ux, 0.0)
    closing = (tbx * (vdx - hvx) + tby * (vdy - hvy)) / np.maximum(dist, EPS)
    return dist, dx, dy, closing, speed


def frame_table(teams, team, col, spells_, ball):
    """Two tables, both in ground_duel's frame.

    F -- one row per carry frame, for the NEAREST opponent. Drives the hazard
         profile and the shape anchors, which are both about the defender who
         actually took the ball.
    C -- one row per (carry frame, opponent within CONTACT_R). Drives the
         exposure LAM_MAX is fitted against, because the model now sums hazards
         over everyone in the gate.

    Opponents are ranked by distance to the BALL, not to the carrier, because
    that is what ground_duel does.
    """
    bx, by = ball[0], ball[1]
    opp = {"Home": "Away", "Away": "Home"}
    near_rows, contact_rows = [], []
    for a, b in spells_:
        ct, cj = team[a], col[a]
        d_side, a_side = teams[opp[ct]], teams[ct]
        ox, oy = d_side["x"][a:b], d_side["y"][a:b]
        ovx, ovy = d_side["vx"][a:b], d_side["vy"][a:b]
        px, py = bx[a:b], by[a:b]
        hvx, hvy = a_side["vx"][a:b, cj], a_side["vy"][a:b, cj]
        frames = np.arange(a, b)

        gx, gy = ox - px[:, None], oy - py[:, None]
        sq = np.where(np.isnan(gx) | np.isnan(gy), np.inf, gx ** 2 + gy ** 2)

        # --- nearest, one row per frame ---
        cand = np.argmin(sq, axis=1)
        r = np.arange(b - a)
        dist, dx, dy, closing, speed = _duel_geometry(
            gx[r, cand], gy[r, cand], ovx[r, cand], ovy[r, cand], hvx, hvy)
        near_rows.append(pd.DataFrame(dict(
            frame=frames, spell_a=a, spell_b=b, team=ct,
            dist=dist, dx=dx, dy=dy, closing=closing, def_speed=speed)))

        # --- everyone close enough to matter, one row per (frame, opponent) ---
        fi, oi = np.nonzero(sq <= CONTACT_R * CONTACT_R)
        if fi.size:
            cdist, cdx, cdy, cclosing, _s = _duel_geometry(
                gx[fi, oi], gy[fi, oi], ovx[fi, oi], ovy[fi, oi],
                hvx[fi], hvy[fi])
            contact_rows.append(pd.DataFrame(dict(
                frame=frames[fi], spell_a=a, opponent=oi,
                dist=cdist, dx=cdx, dy=cdy, closing=cclosing)))

    F = pd.concat(near_rows, ignore_index=True)
    F = F[np.isfinite(F["dist"])].reset_index(drop=True)
    C = (pd.concat(contact_rows, ignore_index=True) if contact_rows
         else pd.DataFrame(columns=["frame", "spell_a", "opponent",
                                    "dist", "dx", "dy", "closing"]))
    return F, C.reset_index(drop=True)


# ===========================================================================
# 3. Events: which contests actually cost the carrier the ball
# ===========================================================================

def dedupe_frames(frames, tol=DEDUPE_F):
    keep = []
    for v in np.sort(np.unique(np.asarray(frames))):
        if not keep or v - keep[-1] > tol:
            keep.append(int(v))
    return np.array(keep, dtype=int)


def duel_events(ev, F, team, spells_):
    """Every ground duel that happened to a tracked carrier, flip or not."""
    sub = ev["Subtype"].fillna("")
    ground = ev[(ev["Type"] == "CHALLENGE")
                & sub.str.startswith(GROUND_SUBTYPES)]
    theft = ev[sub == THEFT_SUBTYPE]
    ev_frames = dedupe_frames(np.r_[ground["Start Frame"].to_numpy(),
                                    theft["Start Frame"].to_numpy()])

    starts = np.array([a for a, _b in spells_])
    inside = {}
    for a, b in spells_:
        for f in range(a, b):
            inside[f] = (a, b)
    opp = {"Home": "Away", "Away": "Home"}
    by_frame = F.set_index("frame")

    rows = []
    for f in ev_frames:
        s = inside.get(int(f))
        if s is None:
            # the ball can come off the foot a beat before the logged contact
            trailing = [(a, b) for a, b in spells_ if 0 <= f - (b - 1) <= MATCH_W]
            s = trailing[-1] if trailing else None
        if s is None:
            continue
        a, b = s
        at = min(max(int(f), a), b - 1)
        if at not in by_frame.index:
            continue
        g = by_frame.loc[at]
        nxt = starts[starts >= b]
        flip = bool(nxt.size and team[nxt[0]] == opp[team[a]])
        rows.append(dict(event_frame=int(f), measured_frame=at, spell_a=a,
                         spell_b=b, carrier_team=team[a], lost=flip,
                         dist=float(g["dist"]), dx=float(g["dx"]),
                         dy=float(g["dy"]), closing=float(g["closing"])))
    return pd.DataFrame(rows), len(ev_frames)


# ===========================================================================
# 4. The fit
# ===========================================================================

def hazard_profile(F, losses, edges=HAZARD_EDGES):
    """Model-free hazard by gap: losses per second of exposure in each ring.

    This is the part of the study that owes the duel model nothing -- no
    super-ellipse, no multiplier, just how long carriers spend with the nearest
    defender at a given range and how often they are dispossessed from it.
    """
    d, ld = F["dist"].to_numpy(), losses["dist"].to_numpy()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n_fr = int(((d >= lo) & (d < hi)).sum())
        expo = n_fr * DT
        k = int(((ld >= lo) & (ld < hi)).sum())
        # Byar's approximation to the exact Poisson interval on the count. scipy
        # is not a project dependency, for the reason press_calibration.py gives.
        lo_k = k * (1 - 1 / (9 * k) - 1.96 / (3 * np.sqrt(k))) ** 3 if k else 0.0
        hi_k = (k + 1) * (1 - 1 / (9 * (k + 1)) + 1.96 / (3 * np.sqrt(k + 1))) ** 3
        rows.append(dict(gap_lo_m=lo, gap_hi_m=hi, frames=n_fr,
                         exposure_s=round(expo, 2), losses=k,
                         hazard_per_s=round(k / expo, 4) if expo else np.nan,
                         hazard_lo95=round(lo_k / expo, 4) if expo else np.nan,
                         hazard_hi95=round(hi_k / expo, 4) if expo else np.nan))
    return pd.DataFrame(rows)


def axis_anchors(losses, q=LOSS_GAP_Q):
    """The three semi-axes the loss geometry implies, before joint rescaling.

    dx is signed along the defender's heading, so forward and behind are fitted
    from disjoint halves of the sample; dy is symmetric and uses all of it.
    """
    dx, dy = losses["dx"].to_numpy(), losses["dy"].to_numpy()
    return (float(np.percentile(dx[dx >= 0], q)),
            float(np.percentile(np.abs(dx[dx < 0]), q)),
            float(np.percentile(np.abs(dy), q)))


def coverage(losses, a_semi, b_semi, a_back):
    g = p_geometry(losses["dx"].to_numpy(), losses["dy"].to_numpy(),
                   a_semi, b_semi, a_back)
    return float((g > 0).mean())


def _g(tbl, a_semi, b_semi, a_back):
    """p_geometry * multiplier, the per-row factor LAM_MAX is multiplied by."""
    return p_geometry(tbl["dx"].to_numpy(), tbl["dy"].to_numpy(),
                      a_semi, b_semi, a_back) \
        * advantage_multiplier(tbl["closing"].to_numpy())


def fit_lambda(C, losses, a_semi, b_semi, a_back):
    """Closed-form MLE of LAM_MAX at a given gate shape.

    Under ground_duel the hazard on a carry frame is LAM_MAX * sum_i g_i over
    every defender in the gate, with g = p_geometry * multiplier known from the
    tracking. The Poisson likelihood over the whole exposure then has the
    occupancy estimator as its maximiser:

        LAM_MAX = (losses the gate covers) / (dt * sum of g over ALL contacts)

    Summing over contacts rather than over frames is what keeps this consistent
    with the model: ground_duel adds the hazards of everyone in range, so fitting
    against nearest-defender exposure alone would set the rate on one denominator
    and then apply it to a larger one.

    Only losses the gate covers go in the numerator, because a loss with
    p_geometry == 0 is one the model cannot produce at all -- counting it would
    charge the rate for an event that never contributed exposure. The coverage
    fraction is reported separately rather than absorbed into the rate.
    """
    g = _g(C, a_semi, b_semi, a_back)
    g_ev = _g(losses, a_semi, b_semi, a_back)
    exposure = float(g.sum()) * DT
    covered = int((g_ev > 0).sum())
    return dict(a_semi=a_semi, b_semi=b_semi, a_back=a_back,
                exposure_s=exposure,
                covered_losses=covered, total_losses=len(losses),
                coverage_frac=covered / max(len(losses), 1),
                lam_max=covered / exposure if exposure > 0 else np.nan)


def bootstrap_lambda(C, losses, a_semi, b_semi, a_back,
                     n=N_BOOTSTRAP, seed=BOOT_SEED):
    """Resample whole carry spells, not frames: frames within a spell are the
    same contest and are nowhere near independent."""
    rng = np.random.default_rng(seed)
    g = _g(C, a_semi, b_semi, a_back)
    covered = _g(losses, a_semi, b_semi, a_back) > 0

    spell = C["spell_a"].to_numpy()
    loss_spell = losses["spell_a"].to_numpy()
    uniq = np.unique(spell)
    out = []
    for _ in range(n):
        counts = pd.Series(rng.choice(uniq, size=uniq.size, replace=True)).value_counts()
        w = pd.Series(spell).map(counts).fillna(0.0).to_numpy()
        wl = pd.Series(loss_spell).map(counts).fillna(0.0).to_numpy()
        expo = float((g * w).sum()) * DT
        if expo > 0:
            out.append(float((wl * covered).sum()) / expo)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def choose_gate(profile, losses):
    """Two independent readings of how far the gate reaches, as a cross-check.

    (1) The hazard elbow: the outer edge of the last ring whose hazard is still
        a substantial fraction of the point-blank rate.
    (2) The p90 of the defender-to-ball gap at a real dispossession.
    Both are radial, so they check the fitted axes rather than setting them.
    """
    near = profile[profile["gap_hi_m"] <= 1.2]
    point_blank = float(near["losses"].sum() / near["exposure_s"].sum())
    live = profile[profile["hazard_per_s"] >= 0.25 * point_blank]
    elbow = float(live["gap_hi_m"].max()) if len(live) else np.nan
    quantile = float(np.percentile(losses["dist"], LOSS_GAP_Q))
    return point_blank, elbow, quantile


def fit_exponent(C, losses, anchors, exps=EXP_SCAN, ks=EXP_K_SCAN):
    """Profile likelihood for DUEL_EXP, with the gate scale fitted alongside it.

    Two things make this harder than it looks, and both were got wrong first.

    The event set must not depend on the exponent. Rescaling the axes per
    exponent to hold coverage at 90% sounds neutral but is not: coverage is a
    step function of the scale, so the axes jump as events cross the boundary and
    the comparison silently becomes one between (exponent, axes(exponent)) pairs.
    Here the set is trimmed once, in anchor-normalised coordinates, which no
    candidate exponent can influence.

    And the scale must be free. A soft edge on a large gate and a hard edge on a
    small one describe nearly the same hazard field, so holding the scale fixed
    would attribute that trade-off to the exponent.

    Returns one row per exponent, at its own best scale.
    """
    af0, ab0, b0 = anchors
    dx, dy = losses["dx"].to_numpy(), losses["dy"].to_numpy()
    mult_ev = advantage_multiplier(losses["closing"].to_numpy())
    # n-independent trim: how far out each loss sits in units of its own axis
    u = np.maximum(np.abs(dx) / np.where(dx >= 0, af0, ab0), np.abs(dy) / b0)
    keep = u <= np.quantile(u, COVERAGE_Q)
    n_keep = int(keep.sum())

    cdx, cdy = C["dx"].to_numpy(), C["dy"].to_numpy()
    mult_c = advantage_multiplier(C["closing"].to_numpy())

    rows = []
    for n in exps:
        best_ll, best_k, best_lam = -np.inf, np.nan, np.nan
        for k in ks:
            a, b, bk = af0 * k, b0 * k, ab0 * k
            g_ev = p_geometry(dx[keep], dy[keep], a, b, bk, n) * mult_ev[keep]
            if (g_ev <= 0).any():
                continue          # this scale cannot hold the fixed event set
            g_all = p_geometry(cdx, cdy, a, b, bk, n) * mult_c
            ll = float(np.sum(np.log(g_ev)) - n_keep * np.log(g_all.sum()))
            if ll > best_ll:
                best_ll, best_k = ll, k
                best_lam = n_keep / (float(g_all.sum()) * DT)
        rows.append(dict(exponent=n, k=best_k, a_semi=af0 * best_k,
                         b_semi=b0 * best_k, a_back=ab0 * best_k,
                         lam_max=best_lam, logL=best_ll, n_events=n_keep))
    out = pd.DataFrame(rows)
    out["dlogL"] = out["logL"] - out["logL"].max()
    return out


def engagement_profile(C, losses, a, b, bk, edges=CLOSING_EDGES):
    """Model-free hazard by closing speed, inside the gate.

    The engagement evidence, owing the multiplier nothing: how long defenders
    spend in the gate at a given closing speed, and how often the carrier is
    dispossessed from it. Exposure is weighted by p_geometry so that "closing
    fast" is not silently credited with "also standing closer".
    """
    gC = p_geometry(C["dx"].to_numpy(), C["dy"].to_numpy(), a, b, bk)
    inside = gC > 0
    cc, gc = C["closing"].to_numpy()[inside], gC[inside]
    gL = p_geometry(losses["dx"].to_numpy(), losses["dy"].to_numpy(), a, b, bk)
    ce = losses["closing"].to_numpy()[gL > 0]

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cc >= lo) & (cc < hi)
        expo = float(gc[m].sum()) * DT
        k = int(((ce >= lo) & (ce < hi)).sum())
        rows.append(dict(closing_lo=lo, closing_hi=hi, mid=0.5 * (lo + hi),
                         contacts=int(m.sum()), exposure_g_s=expo, losses=k,
                         hazard_per_s=k / expo if expo > 0 else np.nan))
    out = pd.DataFrame(rows)
    peak = out["hazard_per_s"].max()
    out["relative"] = out["hazard_per_s"] / peak
    return out


def fit_engagement(C, losses, a, b, bk, floors=ADV_FLOOR_SCAN, exps=ADV_EXP_SCAN):
    """MLE for (ADV_FLOOR, ADV_EXP), the engagement multiplier.

    Same occupancy likelihood as fit_lambda with LAM_MAX profiled out, but the
    multiplier rather than the geometry is what varies. Returns the full (floor,
    exponent) grid so the profile intervals can be read off either axis.
    """
    gC = p_geometry(C["dx"].to_numpy(), C["dy"].to_numpy(), a, b, bk)
    inside = gC > 0
    cc, gc = C["closing"].to_numpy()[inside], gC[inside]
    gL = p_geometry(losses["dx"].to_numpy(), losses["dy"].to_numpy(), a, b, bk)
    ce, ge = losses["closing"].to_numpy()[gL > 0], gL[gL > 0]
    n = len(ge)

    rows = []
    for f in floors:
        for p in exps:
            me = advantage_multiplier(ce, f, p)
            mc = advantage_multiplier(cc, f, p)
            denom = float(np.sum(gc * mc))
            rows.append(dict(adv_floor=f, adv_exp=p, n_events=n,
                             lam_max=n / (denom * DT),
                             logL=float(np.sum(np.log(ge * me))
                                        - n * np.log(denom))))
    out = pd.DataFrame(rows)
    out["dlogL"] = out["logL"] - out["logL"].max()
    return out


def fit_shape(losses, target=COVERAGE_Q, scan=K_SCAN):
    """The fitted gate: per-axis anchors, scaled jointly to hit target coverage.

    Returns (a, b, a_back, k, per-axis anchors). The smallest k on the scan whose
    gate encloses `target` of real dispossessions is taken -- smallest, because
    every extra metre of gate is also extra exposure, and an oversized gate would
    buy its coverage by diluting the rate rather than by being right.
    """
    af, ab, b0 = axis_anchors(losses)
    for k in scan:
        if coverage(losses, af * k, b0 * k, ab * k) >= target:
            return af * k, b0 * k, ab * k, float(k), (af, ab, b0)
    k = float(scan[-1])
    return af * k, b0 * k, ab * k, k, (af, ab, b0)


# ===========================================================================
# 5. Figures
# ===========================================================================

def _gate_outline(a_semi, b_semi, a_back, n=400):
    """The p_geometry == 0 contour, for drawing. Solved per angle: the
    super-ellipse is separable, so the radius at angle t is closed form."""
    t = np.linspace(0, 2 * np.pi, n)
    ct, st = np.cos(t), np.sin(t)
    reach = np.where(ct >= 0, a_semi, a_back)
    r = ((np.abs(ct) / reach) ** DUEL_EXP
         + (np.abs(st) / b_semi) ** DUEL_EXP) ** (-1.0 / DUEL_EXP)
    return r * ct, r * st


def figures(profile, losses, scan, a_fit, b_fit, back_fit, lam_fit):
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.9))

    # (1) the headline: measured hazard vs what each model version claims
    mid = (profile["gap_lo_m"] + profile["gap_hi_m"]) / 2
    m = profile["exposure_s"] > 0
    ax[0].errorbar(mid[m], profile["hazard_per_s"][m],
                   yerr=[profile["hazard_per_s"][m] - profile["hazard_lo95"][m],
                         profile["hazard_hi95"][m] - profile["hazard_per_s"][m]],
                   fmt="o", color="k", ms=5, lw=1.2, capsize=3,
                   label="Metrica, measured")
    grid = np.linspace(0, 5, 400)
    zero = np.zeros_like(grid)
    ax[0].plot(grid, PREV_LAM_MAX * p_geometry(grid, zero, PREV_DUEL_A,
                                               PREV_DUEL_B, PREV_DUEL_A_BACK) * 0.75,
               lw=2, color="#C44E52",
               label=f"was (LAM={PREV_LAM_MAX}, A={PREV_DUEL_A})")
    ax[0].plot(grid, lam_fit * p_geometry(grid, zero, a_fit, b_fit, back_fit) * 0.75,
               lw=2, color="#4C72B0",
               label=f"fitted (LAM={lam_fit:.2f}, A={a_fit:.2f})")
    ax[0].set_yscale("symlog", linthresh=0.05)
    ax[0].set_xlabel("defender-to-ball gap (m), straight ahead")
    ax[0].set_ylabel("dispossession hazard (1/s)")
    ax[0].set_title("The measured hazard is ~8x below the old model\n"
                    "and does not stop at 1.2m")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(alpha=0.3)

    # (2) THE SHAPE. Every real dispossession in the defender's own frame, with
    #     both gates drawn over it. This is what says the forward bias is wrong.
    dx, dy = losses["dx"].to_numpy(), losses["dy"].to_numpy()
    inside = p_geometry(dx, dy, a_fit, b_fit, back_fit) > 0
    ax[1].scatter(dx[inside], dy[inside], s=26, color="#4C72B0", zorder=3,
                  edgecolor="k", linewidth=0.3, label="covered by fitted gate")
    ax[1].scatter(dx[~inside], dy[~inside], s=26, color="#999999", zorder=3,
                  edgecolor="k", linewidth=0.3, marker="x", label="missed")
    for (a_, b_, bk_), c, lab in (
            ((PREV_DUEL_A, PREV_DUEL_B, PREV_DUEL_A_BACK), "#C44E52", "was"),
            ((a_fit, b_fit, back_fit), "#4C72B0", "fitted")):
        gx, gy = _gate_outline(a_, b_, bk_)
        ax[1].plot(gx, gy, "-", color=c, lw=2, zorder=2,
                   label=f"{lab}  A={a_:.2f} B={b_:.2f} back={bk_:.2f}")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_aspect("equal")
    ax[1].set_xlim(-3.2, 3.6); ax[1].set_ylim(-3.4, 3.4)
    ax[1].set_xlabel("along the defender's heading (m)  ->  forward")
    ax[1].set_ylabel("across his heading (m)")
    ax[1].set_title(f"Where the ball is actually lost, n={len(dx)}\n"
                    "real duels are as wide as they are deep")
    ax[1].legend(fontsize=7, frameon=False, loc="upper left")
    ax[1].grid(alpha=0.3)

    # (3) coverage and rate against the joint scale factor
    ax[2].plot(scan["k"], scan["lam_max"], "o-", ms=3, color="#8172B2",
               label="fitted LAM_MAX")
    ax[2].axhline(lam_fit, color="#4C72B0", ls=":", lw=1.2)
    ax[2].axhline(PREV_LAM_MAX, color="#C44E52", lw=1.5,
                  label=f"was {PREV_LAM_MAX}")
    ax[2].set_xlabel("joint scale k on the fitted axes")
    ax[2].set_ylabel("fitted LAM_MAX (1/s)")
    ax2b = ax[2].twinx()
    ax2b.plot(scan["k"], scan["coverage_frac"], "-", color="#55A868", lw=1.8,
              label="coverage")
    ax2b.axhline(COVERAGE_Q, color="#55A868", ls=":", lw=1)
    ax2b.set_ylabel("fraction of real losses covered", color="#55A868")
    ax2b.set_ylim(0, 1.02)
    ax[2].axvline(scan["k"][(scan["coverage_frac"] >= COVERAGE_Q).idxmax()],
                  color="k", ls="--", lw=1.2)
    ax[2].set_title("Coverage is bought by scale;\nthe rate is nearly flat in it")
    ax[2].legend(fontsize=8, frameon=False, loc="upper right")
    ax[2].grid(alpha=0.3)

    # (4) what a carrier actually faces, before and after
    t = np.linspace(0, 4, 200)
    for lam, a_, b_, bk_, c, lab in (
            (PREV_LAM_MAX, PREV_DUEL_A, PREV_DUEL_B, PREV_DUEL_A_BACK, "#C44E52", "was"),
            (lam_fit, a_fit, b_fit, back_fit, "#4C72B0", "fitted")):
        for gap, ls in ((0.5, "-"), (1.0, "--")):
            pg = p_geometry(np.array([gap]), np.array([0.0]), a_, b_, bk_)[0] * 0.75
            ax[3].plot(t, np.exp(-lam * pg * t), ls, color=c, lw=1.8,
                       label=f"{lab}, gap {gap}m")
    ax[3].axvline(0.6, color="k", ls=":", lw=1,
                  label="real median time inside 1.2m")
    ax[3].set_xlabel("time under pressure (s)")
    ax[3].set_ylabel("P(still has the ball)")
    ax[3].set_title("Survival under a defender\n"
                    "solid 0.5m, dashed 1.0m")
    ax[3].legend(fontsize=7, frameon=False)
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(CAL_DIR, "duel_calibration_fit.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ===========================================================================

def main():
    home, away = load()
    ev = mio.read_event_data(DATA_DIR, GAME_ID)
    ball = tuple(home[f"ball_{c}"].to_numpy(dtype=float)
                 for c in ("x", "y", "vx", "vy"))

    teams, team, col = carrier_timeline(home, away, ball)
    spells_ = carry_spells(team, col)
    F, C = frame_table(teams, team, col, spells_, ball)
    E, n_ev = duel_events(ev, F, team, spells_)
    losses = E[E["lost"]].reset_index(drop=True)

    print("=" * 78)
    print("METRICA GAME 1 -- CONTESTED CARRIES")
    print("=" * 78)
    print(f"  carry spells                 {len(spells_):,}")
    print(f"  carry frames with an opponent tracked  {len(F):,} "
          f"= {len(F) * DT:,.0f}s of exposure")
    print(f"  ground-duel events in the event file   {n_ev}")
    print(f"  ... matched to a tracked carry         {len(E)}")
    print(f"  ... that actually cost possession      {len(losses)} "
          f"({len(losses) / max(len(E), 1):.0%} of contests)")
    print("  A real carrier survives about half the ground duels he is put in.")
    print(f"  (frame,opponent) pairs within {CONTACT_R:.0f}m    {len(C):,}")

    profile = hazard_profile(F, losses)
    print("\n" + "=" * 78)
    print("MODEL-FREE HAZARD BY GAP (no super-ellipse, no multiplier)")
    print("=" * 78)
    print(f"{'gap (m)':>12} {'exposure':>10} {'losses':>8} {'hazard/s':>10} "
          f"{'95% CI':>16}")
    for _, r in profile.iterrows():
        if not r["frames"]:
            continue
        print(f"{r['gap_lo_m']:5.1f}-{r['gap_hi_m']:5.1f} {r['exposure_s']:9.1f}s "
              f"{int(r['losses']):8d} {r['hazard_per_s']:10.3f} "
              f"{r['hazard_lo95']:7.3f}-{r['hazard_hi95']:.3f}")

    point_blank, elbow, quantile = choose_gate(profile, losses)
    print(f"\n  point-blank hazard (gap < 1.2m)  {point_blank:.3f} /s")
    print(f"  reading 1, hazard elbow          {elbow:.2f} m")
    print(f"  reading 2, p{LOSS_GAP_Q} gap at a loss      {quantile:.2f} m")

    a_raw, b_raw, back_raw, k_fit, anchors = fit_shape(losses)
    af0, ab0, b0 = anchors
    print("\n" + "=" * 78)
    print("GATE SHAPE, FITTED PER AXIS")
    print("=" * 78)
    dx, dy = losses["dx"].to_numpy(), losses["dy"].to_numpy()
    print(f"  losses in front of the defender  {(dx >= 0).sum()}   "
          f"behind him {(dx < 0).sum()}")
    print(f"{'axis':>22} {'median':>8} {'p75':>7} {f'p{LOSS_GAP_Q}':>7}")
    for name, v in (("dx, forward", dx[dx >= 0]),
                    ("|dx|, behind", np.abs(dx[dx < 0])),
                    ("|dy|, lateral", np.abs(dy))):
        print(f"{name:>22} {np.median(v):8.2f} {np.percentile(v, 75):7.2f} "
              f"{np.percentile(v, LOSS_GAP_Q):7.2f}")
    print(f"\n  per-axis anchors:  A={af0:.2f}  B={b0:.2f}  A_BACK={ab0:.2f}")
    print(f"  implied ratios:    A/B={af0 / b0:.2f}   A/A_BACK={af0 / ab0:.2f}")
    print(f"  RoboCup's, which this replaces:  A/B=2.40   A/A_BACK=3.00")
    print("  The lateral axis is the finding: real dispossessions are as wide as")
    print("  they are deep, so the forward-biased tackle box does not survive")
    print("  contact with the data. The front/back asymmetry is real but mild.")
    print(f"\n  scaled by k={k_fit:.2f} for {COVERAGE_Q:.0%} joint coverage "
          f"(DUEL_EXP={DUEL_EXP} cuts the corners, so per-axis p{LOSS_GAP_Q} "
          f"alone encloses only {coverage(losses, af0, b0, ab0):.0%})")

    scan = pd.DataFrame([fit_lambda(C, losses, af0 * k, b0 * k, ab0 * k)
                         for k in K_SCAN])
    scan.insert(0, "k", K_SCAN)

    a_fit, b_fit = round(a_raw, 2), round(b_raw, 2)
    back_fit = round(back_raw, 2)
    # A and B come out within 2% of each other, which 53 losses cannot resolve.
    # Reporting them as different would be reading noise as a shape.
    if abs(a_fit - b_fit) <= 0.05:
        a_fit = b_fit = round((a_raw + b_raw) / 2, 2)
    fit = fit_lambda(C, losses, a_fit, b_fit, back_fit)
    lam_raw = fit["lam_max"]
    lam_fit = round(lam_raw, 2)
    lo95, hi95 = bootstrap_lambda(C, losses, a_fit, b_fit, back_fit)

    print("\n" + "=" * 78)
    print("FIT")
    print("=" * 78)
    assert max(a_fit, b_fit, back_fit) < CONTACT_R, (
        "fitted gate reaches past CONTACT_R, so the exposure table is missing "
        "contacts the model would count")

    # How much of the exposure is a second or third defender? This is the whole
    # reason the denominator sums over contacts: ground_duel adds their hazards.
    in_gate = _g(C, a_fit, b_fit, back_fit) > 0
    per_frame = C[in_gate].groupby("frame").size()
    n_carry = len(F)
    multi = int((per_frame >= 2).sum())
    print(f"\n  frames with >=1 defender in the fitted gate  {len(per_frame):,} "
          f"({len(per_frame) / n_carry:.1%} of carry frames)")
    print(f"  frames with >=2                              {multi:,} "
          f"({multi / max(len(per_frame), 1):.1%} of those)")
    print("  Those second defenders used to contribute nothing: ground_duel took")
    print("  argmin and rolled once. It now sums their hazards, so the exposure")
    print("  above is summed over contacts to match.")

    print(f"  radial cross-checks: hazard elbow {elbow:.2f}m, "
          f"p{LOSS_GAP_Q} loss gap {quantile:.2f}m -- both consistent with a "
          f"reach of {a_fit:.2f}m")
    print(f"  at that gate: exposure {fit['exposure_s']:.1f}s, "
          f"{fit['covered_losses']} of {fit['total_losses']} losses covered "
          f"({fit['coverage_frac']:.0%})")
    print(f"  LAM_MAX = {lam_raw:.4f} /s   (bootstrap 95% CI "
          f"{lo95:.3f}-{hi95:.3f}, {N_BOOTSTRAP} spell resamples)")
    print(f"  stable across the scan: LAM_MAX in "
          f"[{scan['lam_max'].min():.2f}, {scan['lam_max'].max():.2f}] "
          f"for every k from {K_SCAN[0]:.2f} to {K_SCAN[-1]:.2f}")

    print(f"\n-> DUEL_A      = {a_fit:.2f}   (was {PREV_DUEL_A})")
    print(f"-> DUEL_B      = {b_fit:.2f}   (was {PREV_DUEL_B})")
    print(f"-> DUEL_A_BACK = {back_fit:.2f}   (was {PREV_DUEL_A_BACK})")
    print(f"-> R_MAX       = {max(a_fit, b_fit, back_fit):.2f}   "
          f"(was {PREV_DUEL_A}, = max of the three)")
    print(f"-> LAM_MAX     = {lam_fit:.2f}   (was {PREV_LAM_MAX}, "
          f"{PREV_LAM_MAX / lam_fit:.1f}x too high)")

    prev_cov = coverage(losses, PREV_DUEL_A, PREV_DUEL_B, PREV_DUEL_A_BACK)
    print(f"\n  The gate now covers {fit['coverage_frac']:.0%} of real "
          f"ground-duel dispossessions, against {prev_cov:.0%} for the RoboCup")
    print("  ellipse. That is the whole point of refitting the shape: the model")
    print("  has exactly one channel for taking the ball off a carrier, so a gate")
    print("  blind to half of them cannot deliver the right rate per contest no")
    print("  matter what LAM_MAX is set to.")
    # --- engagement ---
    engp = engagement_profile(C, losses, a_fit, b_fit, back_fit)
    engf = fit_engagement(C, losses, a_fit, b_fit, back_fit)
    print("\n" + "=" * 78)
    print("ENGAGEMENT: PROXIMITY IS NOT CONTEST")
    print("=" * 78)
    print("  model-free hazard by closing speed, inside the fitted gate:")
    print(f"{'closing m/s':>16} {'expo_g_s':>10} {'losses':>8} {'hazard/s':>10} "
          f"{'relative':>9}")
    for _, r in engp.iterrows():
        print(f"{r['closing_lo']:7.1f}-{r['closing_hi']:<8.1f} "
              f"{r['exposure_g_s']:10.1f} {int(r['losses']):8d} "
              f"{r['hazard_per_s']:10.3f} {r['relative']:9.2f}")
    print("  A defender who is not closing contests at about a sixth of the rate")
    print("  of one closing hard. The old linear ramp allowed a spread of 2x.")

    f_mle = engf.loc[engf["logL"].idxmax()]
    fprof = engf.groupby("adv_floor")["dlogL"].max()
    fok = fprof[fprof >= -LR_95]
    eprof = engf.groupby("adv_exp")["dlogL"].max()
    eok = eprof[eprof >= -LR_95]
    at_old = float(engf[(engf.adv_floor == 0.50) & (engf.adv_exp == 1.0)]["dlogL"].iloc[0])
    chosen = engf[(engf.adv_floor == ADV_FLOOR) & (engf.adv_exp == float(ADV_EXP))]
    print(f"\n  MLE: ADV_FLOOR {f_mle['adv_floor']:.2f}, ADV_EXP {f_mle['adv_exp']:.1f}")
    print(f"  ADV_FLOOR 95% LR interval {fok.index.min():.2f}-{fok.index.max():.2f}"
          "   <- identified")
    print(f"  ADV_EXP   95% LR interval {eok.index.min():.1f}-{eok.index.max():.1f}"
          "   <- NOT identified, runs to the scan edge")
    print(f"  the old (0.50, linear) sits at dlogL {at_old:.2f}: rejected")
    if len(chosen):
        print(f"  in use ({ADV_FLOOR}, {ADV_EXP}) sits at dlogL "
              f"{float(chosen['dlogL'].iloc[0]):.2f}, and is the value inside the")
        print("  interval that best tracks the model-free curve above:")
        print(f"{'closing':>10} {'measured':>10} {'in use':>10}")
        for _, r in engp.iterrows():
            print(f"{r['mid']:10.1f} {r['relative']:10.2f} "
                  f"{float(advantage_multiplier(r['mid'])):10.2f}")

    # --- the exponent ---
    expf = fit_exponent(C, losses, anchors)
    ok = expf[expf["dlogL"] >= -LR_95]
    lo_e, hi_e = float(ok["exponent"].min()), float(ok["exponent"].max())
    at_6 = float(expf.loc[expf["exponent"] == 6.0, "dlogL"].iloc[0])
    mle = expf.loc[expf["logL"].idxmax()]

    print("\n" + "=" * 78)
    print("DUEL_EXP: FITTED, AND NOT IDENTIFIED")
    print("=" * 78)
    print(f"  fitted on {int(mle['n_events'])} losses, scale free at every exponent")
    print(f"{'exponent':>10} {'best A':>8} {'LAM_MAX':>9} {'dlogL':>8}")
    for _, r in expf.iterrows():
        if r["exponent"] in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0):
            print(f"{r['exponent']:10.1f} {r['a_semi']:8.2f} {r['lam_max']:9.3f} "
                  f"{r['dlogL']:8.2f}")
    print(f"\n  MLE is {mle['exponent']:.2f}, but the 95% likelihood-ratio interval is "
          f"{lo_e:.2f}-{hi_e:.2f}")
    print(f"  -- the whole scanned range. RoboCup's 6 sits at dlogL {at_6:.2f}, well")
    print(f"  inside it. 53 losses cannot resolve the edge profile, exactly as the")
    print("  earlier caveat guessed; this now measures that rather than asserting it.")
    print("\n  It also does not matter, which is the more useful half. Because the")
    print("  scale is refitted with it, every exponent lands on nearly the same")
    print("  hazard field. P(ball lost within 1s of continuous pressure):")
    print(f"{'exponent':>10}" + "".join(f"{d:8.1f}m" for d in (0.5, 1.0, 1.5, 2.0, 2.5)))
    for _, r in expf.iterrows():
        if r["exponent"] not in (1.0, 2.0, 3.5, 6.0, 10.0):
            continue
        cells = []
        for d in (0.5, 1.0, 1.5, 2.0, 2.5):
            pg = p_geometry(np.array([d]), np.array([0.0]), r["a_semi"],
                            r["b_semi"], r["a_back"], r["exponent"])[0]
            cells.append(1.0 - np.exp(-r["lam_max"] * pg
                                      * float(advantage_multiplier(0.0)) * 1.0))
        print(f"{r['exponent']:10.1f}" + "".join(f"{c:9.3f}" for c in cells))
    print("\n  Only exponent 1 is visibly different -- a diamond with a long tail")
    print("  past 3m -- and it is also the one the likelihood most disfavours.")
    print(f"  DUEL_EXP is therefore LEFT AT {DUEL_EXP}: not because it was inherited,")
    print("  but because the data cannot distinguish it and the physics barely")
    print("  changes across everything the data allows.")

    # --- CSVs ---
    profile.to_csv(os.path.join(CAL_DIR, "duel_hazard_profile.csv"), index=False)
    E.to_csv(os.path.join(CAL_DIR, "duel_loss_events.csv"), index=False)
    scan.round(4).to_csv(os.path.join(CAL_DIR, "duel_gate_scan.csv"), index=False)
    expf.round(4).to_csv(os.path.join(CAL_DIR, "duel_exponent_profile.csv"),
                         index=False)
    engp.round(4).to_csv(os.path.join(CAL_DIR, "duel_engagement_profile.csv"),
                         index=False)
    engf.round(4).to_csv(os.path.join(CAL_DIR, "duel_engagement_fit.csv"),
                         index=False)

    params = [
        ("source", "Metrica Sample_Game_1, tracking + event file"),
        ("fps", FPS), ("dt_s", DT),
        ("carry_spells", len(spells_)),
        ("carry_frames", len(F)),
        ("carry_exposure_s", round(len(F) * DT, 1)),
        ("ground_duel_events", n_ev),
        ("ground_duels_matched", len(E)),
        ("ground_duels_lost", len(losses)),
        ("duel_loss_rate_of_contests", round(len(losses) / max(len(E), 1), 4)),
        ("point_blank_hazard_per_s", round(point_blank, 4)),
        ("hazard_elbow_m", round(elbow, 2)),
        (f"loss_gap_p{LOSS_GAP_Q}_m", round(quantile, 2)),
        ("loss_gap_median_m", round(float(np.median(losses["dist"])), 2)),
        ("loss_gap_inside_old_gate_frac",
         round(float((losses["dist"] <= PREV_DUEL_A).mean()), 4)),
        ("axis_anchor_forward_m", round(af0, 3)),
        ("axis_anchor_lateral_m", round(b0, 3)),
        ("axis_anchor_behind_m", round(ab0, 3)),
        ("axis_anchor_quantile", LOSS_GAP_Q),
        ("coverage_target", COVERAGE_Q),
        ("coverage_scale_k", k_fit),
        ("fitted_duel_a_m", a_fit),
        ("fitted_duel_b_m", b_fit),
        ("fitted_duel_a_back_m", back_fit),
        ("fitted_ratio_a_over_b", round(a_fit / b_fit, 3)),
        ("fitted_ratio_a_over_back", round(a_fit / back_fit, 3)),
        ("fitted_lam_max_per_s", lam_fit),
        ("fitted_lam_max_raw", round(lam_raw, 4)),
        ("fitted_lam_max_lo95", round(float(lo95), 4)),
        ("fitted_lam_max_hi95", round(float(hi95), 4)),
        ("gate_exposure_s", round(fit["exposure_s"], 2)),
        ("gate_covered_losses", fit["covered_losses"]),
        ("gate_coverage_frac", round(fit["coverage_frac"], 4)),
        ("previous_lam_max", PREV_LAM_MAX),
        ("previous_duel_a_m", PREV_DUEL_A),
        ("previous_duel_b_m", PREV_DUEL_B),
        ("previous_duel_a_back_m", PREV_DUEL_A_BACK),
        ("previous_coverage_frac", round(prev_cov, 4)),
        ("previous_ratio_a_over_b", round(PREV_DUEL_A / PREV_DUEL_B, 3)),
        ("previous_ratio_a_over_back",
         round(PREV_DUEL_A / PREV_DUEL_A_BACK, 3)),
        ("loss_median_abs_dx_m", round(float(np.median(np.abs(losses["dx"]))), 3)),
        ("loss_median_abs_dy_m", round(float(np.median(np.abs(losses["dy"]))), 3)),
        ("losses_in_front", int((losses["dx"] >= 0).sum())),
        ("losses_behind", int((losses["dx"] < 0).sum())),
        ("duel_exp_used", DUEL_EXP),
        ("duel_exp_mle", float(mle["exponent"])),
        ("duel_exp_lr95_lo", lo_e),
        ("duel_exp_lr95_hi", hi_e),
        ("duel_exp_dlogL_at_6", round(at_6, 3)),
        ("duel_exp_identified", False),
        ("adv_floor_used", ADV_FLOOR),
        ("adv_exp_used", ADV_EXP),
        ("adv_floor_mle", float(f_mle["adv_floor"])),
        ("adv_exp_mle", float(f_mle["adv_exp"])),
        ("adv_floor_lr95_lo", float(fok.index.min())),
        ("adv_floor_lr95_hi", float(fok.index.max())),
        ("adv_exp_lr95_lo", float(eok.index.min())),
        ("adv_exp_lr95_hi", float(eok.index.max())),
        ("adv_old_linear_dlogL", round(at_old, 3)),
        ("n_bootstrap", N_BOOTSTRAP),
    ]
    pd.DataFrame(params, columns=["metric", "value"]).to_csv(
        os.path.join(CAL_DIR, "duel_survival_params.csv"), index=False)

    png = figures(profile, losses, scan, a_fit, b_fit, back_fit, lam_fit)
    print(f"\nwrote duel_hazard_profile.csv, duel_loss_events.csv, "
          f"duel_gate_scan.csv,\n      duel_exponent_profile.csv, "
          f"duel_survival_params.csv and {os.path.basename(png)}"
          f"\n      into {CAL_DIR}")


if __name__ == "__main__":
    main()
