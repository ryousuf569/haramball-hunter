import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

RADIUS_COLORS = ("#9ecae1", "#6baed6", "#3182bd", "#08519c")

# Only the scripted band is a real gate; see the note at the bottom of main()
SCRIPTED_BAND = (0.30, 0.50)


def load(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return list(csv.DictReader(f))


def plot_arrival(summary):
    # arrival_rate does not depend on theta, so collapse to one row per disc
    arr = {}
    for r in summary:
        arr[(r["policy"], float(r["x"]), float(r["radius"]))] = float(r["arrival_rate"])
    xs = sorted({k[1] for k in arr})
    radii = sorted({k[2] for k in arr})

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for col, pol in enumerate(("random", "scripted")):
        for i, x in enumerate(xs):
            ys = [arr[(pol, x, rr)] for rr in radii]
            axes[0][col].plot(radii, ys, "-o", color=RADIUS_COLORS[i],
                              label=f"x = {x:.0f}")
            # same numbers against the near edge: if x - r is what sets arrival,
            # the four curves collapse onto one
            axes[1][col].plot([x - rr for rr in radii], ys, "-o",
                              color=RADIUS_COLORS[i], label=f"x = {x:.0f}")
        axes[0][col].set_title(pol)
        axes[0][col].set_xlabel("disc radius (m)")
        axes[1][col].set_xlabel("near edge x - r (m)")
        # where defenders.py holds its midfield and back lines
        axes[1][col].axvline(74, color="grey", linestyle=":")
        axes[1][col].axvline(84, color="grey", linestyle=":")
        for ax in (axes[0][col], axes[1][col]):
            ax.grid(alpha=0.3)

    axes[0][0].set_ylabel("ball ever arrives held")
    axes[1][0].set_ylabel("ball ever arrives held")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Arrival rate (geometry half). Bottom row: does it collapse "
                 "on near edge?")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "zone_probe_arrival.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)


def plot_theta(summary):
    xs = sorted({float(r["x"]) for r in summary})
    radii = sorted({float(r["radius"]) for r in summary})
    fig, axes = plt.subplots(1, len(xs), figsize=(13, 3.8), sharey=True)

    for ax, x in zip(axes, xs):
        for i, rr in enumerate(radii):
            for pol, style in (("scripted", "-"), ("random", "--")):
                pts = sorted((float(s["theta"]), float(s["success_rate"]))
                             for s in summary
                             if s["policy"] == pol and float(s["x"]) == x
                             and float(s["radius"]) == rr)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        color=RADIUS_COLORS[i],
                        label=f"r = {rr:.0f}" if pol == "scripted" else None)
        ax.axhspan(*SCRIPTED_BAND, color="orange", alpha=0.12)
        ax.set_title(f"zone x = {x:.0f}")
        ax.set_xlabel("theta")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Success vs theta -- solid scripted, dashed random, "
                 "band = scripted target")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "zone_probe_theta.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)


def plot_decision(ranked):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.axhspan(*SCRIPTED_BAND, color="orange", alpha=0.12)

    inb = [d for d in ranked
           if SCRIPTED_BAND[0] <= d["scripted"] <= SCRIPTED_BAND[1]]
    out = [d for d in ranked if d not in inb]
    ax.scatter([d["random"] for d in out], [d["scripted"] for d in out],
               s=26, color="grey", alpha=0.4, label="outside target")
    ax.scatter([d["random"] for d in inb], [d["scripted"] for d in inb],
               s=52, color="darkorange", edgecolor="white",
               label="scripted in target", zorder=3)

    # label the shortlist only
    for d in ranked[:5]:
        ax.annotate(f"x{d['x']:.0f} r{d['radius']:.0f} th{d['theta']:.2f}",
                    (d["random"], d["scripted"]), textcoords="offset points",
                    xytext=(7, 3), fontsize=7.5)

    ax.set_xlabel("random success rate")
    ax.set_ylabel("scripted success rate")
    ax.set_title("Every (x, r, theta): the skill gap")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "zone_probe_decision.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)


def rank(summary):
    cells = {}
    for s in summary:
        key = (float(s["x"]), float(s["y"]), float(s["radius"]),
               float(s["theta"]))
        cells.setdefault(key, {})[s["policy"]] = (float(s["success_rate"]),
                                                 float(s["arrival_rate"]))
    out = []
    for (x, y, r, th), by in cells.items():
        if "random" not in by or "scripted" not in by:
            continue
        out.append({"x": x, "y": y, "radius": r, "theta": th,
                    "random": by["random"][0], "scripted": by["scripted"][0],
                    "scripted_arrival": by["scripted"][1]})
    # scripted inside the band first, then the widest gap over random
    out.sort(key=lambda d: (
        not (SCRIPTED_BAND[0] <= d["scripted"] <= SCRIPTED_BAND[1]),
        -(d["scripted"] - d["random"])))
    return out


def main():
    summary = load("zone_probe_summary.csv")
    ranked = rank(summary)

    plot_arrival(summary)
    plot_theta(summary)
    plot_decision(ranked)

    ys = sorted({float(s["y"]) for s in summary})
    print(f"\ny swept: {ys} -- nothing to choose if there is only one")
    print(f"target: scripted in {SCRIPTED_BAND}, random low but non-zero\n")
    print(f"{'x':>4} {'r':>4} {'theta':>6} {'random':>8} {'scripted':>9} "
          f"{'gap':>7} {'scr arr':>8}")
    for d in ranked[:12]:
        print(f"{d['x']:4.0f} {d['radius']:4.0f} {d['theta']:6.2f} "
              f"{d['random']:8.1%} {d['scripted']:9.1%} "
              f"{d['scripted'] - d['random']:7.1%} {d['scripted_arrival']:8.1%}")


if __name__ == "__main__":
    main()
