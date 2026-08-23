import argparse
import csv
import io
import os

import matplotlib.pyplot as plt

X = "step"
COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def read_metrics(path):
    with io.open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("no rows in %s" % path)
    cols = {}
    for name in rows[0]:
        cols[name] = [float(r[name]) if r[name] not in ("", "nan") else float("nan")
                      for r in rows]
    return cols


def smooth(y, window):
    if window <= 1:
        return y
    out = []
    for i in range(len(y)):
        lo = max(0, i - window + 1)
        chunk = [v for v in y[lo:i + 1] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def plot(path, columns, window=1, out=None, title=None):
    cols = read_metrics(path)
    if X not in cols:
        raise ValueError("%s has no '%s' column" % (path, X))
    missing = [c for c in columns if c not in cols]
    if missing:
        raise ValueError("no column %s in %s; available: %s"
                         % (missing, path, ", ".join(k for k in cols if k != X)))
    if not 1 <= len(columns) <= 3:
        raise ValueError("plot 1 to 3 columns, got %d" % len(columns))

    x = cols[X]
    fig, axes = plt.subplots(len(columns), 1, sharex=True, squeeze=False,
                             figsize=(9, 2.6 * len(columns)))
    axes = [a[0] for a in axes]

    for ax, name, color in zip(axes, columns, COLORS):
        y = cols[name]
        if window > 1:
            ax.plot(x, y, color=color, alpha=0.25, linewidth=1.0)
            ax.plot(x, smooth(y, window), color=color, linewidth=1.8)
        else:
            ax.plot(x, y, color=color, linewidth=1.4)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        finite = [v for v in y if v == v]
        if finite:
            ax.set_title("%s: last %.4g | min %.4g | max %.4g"
                         % (name, finite[-1], min(finite), max(finite)),
                         fontsize=8, loc="left")

    axes[-1].set_xlabel(X)
    fig.suptitle(title or os.path.dirname(os.path.abspath(path)).split(os.sep)[-1])
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=140)
        print("wrote %s" % out)
    else:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("columns", nargs="+")
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    plot(args.csv, args.columns, window=args.smooth,
         out=args.out or None, title=args.title or None)
