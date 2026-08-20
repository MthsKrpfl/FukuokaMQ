#!/usr/bin/env python3
"""
Two diagrams for one instance size, sharing the same k axis.

  A  the measured GB times over k with an exponential trend through them, plus
     the guess count q^k on its own axis -- the two factors of the cost.
  B  their product q^k * t(k), the U-shaped cost curve, with k* marked at its
     minimum and the measured minimum beside it.

Reads one group file, results/type<T>/<label>_measurements.csv, and writes
figures/type<T>/t<T>q<q>n<n>m<m>.png. Without arguments it walks every group
under results/.

params:
  input           ; val is a group csv, e.g. results/type1/n55_measurements.csv
  --file val      ; val is an instance file, whose group is plotted instead
  --cutoff-k val  ; val is the largest k used for the trend fit
  --degree val    ; val is the degree of the trend polynomial, 1 to 6
  --out val       ; val is the png to write, single group only
  --force         ; redraw even when the figure is newer than the data
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, LogLocator,
                               MaxNLocator, NullFormatter)

sys.path.insert(0, str(Path(__file__).resolve().parent / "subprograms"))
import trendline as T
import bookkeeping

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

C_TYPE = {1: "#4C9BE8", 4: "#4C9BE8",
          2: "#F4A24C", 5: "#F4A24C",
          3: "#5FBF8F", 6: "#5FBF8F"}
C_MEAS = "#4C9BE8"
C_QK = "#F4A24C"
C_COST = "#8E6FD8"
C_MARK = "#999999"
C_ARGMIN = "#D94141"
C_TIMEOUT = "#E8E8E8"


# accept windows style paths as well
def _norm(p: str) -> str:
    return p.replace("\\", "/") if p else p


def _finite(ks, ys):
    xs, vs = [], []
    for k, y in zip(ks, ys):
        if math.isfinite(y) and y > 0:
            xs.append(k)
            vs.append(y)
    return xs, vs


# one look for both panels
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.35)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(left=0)
    _log_ticks(ax)


# decade ticks that survive a span of many decades
def _log_ticks(ax):
    lo, hi = ax.get_ylim()
    if not (lo > 0 and hi > lo):
        return
    dec = math.log10(hi) - math.log10(lo)
    if dec <= 8:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
    else:
        step = max(1, int(math.ceil(dec / 8.0)))
        ax.yaxis.set_major_locator(LogLocator(base=10.0 ** step, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, numticks=int(dec) + 2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)


# argmin of the cost over the k actually measured
def measured_kstar(med, q):
    lq = math.log(q)
    return min(med, key=lambda k: k * lq + math.log(med[k]))


# fit up to the measured cost minimum: beyond it the curve bends the other way
def choose_cutoff(med, n, q, degree=2, minpts=6):
    ks = sorted(med)
    kmeas = measured_kstar(med, q)
    if kmeas in (ks[0], ks[-1]) or degree != 2:
        return T.auto_cutoff(med, 1.5), kmeas
    lq = math.log(q)
    best = None
    for cut in range(ks[-1], kmeas, -1):
        elig = [k for k in ks if k <= cut]
        if len(elig) < minpts:
            continue
        co = T.polyfit_vertex([float(k) for k in elig],
                              [math.log(med[k]) for k in elig], n)
        if co is None or co[2] <= 0:
            continue
        tks = min(range(1, n + 1), key=lambda k: k * lq + T.polyval(co, k))
        score = (abs(tks - kmeas), -len(elig))
        if best is None or score < best[0]:
            best = (score, cut)
    if best is None:
        return T.auto_cutoff(med, 1.5), kmeas
    return best[1], kmeas


# n55 out of results/type1/n55_measurements.csv
def group_label(path: Path) -> str:
    return path.stem.replace("_measurements", "")


# the group file an instance file belongs to
def group_files_for_file(file_path: str):
    fp = Path(_norm(file_path))
    info = bookkeeping.parse_name(fp.stem)
    nmq = bookkeeping.peek_header_nmq(fp)
    if info is None or nmq is None:
        return []
    typ = info[0]
    n, m, _q = nmq
    cands = [RESULTS / f"type{typ}" / f"n{n}_measurements.csv",
             RESULTS / f"type{typ}" / f"m{m}_measurements.csv"]
    return [p for p in cands if p.is_file()]


# every group file under results/
def all_group_files():
    return sorted(RESULTS.glob("type*/*_measurements*.csv"))


# read one group, fit the trend, draw both panels
def plot_group(path: Path, degree, cutoff_k, force, out=None):
    rows, meta = T.read_points(path)
    if not rows:
        return None, "empty"
    n, m, q = meta["n"], meta["m"], meta["q"]

    if out is None:
        out = (FIGURES / f"type{meta['type']}"
               / f"t{meta['type']}q{q}n{n}m{m}.png")
    out = Path(out)
    if out.exists() and not force:
        return out, "skip"

    med = {k: statistics.median([r["t"] for r in rows if r["k"] == k])
           for k in {r["k"] for r in rows}}
    if len(med) < 2:
        return out, "few"

    if cutoff_k is not None:
        cutoff = cutoff_k
    else:
        cutoff = max(med)
    elig = sorted(k for k in med if k <= cutoff)
    deg = degree if len(elig) >= degree + 2 else 1
    if len(elig) < deg + 2:
        return out, "few"
    xs = [float(k) for k in elig]
    ys = [math.log(med[k]) for k in elig]
    floor, coeffs, rmse = T.fit_floor(xs, ys, deg)
    curv = T.fit_floor(xs, ys, 2)[1][2] if len(xs) >= 4 else 0.0
    j = 2 if len(xs) >= 6 else 0
    T_trend = T.make_trend(floor, coeffs, min(xs), max(xs), curv=curv,
                           join_lo=xs[j], join_hi=xs[-1])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"wspace": 0.25})

    kb, yb = _finite(range(0, n + 1),
                     [float(q) ** k * T_trend(k) for k in range(0, n + 1)])
    kstar = min(med, key=lambda k: k * math.log(q) + math.log(med[k]))
    kstar_trend = kb[min(range(len(yb)), key=lambda i: yb[i])]
    k_measured_lo = min(med)
    censored = kstar <= k_measured_lo

    c_meas = C_TYPE.get(int(meta["type"]), C_MEAS)
    axA.plot(elig, [med[k] for k in elig], "o", ms=3.5, color=c_meas,
             label="measured (median)")
    ktr, ytr = _finite(range(0, n + 1), [T_trend(k) for k in range(0, n + 1)])
    axA.plot(ktr, ytr, ":", lw=1.0, color=c_meas, label="trend")
    axA.set_xlabel("$k$")
    axA.set_ylabel("$t_{GB}(k)$ [s]")
    axA.legend(frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), ncol=2)
    _style(axA)

    axB.plot(kb, yb, "-", lw=1.0, color=C_COST, label="trend cost")
    axB.plot(elig, [float(q) ** k * med[k] for k in elig], "o", ms=3.5,
             color=C_COST, label="measured cost")
    axB.plot([kstar], [float(q) ** kstar * med[kstar]], "o", ms=7.0,
             color=C_ARGMIN, mec="white", mew=0.8, zorder=5,
             label=f"measured min $k$"
                   + (" (edge of data)" if censored else ""))
    axB.set_xlabel("$k$")
    axB.set_ylabel("$T(k)$")
    axB.axvline(kstar_trend, ls=":", lw=1.2, color=C_MARK, zorder=3,
                label=f"trend $k^*$")
    _style(axB)
    axB.legend(frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), ncol=2)

    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] {group_label(path)}: trend k*={kstar_trend}, measured k*={kstar} "
          f"(k {min(med)}..{max(med)}, fit k≤{cutoff} deg {deg})")
    return out, "ok"


# one group, one instance file, or everything
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=None,
                    help="a single group csv, e.g. results/type1/n55_measurements.csv; "
                         "omit it and --file to plot every group")
    ap.add_argument("--file", default=None,
                    help="instance path (sweep format, backslashes/.bz2 ok); "
                         "plot only that instance's group")
    ap.add_argument("--cutoff-k", type=int, default=None,
                    help="eligible points are k <= this; auto-detected if omitted")
    ap.add_argument("--degree", type=int, default=2, choices=(1, 2, 3, 4, 5, 6),
                    help="degree of the polynomial in the exponent of the trend "
                         "t(k) = c + exp(p(k)). 2 (default) is the smooth shape; "
                         "5-6 follows the local structure of the measured points "
                         "as well, at the price of describing one sweep's noise "
                         "rather than a reproducible feature")
    ap.add_argument("--out", default=None,
                    help="figure path, single group only, default "
                         "figures/type<T>/t<T>q<q>n<n>m<m>.png")
    ap.add_argument("--force", action="store_true",
                    help="re-plot even if the target PNG already exists")
    args = ap.parse_args(argv)

    if args.file:
        groups = group_files_for_file(args.file)
        if not groups:
            print(f"[plot] no results group found for {args.file}", file=sys.stderr)
            return 1
    elif args.input:
        groups = [Path(_norm(args.input))]
    else:
        groups = all_group_files()
        if not groups:
            print("[plot] no measurement groups under results/", file=sys.stderr)
            return 1

    targeted = bool(args.file or args.input)
    force = args.force or targeted
    n_ok = n_skip = n_bad = 0
    for src in groups:
        out = Path(args.out) if (targeted and args.out) else None
        try:
            path_out, status = plot_group(src, args.degree, args.cutoff_k, force, out)
        except Exception as e:
            print(f"[plot] {src}: FAILED ({e})", file=sys.stderr)
            n_bad += 1
            continue
        if status == "skip":
            print(f"[plot] {group_label(src)}: exists, skipped -> {path_out}")
            n_skip += 1
        elif status in ("empty", "few"):
            print(f"[plot] {group_label(src)}: {status}, skipped", file=sys.stderr)
            n_bad += 1
        else:
            print(f"[plot] wrote {path_out}")
            n_ok += 1

    if len(groups) > 1:
        print(f"[plot] {n_ok} written, {n_skip} skipped (exist), {n_bad} empty/failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
