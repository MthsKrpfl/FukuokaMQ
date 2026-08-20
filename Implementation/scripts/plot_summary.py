#!/usr/bin/env python3
"""
The figures that span instances instead of belonging to one size, all written to
figures/summary/:

  guesses_qk.png            the worst case guess count q^k for q = 2, 31, 256
  kstar_over_n_type<T>.png  k* over the instance size, one figure per type
  scaling_encryption.png    the three encryption types at one size, and
  scaling_signature.png     the three signature types at one size
  scaling_types<a>_<b>.png  measured GB time over k, encryption next to
                            signature for the same field, three sizes each

The scaling figures use the measurements directly -- the median time of every
instance in scope at every measured k -- so what they show is the growth itself,
not a derived k*.

params:
  --decades val       ; val is how many decades the q^k figure spans
  --kmax val          ; val is the largest k in the q^k figure
  --scope val         ; val is toy, challenge or all, for the scaling figures
  --kstar-scope val   ; val is toy, challenge or all, for the k* figures
  --degree-trend val  ; val is the trend degree, matching plot_instance.py
  --enc-size val      ; val is the n used for the encryption comparison
  --sig-size val      ; val is the m used for the signature comparison
  --free-scale        ; let every pair figure pick its own time axis
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, FuncFormatter,
                               LogLocator, MaxNLocator, MultipleLocator,
                               NullFormatter)

sys.path.insert(0, str(Path(__file__).resolve().parent / "subprograms"))

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "summary"

C_Q = {2: "#4C9BE8", 31: "#5FBF8F", 256: "#F4A24C"}
C_TYPE = {1: "#4C9BE8", 2: "#F4A24C", 3: "#5FBF8F",
          4: "#4C9BE8", 5: "#F4A24C", 6: "#5FBF8F"}

ENCRYPTION = (1, 2, 3)
SIGNATURE = (4, 5, 6)


# one look for every figure here
def _style(ax):
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.35)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# decade ticks that stay readable over a large span
def _log_yticks(ax):
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
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)


# times per type, size and k, read from the results tree, ok rows only
def read_measurements(types, scope="toy"):
    out = defaultdict(lambda: defaultdict(list))
    for typ in types:
        for p in sorted((RESULTS / f"type{typ}").glob("*measurements*.csv")):
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("status") != "ok":
                        continue
                    toy = str(r.get("instance_id", "")).lower().startswith("toy")
                    if scope == "toy" and not toy:
                        continue
                    if scope == "challenge" and toy:
                        continue
                    try:
                        k = int(r["k"])
                        n = int(r["n"])
                        t = float(r["gb_time_s"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if t <= 0:
                        continue
                    out[(typ, n)][k].append(t)
    return out


# q^k over k: instance independent, so it is shown once here
def plot_guesses(decades: int, kmax: int | None, out: Path):
    if kmax is None:
        kmax = int(math.ceil(decades / math.log10(2)))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ks = [k * 0.05 for k in range(0, int(kmax / 0.05) + 1)]
    for q in (2, 31, 256):
        ax.plot(ks, [float(q) ** k for k in ks], "-", lw=1.8, color=C_Q[q],
                label=f"$q={q}$")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$q^k$")
    ax.set_xlim(0, kmax)
    ax.set_ylim(1, 10.0 ** decades)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=decades + 1))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=10 * decades))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: "1" if y == 1 else rf"$10^{{{int(round(math.log10(y)))}}}$"))
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[summary] {out.relative_to(ROOT)}  (q^k, k=0..{kmax}, 10^{decades} ceiling)")


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


# one shade per size, same hue per type
def _shades(hexcolor: str, count: int):
    r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    out = []
    for i in range(count):
        f = 0.45 + 0.55 * (i / max(1, count - 1))
        out.append((r + (1 - r) * (1 - f), g + (1 - g) * (1 - f),
                    b + (1 - b) * (1 - f)))
    return out


def _q_of(typ: int) -> int:
    return {1: 2, 4: 2, 2: 256, 5: 256, 3: 31, 6: 31}[typ]


def _size_label(typ: int) -> str:
    return "n" if typ in ENCRYPTION else "m"


# what one guessed variable buys, per step
def _growth(ks, med) -> float:
    if len(ks) < 2:
        return float("nan")
    b = np.polyfit(np.array(ks, float), np.log(np.array(med, float)), 1)[0]
    return float(math.exp(-b))


def _xstep(xmax: float) -> int:
    for step in (2, 5, 10, 20, 25, 50):
        if xmax / step <= 12:
            return step
    return 100


def _medians(by_k):
    ks = sorted(by_k)
    return ks, [statistics.median(by_k[k]) for k in ks]


# one line per instance size
def _draw_type(ax, data, typ, shaded: bool, sizes=None):
    have = sorted(n for (t, n) in data if t == typ)
    if sizes is not None:
        have = [n for n in have if n in sizes]
    cols = _shades(C_TYPE[typ], len(have)) if shaded else [C_TYPE[typ]] * len(have)
    xmax, ymin, ymax = 0, math.inf, 0.0
    for i, n in enumerate(have):
        ks, med = _medians(data[(typ, n)])
        lab = f"{_size_label(typ)}={n}" if shaded else \
            f"type {typ} ($q={_q_of(typ)}$)  {_size_label(typ)}={n}"
        ax.plot(ks, med, "-", lw=1.8, color=cols[i],
                label=f"{lab}")
        xmax = max(xmax, n)
        ymin = min(ymin, min(med))
        ymax = max(ymax, max(med))
    return xmax, ymin, ymax


def _finish(ax, xmax, ymin, ymax, ylim=None, ylabel=True):
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    if ylabel:
        ax.set_ylabel("$t_{GB}(k)$ [s]")
    ax.set_xlim(0, xmax)
    if ylim:
        ax.set_ylim(*ylim)
    elif math.isfinite(ymin) and ymax > 0:
        ax.set_ylim(10.0 ** math.floor(math.log10(ymin)),
                    10.0 ** math.ceil(math.log10(ymax)))
    ax.xaxis.set_major_locator(MultipleLocator(_xstep(max(1.0, xmax))))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    _log_yticks(ax)
    _style(ax)


# the k* summary, if a figure wants it
def read_estimates(path: Path):
    if not path.exists():
        sys.exit(f"no {path} -- run scripts/estimate_kstar.py first")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"type": int(r["type"]), "n": int(r["n"]),
                             "m": int(r["m"]), "q": int(r["q"]),
                             "kstar": int(r["kstar"]),
                             "method": r.get("method", ""),
                             "reach": r.get("reach", ""),
                             "note": r.get("note", "")})
            except (KeyError, ValueError, TypeError):
                continue
    return rows


# measured argmin and trend minimum, censored means argmin at the smallest k
def _kstars(med, n, q, deg):
    import trendline as T
    ks = sorted(med)
    xs = [float(k) for k in ks]
    ys = [math.log(med[k]) for k in ks]
    lq = math.log10(q)
    kmeas = min(ks, key=lambda k: k * lq + math.log10(med[k]))
    d = deg if len(xs) >= deg + 2 else 1
    floor, co, _ = T.fit_floor(xs, ys, d)
    curv = T.fit_floor(xs, ys, 2)[1][2] if len(xs) >= 4 else 0.0
    j = 2 if len(xs) >= 6 else 0
    f = T.make_trend(floor, co, min(xs), max(xs), curv=curv,
                     join_lo=xs[j], join_hi=xs[-1])
    ktrend = min(range(0, n), key=lambda k: k * lq + math.log10(f(k)))
    return kmeas, ktrend, ks[0]


# k* over the instance size for one type
def plot_kstar(typ, out: Path, scope="toy", degree=6, tol=3):
    measured_line = typ in (1, 4)
    if not measured_line:
        scope = "all"
    data = read_measurements((typ,), scope=scope)
    if not data:
        print(f"[summary] {out.name}: no measurements in scope '{scope}'")
        return
    q = _q_of(typ)
    rows = []
    for (t, n), by_k in sorted(data.items(), key=lambda kv: kv[0][1]):
        med = {k: statistics.median(v) for k, v in by_k.items()}
        if len(med) < 4:
            continue
        km, kt, kmin = _kstars(med, n, q, degree)
        rows.append((n, km if measured_line else kt,
                     kt if measured_line else km, kmin))
    if not rows:
        print(f"[summary] {out.name}: too few points")
        return
    kind = "encryption" if typ in ENCRYPTION else "signature"
    c = C_TYPE[typ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "-", lw=1.8, color=c,
            label="measured $k^{*}$" if measured_line else "predicted $k^{*}$")
    near = [] if measured_line else [r for r in rows
                                     if r[3] == 0
                                     and r[2] != r[1]
                                     and abs(r[2] - r[1]) <= tol]
    for n, main, other, _ in near:
        ax.plot([n, n], [main, other], "-", lw=0.9, color=c, alpha=0.55)
    if near:
        ax.plot([r[0] for r in near], [r[2] for r in near], "o", ms=3.0,
                color=c, mec=c,
                label="measured $k^{*}$ (complete sweep, "
                      f"within $\\pm${tol})")
    ax.set_xlabel(f"${_size_label(typ)}$")
    ax.set_ylabel("$k^{*}$")
    ax.set_title(f"type {typ} - {kind}", fontsize=10)
    ns = [r[0] for r in rows]
    pad = max(1, round(0.04 * (max(ns) - min(ns)))) if len(ns) > 1 else 1
    ax.set_xlim(min(ns) - pad, max(ns) + pad)
    if measured_line:
        ymax = max(r[1] for r in rows)
        ax.set_ylim(-0.04 * max(ymax, 1), None)
    else:
        ax.set_ylim(-0.32, 8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", length=5, width=0.8)
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[summary] {out.relative_to(ROOT)}  ({len(rows)} sizes, "
          f"{'measured' if measured_line else 'predicted'} line, "
          f"{len(near)} within +-{tol})")


# the three types of one group at the same size, colour by field
def plot_fixed_size(types, size, out: Path, scope="toy"):
    data = read_measurements(types, scope=scope)
    drawn = 0
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xmax, ymin, ymax = 0, math.inf, 0.0
    for typ in types:
        by_k = data.get((typ, size))
        if not by_k:
            print(f"[summary] {out.name}: no size {size} for type {typ}")
            continue
        ks, med = _medians(by_k)
        ax.plot(ks, med, "-o", lw=1.8, ms=4.5, color=C_TYPE[typ],
                mec=C_TYPE[typ], label=f"$q={_q_of(typ)}$")
        xmax = max(xmax, max(ks))
        ymin, ymax = min(ymin, min(med)), max(ymax, max(med))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$t_{GB}(k)$ [s]")
    ax.set_xlim(0, xmax)
    ax.set_ylim(10.0 ** math.floor(math.log10(ymin)),
                10.0 ** math.ceil(math.log10(ymax)))
    ax.xaxis.set_major_locator(MultipleLocator(_xstep(xmax)))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    _log_yticks(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[summary] {out.relative_to(ROOT)}  ({drawn} types at size {size})")


# smallest, middle and largest size, skipping sweeps that stop short
def pick_sizes(data, typ, count=3):
    sizes = sorted(n for (t, n) in data if t == typ)
    full = [n for n in sizes if max(data[(typ, n)]) == n - 1]
    pool = full if len(full) >= count else sizes
    if len(pool) <= count:
        return set(pool)
    return {pool[0], pool[len(pool) // 2], pool[-1]}


# encryption next to signature, two panels on one time axis
def plot_pair(enc_typ, sig_typ, out: Path, scope="challenge", ylim=None):
    data = read_measurements((enc_typ, sig_typ), scope=scope)
    if not data:
        print(f"[summary] {out.name}: no measurements in scope '{scope}'")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, typ, kind in ((axes[0], enc_typ, "encryption"),
                          (axes[1], sig_typ, "signature")):
        sel = pick_sizes(data, typ)
        xmax, ymin, ymax = _draw_type(ax, data, typ, shaded=True, sizes=sel)
        ax.set_title(f"type {typ} - {kind}", fontsize=10)
        _finish(ax, xmax, ymin, ymax, ylim=ylim, ylabel=(ax is axes[0]))
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[summary] {out.relative_to(ROOT)}  (types {enc_typ} and {sig_typ})")


# one time range for all pair figures, so they stay comparable
def common_ylim(scope="challenge"):
    ymin, ymax = math.inf, 0.0
    for typ in ENCRYPTION + SIGNATURE:
        data = read_measurements((typ,), scope=scope)
        sel = pick_sizes(data, typ)
        for (t, n), by_k in data.items():
            if n not in sel:
                continue
            _, med = _medians(by_k)
            ymin, ymax = min(ymin, min(med)), max(ymax, max(med))
    if not math.isfinite(ymin) or ymax <= 0:
        return None
    return (10.0 ** math.floor(math.log10(ymin)),
            10.0 ** math.ceil(math.log10(ymax)))


# draw every summary figure
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decades", type=int, default=8,
                    help="how many decades the q^k figure shows; "
                         "keep it small or the within-decade log ticks collapse")
    ap.add_argument("--kmax", type=int, default=None,
                    help="k range of the q^k figure (default: where q=2 reaches "
                         "the decade ceiling)")
    ap.add_argument("--scope", choices=("toy", "challenge", "all"),
                    default="challenge",
                    help="which instances the scaling figures use; three sizes "
                         "(smallest, middle, largest) are drawn from it")
    ap.add_argument("--kstar-scope", choices=("toy", "challenge", "all"),
                    default="challenge",
                    help="which instances the k* figures use (default: challenge -- "
                         "the toy sizes are too small for k* to be interesting)")
    ap.add_argument("--degree-trend", type=int, default=6,
                    help="degree of the trend used for the k* figures; must match "
                         "what scripts/plot_instance.py was run with (default 6)")
    ap.add_argument("--enc-size", type=int, default=20,
                    help="n for the encryption comparison figure (default 20)")
    ap.add_argument("--sig-size", type=int, default=10,
                    help="m for the signature comparison figure (default 10)")
    ap.add_argument("--free-scale", action="store_true",
                    help="let every pair figure pick its own time axis instead of "
                         "sharing one range (sharing is what makes them comparable)")
    args = ap.parse_args(argv)

    tag = "" if args.scope == "all" else f" -- {args.scope} instances"
    plot_guesses(args.decades, args.kmax, FIGURES / "guesses_qk.png")
    for typ in ENCRYPTION + SIGNATURE:
        plot_kstar(typ, FIGURES / f"kstar_over_n_type{typ}.png",
                   scope=args.kstar_scope, degree=args.degree_trend)

    plot_fixed_size(ENCRYPTION, args.enc_size,
                    FIGURES / "scaling_encryption.png", scope="all")
    plot_fixed_size(SIGNATURE, args.sig_size,
                    FIGURES / "scaling_signature.png", scope="all")

    ylim = None if args.free_scale else common_ylim(args.scope)
    for enc, sig in ((1, 4), (2, 5), (3, 6)):
        plot_pair(enc, sig, FIGURES / f"scaling_types{enc}_{sig}.png",
                  scope=args.scope, ylim=ylim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
