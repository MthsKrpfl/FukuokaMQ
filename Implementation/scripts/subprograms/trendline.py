#!/usr/bin/env python3
r"""
Fits an exponential trendline to the usable GB times of one size group and
extrapolates it over the whole k range, so the cost model has a time for every
k -- including the ones far too slow to measure.

    ln t(k) = a + b*k             degree 1, the Excel exponential trendline
    ln t(k) = a + b*k + c*k^2     degree 2, when the tail curves

At large k the time sits on a flat ~ms overhead plateau, only the steep points
above it are usable. Imported by plot_instance.py and plot_summary.py, and runs
standalone as well, writing <label>_trend.csv plus the equation, R^2, the
growth factor per step and an implied k*.

params:
  input               ; a group csv, e.g. results/type1/n55_measurements.csv
  --cutoff-k val      ; val is the largest k still used for the fit
  --ratio val         ; val is the step factor marking the steep regime, used
                        only when --cutoff-k is omitted
  --degree val        ; val is the degree of the polynomial in the exponent, 1 or 2
  --predict-kmin val  ; val is the lowest k to predict down to
  --out val           ; val is the csv the predictions go to
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


# median gb time per k, read from whatever shape the input has
def read_points(path: Path):
    files = [path] if path.is_file() else []
    if not files:
        print(f"[trend] nothing to read at {path}", file=sys.stderr)
        return [], {}

    rows, meta = [], {}
    for f in files:
        with open(f, "r", newline="", encoding="utf-8") as fh:
            text = fh.read().replace(";\n", "\n")
        for r in csv.DictReader(text.splitlines()):
            if r.get("status", "ok") != "ok":
                continue
            try:
                k = int(r["k"]); t = float(r["gb_time_s"])
            except (KeyError, ValueError):
                continue
            if t <= 0:
                continue
            rows.append({"k": k, "t": t})
            if not meta:
                meta = {"type": r.get("type"), "q": int(r["q"]),
                        "n": int(r["n"]), "m": int(r["m"])}
    return rows, meta


# least squares fit without numpy
def polyfit(xs, ys, deg):
    N = deg + 1
    powsum = [sum(x ** p for x in xs) for p in range(2 * deg + 1)]
    A = [[powsum[i + j] for j in range(N)] for i in range(N)]
    v = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(N)]
    for c in range(N):
        p = max(range(c, N), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]; v[c], v[p] = v[p], v[c]
        for r in range(N):
            if r == c:
                continue
            f = A[r][c] / A[c][c]
            for j in range(c, N):
                A[r][j] -= f * A[c][j]
            v[r] -= f * v[c]
    return [v[i] / A[i][i] for i in range(N)]


def polyval(coeffs, x):
    return sum(a * x ** i for i, a in enumerate(coeffs))


# quadratic forced through a given vertex, so the trend cannot turn back up
def polyfit_vertex(xs, ys, vertex):
    v = float(vertex)
    u = [(x - v) ** 2 for x in xs]
    s0 = float(len(xs))
    s1 = sum(u)
    s2 = sum(uu * uu for uu in u)
    ty = sum(ys)
    tuy = sum(uu * yy for uu, yy in zip(u, ys))
    det = s0 * s2 - s1 * s1
    if det == 0:
        return None
    A = (ty * s2 - tuy * s1) / det
    c = (s0 * tuy - s1 * ty) / det
    return [A + c * v * v, -2.0 * c * v, c]


def polyder(coeffs):
    return [i * a for i, a in enumerate(coeffs)][1:] or [0.0]


# t(k) = c + exp(p(k)), continued linearly beyond the fitted range
def make_trend(c, coeffs, x_lo, x_hi, curv=0.0, join_lo=None, join_hi=None):
    if join_lo is None:
        join_lo = x_lo
    if join_hi is None:
        join_hi = x_hi
    d1 = polyder(coeffs)

    def g(x):
        if x < join_lo:
            dx = x - join_lo
            return (polyval(coeffs, join_lo) + polyval(d1, join_lo) * dx
                    + max(curv, 0.0) * dx * dx)
        if x > join_hi:
            dx = x - join_hi
            return polyval(coeffs, join_hi) + polyval(d1, join_hi) * dx
        return polyval(coeffs, x)

    def t(x):
        try:
            return c + math.exp(g(x))
        except OverflowError:
            return float("inf")
    return t


# search the constant overhead floor c that makes the fit best
def fit_floor(xs, ys, deg=2, steps=60, refine=6):
    ts = [math.exp(y) for y in ys]
    tmin = min(ts)
    lo, hi = 0.0, tmin * 0.999
    best = None
    for _ in range(refine):
        for i in range(steps + 1):
            c = lo + (hi - lo) * i / steps
            pts = [(x, t - c) for x, t in zip(xs, ts) if t - c > 0]
            if len(pts) < deg + 2:
                continue
            co = polyfit([x for x, _ in pts], [math.log(r) for _, r in pts], deg)
            se = 0.0
            for x, y in zip(xs, ys):
                se += (math.log(c + math.exp(polyval(co, x))) - y) ** 2
            rmse = math.sqrt(se / len(xs))
            if best is None or rmse < best[2]:
                best = (c, co, rmse)
        if best is None:
            return 0.0, polyfit(xs, ys, deg), float("inf")
        span = (hi - lo) / steps
        lo, hi = max(0.0, best[0] - span), min(tmin * 0.999, best[0] + span)
    return best


def step_ratios(med_by_k):
    return {k: med_by_k[k] / med_by_k[k + 1]
            for k in med_by_k if (k + 1) in med_by_k}


# largest k whose steps still grow by at least `ratio`
def auto_cutoff(med_by_k, ratio, run=3):
    r = step_ratios(med_by_k)
    ks = sorted(med_by_k, reverse=True)
    for hi in ks:
        seg = [r[hi - j] for j in range(run) if (hi - j) in r]
        if len(seg) >= run and statistics.median(seg) >= ratio:
            return hi
    return ks[-1]


# growth factor per step, to check an auto cutoff by eye
def print_ratio_table(med_by_k, n, cutoff):
    r = step_ratios(med_by_k)
    nprime = "n'"
    print(f"  {'k':>4} {nprime:>4} {'median t':>12} {'x/step':>7}  fit?")
    for k in sorted(med_by_k, reverse=True):
        rr = f"{r[k]:.2f}" if k in r else "-"
        mark = "  <<" if k == cutoff else ""
        print(f"  {k:>4} {n - k:>4} {med_by_k[k]:>12.5f} {rr:>7}  "
              f"{'yes' if k <= cutoff else '   '}{mark}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a group csv, e.g. results/type1/n55_measurements.csv")
    ap.add_argument("--cutoff-k", type=int, default=None,
                    help="usable points are k <= this (the exponential regime); "
                         "if omitted, auto-detected from --ratio")
    ap.add_argument("--ratio", type=float, default=1.5,
                    help="step factor that marks the start of the steep regime "
                         "(used only when --cutoff-k is omitted)")
    ap.add_argument("--degree", type=int, default=1, choices=(1, 2),
                    help="1 = log-linear (Excel exponential trendline), "
                         "2 = log-quadratic (if the tail curves)")
    ap.add_argument("--predict-kmin", type=int, default=None,
                    help="lowest k to predict down to (default: k_min = max(0, n-m))")
    ap.add_argument("--out", default=None,
                    help="predictions CSV, default <label>_trend.csv beside the input")
    args = ap.parse_args(argv)

    path = Path(args.input)
    rows, meta = read_points(path)
    if not rows:
        return 1
    n, m, q = meta["n"], meta["m"], meta["q"]

    med_by_k = {k: statistics.median([r["t"] for r in rows if r["k"] == k])
                for k in {r["k"] for r in rows}}
    if args.cutoff_k is not None:
        cutoff = args.cutoff_k
    else:
        cutoff = auto_cutoff(med_by_k, args.ratio)
        print(f"[trend] no --cutoff-k given; auto-detected k <= {cutoff} "
              f"(sustained x>={args.ratio}). Verify against the table and "
              f"override with --cutoff-k if needed:")
        print_ratio_table(med_by_k, meta["n"], cutoff)

    usable = [r for r in rows if r["k"] <= cutoff]
    if len({r["k"] for r in usable}) < args.degree + 1:
        print(f"[trend] too few usable k values ({len({r['k'] for r in usable})}) "
              f"for degree {args.degree}", file=sys.stderr)
        return 1

    xs = [r["k"] for r in usable]
    ys = [math.log(r["t"]) for r in usable]
    coeffs = polyfit(xs, ys, args.degree)

    ybar = sum(ys) / len(ys)
    ss_res = sum((y - polyval(coeffs, x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"[trend] {path}  (type {meta['type']}, q={q}, n={n}, m={m})")
    print(f"[trend] usable: k <= {cutoff}  ({len(usable)} points over "
          f"{len(set(xs))} distinct k, k {min(xs)}..{max(xs)})")
    a, b = coeffs[0], coeffs[1]
    if args.degree == 1:
        print(f"[trend] fit  gb_time = {math.exp(a):.6g} * exp({b:+.5f} * k)")
        print(f"        Excel:  y = {math.exp(a):.6g}*EXP({b:+.5f}*x)   (x = k)")
        print(f"        per -1 in k, time x{math.exp(-b):.3f}   (per +1 in n' = n-k)")
    else:
        c = coeffs[2]
        print(f"[trend] fit  ln(gb_time) = {a:.5g} {b:+.5g}*k {c:+.5g}*k^2")
        print(f"        Excel:  y = EXP({a:.5g} {b:+.5g}*x {c:+.5g}*x^2)   (x = k)")
    print(f"        R^2 (log space) = {r2:.5f}")

    k_floor = max(0, n - m) if args.predict_kmin is None else args.predict_kmin
    k_ceil = max(med_by_k)
    pred = {k: math.exp(polyval(coeffs, k)) for k in range(k_floor, k_ceil + 1)}

    out = Path(args.out) if args.out else \
        path.with_name(path.stem.replace("_measurements", "") + "_trend.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("k;n_prime;t_measured_med;t_pred;used_in_fit;\n")
        for k in range(k_ceil, k_floor - 1, -1):
            tm = med_by_k.get(k)
            used = int(tm is not None and k <= cutoff)
            fh.write(f"{k};{n - k};{'' if tm is None else repr(tm)};"
                     f"{repr(pred[k])};{used};\n")
    print(f"[trend] wrote {out}  (k {k_ceil}..{k_floor}, {k_ceil - k_floor + 1} rows)")

    if args.degree == 1:
        factor = math.exp(-b)
        if factor > q:
            lean = f"factor {factor:.3f} > q={q}  ->  guessing pays; k* toward high k"
        elif factor < q:
            lean = f"factor {factor:.3f} < q={q}  ->  guessing costs; k* toward k_min={k_floor}"
        else:
            lean = f"factor {factor:.3f} == q={q}  ->  cost flat in k"
        print(f"[trend] {lean}  (margin {factor - q:+.3f}; near 0 = ambiguous)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
