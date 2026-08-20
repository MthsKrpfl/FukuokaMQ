#!/usr/bin/env python3
r"""
The same k-sweep as measure_file.py, run with two Groebner engines on exactly
the same residual systems, to show how the choice of algorithm scales:

    libsingular:std      Buchberger with the classical criteria
    libsingular:slimgb   the engine Sage's groebner_basis() picks by default

Both live in the same compiled library, so the difference between the curves is
the algorithm and not the implementation language. Every k and algorithm is
solved --repeats times on one specialization and the median is reported. A
fresh ideal is built for each solve, because Sage caches the basis on the ideal.

Measuring needs Sage, plotting does not:

    sage -python scripts/algorithm_comparison.py --file <instance>
    python3 scripts/algorithm_comparison.py --plot-only

Writes results/algorithm_comparison/algorithm_comparison.csv and
figures/summary/algorithm_comparison.png. Both names are fixed, so a second
instance overwrites the first.

params:
  --file val              ; val is the instance to sweep
  --algorithms val        ; val is a comma separated list of Sage engine names
  --repeats val           ; val is how many solves per k and algorithm
  --kmin val              ; val is the smallest k to measure
  --kmax val              ; val is the largest k to measure
  --per-solve-timeout val ; val is the ceiling per solve in seconds
  --plot-only             ; skip the measuring and plot an existing csv
  --csv val               ; val is the csv to plot with --plot-only
  --logy                  ; logarithmic time axis instead of linear
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ALGO_DIR = RESULTS / "algorithm_comparison"
FIGURES = ROOT / "figures" / "summary"

DEFAULT_FILE = ("samples/toy/ToyExample-type1/ToyExample-type1-n20-seed0")
ALGORITHMS = ("libsingular:std", "libsingular:slimgb")
C_ALGO = {"libsingular:std": "#D94141",
          "libsingular:slimgb": "#4C9BE8"}


# sweep k for every engine on identical systems, needs sage
def measure(file_path, algorithms, repeats, kmin, kmax, timeout):
    sys.path.insert(0, str(ROOT / "scripts" / "subprograms"))
    import random as _pyrandom
    from sage.all import PolynomialRing, alarm, cancel_alarm
    from cysignals.signals import AlarmInterrupt
    import mq_parser as mqparser

    inst = mqparser.parse_instance(str(file_path), order_mode="degrevlex_desc")
    n, m, q, K = inst.n, inst.m, inst.q, inst.ring.base_ring()
    print(f"[algo] {inst.instance_id}: type {inst.typ}, q={q}, n={n}, m={m}")

    def build_ideal(k, spec_seed):
        rng = _pyrandom.Random(f"{spec_seed}-{inst.sha256}")
        guessed = sorted(rng.sample(range(n), k)) if k > 0 else []
        gset = set(guessed)
        free_idx = [i for i in range(n) if i not in gset]
        nf = len(free_idx)

        def rand_val():
            if q == 2:
                return K(rng.randint(0, 1))
            if q == 256:
                v = rng.randint(0, 255)
                try:
                    return K.from_integer(v)
                except AttributeError:
                    return K.fetch_int(v)
            return K(rng.randint(0, q - 1))

        fixed = {i: rand_val() for i in guessed}
        if nf == 0:
            return None
        R2 = PolynomialRing(K, nf, names=[f"y{j}" for j in range(nf)],
                            order="degrevlex")
        sub = [None] * n
        for i in guessed:
            sub[i] = fixed[i]
        for j, i in enumerate(free_idx):
            sub[i] = R2.gen(j)
        residual = [p(*sub) for p in inst.polys]
        extra = [y ** 2 - y for y in R2.gens()] if q == 2 else []
        return R2.ideal([f for f in residual if f != 0] + extra)

    k_hi = (n - 1) if kmax is None else min(n - 1, kmax)
    k_lo = 0 if kmin is None else max(0, kmin)
    rows = []
    for k in range(k_hi, k_lo - 1, -1):
        line = [f"k={k:>3}"]
        for algo in algorithms:
            times = []
            status = "ok"
            for rep in range(repeats):
                I = build_ideal(k, rep)
                if I is None:
                    times.append(0.0)
                    continue
                if timeout:
                    alarm(float(timeout))
                try:
                    t0 = time.perf_counter()
                    I.groebner_basis(algorithm=algo)
                    times.append(time.perf_counter() - t0)
                except AlarmInterrupt:
                    status = "timeout"
                    times.append(float(timeout))
                    break
                finally:
                    if timeout:
                        cancel_alarm()
            med = statistics.median(times) if times else float("nan")
            for rep, t in enumerate(times):
                rows.append({"instance_id": inst.instance_id, "type": inst.typ,
                             "q": q, "n": n, "m": m, "k": k, "algorithm": algo,
                             "rep": rep, "time_s": round(t, 6),
                             "status": status})
            line.append(f"{algo.split(':')[-1]} {med:.4f}s")
        print("  ".join(line), flush=True)
    return rows, inst.instance_id


# one row per k, algorithm and repeat
def write_csv(rows):
    ALGO_DIR.mkdir(parents=True, exist_ok=True)
    out = ALGO_DIR / "algorithm_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[algo] wrote {out}")
    return out


# median time per k, one line per engine, timed out solves are left out
def plot(csv_path: Path, logy=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (AutoMinorLocator, LogLocator, MaxNLocator,
                                   NullFormatter)

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    if not rows:
        sys.exit(f"no rows in {csv_path}")
    n = int(rows[0]["n"])
    by = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        by.setdefault(r["algorithm"], {}).setdefault(int(r["k"]), []).append(
            float(r["time_s"]))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for algo in sorted(by):
        series = by[algo]
        ks = sorted(series)
        ax.plot(ks, [statistics.median(series[k]) for k in ks], "-", lw=1.8,
                color=C_ALGO.get(algo, "#888888"), label=algo.split(":")[-1])
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, None)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel("$k$")
    ax.set_ylabel("$t_{GB}(k)$ [s]")
    ax.set_xlim(0, n - 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.35)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "algorithm_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[algo] wrote {out}")


# measure and plot, or plot an existing csv
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_FILE, help="instance path")
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS),
                    help="comma separated Sage algorithm names")
    ap.add_argument("--repeats", type=int, default=3,
                    help="solves per (k, algorithm); the median is plotted")
    ap.add_argument("--kmin", type=int, default=None)
    ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--per-solve-timeout", type=float, default=600.0)
    ap.add_argument("--plot-only", action="store_true",
                    help="skip the measuring and plot an existing CSV")
    ap.add_argument("--csv", default=None, help="CSV for --plot-only")
    ap.add_argument("--logy", action="store_true",
                    help="logarithmic time axis (default: linear)")
    args = ap.parse_args(argv)

    if args.plot_only:
        path = Path(args.csv) if args.csv else None
        if path is None:
            path = ALGO_DIR / "algorithm_comparison.csv"
        if not path.exists():
            sys.exit(f"no {path} -- run the measuring step first "
                     "(or pass --csv for a file kept under another name)")
        plot(path, logy=args.logy)
        return 0

    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    if not file_path.exists():
        sys.exit(f"no such instance: {file_path}")
    rows, _ = measure(file_path, args.algorithms.split(","), args.repeats,
                      args.kmin, args.kmax, args.per_solve_timeout)
    plot(write_csv(rows), logy=args.logy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
