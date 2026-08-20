#!/usr/bin/env python3
"""
Turns the measured GB times into an optimal guess count k* per instance size.

For every group of type, n and m the full types 1 and 4 get the cost model
fitted on the points above the gate, extrapolated over the whole range
[k_min, n-1], and the argmin reported with a bootstrap confidence interval.
The light types 2, 3, 5 and 6 only minimise the measured cost over the few k
near k_min, where their k* sits anyway.

Writes results/estimates.csv and results/estimate_details.json, the arrays
plot_instance.py draws.

params:
  --config val         ; val is the config file, default scripts/config.json
  --type val           ; val is a comma separated list of types to estimate
  --n_min val          ; val is the smallest instance size to include
  --n_max val          ; val is the largest instance size to include
  --gate val           ; val is the time in seconds a point must exceed to be fitted
  --fit-floor-abs val  ; val is the absolute signal floor in seconds
  --fit-floor-rel val  ; val is the floor as a multiple of the fastest measurement
  --n-boot val         ; val is how many bootstrap resamples the interval uses
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "subprograms"))
import numpy as np
import bookkeeping
import cost_model


# every measurement shard belonging to this record
def read_rows(csv_path: Path):
    if not csv_path.exists():
        print(f"No measurements at {csv_path}", file=sys.stderr)
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# repeats resolved, so a re-measured point counts once
def dedup_rows(rows):
    kept = list(bookkeeping.dedup_records(rows).values())
    kept += [r for r in rows if r.get("status") not in ("ok", "timeout")]
    return kept


# group strictly by type, n and m, different sizes never mix
def group_rows(rows):
    groups = defaultdict(list)
    unusable = 0
    for r in rows:
        try:
            key = (int(r["type"]), int(r["n"]), int(r["m"]))
        except (KeyError, ValueError, TypeError):
            unusable += 1
            continue
        groups[key].append(r)
    if unusable:
        print(f"[estimate] {unusable} row(s) skipped: no usable type/n/m "
              f"(re-measure or backfill to use them)", file=sys.stderr)
    return groups


# fit the cost model and extrapolate it to find k*
def estimate_full(rows, typ, n, m, q, gate, n_boot, floor_abs, floor_rel):
    by_np = cost_model.aggregate_times(rows)
    if not by_np:
        return None
    nps_all, med_all = cost_model.median_points(by_np)

    min_med = float(med_all.min())
    floor = max(floor_abs, floor_rel * min_med)
    signal = {nf: v for nf, v in by_np.items()
              if float(np.median(v)) >= floor}
    note = ""
    fit_src = signal
    if len(signal) < 3:
        fit_src = by_np
        note = f"few signal pts (>= {floor:.3g}s); fit uses all {len(by_np)}"

    nps, med = cost_model.median_points(fit_src)
    k_min = max(0, n - m)
    try:
        model = cost_model.fit_model(nps, med, q=q, n=n, m=m, k_min=k_min)
    except cost_model.InsufficientData as e:
        print(f"[type {typ} n={n} m={m}] no fit: {e}", file=sys.stderr)
        return None
    if model.flag:
        note = (note + "; " if note else "") + model.flag

    yhat = model.log_time(nps)
    y = np.log(med)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    kstar, cost_at, ks, log10cost = model.kstar()
    klo, klo5, khi95, _ = cost_model.bootstrap_kstar(
        fit_src, q=q, n=n, m=m, k_min=k_min, n_boot=n_boot)

    measured_set = {n - int(x) for x in by_np}
    fit_set = {int(x) for x in nps}
    fit_np_lo = min(fit_set)
    fit_np_hi = max(fit_set)

    kstar_np = n - kstar
    if fit_np_lo <= kstar_np <= fit_np_hi:
        reach = "interpolated"
    else:
        gap = kstar_np - fit_np_hi if kstar_np > fit_np_hi else fit_np_lo - kstar_np
        reach = f"extrapolated+{gap}"
    details = {
        "type": typ, "n": n, "m": m, "q": q, "k_min": k_min, "method": "full",
        "measured": {
            "nprime": [int(x) for x in nps_all],
            "time_med": [float(x) for x in med_all],
            "time_min": [float(min(by_np[int(x)])) for x in nps_all],
            "time_max": [float(max(by_np[int(x)])) for x in nps_all],
            "in_fit": [bool(int(x) in fit_set) for x in nps_all],
        },
        "gate": gate,
        "fit": {"coeffs": [float(c) for c in model.coeffs],
                "degree": model.degree, "r2": r2, "floor": floor,
                "np_lo": fit_np_lo, "np_hi": n - k_min,
                "np_fit_hi": fit_np_hi, "n_points": len(nps)},
        "kstar_nprime": kstar_np, "reach": reach,
        "curve": {
            "k": [int(x) for x in ks],
            "log10cost": [float(x) for x in log10cost],
            "measured_mask": [bool(int(x) in measured_set) for x in ks],
        },
        "kstar": kstar, "kstar_lo": klo5, "kstar_hi": khi95,
        "cost_log10_at_kstar": cost_at,
        "note": note,
    }
    summary = {
        "type": typ, "n": n, "m": m, "q": q, "k_min": k_min, "method": "full",
        "kstar": kstar, "kstar_lo": klo5, "kstar_hi": khi95,
        "cost_log10_at_kstar": round(cost_at, 3),
        "fit_degree": model.degree, "fit_r2": round(r2, 4),
        "fit_coeffs": ";".join(f"{c:.6g}" for c in model.coeffs),
        "n_fit_points": len(nps), "n_measured_nprime": len(by_np),
        "fit_floor_s": round(floor, 4),
        "kstar_nprime": kstar_np, "fit_np_lo": fit_np_lo, "fit_np_hi": fit_np_hi,
        "reach": reach, "note": note,
    }
    return summary, details


# no fit: take the smallest measured cost, k* sits at the boundary here
def estimate_light(rows, typ, n, m, q):
    costs = cost_model.direct_cost_by_k(rows, q)
    if not costs:
        return None
    kstar = min(costs, key=costs.get)
    k_min = max(0, n - m)
    details = {
        "type": typ, "n": n, "m": m, "q": q, "k_min": k_min, "method": "light",
        "light_costs": {int(k): float(v) for k, v in sorted(costs.items())},
        "kstar": int(kstar), "kstar_lo": int(kstar), "kstar_hi": int(kstar),
        "cost_log10_at_kstar": float(costs[kstar]),
    }
    summary = {
        "type": typ, "n": n, "m": m, "q": q, "k_min": k_min, "method": "light",
        "kstar": int(kstar), "kstar_lo": int(kstar), "kstar_hi": int(kstar),
        "cost_log10_at_kstar": round(float(costs[kstar]), 3),
        "fit_degree": "", "fit_r2": "", "fit_coeffs": "", "n_fit_points": "",
        "n_measured_nprime": len(costs), "fit_floor_s": "",
        "note": f"observed min over k={sorted(costs)}",
    }
    return summary, details


# columns of results/estimates.csv
SUMMARY_COLS = ["type", "n", "m", "q", "k_min", "method", "kstar", "kstar_lo",
                "kstar_hi", "cost_log10_at_kstar", "reach", "kstar_nprime",
                "fit_np_lo", "fit_np_hi", "fit_degree", "fit_r2",
                "fit_coeffs", "n_fit_points", "n_measured_nprime", "fit_floor_s",
                "note"]


# estimate every group and write the summary plus the details
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--type", dest="types", default=None, help="comma list or 'all'")
    ap.add_argument("--n_min", type=int, default=None, help="only sizes with n >= this")
    ap.add_argument("--n_max", type=int, default=None, help="only sizes with n <= this")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--fit-floor-abs", type=float, default=None,
                    help="absolute GB-time floor (s) below which points are "
                         "excluded from the fit (noise/overhead)")
    ap.add_argument("--fit-floor-rel", type=float, default=None,
                    help="relative floor: drop points below this x the fastest "
                         "measured GB (removes the flat plateau)")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args(argv)

    cfg = bookkeeping.load_config(args.config)
    gate = args.gate if args.gate is not None else cfg.get("gate_seconds", 10.0)
    floor_abs = args.fit_floor_abs if args.fit_floor_abs is not None \
        else cfg.get("fit_floor_abs_s", 0.05)
    floor_rel = args.fit_floor_rel if args.fit_floor_rel is not None \
        else cfg.get("fit_floor_rel", 10.0)
    full_types = set(cfg.get("full_types", [1, 4]))
    want_types = None if args.types in (None, "all") else \
        {int(x) for x in args.types.split(",")}

    csv_path = bookkeeping.resolve(cfg["results_csv"])
    files = bookkeeping.measurement_files(csv_path) or [csv_path]
    rows = []
    for f in files:
        rows.extend(read_rows(f))
    n_raw = len(rows)
    rows = dedup_rows(rows)
    shown = ", ".join(str(f.relative_to(csv_path.parent)) for f in files[:6])
    if len(files) > 6:
        shown += f", ... (+{len(files) - 6} more)"
    print(f"[estimate] read {n_raw} rows from {len(files)} file(s): {shown}")
    if n_raw != len(rows):
        print(f"[estimate] dropped {n_raw - len(rows)} duplicate measurement(s) "
              f"on (type, instance_id, k, seed)")
    groups = group_rows(rows)
    if not groups:
        print("No usable measurement rows.", file=sys.stderr)
        return 1

    summaries, details_all = [], {}
    for (typ, n, m), grows in sorted(groups.items()):
        if want_types and typ not in want_types:
            continue
        if args.n_min is not None and n < args.n_min:
            continue
        if args.n_max is not None and n > args.n_max:
            continue
        try:
            q = int(next(r["q"] for r in grows if r.get("q") not in (None, "", "None")))
        except (StopIteration, ValueError):
            print(f"[type {typ} n={n}] no q in rows, skipping", file=sys.stderr)
            continue

        if typ in full_types:
            res = estimate_full(grows, typ, n, m, q, gate, args.n_boot,
                                floor_abs, floor_rel)
        else:
            res = estimate_light(grows, typ, n, m, q)
        if res is None:
            print(f"[type {typ} n={n} m={m}] no usable data, skipped")
            continue
        summary, details = res
        summaries.append(summary)
        details_all[f"{typ}-{n}-{m}"] = details
        print(f"[type {typ} n={n} m={m}] method={summary['method']} "
              f"k*={summary['kstar']} "
              f"(CI {summary['kstar_lo']}..{summary['kstar_hi']}) "
              f"log10 cost={summary['cost_log10_at_kstar']} {summary['note']}")

    est_csv = bookkeeping.resolve(cfg["estimates_csv"])
    est_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(est_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    details_path = est_csv.parent / "estimate_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details_all, f, indent=2)
    print(f"\nWrote {est_csv} ({len(summaries)} rows) and {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
