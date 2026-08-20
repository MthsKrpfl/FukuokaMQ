#!/usr/bin/env python3
r"""
Times one Groebner basis per k and spec for a single MQ instance, inside ONE
persistent Sage process: Sage and the instance are loaded once, and only the
groebner_basis() call is timed. Rows are appended to
results/type<T>/<label>_measurements.csv, the file estimate_kstar.py reads.

The sweep runs from easy at large k to hard and stops at the first timeout,
since no harder residual can be faster. It resumes, any k and spec already
recorded for this instance is skipped.

Requires Sage, run it under Sage's Python:

    sage -python scripts/measure_file.py --file <instance> --kmin 16 --kmax 45

params:
  --file val              ; val is the instance file, plain or .bz2
  --kmin val              ; val is the smallest k to measure
  --kmax val              ; val is the largest k to measure
  --specs val             ; val is how many random specializations per k
  --spec-start val        ; val is the first spec index, for splitting the work
  --per-solve-timeout val ; val is the ceiling per solve in seconds
  --zero-surplus          ; fix the n-m surplus variables to 0, types 4-6
  --zero-first val        ; val is how many leading variables to fix to 0
  --out val               ; val is the csv to append to
  --gb-dir val            ; val is where a solved k=0 basis is dumped
  --no-resume             ; measure again even where rows exist
  --no-warmup             ; skip the discarded warmup solve
  --no-stop               ; keep going past a timeout
  --root val              ; val is the project root, used for imports
  --order-mode val        ; val is the monomial order used when parsing
"""
import argparse
import csv
import datetime
import os
import sys
import time


# progress goes to stderr, the measurements to stdout
def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# command line
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="instance path (.bz2 ok)")
    ap.add_argument("--zero-surplus", action="store_true",
                    help="fix the first (n-m) variables to 0 before sweeping, "
                         "reducing an underdetermined system (n>m, types 4-6) to "
                         "a square m-variable subsystem; k* is then measured on "
                         "that subsystem exactly as for type 1")
    ap.add_argument("--zero-first", type=int, default=0, dest="zero_first",
                    help="fix the first D variables to 0 explicitly "
                         "(--zero-surplus sets D=n-m automatically)")
    ap.add_argument("--kmin", type=int, default=None,
                    help="smallest k to measure (default: analysis floor, >=0)")
    ap.add_argument("--kmax", type=int, default=None,
                    help="largest k to measure (default: n-1)")
    ap.add_argument("--specs", type=int, default=3,
                    help="number of random specializations per k")
    ap.add_argument("--spec-start", type=int, default=0,
                    help="first spec seed; this run covers spec-start..+specs-1")
    ap.add_argument("--per-solve-timeout", type=float, default=None,
                    help="per-solve ceiling in seconds (Sage alarm); default none")
    ap.add_argument("--stop-on-timeout", dest="stop_on_timeout",
                    action="store_true", default=True,
                    help="stop the sweep at the first timeout (default)")
    ap.add_argument("--no-stop", dest="stop_on_timeout", action="store_false",
                    help="keep going past a timeout (risks a dirty Singular state)")
    ap.add_argument("--warmup", dest="warmup", action="store_true", default=True,
                    help="one discarded solve before timing (default)")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--out", default=None,
                    help="output CSV, default results/type<T>/"
                         "<label>_measurements.csv -- the file "
                         "scripts/estimate_kstar.py reads, so a sweep is a "
                         "drop-in for the measurement stage")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="do not skip (k, spec) already present in --out")
    ap.add_argument("--gb-dir", default=None,
                    help="where to dump the Groebner basis whenever a k=0 solve "
                         "succeeds (the full system is solved directly, so its GB "
                         "carries the solution); default results/gb_solutions/")
    ap.add_argument("--root", default=None, help="project root (for imports)")
    ap.add_argument("--order-mode", default="degrevlex_desc")
    return ap.parse_args(argv)


# same schema as subprograms/bookkeeping.py, so a sweep is a drop-in
FIELDNAMES = ["type", "instance_id", "q", "n", "m", "k", "n_minus_k",
              "seed", "gb_time_s", "gb_size", "status", "hash"]


# how many specs of each k this instance already has, for the resume
def load_done(out_path, instance_id):
    done = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("ok", "timeout"):
                continue
            if row.get("instance_id") != instance_id:
                continue
            try:
                k = int(row["k"])
            except (KeyError, TypeError, ValueError):
                continue
            done[k] = done.get(k, 0) + 1
    return done


# append immediately, so a killed run keeps everything it measured
def append_row(out_path, row):
    new = not os.path.exists(out_path)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# a solved k=0 basis carries the solution, so it is dumped to its own file
def store_gb(gb_dir, inst, spec, n_eff, D, gb_time, gb):
    os.makedirs(gb_dir, exist_ok=True)
    path = os.path.join(gb_dir, f"{inst.instance_id}_spec{spec}.csv")
    polys = list(gb)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "instance_id", "q", "n", "m", "zeroed", "spec_seed",
                    "gb_time_s", "gb_size", "poly_index", "polynomial"])
        for i, g in enumerate(polys):
            w.writerow([inst.typ, inst.instance_id, inst.q, n_eff, inst.m, D,
                        spec, gb_time, len(polys), i, str(g)])
    return path


# load sage once, parse once, then time one solve per k and spec
def main(argv=None):
    args = parse_args(argv)
    args.file = args.file.replace("\\", "/")

    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprograms = os.path.join(root, "scripts", "subprograms")
    if subprograms not in sys.path:
        sys.path.insert(0, subprograms)

    t0 = time.perf_counter()
    from sage.all import PolynomialRing, alarm, cancel_alarm
    from cysignals.signals import AlarmInterrupt
    import random as _pyrandom
    import mq_parser as mqparser
    eprint(f"[sweep] imported sage in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    inst = mqparser.parse_instance(args.file, order_mode=args.order_mode)
    n, m, q, K = inst.n, inst.m, inst.q, inst.ring.base_ring()
    eprint(f"[sweep] parsed {inst.instance_id} in {time.perf_counter() - t0:.2f}s "
           f"(type {inst.typ}, q={q}, n={n}, m={m})")

    # zeroing the surplus variables leaves a square subsystem, types 4-6
    D = (n - m) if args.zero_surplus else max(0, args.zero_first)
    if D < 0:
        D = 0
    if D >= n:
        eprint(f"[sweep] D={D} >= n={n}: nothing left to solve")
        return 1
    active = list(range(D, n))
    n_eff = n - D
    if D:
        eprint(f"[sweep] zeroing first D={D} vars -> {n_eff}-variable subsystem "
               f"(m={m} equations); k and n' are on this subsystem")

    # one file per size group, the seed column tells the instances apart
    import bookkeeping
    seed = None
    info = bookkeeping.parse_name(inst.instance_id)
    if info is not None:
        seed = info[2]
    if seed is None:
        seed = 0
        eprint(f"[sweep] no seed in id {inst.instance_id!r}; recording seed=0")
    out_path = args.out
    if out_path is None:
        base = bookkeeping.resolve("results/measurements.csv")
        if D:
            out_path = str(base.parent / f"type{inst.typ}"
                            / f"m{m}_{base.stem}.csv")
        else:
            out_path = str(bookkeeping.group_csv(
                base, inst.typ, n_eff, m, instance_id=inst.instance_id))
    eprint(f"[sweep] writing to {out_path}")

    gb_dir = args.gb_dir or os.path.join(root, "results", "gb_solutions")

    # fix k random variables and rebuild the system over the n-k free ones
    def build_ideal(k, spec_seed):
        rng = _pyrandom.Random(f"{spec_seed}-{inst.sha256}")
        guessed = sorted(rng.sample(active, k)) if k > 0 else []
        gset = set(guessed)
        free_idx = [i for i in active if i not in gset]
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

        fixed_val = {i: rand_val() for i in guessed}
        if nf == 0:
            return None, 0
        R2 = PolynomialRing(K, nf, names=[f"y{j}" for j in range(nf)],
                            order="degrevlex")
        sub = [None] * n
        for i in range(D):
            sub[i] = K(0)
        for i in guessed:
            sub[i] = fixed_val[i]
        for j, i in enumerate(free_idx):
            sub[i] = R2.gen(j)
        residual = [p(*sub) for p in inst.polys]
        gens_extra = [y ** 2 - y for y in R2.gens()] if q == 2 else []
        system = [f for f in residual if f != 0] + gens_extra
        return R2.ideal(system), nf

    # the timer covers the groebner basis and nothing else
    def timed_solve(k, spec_seed):
        I, nf = build_ideal(k, spec_seed)
        if I is None:
            return "ok", 0.0, 0, None
        if args.per_solve_timeout:
            alarm(float(args.per_solve_timeout))
        try:
            t = time.perf_counter()
            gb = I.groebner_basis()
            dt = time.perf_counter() - t
        except AlarmInterrupt:
            return "timeout", float(args.per_solve_timeout), nf, None
        finally:
            if args.per_solve_timeout:
                cancel_alarm()
        return "ok", dt, nf, gb

    # k range, swept from easy at large k to hard
    k_floor = max(0, n_eff - m) if args.kmin is None else max(0, args.kmin)
    k_ceil = (n_eff - 1) if args.kmax is None else min(n_eff - 1, args.kmax)
    if k_floor > k_ceil:
        eprint(f"[sweep] empty k range [{k_floor}, {k_ceil}]")
        return 1
    ks = list(range(k_ceil, k_floor - 1, -1))
    specs = list(range(args.spec_start, args.spec_start + args.specs))
    eprint(f"[sweep] k {k_ceil}..{k_floor} (n' {n_eff - k_ceil}..{n_eff - k_floor}), "
           f"specs {specs}, {len(ks) * len(specs)} solves"
           + (f", per-solve timeout {args.per_solve_timeout:.0f}s"
              if args.per_solve_timeout else ""))

    last_wall = None
    # one discarded solve first, so no startup cost lands in a measurement
    if args.warmup:
        t = time.perf_counter()
        st, dt, _nf, _ = timed_solve(k_ceil, args.spec_start)
        last_wall = time.perf_counter() - t
        eprint(f"[sweep] warmup k={k_ceil} spec={args.spec_start}: {st} "
               f"gb={dt:.4f}s (wall {last_wall:.2f}s) -- discarded")

    done = load_done(out_path, inst.instance_id) if args.resume else {}

    # measure every k and spec, stop at the first timeout
    stopped = False
    for k in ks:
        if stopped:
            break
        for idx, spec in enumerate(specs):
            if idx < done.get(k, 0):
                continue
            now = datetime.datetime.now()
            if last_wall is not None:
                eta = now + datetime.timedelta(seconds=2.0 * last_wall)
                eprint(f"[sweep] k={k} n'={n_eff - k} spec={spec} start "
                       f"{now:%H:%M:%S} -> est. done ~{eta:%H:%M:%S} "
                       f"(2x last {last_wall:.0f}s)")
            wall0 = time.perf_counter()
            status, gb_time, _nf, gb = timed_solve(k, spec)
            wall = time.perf_counter() - wall0
            last_wall = wall
            gb_size = int(len(gb)) if gb else 0
            row = {
                "type": inst.typ, "instance_id": inst.instance_id,
                "q": q, "n": n_eff, "m": m, "k": k, "n_minus_k": n_eff - k,
                "seed": seed, "gb_time_s": gb_time,
                "gb_size": gb_size, "status": status, "hash": inst.sha256,
            }
            append_row(out_path, row)
            done[k] = done.get(k, 0) + 1
            if k == 0 and status == "ok" and gb is not None:
                gb_path = store_gb(gb_dir, inst, spec, n_eff, D, gb_time, gb)
                eprint(f"[sweep] k=0 solved: stored {gb_size}-poly Groebner "
                       f"basis -> {gb_path}")
            print(f"k={k} n'={n_eff - k} spec={spec}: {status} "
                  f"t={gb_time:.4f}s (wall {wall:.2f}s)", flush=True)
            if status == "timeout" and args.stop_on_timeout:
                eprint(f"[sweep] timeout at k={k} (n'={n_eff - k}); "
                       f"harder residuals will too -- stopping")
                stopped = True
                break

    eprint("[sweep] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
