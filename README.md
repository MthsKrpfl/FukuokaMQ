# Optimal guess count k* for hybrid MQ solving

A hybrid attack on a multivariate quadratic system fixes k of the n variables and
solves the remaining system with a Gröbner basis, once for every one of the q^k
guesses. A larger k buys a cheaper solve and pays for it with more guesses, so the
total work is a product of two factors that pull in opposite directions.

```
Cost(k) = q^k * T_GB(n - k)
k*      = argmin Cost(k),   k in [k_min, n-1]
```

The MQ Challenge suggests that variable guessing is mostly effective over GF(2),
because the solving time has to fall faster than the factor q that each guessed
variable adds to the search space. The expectation entering these experiments is
therefore an interior optimum k* > 0 mainly over the small fields, and next to no
gain over the large ones.

What can be measured cheaply is the easy end at large k. This project
measures that end, fits an exponential trend through it, extrapolates it over the
whole k range, forms the cost curve and reports the k* that minimises it.

## The six challenge types

Instances come from the Fukuoka MQ Challenge. Types 1 to 3 are the overdetermined
encryption case, types 4 to 6 the underdetermined signature case, and the three
fields repeat in the same order.

```
type 1   GF(2)     m = 2n      encryption
type 2   GF(2^8)   m = 2n      encryption
type 3   GF(31)    m = 2n      encryption
type 4   GF(2)     n = 1.5m    signature
type 5   GF(2^8)   n = 1.5m    signature
type 6   GF(31)    n = 1.5m    signature
```

Types 1 and 4 are the full types. They get the fit and the extrapolation, and the
sweeps cover every size from 55 up to 100. The other four types are only confirmed
on a handful of k near their floor, which is where their optimum sits anyway.

## Layout

```
scripts/
  measure_file.py           one instance, one solve per k, appends rows
  measure_range.sh          drives a contiguous size range over seeds 0, 1 and 2
  estimate_kstar.py         measured rows to k* per size
  plot_instance.py          two diagrams for one size
  plot_summary.py           the figures that span instances
  algorithm_comparison.py   the same sweep with two Gröbner engines
  config.json               gate, signal floors, paths, which types are full
  subprograms/              parser, instance container, cost model, trendline
samples/challenge/          the challenge instances, plain or bz2
samples/toy/                smaller instances 
results/type<T>/            one measurements csv per instance size
figures/type<T>/            one figure per instance size
figures/summary/            the figures across sizes
```

## Workflow

One call sweeps a single instance from easy to hard, writes one row per k into the group file of that size and stops at the first
timeout, because no harder residual can be faster. Every call is resumable, so a crash or a reboot just continues where it stopped.

```
sage -python scripts/measure_file.py --file samples/challenge/Challenge-type1/Challenge-type1-n60/challenge-1-60-0.bz2 --per-solve-timeout 14800
```

Underdetermined instances need one extra flag. It fixes the surplus n minus m
variables to zero, which leaves a square subsystem that is swept exactly like a
type 1 instance.

```
sage -python scripts/measure_file.py --zero-surplus --file samples/challenge/Challenge-type4/Challenge-type4-m60/challenge-4-60-0.bz2 --per-solve-timeout 14800
```

The bash script does the same for a whole size range, one seed at a time, and covers type 1 and type 4 at every size. Give each terminal a disjoint range, since all seeds of one size share one file.

```
bash scripts/measure_range.sh 56 70
bash scripts/measure_range.sh 71 85
```

Once the rows are there the rest is fast and needs no Sage.

```
python scripts/estimate_kstar.py     # writes results/estimates.csv and the details json
python scripts/plot_instance.py      # one figure per size, into figures/type<T>/
python scripts/plot_summary.py       # the figures across sizes, into figures/summary/
```

The per size figure has two panels on a shared k axis. Panel A shows the measured solving times with their trend and the guess count q^k beside them, so both factors of the cost are visible. Panel B shows their product, the U shaped cost curve, with k* marked at its minimum.

A separate script runs the same sweep with two Gröbner engines on identical systems, so the difference between the curves is the algorithm and not the implementation.

```
sage -python scripts/algorithm_comparison.py --file <instance>
python3 scripts/algorithm_comparison.py --plot-only
```

## Hardware

Everything that touches Sage runs inside WSL2 with the conda environment active. Plotting alone works with plain Python. All measurements were taken on a single Windows desktop.

```
CPU         Intel Core i5 12400F, six cores and twelve threads at around 4.2 GHz
RAM         64 GB DDR4 at around 3600 MHz
SageMath    10.7, released on 9 August 2025
Conda       26.1.0
Environment Conda 4.12.0 in WSL2 running Ubuntu
```

The runs were driven by the bash script above, which deals the instance sizes out to three terminals, so every run has at least two physical cores available. Each sweep stores its measurements in its own csv file.