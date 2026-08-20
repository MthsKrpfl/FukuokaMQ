#!/bin/bash
#
# Measures a contiguous size range for seeds 0, 1 and 2, type 1 named by n and
# type 4 named by m with --zero-surplus, at every size. One solve per k and
# seed, which is exactly one row in the group's measurements.csv.
#
# The order follows the result files, one seed at a time:
#
#   measure 56 0, measure 57 0, ... measure 70 0
#   measure 56 1, measure 57 1, ... measure 70 1
#   measure 56 2, measure 57 2, ... measure 70 2
#
# Every run appends to results/type<T>/<label>_measurements.csv and is
# resumable, so a crash or reboot just continues where it stopped.
#
# Give each terminal a disjoint range, all seeds of one size share one group
# file and two terminals on the same size would interleave their appends:
#
#   bash scripts/measure_range.sh 56 70
#   bash scripts/measure_range.sh 71 85
#
# params:
#   n_min n_max  ; the size range, both included
#   SEEDS        ; env, which seeds to measure, default "0 1 2"
#   TIMEOUT      ; env, per-solve ceiling in seconds, default 14800
#
# Run from the project root with Sage active: conda activate sage

set -uo pipefail

N_MIN="${1:-}"
N_MAX="${2:-}"
SEEDS="${SEEDS:-0 1 2}"
TIMEOUT="${TIMEOUT:-14800}"

if ! [[ "$N_MIN" =~ ^[0-9]+$ ]] || ! [[ "$N_MAX" =~ ^[0-9]+$ ]] \
   || [ "$N_MIN" -gt "$N_MAX" ]; then
    echo "Usage: bash scripts/measure_range.sh <n_min> <n_max>   e.g. bash scripts/measure_range.sh 56 70"
    exit 1
fi

echo "sizes $N_MIN..$N_MAX, seeds $SEEDS, per-solve timeout ${TIMEOUT}s"
echo ""

# measure one instance file, one solve per k, appended to its group file
measure() {          # measure <type> <label n|m> <size> <seed> [extra flags...]
    local typ="$1" label="$2" size="$3" seed="$4"; shift 4
    local dir="samples/challenge/Challenge-type${typ}/Challenge-type${typ}-${label}${size}"
    local file="${dir}/challenge-${typ}-${size}-${seed}.bz2"
    local out="results/type${typ}/${label}${size}_measurements.csv"
    if [ ! -f "$file" ]; then
        echo "  !! measure $size $seed, type $typ: no file $file, skipping"
        return
    fi
    echo ">>> [$(date '+%F %T')] measure $size $seed, type $typ ${label}${size} -> $out"
    sage -python scripts/measure_file.py "$@" \
        --file "$file" \
        --specs 1 \
        --per-solve-timeout "$TIMEOUT"
    echo "<<< [$(date '+%F %T')] measure $size $seed, type $typ done, rc=$?"
    echo ""
}

# measure files from n_min to n_max for all seeds, seed by seed, so an aborted
# run leaves one finished seed instead of a fragment of each
for seed in $SEEDS; do
    for (( size = N_MIN; size <= N_MAX; size++ )); do
        measure 1 n "$size" "$seed"
        measure 4 m "$size" "$seed" --zero-surplus
    done
done

echo "finished sizes $N_MIN..$N_MAX for seeds $SEEDS"
echo "next:  python scripts/estimate_kstar.py && python scripts/plot_instance.py"
