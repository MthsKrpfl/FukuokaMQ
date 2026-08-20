"""
Shared bookkeeping, config, instance file names, the layout of the results
tree, and the CSV schema every measurement row follows.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

# this file sits in scripts/subprograms/, so the root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# one row per completed measurement, keyed by instance, k and spec
FIELDNAMES = [
    "type", "instance_id", "q", "n", "m", "k", "n_minus_k",
    "seed", "gb_time_s", "gb_size", "status", "hash",
]


# read scripts/config.json, which sits one level up from here
def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else (Path(__file__).resolve().parent.parent
                                        / "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# make a config path absolute
def resolve(cfg_path_value: str) -> Path:
    p = Path(cfg_path_value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# the middle number in a file name is only a label, the header decides n and m
_NAME_RE = re.compile(
    r"(?:challenge|ToyExample)-(?:type)?(?P<type>\d+)-n?(?P<param>\d+)"
    r"[-_](?:seed)?(?P<seed>\d+)", re.IGNORECASE)

_H_N = re.compile(r"Number of variables.*?:\s*(\d+)", re.IGNORECASE)
_H_M = re.compile(r"Number of (?:polynomials|equations).*?:\s*(\d+)", re.IGNORECASE)
_H_FIELD = re.compile(r"Galois\s+Field\s*:\s*(.+)", re.IGNORECASE)


# type, size label and seed from a file name
def parse_name(stem: str):
    m = _NAME_RE.search(stem)
    if not m:
        return None
    return int(m.group("type")), int(m.group("param")), int(m.group("seed"))


# field size from the header line
def _q_from_field(field_raw: str):
    fr = field_raw.replace(" ", "")
    if fr.startswith("GF(2)[x]") or "GF(2^8)" in fr or "GF(256)" in fr:
        return 256
    mq = re.match(r"GF\((\d+)\)", fr)
    return int(mq.group(1)) if mq else None


# true n, m and q from the header, a timed out run never reports them
def peek_header_nmq(path: Path):
    import bz2
    opener = (lambda: bz2.open(path, "rt", encoding="utf-8")) if path.suffix == ".bz2" \
        else (lambda: open(path, "rt", encoding="utf-8"))
    n = m = q = None
    with opener() as f:
        for _ in range(12):
            line = f.readline()
            if not line:
                break
            if n is None:
                mm = _H_N.search(line)
                if mm:
                    n = int(mm.group(1))
            if m is None:
                mm = _H_M.search(line)
                if mm:
                    m = int(mm.group(1))
            if q is None:
                mm = _H_FIELD.search(line)
                if mm:
                    q = _q_from_field(mm.group(1).strip())
            if n is not None and m is not None and q is not None:
                break
    if n is None or m is None:
        return None
    return n, m, q


# identity of a measurement, for resume and dedup
def row_key(row: dict):
    return (str(row.get("type")), str(row.get("instance_id")),
            str(row.get("k")), str(row.get("seed")))


# challenge types 4-6 are distributed named by m, toy files always by n
LABEL_BY_M_TYPES = {4, 5, 6}


def is_toy_id(instance_id) -> bool:
    return "toyexample" in str(instance_id or "").lower()


# directory label of one size group: m67 for challenge types 4-6, else n55
def group_label(typ, n, m, instance_id=None) -> str:
    try:
        typ_i = int(typ)
    except (TypeError, ValueError):
        typ_i = -1
    if typ_i in LABEL_BY_M_TYPES and not is_toy_id(instance_id):
        return f"m{int(m)}"
    return f"n{int(n)}"


# one file per size group, holding every seed of that size
def group_csv(base_csv: Path, typ, n, m, instance_id=None) -> Path:
    label = group_label(typ, n, m, instance_id)
    return base_csv.parent / f"type{typ}" / f"{label}_{base_csv.stem}.csv"


# every csv shard the record is made of, older flat files included
def measurement_files(csv_path: Path):
    parent = csv_path.parent
    if not parent.exists():
        return [csv_path] if csv_path.exists() else []
    stem = csv_path.stem
    files = sorted(parent.glob(f"type*/*{stem}*.csv"))
    files += sorted(parent.glob(f"{stem}*.csv"))
    return files or ([csv_path] if csv_path.exists() else [])


def _gb_time(row: dict) -> float:
    try:
        return float(row["gb_time_s"])
    except (KeyError, TypeError, ValueError):
        return float("inf")


# a repeat re-times an identical computation: fastest wins, ok beats timeout
def dedup_records(rows: Iterable[dict]) -> dict:
    best = {}
    for row in rows:
        if row.get("status") not in ("ok", "timeout"):
            continue
        key = row_key(row)
        cur = best.get(key)
        if cur is None:
            best[key] = row
            continue
        if cur.get("status") != "ok" and row.get("status") == "ok":
            best[key] = row
        elif cur.get("status") == row.get("status") and _gb_time(row) < _gb_time(cur):
            best[key] = row
    return best
