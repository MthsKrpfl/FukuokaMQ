"""
SHA-256 of a file, read in chunks so a large .bz2 never sits in memory as a whole.
The hash identifies the exact input a measurement was taken on.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
