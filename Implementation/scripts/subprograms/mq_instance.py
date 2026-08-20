"""
Container for one parsed MQ instance, its ring, polynomials and the known
solution if there is one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple


@dataclass
class Instance:

    instance_id: str
    typ: int
    q: int
    n: int
    m: int
    d: int
    ring: Any
    polys: Sequence[Any]
    solution: Optional[Tuple[Any, ...]] = None
    source_path: str = ""
    sha256: str = ""

    # smallest k worth guessing: the surplus n-m variables are free
    @property
    def k_min(self) -> int:
        return max(0, self.n - self.m)

    def __repr__(self) -> str:
        sol = "solved" if self.solution is not None else "unsolved"
        return (f"Instance({self.instance_id}, type={self.typ}, q={self.q}, "
                f"n={self.n}, m={self.m}, {sol})")
