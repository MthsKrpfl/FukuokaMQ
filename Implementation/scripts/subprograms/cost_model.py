"""
The empirical cost model behind estimate_kstar.py.

    T_GB(n') ~ exp(a + b*n' + c*n'^2)    n' = n - k, the residual size
    Cost(k)  = q^k * T_GB(n-k)           total work of the hybrid attack
    k*       = argmin Cost(k),  k in [k_min, n-1]

Fitted in log space by least squares, the quadratic term is dropped when there
are too few distinct residual sizes.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Dict, List, Optional, Sequence

import numpy as np


# {residual size n' -> [gb times]} over the ok rows
def aggregate_times(rows: Sequence[dict]) -> Dict[int, List[float]]:
    by_np: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        t = r.get("gb_time_s")
        nmk = r.get("n_minus_k")
        if t in (None, "", "None") or nmk in (None, "", "None"):
            continue
        try:
            t = float(t)
            nmk = int(nmk)
        except (TypeError, ValueError):
            continue
        if t <= 0 or nmk < 1:
            continue
        by_np.setdefault(nmk, []).append(t)
    return by_np


# one median point per residual size
def median_points(by_np: Dict[int, List[float]]):
    nps = sorted(by_np)
    med = [median(by_np[k]) for k in nps]
    return np.array(nps, dtype=float), np.array(med, dtype=float)


class InsufficientData(ValueError):
    """Not enough distinct residual sizes to identify a slope."""


# the fitted log-time model and the cost curve it implies
class CostModel:

    def __init__(self, coeffs: np.ndarray, q: int, n: int, m: int,
                 k_min: Optional[int] = None, flag: str = ""):
        self.coeffs = coeffs
        self.q = q
        self.n = n
        self.m = m
        self.k_min = max(0, n - m) if k_min is None else k_min
        self.flag = flag

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def log_time(self, nprime):
        return np.polyval(self.coeffs, np.asarray(nprime, dtype=float))

    def predict_time(self, nprime):
        return np.exp(self.log_time(nprime))

    # log10 cost over k, k*log10(q) + log10 t(n-k)
    def cost_curve(self):
        ks = np.arange(self.k_min, self.n)
        nprime = self.n - ks
        log10_cost = ks * math.log10(self.q) + self.log_time(nprime) / math.log(10.0)
        return ks, log10_cost

    # the k where that curve is lowest
    def kstar(self):
        ks, log10_cost = self.cost_curve()
        i = int(np.argmin(log10_cost))
        return int(ks[i]), float(log10_cost[i]), ks, log10_cost


# least squares in log space, a wrong-way curvature falls back to linear
def fit_model(nprimes: np.ndarray, times: np.ndarray, q: int, n: int, m: int,
              k_min: Optional[int] = None, max_degree: int = 2) -> CostModel:
    nprimes = np.asarray(nprimes, dtype=float)
    logt = np.log(np.asarray(times, dtype=float))
    ndist = len(np.unique(nprimes))
    if ndist < 2:
        raise InsufficientData(
            f"only {ndist} distinct n' point(s); need >= 2 to identify a slope")
    deg = min(max_degree, max(1, ndist - 1))
    coeffs = np.polyfit(nprimes, logt, deg)
    flag = ""
    if deg == 2 and coeffs[0] <= 0:
        coeffs = np.polyfit(nprimes, logt, 1)
        flag = "curvature<=0 -> linear (boundary k*)"
    return CostModel(coeffs, q=q, n=n, m=m, k_min=k_min, flag=flag)


# resample the times per residual size to get a confidence interval for k*
def bootstrap_kstar(by_np: Dict[int, List[float]], q: int, n: int, m: int,
                    k_min: Optional[int] = None, n_boot: int = 300,
                    max_degree: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    nps = sorted(by_np)
    ksamples = []
    for _ in range(n_boot):
        res_nps, res_times = [], []
        for npv in nps:
            arr = by_np[npv]
            draw = rng.choice(arr, size=len(arr), replace=True)
            res_nps.append(npv)
            res_times.append(float(np.median(draw)))
        try:
            model = fit_model(np.array(res_nps), np.array(res_times),
                              q=q, n=n, m=m, k_min=k_min, max_degree=max_degree)
            kstar, *_ = model.kstar()
            ksamples.append(kstar)
        except Exception:
            continue
    if not ksamples:
        return None, None, None, []
    arr = np.array(ksamples)
    return (int(np.median(arr)), int(np.percentile(arr, 5)),
            int(np.percentile(arr, 95)), ksamples)


# cost at every measured k directly, without any fit
def direct_cost_by_k(rows: Sequence[dict], q: int):
    by_k: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        t = r.get("gb_time_s")
        k = r.get("k")
        try:
            t = float(t)
            k = int(k)
        except (TypeError, ValueError):
            continue
        if t <= 0:
            continue
        by_k.setdefault(k, []).append(t)
    out = {}
    for k, ts in by_k.items():
        out[k] = k * math.log10(q) + math.log10(median(ts))
    return out
