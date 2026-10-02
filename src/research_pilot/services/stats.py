"""Pure-Python statistics for the Quantitative Analyst (no model calls, no SciPy dependency)."""

import math
from typing import List, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: float) -> float:
    if df <= 0 or math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_ppf(q: float, df: float) -> float:
    """Inverse CDF of Student's t (q in (0.5, 1)), found by bisection on the two-sided tail."""
    tail = 2.0 * (1.0 - q)
    lo, hi = 0.0, 1.0
    while t_sf_two_sided(hi, df) > tail:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_sf_two_sided(mid, df) > tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float | None, float | None]:
    n = len(values)
    if n < 2:
        return None, None
    half = t_ppf(0.5 + confidence / 2.0, n - 1) * stdev(values) / math.sqrt(n)
    m = mean(values)
    return m - half, m + half


def welch(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> dict:
    """Welch's t-test for mean(a) - mean(b), with a confidence interval of the difference."""
    n1, n2 = len(a), len(b)
    diff = mean(a) - mean(b)
    out = {"diff": diff, "t": None, "df": None, "p_value": None, "ci_low": None, "ci_high": None}
    if n1 < 2 or n2 < 2:
        return out
    v1, v2 = stdev(a) ** 2 / n1, stdev(b) ** 2 / n2
    se = math.sqrt(v1 + v2)
    if se == 0:
        out.update({"t": math.inf if diff else 0.0, "df": float(n1 + n2 - 2), "p_value": 0.0 if diff else 1.0, "ci_low": diff, "ci_high": diff})
        return out
    df = (v1 + v2) ** 2 / ((v1**2) / (n1 - 1) + (v2**2) / (n2 - 1))
    t = diff / se
    half = t_ppf(0.5 + confidence / 2.0, df) * se
    out.update({"t": t, "df": df, "p_value": t_sf_two_sided(t, df), "ci_low": diff - half, "ci_high": diff + half})
    return out


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float | None:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    pooled = math.sqrt(((n1 - 1) * stdev(a) ** 2 + (n2 - 1) * stdev(b) ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return None
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return (mean(a) - mean(b)) / pooled * correction


def holm(p_values: Sequence[float | None]) -> List[float | None]:
    """Holm-Bonferroni adjusted p-values (None entries are passed through)."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: List[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, idx) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[idx] = running
    return adjusted


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    m = mean(values)
    if len(values) < 2 or m == 0:
        return None
    return abs(stdev(values) / m)
