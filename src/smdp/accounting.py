from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr


def gdp_delta(mu: float, eps: float) -> float:
    if mu <= 0:
        return 0.0
    a = -eps / mu + mu / 2.0
    b = -eps / mu - mu / 2.0
    la = log_ndtr(a)
    lb = eps + log_ndtr(b)
    if lb >= la:
        return 0.0
    return float(math.exp(la) * (-math.expm1(lb - la)))


def gdp_mu(eps: float, delta: float, tol: float = 1e-12) -> float:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    lo, hi = 0.0, 1.0
    while gdp_delta(hi, eps) < delta:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gdp_delta(mid, eps) <= delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def noise_multiplier(eps: float, delta: float) -> float:
    return 1.0 / gdp_mu(eps, delta)


def gdp_to_eps(mu: float, delta: float) -> float:
    lo, hi = 0.0, 1.0
    while gdp_delta(mu, hi) > delta:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gdp_delta(mu, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def rdp_noise_multiplier(eps: float, delta: float) -> float:
    orders = np.arange(1.01, 512.0, 0.01)

    def eps_of(z: float) -> float:
        return float(np.min(orders / (2 * z * z) + np.log(1 / delta) / (orders - 1)))

    lo, hi = 1e-3, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eps_of(mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class BudgetLedger:

    delta_target: float
    pure_eps: list[tuple[str, float]] = field(default_factory=list)
    gauss_mu: list[tuple[str, float]] = field(default_factory=list)

    def add_pure(self, name: str, eps: float) -> None:
        self.pure_eps.append((name, float(eps)))

    def add_gaussian(self, name: str, sensitivity: float, sigma: float) -> None:
        self.gauss_mu.append((name, float(sensitivity) / float(sigma)))

    def total(self) -> tuple[float, float]:
        e_pure = sum(e for _, e in self.pure_eps)
        if not self.gauss_mu:
            return e_pure, 0.0
        mu = math.sqrt(sum(m * m for _, m in self.gauss_mu))
        return e_pure + gdp_to_eps(mu, self.delta_target), self.delta_target

    def describe(self) -> str:
        e, d = self.total()
        parts = [f"{n}:eps={v:.4g}" for n, v in self.pure_eps]
        parts += [f"{n}:mu={v:.4g}" for n, v in self.gauss_mu]
        return f"total (eps={e:.4g}, delta={d:.1e}) <- " + ", ".join(parts)
