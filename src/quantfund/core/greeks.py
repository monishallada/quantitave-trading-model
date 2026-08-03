"""Black-Scholes pricing, Greeks, and implied volatility (no scipy dependency).

Conventions:
  * ``theta`` is PER CALENDAR DAY (annual theta / 365), per share.
  * ``vega`` is per 1 percentage point of vol (i.e. dPrice for sigma +0.01), per share.
  * ``delta``/``gamma`` are per share of underlying.
  * All values are per-share; multiply by qty * 100 for a contract position.

Used as the fallback pricing model in backtests and whenever the live chain
does not carry Greeks. Live trading prefers the Greeks delivered on the
Alpaca option chain snapshot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from quantfund.core.instruments import OptionRight

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day, per share
    vega: float   # per 1 vol point (0.01), per share
    rho: float = 0.0

    @staticmethod
    def zero() -> "Greeks":
        return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)


def bs_price(
    right: OptionRight,
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float = 0.04,
    q: float = 0.0,
) -> float:
    """Black-Scholes(-Merton) price. ``t`` in years, ``sigma`` annualized."""
    if s <= 0 or k <= 0:
        raise ValueError("spot and strike must be positive")
    if t <= 0 or sigma <= 0:
        # At/after expiry (or degenerate vol): intrinsic value.
        if right == OptionRight.CALL:
            return max(0.0, s - k)
        return max(0.0, k - s)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if right == OptionRight.CALL:
        return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)


def bs_greeks(
    right: OptionRight,
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float = 0.04,
    q: float = 0.0,
) -> Greeks:
    """Analytic Black-Scholes Greeks. Degenerate inputs return step-function delta."""
    if s <= 0 or k <= 0:
        raise ValueError("spot and strike must be positive")
    if t <= 0 or sigma <= 0:
        itm = s > k if right == OptionRight.CALL else s < k
        d = (1.0 if right == OptionRight.CALL else -1.0) if itm else 0.0
        return Greeks(delta=d, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = _norm_pdf(d1)
    if right == OptionRight.CALL:
        delta = math.exp(-q * t) * _norm_cdf(d1)
        theta_annual = (
            -(s * math.exp(-q * t) * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * k * math.exp(-r * t) * _norm_cdf(d2)
            + q * s * math.exp(-q * t) * _norm_cdf(d1)
        )
        rho = k * t * math.exp(-r * t) * _norm_cdf(d2) / 100.0
    else:
        delta = -math.exp(-q * t) * _norm_cdf(-d1)
        theta_annual = (
            -(s * math.exp(-q * t) * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * k * math.exp(-r * t) * _norm_cdf(-d2)
            - q * s * math.exp(-q * t) * _norm_cdf(-d1)
        )
        rho = -k * t * math.exp(-r * t) * _norm_cdf(-d2) / 100.0
    gamma = math.exp(-q * t) * pdf_d1 / (s * sigma * sqrt_t)
    vega = s * math.exp(-q * t) * pdf_d1 * sqrt_t / 100.0  # per 1 vol point
    return Greeks(delta=delta, gamma=gamma, theta=theta_annual / 365.0, vega=vega, rho=rho)


def implied_vol(
    right: OptionRight,
    price: float,
    s: float,
    k: float,
    t: float,
    r: float = 0.04,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Implied volatility via bisection. Returns None if no vol in [1e-4, 5.0] fits."""
    if t <= 0 or price <= 0:
        return None
    lo, hi = 1e-4, 5.0
    p_lo = bs_price(right, s, k, t, lo, r, q)
    p_hi = bs_price(right, s, k, t, hi, r, q)
    if not (p_lo <= price <= p_hi):
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(right, s, k, t, mid, r, q)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
