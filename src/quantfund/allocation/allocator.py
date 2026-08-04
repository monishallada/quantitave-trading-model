"""Capital allocator: distributes simulated capital across sleeves by
risk-adjusted performance and correlation, concentrating into what's working,
under hard per-sleeve caps. Deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantfund.core.config import RiskLimits


@dataclass
class SleevePerf:
    strategy_id: str
    daily_returns: list[float]  # oldest first


class CapitalAllocator:
    def __init__(self, limits: RiskLimits, lookback: int = 60,
                 min_weight: float = 0.05,
                 base_weights: dict[str, float] | None = None):
        """``base_weights`` sets the intended capital split BEFORE any live
        performance history exists (equal-weighting a 4-sleeve book caps the
        main options engine at 25% of equity, which makes a 50%-in-options
        target unreachable). Once returns accumulate, performance and
        correlation scoring blend in and can move capital away from a base
        weight that isn't earning."""
        self.limits = limits
        self.lookback = lookback
        self.min_weight = min_weight
        self.base_weights = base_weights or {}

    def allocate(self, total_equity: float, sleeves: list[SleevePerf]) -> dict[str, float]:
        if total_equity <= 0 or not sleeves:
            return {s.strategy_id: 0.0 for s in sleeves}
        n = len(sleeves)
        if self.base_weights:
            base = np.array([self.base_weights.get(s.strategy_id, 1.0 / n)
                             for s in sleeves], dtype=float)
            base = base / base.sum() if base.sum() > 0 else np.full(n, 1.0 / n)
        else:
            base = np.full(n, 1.0 / n)
        equal = base  # the "no evidence yet" anchor the scores shrink toward

        # ── raw risk-adjusted score with shrinkage toward equal weight ────
        scores = np.zeros(n)
        history = []
        for i, s in enumerate(sleeves):
            rets = np.asarray(s.daily_returns[-self.lookback:], dtype=float)
            history.append(rets)
            if rets.size < 5:
                scores[i] = 0.0  # no evidence → pure equal weight via shrinkage
                continue
            std = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
            mean = float(rets.mean())
            if std > 1e-12:
                sharpe = mean / std * np.sqrt(252.0)
            else:
                # zero-variance stream: sign of the mean decides, capped
                sharpe = float(np.sign(mean)) * 10.0 if abs(mean) > 1e-12 else 0.0
            confidence = min(1.0, rets.size / self.lookback)
            scores[i] = sharpe * confidence

        # ── correlation penalty: crowded sleeves score lower ─────────────
        penalties = np.zeros(n)
        min_len = min((h.size for h in history), default=0)
        if n > 1 and min_len >= 10:
            aligned = np.vstack([h[-min_len:] for h in history])
            stds = aligned.std(axis=1, ddof=1)
            valid = stds > 1e-12
            if valid.sum() > 1:
                corr = np.corrcoef(aligned[valid])
                idx = np.flatnonzero(valid)
                for k, i in enumerate(idx):
                    others = [corr[k, j] for j in range(len(idx)) if j != k]
                    penalties[i] = max(0.0, float(np.mean(others)))

        # score → weight: softmax-free monotone map, shrunk toward equal
        adj = scores * (1.0 - 0.5 * penalties)
        # map to positive mass: shift so the worst sleeve still gets floor weight
        mass = np.maximum(adj - adj.min() + 0.1, 0.1)
        raw = mass / mass.sum()
        # shrink toward equal weight by (1 - avg confidence) so thin history
        # cannot cause concentration
        avg_conf = float(np.mean([min(1.0, h.size / self.lookback) for h in history])) \
            if history else 0.0
        weights = avg_conf * raw + (1.0 - avg_conf) * equal

        # ── floors and caps, then renormalize ────────────────────────────
        cap = self.limits.max_strategy_pct
        floor = min(self.min_weight, float(base.min()))
        weights = np.clip(weights, floor, cap)
        # iterative renormalization under the cap
        for _ in range(10):
            total = weights.sum()
            if total <= 1.0 + 1e-9:
                break
            over = weights > floor
            excess = total - 1.0
            reducible = weights[over] - floor
            if reducible.sum() <= 1e-12:
                break
            weights[over] -= excess * reducible / reducible.sum()
            weights = np.clip(weights, floor, cap)
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

        # ── portfolio vol targeting ──────────────────────────────────────
        # Scale total deployment toward target_portfolio_vol: vol-managed
        # portfolios earn better risk-adjusted returns and cut tail risk.
        # Needs >= 20 aligned observations; never scales UP above 1.0 and
        # never below a 30% deployment floor.
        deployment = 1.0
        min_len = min((h.size for h in history), default=0)
        target = getattr(self.limits, "target_portfolio_vol", 0.0)
        if target > 0 and min_len >= 20:
            aligned = np.vstack([h[-min_len:] for h in history])
            port_rets = weights @ aligned
            port_vol = float(port_rets.std(ddof=1)) * np.sqrt(252.0)
            if port_vol > 1e-9:
                deployment = float(np.clip(target / port_vol, 0.30, 1.0))

        return {
            s.strategy_id: float(round(w * deployment * total_equity, 2))
            for s, w in zip(sleeves, weights)
        }
