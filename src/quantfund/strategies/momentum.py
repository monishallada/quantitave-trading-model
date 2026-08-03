"""Cross-sectional momentum sleeve, upgraded:

- 12-1 style scoring (126-bar return, skipping the last 5 bars).
- ABSOLUTE momentum filter: only names whose own score is positive are held —
  in a broad downtrend the sleeve goes to cash rather than buying the
  "least bad" losers (dual-momentum evidence: cuts deep drawdowns).
- Inverse-volatility position sizing: each name's weight scales with the
  inverse of its 20-bar realized vol, normalized across the book
  (vol-managed sizing improves risk-adjusted returns).
- Regime filter: weights halve when the shared trend regime is "down".
"""
from __future__ import annotations

import math

from quantfund.core.instruments import Equity
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy


class MomentumStrategy(Strategy):
    strategy_id = "momentum"
    name = "Momentum (12-1, vol-scaled)"
    warmup_bars = 130
    uses_options = False

    def __init__(self, top_n: int = 5, deploy_fraction: float = 0.90,
                 max_name_weight: float = 0.25, lookback: int = 126,
                 skip: int = 5):
        self.top_n = top_n
        self.deploy_fraction = deploy_fraction
        self.max_name_weight = max_name_weight
        self.lookback = lookback
        self.skip = skip

    @staticmethod
    def _realized_vol(closes: list[float], window: int = 20) -> float | None:
        if len(closes) < window + 1:
            return None
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(-window, 0)
                if closes[i - 1] > 0]
        if len(rets) < window - 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        vol = math.sqrt(var) * math.sqrt(252.0)
        return vol if vol > 1e-6 else None

    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext) -> list[Signal]:
        scores: dict[str, float] = {}
        vols: dict[str, float] = {}
        for sym in snapshot.bars.keys():
            bars = snapshot.get_bars(sym)
            if len(bars) < self.warmup_bars:
                continue
            closes = [b.close for b in bars]
            past = closes[-(self.lookback + self.skip)]
            recent = closes[-(self.skip + 1)]
            if past <= 0:
                continue
            vol = self._realized_vol(closes)
            if vol is None:
                continue
            scores[sym] = recent / past - 1.0
            vols[sym] = vol
        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        # absolute-momentum filter: positive 12-1 return required to be held
        top = [sym for sym, s in ranked[: self.top_n] if s > 0.0]

        regime = snapshot.get_regime()
        regime_scale = 1.0
        regime_note = "no regime signal"
        if regime is not None:
            regime_note = f"trend={regime.trend_regime} vol={regime.vol_regime}"
            if regime.trend_regime == "down":
                regime_scale = 0.5
                regime_note += " → weights halved (downtrend)"

        # inverse-vol sizing across the accepted names
        weights: dict[str, float] = {}
        if top:
            inv = {sym: 1.0 / vols[sym] for sym in top}
            total_inv = sum(inv.values())
            for sym in top:
                w = self.deploy_fraction * inv[sym] / total_inv * regime_scale
                weights[sym] = min(w, self.max_name_weight)

        signals: list[Signal] = []
        held = {k for k, p in ctx.positions.items() if p.is_open}
        for rank, sym in enumerate(top, start=1):
            signals.append(Signal(
                instrument=Equity(sym),
                action=SignalAction.TARGET_WEIGHT,
                target_weight=weights[sym],
                rationale=(f"momentum rank {rank}/{self.top_n}: 12-1 "
                           f"{scores[sym]:+.1%}, vol {vols[sym]:.0%} → "
                           f"w={weights[sym]:.1%}; {regime_note}"),
                strategy_id=self.strategy_id,
                ts=snapshot.as_of,
                confidence=min(1.0, 0.5 + 0.1 * (self.top_n - rank)),
                rationale_payload={
                    "rank": rank, "score": scores[sym], "vol": vols[sym],
                    "weight": weights[sym], "regime": regime_note,
                    "universe_scored": len(scores),
                    "absolute_filter": "score>0 required",
                },
            ))
        for key in held - set(top):
            pos = ctx.positions[key]
            if isinstance(pos.instrument, Equity):
                signals.append(Signal(
                    instrument=pos.instrument,
                    action=SignalAction.CLOSE,
                    target_weight=0.0,
                    rationale=(f"{key} exited momentum book (out of top "
                               f"{self.top_n} or 12-1 turned negative)"),
                    strategy_id=self.strategy_id,
                    ts=snapshot.as_of,
                    confidence=0.7,
                    rationale_payload={"reason": "rank_or_sign_exit",
                                       "score": scores.get(key)},
                ))
        return signals
