"""Short-horizon mean-reversion sleeve, upgraded: buys 5-day washouts ONLY in
confirmed uptrends (price above its 200-bar average) — dip-buying works in
uptrends and catches knives in downtrends. Long-only; stands down entirely in
high-vol regimes; exits on reversion."""
from __future__ import annotations

import math

from quantfund.core.instruments import Equity
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy


class MeanReversionStrategy(Strategy):
    strategy_id = "mean_reversion"
    name = "Mean Reversion (5d z, trend-filtered)"
    warmup_bars = 210
    uses_options = False

    def __init__(self, entry_z: float = -1.5, exit_z: float = -0.25,
                 max_names: int = 5, trend_window: int = 200):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_names = max_names
        self.trend_window = trend_window

    def _zscore(self, closes: list[float]) -> float | None:
        if len(closes) < 66:
            return None
        r5 = [closes[i] / closes[i - 5] - 1.0 for i in range(5, len(closes))
              if closes[i - 5] > 0]
        if len(r5) < 60:
            return None
        window = r5[-60:]
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
        std = math.sqrt(var)
        if std < 1e-9:
            return None
        return (r5[-1] - mean) / std

    def _in_uptrend(self, closes: list[float]) -> bool:
        if len(closes) < self.trend_window:
            return False
        sma = sum(closes[-self.trend_window:]) / self.trend_window
        return closes[-1] > sma

    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext) -> list[Signal]:
        regime = snapshot.get_regime()
        high_vol = regime is not None and regime.vol_regime == "high"

        zscores: dict[str, float] = {}
        uptrend: dict[str, bool] = {}
        for sym in snapshot.bars.keys():
            bars = snapshot.get_bars(sym)
            if len(bars) < self.warmup_bars:
                continue
            closes = [b.close for b in bars]
            z = self._zscore(closes)
            if z is not None:
                zscores[sym] = z
                uptrend[sym] = self._in_uptrend(closes)

        signals: list[Signal] = []
        held = {k for k, p in ctx.positions.items() if p.is_open}

        # exits first
        for key in held:
            z = zscores.get(key)
            pos = ctx.positions[key]
            if not isinstance(pos.instrument, Equity):
                continue
            if z is None or z > self.exit_z:
                signals.append(Signal(
                    instrument=pos.instrument,
                    action=SignalAction.CLOSE,
                    target_weight=0.0,
                    rationale=(f"mean-reversion exit: z={z:+.2f} > {self.exit_z}"
                               if z is not None else "mean-reversion exit: no data"),
                    strategy_id=self.strategy_id,
                    ts=snapshot.as_of,
                    confidence=0.6,
                    rationale_payload={"z": z, "exit_threshold": self.exit_z},
                ))

        if high_vol:
            # entering washouts during a vol spike is knife-catching — stand down
            return signals

        candidates = sorted(
            ((s, z) for s, z in zscores.items()
             if z < self.entry_z and s not in held and uptrend.get(s, False)),
            key=lambda kv: kv[1],
        )
        slots = max(0, self.max_names - len(held))
        for sym, z in candidates[:slots]:
            w = min(0.10, 0.05 * min(abs(z), 3.0) / abs(self.entry_z))
            signals.append(Signal(
                instrument=Equity(sym),
                action=SignalAction.TARGET_WEIGHT,
                target_weight=w,
                rationale=(f"mean-reversion entry: 5d z={z:+.2f} < {self.entry_z}, "
                           f"price above {self.trend_window}d SMA (uptrend dip) "
                           f"(vol regime={'n/a' if regime is None else regime.vol_regime})"),
                strategy_id=self.strategy_id,
                ts=snapshot.as_of,
                confidence=min(1.0, abs(z) / 3.0),
                rationale_payload={"z": z, "entry_threshold": self.entry_z,
                                   "weight": w, "trend_filter": "close > 200d SMA"},
            ))
        return signals
