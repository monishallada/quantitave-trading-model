"""Cross-sectional momentum sleeve:

- 12-1 style scoring (126-bar return, skipping the last 5 bars).
- ABSOLUTE momentum filter: only names whose own score is positive are held —
  in a broad downtrend the sleeve goes to cash rather than buying the
  "least bad" losers (dual-momentum evidence: cuts deep drawdowns).
- Inverse-volatility position sizing: each name's weight scales with the
  inverse of its 20-bar realized vol, normalized across the book.
- Regime filter: weights halve when the shared trend regime is "down".

``express_via`` selects the INSTRUMENT used to take that exposure:
  "equity"  — buy shares (conservative default).
  "options" — STOCK REPLACEMENT: buy deep-ITM (~70-delta) calls 45-90 DTE
              instead of shares. Same validated signal, ~2-3x the delta per
              dollar, max loss capped at premium, and far less cash tied up
              (which leaves room for the cash-secured put sleeve). Deep ITM +
              long dated is deliberately the LOW-theta way to hold options —
              this is leverage, not a lottery ticket.
"""
from __future__ import annotations

import math

from quantfund.core.instruments import Equity, Option, OptionRight
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy


class MomentumStrategy(Strategy):
    strategy_id = "momentum"
    name = "Momentum (12-1, vol-scaled)"
    warmup_bars = 130
    uses_options = False

    def __init__(self, top_n: int = 5, deploy_fraction: float = 0.90,
                 max_name_weight: float = 0.25, lookback: int = 126,
                 skip: int = 5, express_via: str = "equity",
                 option_delta: float = 0.70, option_min_dte: float = 45,
                 option_max_dte: float = 90, option_leverage: float = 2.5,
                 option_max_premium_weight: float = 0.30,
                 option_roll_dte: float = 21,
                 universe: list[str] | None = None,
                 exclude: list[str] | None = None,
                 strategy_id: str | None = None, name: str | None = None):
        # instance-level id so equity and options expressions of this signal can
        # run side by side as separate sleeves (distinct capital + accounting)
        if strategy_id:
            self.strategy_id = strategy_id
        if name:
            self.name = name
        self.top_n = top_n
        self.deploy_fraction = deploy_fraction
        self.max_name_weight = max_name_weight
        self.lookback = lookback
        self.skip = skip
        if express_via not in ("equity", "options"):
            raise ValueError("express_via must be 'equity' or 'options'")
        self.express_via = express_via
        self.option_delta = option_delta
        self.option_min_dte = option_min_dte
        self.option_max_dte = option_max_dte
        self.option_leverage = option_leverage
        self.option_max_premium_weight = option_max_premium_weight
        self.option_roll_dte = option_roll_dte
        # Universe control: when an equity and an options expression of this
        # signal run side by side they otherwise rank the SAME names, take
        # duplicate exposure, and the equity sleeve eats the shared per-name
        # and per-sector budget the options sleeve needs (2026-08-04: options
        # momentum rejected for "sector TECH > 60%" caused by equity holdings).
        self.universe = set(universe) if universe else None
        self.exclude = set(exclude) if exclude else set()

    @property
    def uses_options(self) -> bool:  # type: ignore[override]
        return self.express_via == "options"

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
            if self.universe is not None and sym not in self.universe:
                continue
            if sym in self.exclude:
                continue
            # In options mode, rank ONLY over names that actually have a chain.
            # Ranking the full universe and then discarding un-optionable
            # winners silently starves the sleeve (2023-25 validation: 2 trades
            # in 3 years because the top ranks were rarely optionable).
            if self.express_via == "options" and \
                    snapshot.get_option_chain(sym) is None:
                continue
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
        # exits, keyed by UNDERLYING so option and equity holdings both resolve
        held_by_underlying: dict[str, list] = {}
        for key, pos in ctx.positions.items():
            if pos.is_open:
                held_by_underlying.setdefault(pos.instrument.underlying, []).append(pos)

        for underlying, positions in held_by_underlying.items():
            for pos in positions:
                inst = pos.instrument
                expiring = (isinstance(inst, Option)
                            and inst.days_to_expiration(snapshot.as_of)
                            <= self.option_roll_dte)
                if underlying in top and not expiring:
                    continue
                reason = ("roll: approaching expiry" if expiring and underlying in top
                          else f"exited momentum top {self.top_n} / 12-1 negative")
                signals.append(Signal(
                    instrument=inst,
                    action=SignalAction.CLOSE,
                    target_weight=0.0,
                    rationale=f"{inst.key} closed — {reason}",
                    strategy_id=self.strategy_id,
                    ts=snapshot.as_of,
                    confidence=0.7,
                    rationale_payload={"reason": reason,
                                       "score": scores.get(underlying)},
                ))

        for rank, sym in enumerate(top, start=1):
            base = {
                "rank": rank, "score": scores[sym], "vol": vols[sym],
                "regime": regime_note, "universe_scored": len(scores),
                "absolute_filter": "score>0 required",
                "express_via": self.express_via,
            }
            if self.express_via == "equity":
                signals.append(Signal(
                    instrument=Equity(sym),
                    action=SignalAction.TARGET_WEIGHT,
                    target_weight=weights[sym],
                    rationale=(f"momentum rank {rank}/{self.top_n}: 12-1 "
                               f"{scores[sym]:+.1%}, vol {vols[sym]:.0%} → "
                               f"w={weights[sym]:.1%}; {regime_note}"),
                    strategy_id=self.strategy_id, ts=snapshot.as_of,
                    confidence=min(1.0, 0.5 + 0.1 * (self.top_n - rank)),
                    rationale_payload={**base, "weight": weights[sym]},
                ))
                continue

            # ── stock replacement via deep-ITM calls ─────────────────────
            chain = snapshot.get_option_chain(sym)
            if chain is None:
                continue
            oq = chain.nearest_delta(OptionRight.CALL, self.option_delta,
                                     min_dte=self.option_min_dte,
                                     max_dte=self.option_max_dte)
            if oq is None or oq.quote.bid <= 0 or oq.quote.ask <= 0:
                # sparse chain: widen the window rather than skip the name,
                # but never accept anything shorter than the roll threshold
                oq = chain.nearest_delta(OptionRight.CALL, self.option_delta,
                                         min_dte=self.option_roll_dte + 7,
                                         max_dte=250)
            if oq is None or oq.quote.bid <= 0 or oq.quote.ask <= 0:
                continue
            inst = oq.instrument
            # skip if this exact contract is already held (avoids churn)
            if any(p.instrument.key == inst.key
                   for p in held_by_underlying.get(sym, [])):
                continue
            mid = oq.quote.mid
            spot = chain.underlying_price
            delta = abs(oq.greeks.delta) if oq.greeks else self.option_delta
            if mid <= 0 or spot <= 0 or delta <= 0:
                continue
            # target DELTA exposure = equity weight x leverage x sleeve capital;
            # convert to a premium weight the executor can size from
            target_delta_dollars = (weights[sym] * self.option_leverage
                                    * ctx.allocated_capital)
            # Options trade in WHOLE contracts. Round the delta-based size up to
            # 1 whole contract when affordable — fractional sizing on expensive
            # underlyings (a 0.4-contract target on $771 SPY) otherwise floors
            # to zero in the executor and the sleeve never trades.
            contracts = max(1, int(target_delta_dollars / (delta * 100.0 * spot)))
            premium_cap = self.option_max_premium_weight * ctx.allocated_capital
            while contracts > 1 and contracts * mid * 100.0 > premium_cap:
                contracts -= 1
            premium_cost = contracts * mid * 100.0
            if premium_cost > premium_cap:
                continue  # even one contract exceeds the sleeve's premium cap
            premium_weight = premium_cost * 1.001 / ctx.allocated_capital
            signals.append(Signal(
                instrument=inst,
                action=SignalAction.TARGET_WEIGHT,
                target_weight=premium_weight,
                rationale=(f"momentum rank {rank}/{self.top_n} via CALL: 12-1 "
                           f"{scores[sym]:+.1%}, vol {vols[sym]:.0%} → "
                           f"{inst.strike:g}C "
                           f"{inst.days_to_expiration(snapshot.as_of):.0f}DTE "
                           f"delta {delta:.2f} @ ~{mid:.2f} "
                           f"(stock replacement, {self.option_leverage:.1f}x "
                           f"target delta, max loss = premium); {regime_note}"),
                strategy_id=self.strategy_id, ts=snapshot.as_of,
                confidence=min(1.0, 0.5 + 0.1 * (self.top_n - rank)),
                rationale_payload={
                    **base, "equity_weight_equiv": weights[sym],
                    "premium_weight": premium_weight, "strike": inst.strike,
                    "delta": delta, "iv": oq.iv, "premium_mid": mid,
                    "dte": inst.days_to_expiration(snapshot.as_of),
                    "leverage": self.option_leverage,
                },
            ))
        return signals
