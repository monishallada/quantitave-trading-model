"""Convexity sleeve (HIGH VARIANCE — built for the explosive paper profile).

Buys short-dated (~7-30 DTE, ~45-delta) LONG options for leveraged, two-sided
exposure: calls on the strongest 20-day momentum names, puts on the weakest —
so the book profits from trends in EITHER direction. The shared regime signal
tilts the call/put mix bearish in downtrends and bullish in uptrends.

Every position is a long option: max loss is the premium paid (defined risk —
this sleeve never violates the naked-short ban). The explosiveness comes from
convexity and concentration, not undefined risk. Expect large swings and heavy
theta bleed in flat markets; this sleeve exists to widen the distribution, not
to raise expected value.

Exits: take-profit at 3x premium, stop at 0.4x, time exit at <=3 DTE, and a
signal exit when the underlying's momentum flips against the position.
"""
from __future__ import annotations

from quantfund.core.instruments import Option, OptionRight
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy


class ConvexMomentumStrategy(Strategy):
    """2023-2025 backtest crackdown (v2): the v1 sleeve lost 45% over three
    years from (a) whipsaw entries off a bare 20d signal, (b) 7-DTE theta
    burn, and (c) buying puts into a bull market. v2 therefore:
    - CALLS require the validated 126d momentum signal (>+10%) CONFIRMED by a
      strong 20d move (>+5%) and a non-down trend regime;
    - PUTS only fire in high-vol or downtrend regimes — the environments
      where long puts historically pay — with a strong negative 20d move;
    - DTE window 14-35 (≈half the daily theta bleed of 7-DTE contracts).
    """

    strategy_id = "convexity"
    name = "Convexity (long calls/puts, 2-way)"
    warmup_bars = 130
    uses_options = True

    def __init__(self, lookback: int = 20, slow_lookback: int = 126,
                 min_abs_score: float = 0.05, slow_min_score: float = 0.10,
                 target_delta: float = 0.45, min_dte: float = 14,
                 max_dte: float = 35, exposure_weight: float = 0.30,
                 max_premium_fraction: float = 0.50,
                 take_profit_mult: float = 3.0, stop_mult: float = 0.4,
                 exit_dte: float = 3.0):
        self.lookback = lookback
        self.slow_lookback = slow_lookback
        self.min_abs_score = min_abs_score
        self.slow_min_score = slow_min_score
        self.target_delta = target_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        # sized by DELTA EXPOSURE (what the risk caps measure), not premium:
        # target |delta|·contracts·100·spot ≈ exposure_weight · sleeve capital
        self.exposure_weight = exposure_weight
        self.max_premium_fraction = max_premium_fraction
        self.take_profit_mult = take_profit_mult
        self.stop_mult = stop_mult
        self.exit_dte = exit_dte

    def _scores(self, snapshot: MarketSnapshot) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym in snapshot.bars.keys():
            bars = snapshot.get_bars(sym)
            if len(bars) < self.warmup_bars:
                continue
            closes = [b.close for b in bars]
            if closes[-(self.lookback + 1)] > 0:
                out[sym] = closes[-1] / closes[-(self.lookback + 1)] - 1.0
        return out

    def _slow_scores(self, snapshot: MarketSnapshot) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym in snapshot.bars.keys():
            bars = snapshot.get_bars(sym)
            if len(bars) < self.warmup_bars:
                continue
            closes = [b.close for b in bars]
            if closes[-(self.slow_lookback + 1)] > 0:
                out[sym] = closes[-1] / closes[-(self.slow_lookback + 1)] - 1.0
        return out

    def _manage_open(self, snapshot: MarketSnapshot, ctx: SleeveContext,
                     scores: dict[str, float]) -> tuple[list[Signal], set[str]]:
        signals: list[Signal] = []
        occupied: set[str] = set()
        for key, pos in ctx.positions.items():
            inst = pos.instrument
            if not (isinstance(inst, Option) and pos.is_open and pos.qty > 0):
                continue
            occupied.add(inst.underlying_symbol)
            dte = inst.days_to_expiration(snapshot.as_of)
            mark = pos.mark if pos.mark is not None else pos.avg_cost
            entry = pos.avg_cost
            score = scores.get(inst.underlying_symbol)
            reason = None
            if entry > 0 and mark >= entry * self.take_profit_mult:
                reason = (f"take profit: {mark:.2f} >= "
                          f"{self.take_profit_mult:.0f}x entry {entry:.2f}")
            elif entry > 0 and mark <= entry * self.stop_mult:
                reason = (f"premium stop: {mark:.2f} <= "
                          f"{self.stop_mult:.1f}x entry {entry:.2f}")
            elif dte <= self.exit_dte:
                reason = f"time exit: {dte:.0f} DTE <= {self.exit_dte:.0f}"
            elif score is not None and (
                    (inst.right == OptionRight.CALL and score < 0)
                    or (inst.right == OptionRight.PUT and score > 0)):
                reason = (f"momentum flipped against position "
                          f"({self.lookback}d score {score:+.1%})")
            if reason:
                signals.append(Signal(
                    instrument=inst, action=SignalAction.CLOSE, target_weight=0.0,
                    rationale=f"convexity {reason}", strategy_id=self.strategy_id,
                    ts=snapshot.as_of, confidence=0.8,
                    rationale_payload={"reason": reason, "entry": entry,
                                       "mark": mark, "dte": dte, "score": score},
                ))
                occupied.discard(inst.underlying_symbol)
        return signals, occupied

    def _pick_contract(self, snapshot: MarketSnapshot, underlying: str,
                       right: OptionRight):
        chain = snapshot.get_option_chain(underlying)
        if chain is None:
            return None
        oq = chain.nearest_delta(right, self.target_delta,
                                 min_dte=self.min_dte, max_dte=self.max_dte)
        if oq is None or oq.quote.bid <= 0 or oq.quote.ask <= 0:
            return None
        return oq

    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext) -> list[Signal]:
        scores = self._scores(snapshot)
        signals, occupied = self._manage_open(snapshot, ctx, scores)
        if not scores or ctx.allocated_capital <= 0:
            return signals

        regime = snapshot.get_regime()
        trend = regime.trend_regime if regime else "sideways"
        vol = regime.vol_regime if regime else "normal"
        n_calls, n_puts = {"up": (2, 1), "down": (0, 3)}.get(trend, (2, 2))
        regime_note = (f"trend={trend} vol={vol} → up to {n_calls} calls / "
                       f"{n_puts} puts")
        slow = self._slow_scores(snapshot)

        # only underlyings with a chain in the snapshot can be traded
        optionable = [s for s in scores if snapshot.get_option_chain(s) is not None]
        ranked = sorted(optionable, key=lambda s: scores[s], reverse=True)
        # CALLS: validated slow momentum + fast confirmation, never in a
        # downtrend regime (2023-25 evidence: unconfirmed calls bleed theta)
        if trend == "down":
            longs: list[str] = []
        else:
            longs = [s for s in ranked
                     if scores[s] > self.min_abs_score
                     and slow.get(s, 0.0) > self.slow_min_score
                     and s not in occupied][:n_calls]
        # PUTS: only where long puts historically pay — vol spikes / downtrends
        if vol == "high" or trend == "down":
            shorts = [s for s in reversed(ranked)
                      if scores[s] < -self.min_abs_score
                      and s not in occupied][:n_puts]
        else:
            shorts = []

        for symbols, right, label in ((longs, OptionRight.CALL, "call"),
                                      (shorts, OptionRight.PUT, "put")):
            for sym in symbols:
                oq = self._pick_contract(snapshot, sym, right)
                if oq is None:
                    continue
                inst = oq.instrument
                mid = oq.quote.mid
                if mid <= 0:
                    continue
                chain = snapshot.get_option_chain(sym)
                under_px = chain.underlying_price if chain else 0.0
                delta_abs = abs(oq.greeks.delta) if oq.greeks else self.target_delta
                expo_per_contract = delta_abs * 100.0 * under_px
                if expo_per_contract <= 0:
                    continue
                target_expo = self.exposure_weight * ctx.allocated_capital
                contracts = max(1, int(target_expo // expo_per_contract))
                premium_cost = contracts * mid * 100.0
                # premium affordability guard (deep-ITM or fat contracts)
                while contracts > 1 and premium_cost > \
                        self.max_premium_fraction * ctx.allocated_capital:
                    contracts -= 1
                    premium_cost = contracts * mid * 100.0
                if premium_cost > self.max_premium_fraction * ctx.allocated_capital:
                    continue
                weight = (premium_cost * 1.001) / ctx.allocated_capital
                if weight > 1.0:
                    continue
                dte = inst.days_to_expiration(snapshot.as_of)
                delta = oq.greeks.delta if oq.greeks else None
                signals.append(Signal(
                    instrument=inst,
                    action=SignalAction.OPEN_LONG,
                    target_weight=weight,
                    rationale=(f"convexity {label}: {sym} {self.lookback}d "
                               f"{scores[sym]:+.1%} → buy {contracts}x "
                               f"{inst.strike:g}{'C' if right == OptionRight.CALL else 'P'} "
                               f"{dte:.0f}DTE @ ~{mid:.2f} (max loss = premium); "
                               f"{regime_note}"),
                    strategy_id=self.strategy_id,
                    ts=snapshot.as_of,
                    confidence=min(1.0, abs(scores[sym]) * 10),
                    rationale_payload={
                        "underlying": sym, "score": scores[sym],
                        "slow_score_126d": slow.get(sym), "right": label,
                        "strike": inst.strike, "dte": dte, "delta": delta,
                        "iv": oq.iv, "premium_mid": mid, "contracts": contracts,
                        "max_loss": contracts * mid * 100.0, "regime": regime_note,
                        "entry_gates": ("calls: 126d>10% + 20d>5% + trend!=down; "
                                        "puts: vol=high or trend=down"),
                        "exits": {"take_profit_mult": self.take_profit_mult,
                                  "stop_mult": self.stop_mult,
                                  "exit_dte": self.exit_dte},
                    },
                ))
        return signals
