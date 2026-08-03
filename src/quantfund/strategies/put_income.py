"""Put-income sleeve: harvests the volatility risk premium by selling
CASH-SECURED ~30-delta 30-45 DTE puts on liquid index ETFs / megacaps.

Why this sleeve exists: implied vol persistently trades above subsequently
realized vol (the variance risk premium) — one of the best-documented market
premia — and its return stream is structurally different from the directional
sleeves, which gives the allocator real diversification to work with.

Risk posture (all defined-risk; passes the platform's naked-short gate):
- Every short put is cash-secured: max loss = strike x 100 - premium.
- Cash committed to strikes capped at ``secured_cash_fraction`` of sleeve capital.
- Premium must clear a minimum annualized yield (don't sell cheap vol).
- No NEW entries in a high-vol regime (existing positions still managed).
- Profit-take at 50% of premium; time exit at <= 7 DTE (rolling off before
  expiry week keeps assignment rare); hard stop if the put trades at 2.5x entry.
"""
from __future__ import annotations

from quantfund.core.instruments import Option, OptionRight
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy


class PutIncomeStrategy(Strategy):
    strategy_id = "put_income"
    name = "Put Income (VRP, cash-secured)"
    warmup_bars = 30
    uses_options = True

    def __init__(self, target_delta: float = 0.30, min_dte: float = 30,
                 max_dte: float = 45, exit_dte: float = 7,
                 profit_take: float = 0.50, loss_stop_mult: float = 2.5,
                 min_annualized_yield: float = 0.08,
                 secured_cash_fraction: float = 0.90, max_underlyings: int = 2,
                 preferred: list[str] | None = None):
        self.target_delta = target_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.exit_dte = exit_dte
        self.profit_take = profit_take
        self.loss_stop_mult = loss_stop_mult
        self.min_annualized_yield = min_annualized_yield
        self.secured_cash_fraction = secured_cash_fraction
        self.max_underlyings = max_underlyings
        # index ETFs first; on small accounts their strikes exceed the secured-
        # cash budget and the loop naturally falls through to cheaper megacaps
        self.preferred = preferred or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA",
                                       "AMZN", "TSLA"]

    # ── exits ────────────────────────────────────────────────────────────

    def _manage_open(self, snapshot: MarketSnapshot,
                     ctx: SleeveContext) -> tuple[list[Signal], set[str]]:
        signals: list[Signal] = []
        open_put_underlyings: set[str] = set()
        for key, pos in ctx.positions.items():
            inst = pos.instrument
            if not (isinstance(inst, Option) and inst.right == OptionRight.PUT
                    and pos.is_open and pos.qty < 0):
                continue
            open_put_underlyings.add(inst.underlying_symbol)
            dte = inst.days_to_expiration(snapshot.as_of)
            mark = pos.mark if pos.mark is not None else pos.avg_cost
            entry = pos.avg_cost
            reason = None
            if entry > 0 and mark <= entry * (1.0 - self.profit_take):
                reason = (f"profit take: put marks {mark:.2f} <= "
                          f"{1 - self.profit_take:.0%} of entry {entry:.2f}")
            elif dte <= self.exit_dte:
                reason = f"time exit: {dte:.0f} DTE <= {self.exit_dte:.0f}"
            elif entry > 0 and mark >= entry * self.loss_stop_mult:
                reason = (f"loss stop: put marks {mark:.2f} >= "
                          f"{self.loss_stop_mult:.1f}x entry {entry:.2f}")
            if reason:
                signals.append(Signal(
                    instrument=inst, action=SignalAction.CLOSE, target_weight=0.0,
                    rationale=f"put-income {reason}", strategy_id=self.strategy_id,
                    ts=snapshot.as_of, confidence=0.8,
                    rationale_payload={"reason": reason, "entry": entry,
                                       "mark": mark, "dte": dte},
                ))
                open_put_underlyings.discard(inst.underlying_symbol)
        return signals, open_put_underlyings

    # ── entries ──────────────────────────────────────────────────────────

    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext) -> list[Signal]:
        signals, occupied = self._manage_open(snapshot, ctx)

        regime = snapshot.get_regime()
        if regime is not None and regime.vol_regime == "high":
            # rich premium but the tail is fat — no new short vol in a spike
            return signals
        if ctx.allocated_capital <= 0:
            return signals

        budget = self.secured_cash_fraction * ctx.allocated_capital
        # cash already securing existing short puts
        for pos in ctx.positions.values():
            inst = pos.instrument
            if (isinstance(inst, Option) and inst.right == OptionRight.PUT
                    and pos.is_open and pos.qty < 0):
                budget -= inst.strike * 100.0 * abs(pos.qty)
        slots = self.max_underlyings - len(occupied)
        if slots <= 0 or budget <= 0:
            return signals

        for underlying in self.preferred:
            if slots <= 0:
                break
            if underlying in occupied:
                continue
            chain = snapshot.get_option_chain(underlying)
            if chain is None:
                continue
            oq = chain.nearest_delta(OptionRight.PUT, self.target_delta,
                                     min_dte=self.min_dte, max_dte=self.max_dte)
            if oq is None or oq.quote.bid <= 0 or oq.quote.ask <= 0:
                continue
            inst = oq.instrument
            premium = oq.quote.bid  # we sell at the bid (cost model enforces)
            dte = inst.days_to_expiration(snapshot.as_of)
            if dte <= 0 or inst.strike <= 0:
                continue
            ann_yield = (premium / inst.strike) * (365.0 / dte)
            if ann_yield < self.min_annualized_yield:
                continue  # don't sell cheap vol
            secured_per_contract = inst.strike * 100.0
            contracts = int(budget // secured_per_contract)
            if contracts < 1:
                continue
            contracts = min(contracts, 3)  # per-name sanity cap
            # weight is denominated at the MID (the executor sizes qty off mid);
            # the 1.001 epsilon survives integer flooring in weight_to_qty
            mid = oq.quote.mid
            weight = -(contracts * mid * 100.0 * 1.001) / ctx.allocated_capital
            if weight >= 0 or weight < -1.0:
                continue
            delta = oq.greeks.delta if oq.greeks else None
            signals.append(Signal(
                instrument=inst,
                action=SignalAction.OPEN_SHORT,
                target_weight=weight,
                rationale=(f"sell {contracts}x cash-secured {underlying} "
                           f"{inst.strike:g}P {dte:.0f}DTE @ ~{premium:.2f} "
                           f"({ann_yield:.0%} ann. yield, "
                           f"delta {delta if delta is None else round(delta, 2)}, "
                           f"vol regime="
                           f"{'n/a' if regime is None else regime.vol_regime})"),
                strategy_id=self.strategy_id,
                ts=snapshot.as_of,
                confidence=0.6,
                rationale_payload={
                    "underlying": underlying, "strike": inst.strike,
                    "premium_bid": premium, "dte": dte, "delta": delta,
                    "iv": oq.iv, "annualized_yield": ann_yield,
                    "contracts": contracts,
                    "cash_secured": secured_per_contract * contracts,
                    "exit_rules": {"profit_take": self.profit_take,
                                   "exit_dte": self.exit_dte,
                                   "loss_stop_mult": self.loss_stop_mult},
                },
            ))
            budget -= secured_per_contract * contracts
            slots -= 1
        return signals
