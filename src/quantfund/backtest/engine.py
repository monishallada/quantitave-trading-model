"""Event-driven daily backtest engine.

Discipline that makes the results honest:
- Strategies only ever see a MarketSnapshot pinned to day d's close.
- Orders generated from day d execute at day d+1 (t+1) — never on the bar that
  produced the signal — at day d+1's OPEN with a synthetic spread, through the
  same CostModel as the sim broker. Zero-cost fills are impossible.
- Every simplification is recorded in BacktestResult.warnings.

The engine deliberately does NOT import risk/allocation/execution modules —
an allocator and risk manager are duck-typed optional injections, so the
harness stays testable in isolation. Sleeve accounting reuses core Portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import math

from quantfund.backtest.metrics import PerfMetrics, compute_metrics
from quantfund.core.costs import CostConfig, CostModel
from quantfund.core.greeks import bs_greeks, bs_price
from quantfund.core.instruments import Equity, Option
from quantfund.core.config import Settings
from quantfund.core.orders import Fill, Order, OrderSide, single_leg
from quantfund.core.portfolio import Portfolio
from quantfund.core.snapshot import Bar, MarketSnapshot, Quote
from quantfund.data.provider import DataProvider
from quantfund.strategies.base import (
    Signal, SignalAction, SleeveContext, Strategy,
)


def _realized_sigma(bars: list[Bar], as_of, window: int = 21) -> Optional[float]:
    closes = [b.close for b in bars if b.ts <= as_of][-(window + 1):]
    if len(closes) < window // 2:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i - 1] > 0]
    if len(rets) < 5:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return max(0.10, math.sqrt(var) * math.sqrt(252.0))


@dataclass
class BacktestResult:
    equity_curves: dict[str, list[tuple[datetime, float]]]
    metrics: dict[str, PerfMetrics]
    trades: list[Fill]
    warnings: list[str] = field(default_factory=list)


@dataclass
class _PendingOrder:
    order: Order
    signal_ts: datetime


class BacktestEngine:
    def __init__(self, provider: DataProvider, settings: Settings,
                 allocator: Optional[object] = None,
                 risk_manager: Optional[object] = None,
                 spread_bps: float = 4.0):
        self.provider = provider
        self.settings = settings
        self.allocator = allocator          # duck-typed .allocate(equity, sleeves)
        self.risk_manager = risk_manager    # duck-typed .check_order(...)
        self.cost_model = CostModel(CostConfig())
        self.spread_bps = spread_bps

    # ── helpers ──────────────────────────────────────────────────────────

    def _exec_quote_from_open(self, open_px: float, ts: datetime) -> Quote:
        half = open_px * (self.spread_bps / 10_000.0) / 2.0
        return Quote(ts=ts, bid=max(0.01, open_px - half), ask=open_px + half)

    def _weight_to_order(self, sig: Signal, sleeve_capital: float,
                         sleeve: Portfolio, snapshot: MarketSnapshot,
                         warnings: list[str]) -> Optional[Order]:
        inst = sig.instrument
        if sig.action == SignalAction.CLOSE:
            pos = sleeve.positions.get(inst.key)
            if pos is None or not pos.is_open:
                return None
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            return single_leg(inst, side, abs(pos.qty), strategy_id=sig.strategy_id)
        # target weight sizing (delta vs current sleeve holding)
        if isinstance(inst, Equity):
            price = snapshot.last_price(inst.symbol)
            mult = 1
        elif isinstance(inst, Option):
            chain = snapshot.get_option_chain(inst.underlying_symbol)
            oq = None if chain is None else next(
                (q for q in chain.quotes if q.instrument.symbol == inst.symbol), None)
            price = oq.quote.mid if oq else None
            mult = inst.multiplier
        else:
            return None
        if price is None or price <= 0:
            warnings.append(f"no price to size {inst.key} at {snapshot.as_of.date()}")
            return None
        target_dollars = sig.target_weight * sleeve_capital
        pos = sleeve.positions.get(inst.key)
        current_dollars = (pos.qty * price * mult) if pos else 0.0
        delta = target_dollars - current_dollars
        qty = int(abs(delta) / (price * mult))
        if qty < 1:
            return None
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        return single_leg(inst, side, float(qty), strategy_id=sig.strategy_id)

    def _bs_option_quote(self, inst: Option, under_px: float,
                         bars: Optional[list[Bar]], as_of) -> Optional[Quote]:
        """Black-Scholes fallback quote when a contract is off today's chain
        (e.g. the chain's strike grid shifted). Spread ~6% like the synthetic
        provider — never a free fill."""
        if under_px <= 0 or bars is None:
            return None
        sigma = _realized_sigma(bars, as_of)
        t = inst.years_to_expiration(as_of)
        if sigma is None or t <= 0:
            return None
        theo = bs_price(inst.right, under_px, inst.strike, t, sigma,
                        self.settings.risk_free_rate)
        half = max(0.01, theo * 0.03)
        return Quote(ts=as_of, bid=max(0.0, theo - half), ask=theo + half)

    def _execute_pending(self, pending: list[_PendingOrder], day_snapshot: MarketSnapshot,
                        day_bars_open: dict[str, float], sleeves: dict[str, Portfolio],
                        trades: list[Fill], warnings: list[str],
                        bars_hist: Optional[dict[str, list[Bar]]] = None) -> None:
        """Fill yesterday's orders at TODAY's open (equities) / today's chain
        (options). day_snapshot is today's close snapshot (used for options)."""
        for p in pending:
            order = p.order
            sleeve = sleeves[order.strategy_id]
            for leg in order.legs:
                inst = leg.instrument
                if isinstance(inst, Equity):
                    open_px = day_bars_open.get(inst.symbol)
                    if open_px is None or open_px <= 0:
                        warnings.append(
                            f"dropped order {inst.key} ({order.strategy_id}): no open "
                            f"price on execution day {day_snapshot.as_of.date()}")
                        continue
                    q = self._exec_quote_from_open(open_px, day_snapshot.as_of)
                else:
                    chain = day_snapshot.get_option_chain(inst.underlying_symbol)
                    oq = None if chain is None else next(
                        (x for x in chain.quotes if x.instrument.symbol == inst.symbol),
                        None)
                    if oq is not None and oq.quote.bid > 0 and oq.quote.ask > 0:
                        q = oq.quote
                    else:
                        under_open = day_bars_open.get(inst.underlying_symbol)
                        q = self._bs_option_quote(
                            inst, under_open or 0.0,
                            (bars_hist or {}).get(inst.underlying_symbol),
                            day_snapshot.as_of)
                        if q is None or q.bid <= 0:
                            warnings.append(
                                f"dropped option order {inst.key} "
                                f"({order.strategy_id}): contract not quoted and "
                                f"no BS fallback on execution day")
                            continue
                        warnings.append(
                            f"BS-fallback fill for {inst.key} (off-chain on "
                            f"execution day)")
                fc = self.cost_model.simulate_fill(inst, leg.side, leg.qty, q)
                fill = Fill(order_id=order.id, instrument=inst, side=leg.side,
                            qty=leg.qty, price=fc.price, ts=day_snapshot.as_of,
                            commission=fc.commission, fees=fc.fees,
                            slippage_cost=fc.slippage_cost,
                            strategy_id=order.strategy_id)
                sleeve.apply_fill(fill)
                trades.append(fill)

    def _mark_unquoted_options(self, sleeve: Portfolio, snapshot: MarketSnapshot,
                               bars_hist: dict[str, list[Bar]]) -> None:
        """BS-mark open option positions the day's chain didn't quote, so
        equity curves never carry stale option marks."""
        for pos in sleeve.open_positions():
            inst = pos.instrument
            if not isinstance(inst, Option) or pos.mark_ts == snapshot.as_of:
                continue
            if inst.is_expired(snapshot.as_of):
                continue  # settlement handles it
            under_px = snapshot.last_price(inst.underlying_symbol)
            bars = bars_hist.get(inst.underlying_symbol)
            if under_px is None or under_px <= 0 or bars is None:
                continue
            sigma = _realized_sigma(bars, snapshot.as_of)
            t = inst.years_to_expiration(snapshot.as_of)
            if sigma is None or t <= 0:
                continue
            theo = bs_price(inst.right, under_px, inst.strike, t, sigma,
                            self.settings.risk_free_rate)
            greeks = bs_greeks(inst.right, under_px, inst.strike, t, sigma,
                               self.settings.risk_free_rate)
            sleeve.mark_position(inst.key, theo, snapshot.as_of, greeks=greeks,
                                 underlying_price=under_px)

    def _settle_expired_options(self, sleeve: Portfolio, snapshot: MarketSnapshot,
                                trades: list[Fill], warnings: list[str]) -> None:
        for pos in sleeve.open_positions():
            inst = pos.instrument
            if not isinstance(inst, Option) or not inst.is_expired(snapshot.as_of):
                continue
            # NEVER settle against a fabricated price: use underlying bars,
            # else the chain's own underlying mark, else leave the position
            # open with a loud warning. (Settling at 0.0 would credit puts
            # with full strike value — fabricated P&L.)
            under_px = snapshot.last_price(inst.underlying_symbol)
            if under_px is None or under_px <= 0:
                chain = snapshot.get_option_chain(inst.underlying_symbol)
                under_px = chain.underlying_price if chain is not None else None
            if under_px is None or under_px <= 0:
                warnings.append(
                    f"cannot settle expired {inst.key}: no underlying price at "
                    f"{snapshot.as_of.date()}; position left open")
                continue
            intrinsic = inst.intrinsic_value(under_px)
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            fill = Fill(order_id=f"expiry-{inst.key}", instrument=inst, side=side,
                        qty=abs(pos.qty), price=intrinsic, ts=snapshot.as_of,
                        commission=0.0, fees=0.0, slippage_cost=0.0,
                        strategy_id="expiration")
            sleeve.apply_fill(fill)
            trades.append(fill)
            warnings.append(
                f"simplified exercise/assignment: {inst.key} cash-settled at "
                f"intrinsic ${intrinsic:.2f} on {snapshot.as_of.date()}")

    # ── main loop ────────────────────────────────────────────────────────

    def run(self, strategies: list[Strategy], start: datetime, end: datetime,
            symbols: Optional[list[str]] = None,
            options_underlyings: Optional[list[str]] = None,
            rebalance_every_n_days: int = 1) -> BacktestResult:
        symbols = symbols or self.settings.universe
        options_underlyings = options_underlyings or []
        # option underlyings must always have bars in the snapshot so marks and
        # expiry settlement can never run without an underlying price
        symbols = list(symbols) + [u for u in options_underlyings if u not in symbols]
        warnings: list[str] = []
        trades: list[Fill] = []

        ref = "SPY" if "SPY" in symbols else symbols[0]
        ref_bars = self.provider.get_bars(ref, start, end)
        days = [b.ts for b in ref_bars if start <= b.ts <= end]
        if len(days) < 3:
            return BacktestResult({}, {}, [], [f"not enough trading days for {ref}"])

        n = max(1, len(strategies))
        starting_per_sleeve = self.settings.starting_cash / n
        sleeves: dict[str, Portfolio] = {
            s.strategy_id: Portfolio(cash=starting_per_sleeve) for s in strategies
        }
        sleeve_capital: dict[str, float] = {
            s.strategy_id: starting_per_sleeve for s in strategies
        }
        curves: dict[str, list[tuple[datetime, float]]] = {
            s.strategy_id: [] for s in strategies
        }
        curves["portfolio"] = []
        curves["buy_and_hold"] = []
        sleeve_daily_returns: dict[str, list[float]] = {
            s.strategy_id: [] for s in strategies
        }
        prev_sleeve_eq: dict[str, float] = dict(sleeve_capital)

        bh = Portfolio(cash=self.settings.starting_cash)
        bh_bought = False

        # open prices per day for t+1 execution
        bars_by_symbol = {
            sym: {b.ts: b for b in self.provider.get_bars(sym, start, end)}
            for sym in set(symbols) | {ref}
        }
        bars_hist = {
            sym: sorted(d.values(), key=lambda b: b.ts)
            for sym, d in bars_by_symbol.items()
        }

        pending: list[_PendingOrder] = []

        for i, day_ts in enumerate(days):
            snapshot = self.provider.build_snapshot(
                symbols, as_of=day_ts,
                options_underlyings=options_underlyings, include_news=False,
            )
            day_open = {
                sym: bars_by_symbol[sym][day_ts].open
                for sym in bars_by_symbol if day_ts in bars_by_symbol[sym]
            }

            # 1. execute yesterday's orders at today's open (t+1)
            if pending:
                self._execute_pending(pending, snapshot, day_open, sleeves,
                                      trades, warnings, bars_hist=bars_hist)
                pending = []
            if not bh_bought and i > 0:
                open_px = day_open.get(ref)
                if open_px:
                    q = self._exec_quote_from_open(open_px, day_ts)
                    qty = int(bh.cash * 0.999 / q.ask)
                    if qty > 0:
                        fc = self.cost_model.simulate_fill(Equity(ref), OrderSide.BUY, qty, q)
                        bh.apply_fill(Fill(order_id="bh", instrument=Equity(ref),
                                           side=OrderSide.BUY, qty=qty, price=fc.price,
                                           ts=day_ts, commission=fc.commission,
                                           fees=fc.fees, slippage_cost=fc.slippage_cost,
                                           strategy_id="buy_and_hold"))
                    bh_bought = True

            # 2. settle expirations, mark everything to today's close
            for sid, sleeve in sleeves.items():
                self._settle_expired_options(sleeve, snapshot, trades, warnings)
                sleeve.mark_from_snapshot(snapshot)
                self._mark_unquoted_options(sleeve, snapshot, bars_hist)
            bh.mark_from_snapshot(snapshot)

            # 3. record curves + sleeve daily returns
            total_eq = 0.0
            for sid, sleeve in sleeves.items():
                eq = sleeve.equity()
                curves[sid].append((day_ts, eq))
                prev = prev_sleeve_eq[sid]
                if prev > 0:
                    sleeve_daily_returns[sid].append(eq / prev - 1.0)
                prev_sleeve_eq[sid] = eq
                total_eq += eq
            curves["portfolio"].append((day_ts, total_eq))
            curves["buy_and_hold"].append((day_ts, bh.equity()))

            # 4. allocator rebalance every 5 days
            if i % 5 == 0 and self.allocator is not None:
                sleeves_perf = [
                    _SleevePerfDuck(sid, sleeve_daily_returns[sid])
                    for sid in sleeves
                ]
                try:
                    alloc = self.allocator.allocate(total_eq, sleeves_perf)
                    for sid in sleeve_capital:
                        sleeve_capital[sid] = alloc.get(sid, sleeve_capital[sid])
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"allocator failed on {day_ts.date()}: {e}")
            elif i % 5 == 0:
                for sid, sleeve in sleeves.items():
                    sleeve_capital[sid] = sleeve.equity()

            # 5. generate signals from TODAY's snapshot; queue for t+1
            if i % rebalance_every_n_days == 0 and i < len(days) - 1:
                for strat in strategies:
                    sleeve = sleeves[strat.strategy_id]
                    ctx = SleeveContext(
                        strategy_id=strat.strategy_id,
                        allocated_capital=sleeve_capital[strat.strategy_id],
                        positions=dict(sleeve.positions),
                        sleeve_equity=sleeve.equity(),
                    )
                    try:
                        signals = strat.generate_signals(snapshot, ctx)
                    except Exception as e:  # noqa: BLE001
                        warnings.append(
                            f"{strat.strategy_id} raised on {day_ts.date()}: {e!r}")
                        continue
                    for sig in signals:
                        order = self._weight_to_order(
                            sig, sleeve_capital[strat.strategy_id], sleeve,
                            snapshot, warnings)
                        if order is None:
                            continue
                        if self.risk_manager is not None:
                            decision = self.risk_manager.check_order(
                                order, sleeve, snapshot)
                            if not getattr(decision, "approved", True):
                                warnings.append(
                                    f"risk rejected {order.legs[0].instrument.key} "
                                    f"({strat.strategy_id}): "
                                    f"{getattr(decision, 'reason', '?')}")
                                continue
                            adj = getattr(decision, "adjusted_qty", None)
                            if adj is not None:
                                order = single_leg(
                                    order.legs[0].instrument, order.legs[0].side,
                                    adj, strategy_id=order.strategy_id)
                        pending.append(_PendingOrder(order=order, signal_ts=day_ts))

        metrics: dict[str, PerfMetrics] = {}
        for key, curve in curves.items():
            key_trades = [t for t in trades if t.strategy_id == key] \
                if key not in ("portfolio", "buy_and_hold") else \
                (trades if key == "portfolio" else [])
            metrics[key] = compute_metrics(curve, key_trades)
        # dedupe warnings, keep order
        seen: set[str] = set()
        warnings = [w for w in warnings if not (w in seen or seen.add(w))]
        return BacktestResult(equity_curves=curves, metrics=metrics,
                              trades=trades, warnings=warnings)


class _SleevePerfDuck:
    """Matches allocation.allocator.SleevePerf's attributes without importing it."""

    def __init__(self, strategy_id: str, daily_returns: list[float]):
        self.strategy_id = strategy_id
        self.daily_returns = daily_returns
