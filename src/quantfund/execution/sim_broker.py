"""Simulated broker: fills orders against the current MarketSnapshot through
the CostModel (spread crossing + slippage + fees). Used by backtests, offline
demo mode, and tests. A fill with zero friction is impossible here by design.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from quantfund.core.costs import CostModel
from quantfund.core.instruments import Equity, Instrument, Option
from quantfund.core.orders import (
    Fill, Order, OrderSide, OrderStatus, OrderType,
)
from quantfund.core.portfolio import Portfolio
from quantfund.core.snapshot import MarketSnapshot, Quote
from quantfund.execution.broker import (
    AccountInfo, Broker, BrokerError, BrokerPosition,
)

log = logging.getLogger(__name__)


class SimBroker(Broker):
    def __init__(self, cost_model: CostModel, starting_cash: float):
        self.cost_model = cost_model
        self.portfolio = Portfolio(cash=starting_cash)
        self.fills: list[Fill] = []
        self._orders: dict[str, Order] = {}
        self._working: list[Order] = []
        self._snapshot: Optional[MarketSnapshot] = None

    # ── market data hookup ───────────────────────────────────────────────

    def set_snapshot(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot
        self.portfolio.mark_from_snapshot(snapshot)
        # re-try working limit orders against the new quotes
        still_working: list[Order] = []
        for order in self._working:
            if not self._try_fill(order):
                still_working.append(order)
        self._working = still_working

    def _quote_for(self, inst: Instrument) -> Optional[Quote]:
        if self._snapshot is None:
            return None
        if isinstance(inst, Equity):
            return self._snapshot.get_quote(inst.symbol)
        if isinstance(inst, Option):
            chain = self._snapshot.get_option_chain(inst.underlying_symbol)
            if chain is None:
                return None
            oq = next((q for q in chain.quotes if q.instrument.symbol == inst.symbol), None)
            return oq.quote if oq else None
        return None

    # ── order handling ───────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Order:
        if self._snapshot is None:
            raise BrokerError("SimBroker has no snapshot; call set_snapshot first")
        order.broker_order_id = order.id
        order.created_at = self._snapshot.as_of
        # validate quotes for every leg up front
        for leg in order.legs:
            q = self._quote_for(leg.instrument)
            if q is None:
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"no quote for {leg.instrument.key}"
                self._orders[order.id] = order
                return order
            if q.bid <= 0 or q.ask <= 0:
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"one-sided quote for {leg.instrument.key}"
                self._orders[order.id] = order
                return order
        order.status = OrderStatus.SUBMITTED
        self._orders[order.id] = order
        if not self._try_fill(order):
            self._working.append(order)
        return order

    def _try_fill(self, order: Order) -> bool:
        assert self._snapshot is not None
        # compute costed execution price per leg
        priced = []
        for leg in order.legs:
            q = self._quote_for(leg.instrument)
            if q is None or q.bid <= 0 or q.ask <= 0:
                return False
            fc = self.cost_model.simulate_fill(leg.instrument, leg.side, leg.qty, q)
            priced.append((leg, fc))
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            limit = order.limit_price
            if len(order.legs) == 1:
                leg, fc = priced[0]
                ok = fc.price <= limit if leg.side == OrderSide.BUY else fc.price >= limit
            else:
                # multi-leg: limit is the NET DEBIT per 1x structure unit,
                # where the unit is the gcd of leg quantities — leg quantities
                # must weight the net price (a 1x2 ratio is not 1x1)
                import math as _math
                qtys = [max(1, int(round(leg.qty))) for leg in order.legs]
                unit = _math.gcd(*qtys) if len(qtys) > 1 else qtys[0]
                net_per_unit = sum(
                    leg.side.sign * fc.price * (max(1, int(round(leg.qty))) / unit)
                    for leg, fc in priced)
                ok = net_per_unit <= limit
            if not ok:
                return False
        ts = self._snapshot.as_of
        for leg, fc in priced:
            fill = Fill(
                order_id=order.id, instrument=leg.instrument, side=leg.side,
                qty=leg.qty, price=fc.price, ts=ts,
                commission=fc.commission, fees=fc.fees,
                slippage_cost=fc.slippage_cost, strategy_id=order.strategy_id,
            )
            self.fills.append(fill)
            self.portfolio.apply_fill(fill)
        order.status = OrderStatus.FILLED
        if self.portfolio.cash < 0:
            log.warning("SimBroker cash negative (%.2f) after order %s — risk layer "
                        "should have prevented this", self.portfolio.cash, order.id)
        return True

    def cancel_order(self, order_id: str) -> bool:
        for o in list(self._working):
            if o.id == order_id or o.broker_order_id == order_id:
                o.status = OrderStatus.CANCELED
                self._working.remove(o)
                return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_fills(self, since: Optional[datetime] = None) -> list[Fill]:
        if since is None:
            return list(self.fills)
        return [f for f in self.fills if f.ts > since]

    # ── account mirror ───────────────────────────────────────────────────

    def get_positions(self) -> list[BrokerPosition]:
        return [
            BrokerPosition(
                instrument=p.instrument, qty=p.qty, avg_entry_price=p.avg_cost,
                market_value=p.market_value, unrealized_pnl=p.unrealized_pnl,
            )
            for p in self.portfolio.open_positions()
        ]

    def get_account(self) -> AccountInfo:
        ts = self._snapshot.as_of if self._snapshot else datetime.now().astimezone()
        return AccountInfo(
            ts=ts, equity=self.portfolio.equity(), cash=self.portfolio.cash,
            buying_power=self.portfolio.buying_power, options_approved_level=2,
        )

    def flatten_all(self, reason: str = "") -> list[Order]:
        for o in list(self._working):
            self.cancel_order(o.id)
        orders: list[Order] = []
        for pos in self.portfolio.open_positions():
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            from quantfund.core.orders import single_leg
            order = single_leg(pos.instrument, side, abs(pos.qty),
                               strategy_id="kill_switch",
                               rationale_id="")
            order.rationale_id = ""
            submitted = self.submit_order(order)
            if submitted.status == OrderStatus.REJECTED:
                log.error("flatten_all could not close %s: %s",
                          pos.instrument.key, submitted.reject_reason)
            orders.append(submitted)
        return orders
