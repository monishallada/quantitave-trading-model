"""Alpaca PAPER broker (alpaca-py). Refuses to construct against a live
account. Handles single-leg equity/option orders and multi-leg (MLEG) option
spreads, order status, fills, position sync, and flatten-all.

Note on costs: Alpaca paper fills report no explicit commission/fee fields, so
Fills from this broker carry commission=fees=slippage_cost=0 — real friction is
embedded in the fill price the paper engine gives us. The CostModel applies to
the SimBroker/backtests only.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass as AlpacaOrderClass,
    OrderSide as AlpacaSide,
    QueryOrderStatus,
    TimeInForce as AlpacaTIF,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
)

from quantfund.core.config import Settings
from quantfund.core.instruments import Equity, Instrument, parse_occ_symbol
from quantfund.core.orders import (
    Fill, Order, OrderSide, OrderStatus, OrderType,
)
from quantfund.execution.broker import (
    AccountInfo, Broker, BrokerError, BrokerPosition,
)

log = logging.getLogger(__name__)

_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


def _to_instrument(symbol: str) -> Instrument:
    try:
        return parse_occ_symbol(symbol)
    except ValueError:
        return Equity(symbol)


def _aware(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        from datetime import timezone
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


class AlpacaPaperBroker(Broker):
    def __init__(self, settings: Settings):
        if not settings.alpaca_paper:
            raise ValueError("AlpacaPaperBroker refuses live trading: paper "
                             "mode required (ALPACA_PAPER must be true)")
        if not settings.has_alpaca:
            raise ValueError("Alpaca API keys missing")
        self._client = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key, paper=True,
        )
        # order_id -> (strategy_id, rationale_id) so fills can be attributed
        self._order_meta: dict[str, tuple[str, str]] = {}
        self._seen_fill_keys: set[str] = set()

    # ── orders ───────────────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Order:
        try:
            side = {OrderSide.BUY: AlpacaSide.BUY, OrderSide.SELL: AlpacaSide.SELL}
            if len(order.legs) == 1:
                leg = order.legs[0]
                kwargs = dict(
                    symbol=leg.instrument.key,
                    qty=leg.qty,
                    side=side[leg.side],
                    time_in_force=AlpacaTIF.DAY,
                )
                if order.order_type == OrderType.LIMIT:
                    req = LimitOrderRequest(limit_price=order.limit_price, **kwargs)
                else:
                    req = MarketOrderRequest(**kwargs)
            else:
                qtys = [int(leg.qty) for leg in order.legs]
                overall = math.gcd(*qtys) if len(qtys) > 1 else qtys[0]
                overall = max(1, overall)
                legs = [
                    OptionLegRequest(
                        symbol=leg.instrument.key,
                        ratio_qty=int(leg.qty) // overall,
                        side=side[leg.side],
                    )
                    for leg in order.legs
                ]
                kwargs = dict(
                    qty=overall,
                    order_class=AlpacaOrderClass.MLEG,
                    legs=legs,
                    time_in_force=AlpacaTIF.DAY,
                )
                if order.order_type == OrderType.LIMIT:
                    req = LimitOrderRequest(limit_price=order.limit_price, **kwargs)
                else:
                    req = MarketOrderRequest(**kwargs)
            resp = self._client.submit_order(req)
            order.broker_order_id = str(resp.id)
            raw_status = getattr(resp.status, "value", str(resp.status))
            order.status = _STATUS_MAP.get(raw_status, OrderStatus.SUBMITTED)
            order.created_at = _aware(getattr(resp, "created_at", None))
            self._order_meta[str(resp.id)] = (order.strategy_id, order.rationale_id)
            return order
        except APIError as e:
            raise BrokerError(f"submit_order failed for "
                              f"{[l.instrument.key for l in order.legs]}: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except APIError as e:
            log.warning("cancel_order %s failed: %s", order_id, e)
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        try:
            resp = self._client.get_order_by_id(order_id)
        except APIError:
            return None
        inst = _to_instrument(resp.symbol) if getattr(resp, "symbol", None) else None
        if inst is None:
            return None
        from quantfund.core.orders import OrderLeg
        raw_status = getattr(resp.status, "value", str(resp.status))
        raw_side = getattr(resp.side, "value", str(resp.side))
        order = Order(
            legs=[OrderLeg(
                instrument=inst,
                side=OrderSide.BUY if raw_side == "buy" else OrderSide.SELL,
                qty=float(resp.qty or resp.filled_qty or 0) or 1.0,
            )],
        )
        order.broker_order_id = str(resp.id)
        order.status = _STATUS_MAP.get(raw_status, OrderStatus.SUBMITTED)
        return order

    # ── fills ────────────────────────────────────────────────────────────

    def get_fills(self, since: Optional[datetime] = None) -> list[Fill]:
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=since, limit=200)
            orders = self._client.get_orders(req)
        except APIError as e:
            raise BrokerError(f"get_fills failed: {e}") from e
        fills: list[Fill] = []
        for o in orders:
            raw_status = getattr(o.status, "value", str(o.status))
            filled_qty = float(o.filled_qty or 0)
            if raw_status not in ("filled", "partially_filled") or filled_qty <= 0:
                continue
            strategy_id, _ = self._order_meta.get(str(o.id), ("", ""))
            legs = getattr(o, "legs", None) or [o]
            for leg in legs:
                lq = float(getattr(leg, "filled_qty", 0) or 0)
                lp = float(getattr(leg, "filled_avg_price", 0) or 0)
                if lq <= 0 or lp <= 0:
                    continue
                key = f"{o.id}:{getattr(leg, 'id', '')}:{lq}"
                if key in self._seen_fill_keys:
                    continue
                self._seen_fill_keys.add(key)
                raw_side = getattr(leg.side, "value", str(leg.side))
                ts = _aware(getattr(leg, "filled_at", None)
                            or getattr(o, "filled_at", None)
                            or getattr(o, "updated_at", None))
                if ts is None:
                    continue
                fills.append(Fill(
                    order_id=str(o.id),
                    instrument=_to_instrument(str(leg.symbol)),
                    side=OrderSide.BUY if raw_side == "buy" else OrderSide.SELL,
                    qty=lq, price=lp, ts=ts,
                    commission=0.0, fees=0.0, slippage_cost=0.0,
                    strategy_id=strategy_id,
                ))
        return fills

    def strategy_for_order(self, broker_order_id: str) -> tuple[str, str]:
        return self._order_meta.get(broker_order_id, ("", ""))

    # ── account/positions ────────────────────────────────────────────────

    def get_positions(self) -> list[BrokerPosition]:
        try:
            raw = self._client.get_all_positions()
        except APIError as e:
            raise BrokerError(f"get_positions failed: {e}") from e
        out: list[BrokerPosition] = []
        for p in raw:
            qty = float(p.qty)
            raw_side = getattr(p.side, "value", str(getattr(p, "side", "long")))
            if raw_side == "short":
                qty = -abs(qty)
            out.append(BrokerPosition(
                instrument=_to_instrument(str(p.symbol)),
                qty=qty,
                avg_entry_price=float(p.avg_entry_price or 0),
                market_value=float(p.market_value or 0),
                unrealized_pnl=float(getattr(p, "unrealized_pl", 0) or 0),
            ))
        return out

    def get_account(self) -> AccountInfo:
        try:
            acct = self._client.get_account()
        except APIError as e:
            raise BrokerError(f"get_account failed: {e}") from e
        from datetime import timezone
        return AccountInfo(
            ts=datetime.now(timezone.utc),
            equity=float(acct.equity or 0),
            cash=float(acct.cash or 0),
            buying_power=float(acct.buying_power or 0),
            options_approved_level=int(
                getattr(acct, "options_approved_level", 0) or 0),
        )

    def flatten_all(self, reason: str = "") -> list[Order]:
        log.critical("FLATTEN ALL (%s)", reason)
        try:
            self._client.cancel_orders()
        except APIError as e:
            log.error("cancel_orders during flatten failed: %s", e)
        try:
            self._client.close_all_positions(cancel_orders=True)
        except APIError as e:
            log.error("close_all_positions during flatten failed: %s", e)
        return []
