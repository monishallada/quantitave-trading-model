"""Order and fill primitives shared by the sim broker, Alpaca broker, and backtester."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from quantfund.core.instruments import Instrument


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class OrderLeg:
    instrument: Instrument
    side: OrderSide
    qty: float  # always positive; shares for equities, contracts for options

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"leg qty must be positive, got {self.qty}")


@dataclass
class Order:
    """A single- or multi-leg order. Multi-leg is options-only (spreads)."""

    legs: list[OrderLeg]
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None  # net price for multi-leg
    tif: TimeInForce = TimeInForce.DAY
    strategy_id: str = ""
    rationale_id: str = ""       # key into the rationale store (why this trade happened)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    created_at: Optional[datetime] = None
    reject_reason: str = ""

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("order must have at least one leg")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit order requires limit_price")

    @property
    def is_multi_leg(self) -> bool:
        return len(self.legs) > 1


@dataclass(frozen=True)
class Fill:
    """One executed leg. ``price`` is the per-share execution price actually paid,
    already including spread crossing and slippage. ``commission`` and ``fees``
    are additional cash costs (always >= 0). ``slippage_cost`` records the dollar
    friction versus the quote mid at decision time, for reporting."""

    order_id: str
    instrument: Instrument
    side: OrderSide
    qty: float
    price: float
    ts: datetime
    commission: float = 0.0
    fees: float = 0.0
    slippage_cost: float = 0.0
    strategy_id: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("fill qty must be positive")
        if self.price < 0:
            raise ValueError("fill price must be non-negative")
        if self.commission < 0 or self.fees < 0:
            raise ValueError("commission/fees must be non-negative")

    @property
    def notional(self) -> float:
        return self.price * self.qty * self.instrument.multiplier

    @property
    def cash_flow(self) -> float:
        """Signed cash impact on the account (negative = cash out)."""
        return -self.side.sign * self.notional - self.commission - self.fees

    @property
    def total_friction(self) -> float:
        """All-in trading cost versus a frictionless mid fill."""
        return self.commission + self.fees + self.slippage_cost


def single_leg(
    instrument: Instrument,
    side: OrderSide,
    qty: float,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Optional[float] = None,
    strategy_id: str = "",
    rationale_id: str = "",
) -> Order:
    return Order(
        legs=[OrderLeg(instrument=instrument, side=side, qty=qty)],
        order_type=order_type,
        limit_price=limit_price,
        strategy_id=strategy_id,
        rationale_id=rationale_id,
    )
