"""Broker interface. Implementations: AlpacaPaperBroker (live paper),
SimBroker (backtests / offline mode). Both must apply realistic costs —
SimBroker via CostModel, Alpaca via real paper fills.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quantfund.core.instruments import Instrument
from quantfund.core.orders import Fill, Order


@dataclass(frozen=True)
class AccountInfo:
    ts: datetime
    equity: float
    cash: float
    buying_power: float
    options_approved_level: int = 0


@dataclass(frozen=True)
class BrokerPosition:
    instrument: Instrument
    qty: float           # signed
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float = 0.0


class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit; returns the order with broker_order_id and status set.
        Raises BrokerError on hard failures."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        ...

    @abstractmethod
    def get_fills(self, since: Optional[datetime] = None) -> list[Fill]:
        """Fills observed since ``since`` (or all recent)."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        ...

    @abstractmethod
    def flatten_all(self, reason: str = "") -> list[Order]:
        """Close every open position with market orders and cancel all working
        orders. Used by the kill switch. Must be idempotent."""


class BrokerError(Exception):
    pass
