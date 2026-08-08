"""Strategy plug-in interface. Strategies are independent sleeves: they see a
point-in-time MarketSnapshot plus their own sleeve context, and emit sized
Signals with a rationale. They never touch the broker or global portfolio.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from quantfund.core.instruments import Instrument
from quantfund.core.orders import OrderType
from quantfund.core.portfolio import Position
from quantfund.core.snapshot import MarketSnapshot


class SignalAction(str, Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"          # close/reduce an existing position
    TARGET_WEIGHT = "target_weight"  # set position to target_weight of sleeve capital


@dataclass(frozen=True)
class Signal:
    """A sized trading intention from one sleeve.

    ``target_weight`` is the desired position size as a signed fraction of the
    sleeve's allocated capital (e.g. +0.10 = long 10% of sleeve capital).
    For CLOSE, target_weight is ignored. The executor converts weights into
    share/contract quantities and applies all risk checks.
    """

    instrument: Instrument
    action: SignalAction
    target_weight: float
    rationale: str
    strategy_id: str
    ts: datetime
    confidence: float = 0.5           # 0..1
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    rationale_payload: dict = field(default_factory=dict)  # full audit detail
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.target_weight <= 1.0):
            raise ValueError("target_weight must be within [-1, 1] of sleeve capital")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class SleeveContext:
    """What a strategy is allowed to know about its own sleeve."""

    strategy_id: str
    allocated_capital: float
    positions: dict[str, Position] = field(default_factory=dict)  # this sleeve's only
    sleeve_equity: float = 0.0
    data_dir: Optional[str] = None    # for strategy-private persistence (LLM memory)


class Strategy(ABC):
    """One sleeve. Implementations must be pure functions of (snapshot, ctx):
    no network calls to data sources, no wall-clock reads — the snapshot's
    ``as_of`` is the only 'now' a strategy may use. (The LLM sleeve calls the
    Anthropic API, which is a reasoning engine, not a data source; everything
    market-related it sees still comes from the snapshot.)
    """

    strategy_id: str = "base"
    name: str = "Base"
    warmup_bars: int = 50            # minimum history before emitting signals
    uses_options: bool = False
    # True => the runner evaluates this sleeve EVERY loop, not only on the
    # slow rebalance cadence. Intraday sleeves (0DTE) are useless at 15-min
    # granularity: their positions live and die within the session.
    fast_cadence: bool = False

    @abstractmethod
    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext) -> list[Signal]:
        ...

    def describe(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "warmup_bars": self.warmup_bars,
            "uses_options": self.uses_options,
        }
