"""Instrument layer: Equity and Option types with a unified interface.

Price convention used across the entire platform:
  * All prices (bars, quotes, option premiums, avg_cost) are quoted PER SHARE.
  * Notional value of a position = price * qty * multiplier
    (multiplier = 1 for equities, 100 for standard equity options).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Union


class AssetClass(str, Enum):
    EQUITY = "equity"
    OPTION = "option"


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Equity:
    """A US common stock / ETF."""

    symbol: str

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITY

    @property
    def multiplier(self) -> int:
        return 1

    @property
    def underlying(self) -> str:
        return self.symbol

    @property
    def key(self) -> str:
        """Unique string key used in position maps."""
        return self.symbol

    def __str__(self) -> str:  # pragma: no cover - repr convenience
        return self.symbol


@dataclass(frozen=True)
class Option:
    """A standard US equity option contract (American exercise, physical delivery).

    ``symbol`` is the OCC/OSI symbol, e.g. ``AAPL240621C00190000``.
    ``strike`` and premiums are per-share; contract notional = premium * 100.
    """

    symbol: str
    underlying_symbol: str
    expiration: date
    strike: float
    right: OptionRight
    style: str = "american"
    _multiplier: int = field(default=100, repr=False)

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.OPTION

    @property
    def multiplier(self) -> int:
        return self._multiplier

    @property
    def underlying(self) -> str:
        return self.underlying_symbol

    @property
    def key(self) -> str:
        return self.symbol

    def days_to_expiration(self, as_of: datetime) -> float:
        """Calendar days until expiration (can be fractional, floored at 0)."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        # Options expire at 16:00 ET; approximate with 20:00 UTC.
        expiry_dt = datetime(
            self.expiration.year, self.expiration.month, self.expiration.day,
            20, 0, tzinfo=timezone.utc,
        )
        return max(0.0, (expiry_dt - as_of).total_seconds() / 86400.0)

    def years_to_expiration(self, as_of: datetime) -> float:
        return self.days_to_expiration(as_of) / 365.0

    def intrinsic_value(self, underlying_price: float) -> float:
        if self.right == OptionRight.CALL:
            return max(0.0, underlying_price - self.strike)
        return max(0.0, self.strike - underlying_price)

    def is_expired(self, as_of: datetime) -> bool:
        return self.days_to_expiration(as_of) <= 0.0

    def __str__(self) -> str:  # pragma: no cover
        return self.symbol


Instrument = Union[Equity, Option]

_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<right>[CP])"
    r"(?P<strike>\d{8})$"
)


def format_occ_symbol(underlying: str, expiration: date, right: OptionRight, strike: float) -> str:
    """Build an OCC/OSI option symbol, e.g. AAPL240621C00190000."""
    strike_int = int(round(strike * 1000))
    r = "C" if right == OptionRight.CALL else "P"
    return f"{underlying.upper()}{expiration.strftime('%y%m%d')}{r}{strike_int:08d}"


def parse_occ_symbol(symbol: str) -> Option:
    """Parse an OCC/OSI option symbol into an Option instrument.

    Raises ValueError for malformed symbols.
    """
    m = _OCC_RE.match(symbol.strip().upper())
    if not m:
        raise ValueError(f"Not a valid OCC option symbol: {symbol!r}")
    yy, mm, dd = int(m["yy"]), int(m["mm"]), int(m["dd"])
    expiration = date(2000 + yy, mm, dd)
    right = OptionRight.CALL if m["right"] == "C" else OptionRight.PUT
    strike = int(m["strike"]) / 1000.0
    return Option(
        symbol=symbol.strip().upper(),
        underlying_symbol=m["root"],
        expiration=expiration,
        strike=strike,
        right=right,
    )


def make_option(underlying: str, expiration: date, right: OptionRight, strike: float) -> Option:
    """Convenience constructor that derives the OCC symbol."""
    return Option(
        symbol=format_occ_symbol(underlying, expiration, right, strike),
        underlying_symbol=underlying.upper(),
        expiration=expiration,
        strike=strike,
        right=right,
    )
