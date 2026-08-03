"""Point-in-time market data view — the platform's core no-look-ahead guard.

Every strategy, allocator, and risk check receives ONLY a ``MarketSnapshot``.
A snapshot is constructed with an ``as_of`` timestamp and rejects (raises
``LookaheadError``) any datum stamped after it. Accessors re-filter
defensively, so even a buggy constructor path cannot leak future data.

All timestamps must be timezone-aware (UTC recommended). Naive datetimes are
rejected outright — silent local-time bugs are how leaks happen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from quantfund.core.greeks import Greeks
from quantfund.core.instruments import Option


class LookaheadError(Exception):
    """Raised when data stamped after the snapshot's as_of time is supplied or requested."""


def _require_aware(ts: datetime, what: str) -> datetime:
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"{what} must be timezone-aware, got naive datetime {ts!r}")
    return ts


@dataclass(frozen=True)
class Bar:
    ts: datetime  # bar CLOSE time (the moment all fields are known)
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        _require_aware(self.ts, "Bar.ts")


@dataclass(frozen=True)
class Quote:
    ts: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    def __post_init__(self) -> None:
        _require_aware(self.ts, "Quote.ts")
        if self.bid < 0 or self.ask < 0:
            raise ValueError(f"negative quote: bid={self.bid} ask={self.ask}")
        if self.ask > 0 and self.bid > self.ask:
            raise ValueError(f"crossed quote: bid={self.bid} > ask={self.ask}")

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.ask or self.bid

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid) if self.ask > 0 else 0.0


@dataclass(frozen=True)
class OptionQuote:
    """One contract on the chain: quote + Greeks + IV as of the chain timestamp."""

    instrument: Option
    quote: Quote
    iv: Optional[float] = None
    greeks: Optional[Greeks] = None
    open_interest: float = 0.0
    day_volume: float = 0.0


@dataclass(frozen=True)
class OptionChain:
    underlying: str
    ts: datetime
    underlying_price: float
    quotes: tuple[OptionQuote, ...] = ()

    def __post_init__(self) -> None:
        _require_aware(self.ts, "OptionChain.ts")

    def expirations(self) -> list:
        return sorted({q.instrument.expiration for q in self.quotes})

    def contracts(
        self,
        right=None,
        expiration=None,
        min_dte: Optional[float] = None,
        max_dte: Optional[float] = None,
    ) -> list[OptionQuote]:
        out = []
        for q in self.quotes:
            inst = q.instrument
            if right is not None and inst.right != right:
                continue
            if expiration is not None and inst.expiration != expiration:
                continue
            dte = inst.days_to_expiration(self.ts)
            if min_dte is not None and dte < min_dte:
                continue
            if max_dte is not None and dte > max_dte:
                continue
            out.append(q)
        return sorted(out, key=lambda q: (q.instrument.expiration, q.instrument.strike))

    def nearest_strike(self, right, expiration, target_strike: float) -> Optional[OptionQuote]:
        cands = self.contracts(right=right, expiration=expiration)
        if not cands:
            return None
        return min(cands, key=lambda q: abs(q.instrument.strike - target_strike))

    def nearest_delta(self, right, target_delta: float, min_dte: float = 0.0,
                      max_dte: Optional[float] = None) -> Optional[OptionQuote]:
        cands = [
            q for q in self.contracts(right=right, min_dte=min_dte, max_dte=max_dte)
            if q.greeks is not None
        ]
        if not cands:
            return None
        return min(cands, key=lambda q: abs(abs(q.greeks.delta) - abs(target_delta)))


@dataclass(frozen=True)
class NewsItem:
    id: str
    published_at: datetime  # the ONLY timestamp that matters for pinning
    headline: str
    summary: str = ""
    symbols: tuple[str, ...] = ()
    source: str = ""
    pinnable: bool = True  # False => timestamp untrustworthy; excluded from snapshots

    def __post_init__(self) -> None:
        _require_aware(self.published_at, "NewsItem.published_at")


@dataclass(frozen=True)
class RegimeState:
    """Shared volatility / trend regime signal, computed only from data <= as_of."""

    as_of: datetime
    vol_regime: str = "normal"      # "low" | "normal" | "high"
    trend_regime: str = "sideways"  # "up" | "down" | "sideways"
    realized_vol_annualized: float = 0.0
    detail: str = ""

    def __post_init__(self) -> None:
        _require_aware(self.as_of, "RegimeState.as_of")


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable, timestamp-pinned view of the market.

    Construction validates that no supplied datum is stamped after ``as_of``
    (raises LookaheadError). Accessors re-filter defensively.
    """

    as_of: datetime
    bars: dict[str, tuple[Bar, ...]] = field(default_factory=dict)
    quotes: dict[str, Quote] = field(default_factory=dict)
    option_chains: dict[str, OptionChain] = field(default_factory=dict)
    news: tuple[NewsItem, ...] = ()
    regime: Optional[RegimeState] = None

    def __post_init__(self) -> None:
        _require_aware(self.as_of, "MarketSnapshot.as_of")
        for symbol, bars in self.bars.items():
            for b in bars:
                if b.ts > self.as_of:
                    raise LookaheadError(
                        f"bar for {symbol} at {b.ts.isoformat()} is after snapshot "
                        f"as_of {self.as_of.isoformat()}"
                    )
        for symbol, q in self.quotes.items():
            if q.ts > self.as_of:
                raise LookaheadError(
                    f"quote for {symbol} at {q.ts.isoformat()} is after snapshot "
                    f"as_of {self.as_of.isoformat()}"
                )
        for underlying, chain in self.option_chains.items():
            if chain.ts > self.as_of:
                raise LookaheadError(
                    f"option chain for {underlying} at {chain.ts.isoformat()} is after "
                    f"snapshot as_of {self.as_of.isoformat()}"
                )
            for oq in chain.quotes:
                if oq.quote.ts > self.as_of:
                    raise LookaheadError(
                        f"option quote {oq.instrument.symbol} at {oq.quote.ts.isoformat()} "
                        f"is after snapshot as_of {self.as_of.isoformat()}"
                    )
        for item in self.news:
            if not item.pinnable:
                raise LookaheadError(
                    f"news item {item.id!r} is not date-pinnable and may not enter a snapshot"
                )
            if item.published_at > self.as_of:
                raise LookaheadError(
                    f"news item {item.id!r} published {item.published_at.isoformat()} is "
                    f"after snapshot as_of {self.as_of.isoformat()}"
                )
        if self.regime is not None and self.regime.as_of > self.as_of:
            raise LookaheadError("regime state computed after snapshot as_of")

    # ── accessors (defensive re-filtering) ───────────────────────────────

    def get_bars(self, symbol: str, lookback: Optional[int] = None) -> list[Bar]:
        """Bars with close time <= as_of, oldest first; optionally the last N."""
        out = [b for b in self.bars.get(symbol, ()) if b.ts <= self.as_of]
        out.sort(key=lambda b: b.ts)
        if lookback is not None:
            out = out[-lookback:]
        return out

    def last_close(self, symbol: str) -> Optional[float]:
        bars = self.get_bars(symbol)
        return bars[-1].close if bars else None

    def get_quote(self, symbol: str) -> Optional[Quote]:
        q = self.quotes.get(symbol)
        if q is not None and q.ts <= self.as_of:
            return q
        return None

    def last_price(self, symbol: str) -> Optional[float]:
        """Best available price: quote mid, else last bar close."""
        q = self.get_quote(symbol)
        if q is not None and q.mid > 0:
            return q.mid
        return self.last_close(symbol)

    def get_option_chain(self, underlying: str) -> Optional[OptionChain]:
        chain = self.option_chains.get(underlying)
        if chain is not None and chain.ts <= self.as_of:
            return chain
        return None

    def get_news(self, symbol: Optional[str] = None, limit: int = 50) -> list[NewsItem]:
        items = [
            n for n in self.news
            if n.published_at <= self.as_of and n.pinnable
            and (symbol is None or symbol in n.symbols)
        ]
        items.sort(key=lambda n: n.published_at, reverse=True)
        return items[:limit]

    def get_regime(self) -> Optional[RegimeState]:
        if self.regime is not None and self.regime.as_of <= self.as_of:
            return self.regime
        return None

    def staleness_seconds(self, symbol: str) -> Optional[float]:
        """Age of the freshest datum for a symbol, in seconds."""
        candidates: list[datetime] = []
        q = self.get_quote(symbol)
        if q:
            candidates.append(q.ts)
        bars = self.get_bars(symbol, lookback=1)
        if bars:
            candidates.append(bars[0].ts)
        if not candidates:
            return None
        return (self.as_of - max(candidates)).total_seconds()


def filter_bars_asof(bars: Iterable[Bar], as_of: datetime) -> tuple[Bar, ...]:
    """Utility for snapshot builders: keep only bars closed at or before as_of."""
    _require_aware(as_of, "as_of")
    return tuple(sorted((b for b in bars if b.ts <= as_of), key=lambda b: b.ts))
