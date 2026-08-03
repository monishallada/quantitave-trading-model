"""Shared offline test fixtures: hand-built snapshots, chains, portfolios."""
from datetime import date, datetime, timedelta, timezone

import pytest

from quantfund.core.greeks import Greeks
from quantfund.core.instruments import OptionRight, make_option
from quantfund.core.snapshot import (
    Bar, MarketSnapshot, OptionChain, OptionQuote, Quote,
)

UTC = timezone.utc
T0 = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)


def make_bars(closes, end_ts=T0, start_px=None):
    """Daily bars ending at end_ts with the given closes (oldest first)."""
    n = len(closes)
    bars = []
    for i, c in enumerate(closes):
        ts = end_ts - timedelta(days=n - 1 - i)
        o = closes[i - 1] if i > 0 else (start_px or c)
        bars.append(Bar(ts=ts, open=o, high=max(o, c) * 1.005,
                        low=min(o, c) * 0.995, close=c, volume=1e6))
    return tuple(bars)


def make_quote(px, ts=T0, spread=0.10):
    return Quote(ts=ts, bid=px - spread / 2, ask=px + spread / 2,
                 bid_size=100, ask_size=100)


def make_chain(underlying="AAPL", spot=200.0, ts=T0, expiry=None,
               strikes=(180.0, 190.0, 200.0, 210.0, 220.0), premium_base=5.0,
               with_greeks=True):
    expiry = expiry or (ts.date() + timedelta(days=45))
    if isinstance(expiry, datetime):
        expiry = expiry.date()
    quotes = []
    for k in strikes:
        d_call = min(0.95, max(0.05, 0.5 + 0.02 * (spot - k)))
        for right in (OptionRight.CALL, OptionRight.PUT):
            inst = make_option(underlying, expiry, right, k)
            moneyness = (spot - k) if right == OptionRight.CALL else (k - spot)
            prem = max(0.30, premium_base + 0.4 * moneyness)
            g = None
            if with_greeks:
                # put delta = call delta - 1 (put-call parity), so an OTM put
                # carries a small negative delta as it should
                d = d_call if right == OptionRight.CALL else d_call - 1.0
                g = Greeks(delta=d, gamma=0.02, theta=-0.05, vega=0.10)
            quotes.append(OptionQuote(
                instrument=inst,
                quote=Quote(ts=ts, bid=prem - 0.10, ask=prem + 0.10,
                            bid_size=50, ask_size=50),
                iv=0.25, greeks=g, open_interest=500, day_volume=100,
            ))
    return OptionChain(underlying=underlying, ts=ts, underlying_price=spot,
                       quotes=tuple(quotes))


def make_snapshot(prices: dict, ts=T0, chains: dict | None = None,
                  n_bars=150, trend=0.0005):
    """Snapshot with n_bars of gently trending history ending at each price."""
    bars = {}
    quotes = {}
    for sym, px in prices.items():
        closes = [px / ((1 + trend) ** (n_bars - 1 - i)) for i in range(n_bars)]
        bars[sym] = make_bars(closes, end_ts=ts)
        quotes[sym] = make_quote(px, ts=ts)
    return MarketSnapshot(as_of=ts, bars=bars, quotes=quotes,
                          option_chains=chains or {})


@pytest.fixture
def t0():
    return T0
