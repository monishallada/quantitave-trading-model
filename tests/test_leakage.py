"""No-look-ahead guard tests. These are the tests that make paper results
meaningful: they FAIL if any path lets a component see data stamped after the
snapshot's as_of time.
"""
from datetime import datetime, timedelta, timezone

import pytest

from quantfund.core.snapshot import (
    Bar,
    LookaheadError,
    MarketSnapshot,
    NewsItem,
    Quote,
)
from quantfund.data.synthetic import SyntheticProvider

UTC = timezone.utc
T0 = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)


def _bar(ts, px=100.0):
    return Bar(ts=ts, open=px, high=px * 1.01, low=px * 0.99, close=px, volume=1e6)


class TestSnapshotConstruction:
    def test_future_bar_rejected(self):
        with pytest.raises(LookaheadError):
            MarketSnapshot(
                as_of=T0,
                bars={"AAPL": (_bar(T0 - timedelta(days=1)), _bar(T0 + timedelta(days=1)))},
            )

    def test_future_quote_rejected(self):
        with pytest.raises(LookaheadError):
            MarketSnapshot(
                as_of=T0,
                quotes={"AAPL": Quote(ts=T0 + timedelta(minutes=1), bid=99, ask=101)},
            )

    def test_future_news_rejected(self):
        with pytest.raises(LookaheadError):
            MarketSnapshot(
                as_of=T0,
                news=(NewsItem(id="n1", published_at=T0 + timedelta(hours=2),
                               headline="future headline"),),
            )

    def test_unpinnable_news_rejected(self):
        # Non-date-pinnable items may NEVER enter a snapshot, even if 'old'.
        with pytest.raises(LookaheadError):
            MarketSnapshot(
                as_of=T0,
                news=(NewsItem(id="n2", published_at=T0 - timedelta(days=1),
                               headline="untrustworthy timestamp", pinnable=False),),
            )

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError):
            MarketSnapshot(as_of=datetime(2024, 6, 3, 21, 0))  # naive

    def test_boundary_inclusive(self):
        snap = MarketSnapshot(as_of=T0, bars={"AAPL": (_bar(T0),)})
        assert len(snap.get_bars("AAPL")) == 1


class TestAccessorDefense:
    def test_accessors_refilter_even_if_bypassed(self):
        """Even if construction validation were bypassed (object.__setattr__),
        accessors still refuse to serve future data."""
        snap = MarketSnapshot(as_of=T0, bars={"AAPL": (_bar(T0 - timedelta(days=1)),)})
        object.__setattr__(
            snap, "bars",
            {"AAPL": (_bar(T0 - timedelta(days=1)), _bar(T0 + timedelta(days=1), px=999.0))},
        )
        object.__setattr__(
            snap, "quotes", {"AAPL": Quote(ts=T0 + timedelta(days=1), bid=998, ask=1000)}
        )
        bars = snap.get_bars("AAPL")
        assert all(b.ts <= T0 for b in bars)
        assert snap.last_close("AAPL") != 999.0
        assert snap.get_quote("AAPL") is None
        # last_price falls back to the last valid close, not the future quote
        assert snap.last_price("AAPL") == bars[-1].close


class TestProviderPinning:
    def test_build_snapshot_never_leaks(self):
        provider = SyntheticProvider(seed=7)
        as_of = datetime(2024, 3, 15, 21, 0, tzinfo=UTC)
        snap = provider.build_snapshot(["AAPL", "SPY"], as_of=as_of,
                                       options_underlyings=["AAPL"])
        assert snap.as_of == as_of
        for sym in ("AAPL", "SPY"):
            assert all(b.ts <= as_of for b in snap.get_bars(sym))
            q = snap.get_quote(sym)
            assert q is None or q.ts <= as_of
        chain = snap.get_option_chain("AAPL")
        assert chain is not None and chain.ts <= as_of
        assert all(oq.quote.ts <= as_of for oq in chain.quotes)
        reg = snap.get_regime()
        assert reg is None or reg.as_of <= as_of

    def test_snapshots_at_earlier_time_see_less(self):
        """Moving as_of back must strictly shrink the visible history —
        catches providers that return a fixed dataset regardless of as_of."""
        provider = SyntheticProvider(seed=7)
        later = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)
        earlier = later - timedelta(days=90)
        snap_late = provider.build_snapshot(["MSFT"], as_of=later)
        snap_early = provider.build_snapshot(["MSFT"], as_of=earlier)
        newest_early = snap_early.get_bars("MSFT")[-1].ts
        newest_late = snap_late.get_bars("MSFT")[-1].ts
        assert newest_early <= earlier
        assert newest_early < newest_late
        # identical overlapping history (no restatement) — compare bars that
        # fall inside BOTH rolling lookback windows
        late_bars = {b.ts: b for b in snap_late.get_bars("MSFT")}
        overlap = [b for b in snap_early.get_bars("MSFT") if b.ts in late_bars]
        assert len(overlap) > 50
        for b in overlap:
            assert late_bars[b.ts].close == b.close

    def test_regime_uses_only_past_data(self):
        provider = SyntheticProvider(seed=11)
        as_of = datetime(2024, 5, 1, 21, 0, tzinfo=UTC)
        regime = provider.compute_regime(as_of)
        assert regime is not None
        assert regime.as_of <= as_of
