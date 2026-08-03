"""Data layer: bar cache round-trips, timestamp normalization, as_of guards."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from quantfund.core.snapshot import Bar
from quantfund.data.alpaca_data import _normalize_bar_ts
from quantfund.data.cache import BarCache

UTC = timezone.utc


def _bar(ts, px=100.0):
    return Bar(ts=ts, open=px, high=px * 1.01, low=px * 0.99, close=px, volume=1e6)


class TestBarCache:
    def test_round_trip(self, tmp_path):
        cache = BarCache(tmp_path)
        now = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)
        bars = [_bar(now - timedelta(days=i)) for i in range(5)]
        cache.put("AAPL", bars, fetched_through=now)
        out = cache.get("AAPL")
        assert out is not None
        got, through = out
        assert len(got) == 5
        assert through == now
        assert got[0].ts < got[-1].ts  # sorted oldest first
        assert got[-1].close == 100.0

    def test_corrupt_file_is_miss(self, tmp_path):
        cache = BarCache(tmp_path)
        (cache.dir / "AAPL.json").write_text("{not json")
        assert cache.get("AAPL") is None

    def test_covers(self, tmp_path):
        cache = BarCache(tmp_path)
        now = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)
        bars = [_bar(now - timedelta(days=i)) for i in range(10)]
        cache.put("AAPL", bars, fetched_through=now)
        assert cache.covers("AAPL", now - timedelta(days=5), now)
        assert not cache.covers("AAPL", now - timedelta(days=30), now)
        assert not cache.covers("AAPL", now - timedelta(days=5),
                                now + timedelta(days=5))
        assert not cache.covers("MSFT", now - timedelta(days=1), now)


class TestBarNormalization:
    def test_midnight_bar_becomes_close_stamp(self):
        raw = datetime(2024, 6, 3, 4, 0, tzinfo=UTC)  # alpaca daily bar stamp
        ts = _normalize_bar_ts(raw)
        assert ts == datetime(2024, 6, 3, 21, 0, tzinfo=UTC)

    def test_naive_treated_as_utc(self):
        raw = datetime(2024, 6, 3, 4, 0)
        ts = _normalize_bar_ts(raw)
        assert ts.tzinfo is not None
        assert ts.hour == 21


class TestCacheGaps:
    def test_disjoint_fetches_do_not_cover_the_gap(self, tmp_path):
        cache = BarCache(tmp_path)
        jan = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
        nov = datetime(2024, 11, 5, 21, 0, tzinfo=UTC)
        jan_bars = [_bar(jan - timedelta(days=i)) for i in range(4)]
        cache.put("SPY", jan_bars, fetched_through=jan,
                  fetched_range=(jan - timedelta(days=10), jan))
        nov_bars = jan_bars + [_bar(nov - timedelta(days=i)) for i in range(4)]
        cache.put("SPY", nov_bars, fetched_through=nov,
                  fetched_range=(nov - timedelta(days=10), nov))
        # each fetched interval is covered…
        assert cache.covers("SPY", jan - timedelta(days=8), jan)
        assert cache.covers("SPY", nov - timedelta(days=8), nov)
        # …but the months-long hole between them is NOT
        mid_a = datetime(2024, 5, 1, tzinfo=UTC)
        mid_b = datetime(2024, 8, 1, tzinfo=UTC)
        assert not cache.covers("SPY", mid_a, mid_b)
        assert not cache.covers("SPY", jan - timedelta(days=8), nov)

    def test_adjacent_ranges_merge(self, tmp_path):
        cache = BarCache(tmp_path)
        t1 = datetime(2024, 3, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)
        bars = [_bar(t1)]
        cache.put("SPY", bars, fetched_through=t2, fetched_range=(t1, t2))
        cache.put("SPY", bars, fetched_through=t2 + timedelta(days=90),
                  fetched_range=(t2 - timedelta(days=1), t2 + timedelta(days=90)))
        assert cache.covers("SPY", t1, t2 + timedelta(days=90))


class TestPartialBarExclusion:
    def _provider_with_fake_bars(self, tmp_path, monkeypatch, bar_dates, now):
        import quantfund.data.alpaca_data as ad
        from quantfund.core.config import Settings

        class FakeStockClient:
            def __init__(self, *a, **k): ...
            def get_stock_bars(self, req):
                data = {"SPY": [
                    SimpleNamespace(timestamp=datetime(d.year, d.month, d.day,
                                                       4, 0, tzinfo=UTC),
                                    open=100.0, high=101.0, low=99.0,
                                    close=100.5, volume=1e6)
                    for d in bar_dates
                ]}
                return SimpleNamespace(data=data)

        monkeypatch.setattr(ad, "StockHistoricalDataClient", FakeStockClient)
        monkeypatch.setattr(ad, "OptionHistoricalDataClient",
                            lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(ad, "_utcnow", lambda: now)
        s = Settings(alpaca_paper=True, data_dir=tmp_path)
        s.alpaca_api_key, s.alpaca_secret_key = "k", "s"
        return ad.AlpacaDataProvider(s, cache_dir=tmp_path / "c")

    def test_in_progress_daily_bar_not_served_or_cached(self, tmp_path, monkeypatch):
        from datetime import date as _date
        today = _date(2024, 6, 3)
        yesterday = _date(2024, 5, 31)
        # 15:00 UTC = mid-session; today's bar (normalized close 21:00) is
        # still in progress and must be excluded
        now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
        p = self._provider_with_fake_bars(tmp_path, monkeypatch,
                                          [yesterday, today], now)
        start = datetime(2024, 5, 25, tzinfo=UTC)
        end = datetime(2024, 6, 4, tzinfo=UTC)
        bars = p.get_bars("SPY", start, end)
        assert [b.ts.date() for b in bars] == [yesterday]
        cached_bars, _ = p.cache.get("SPY")
        assert all(b.ts.date() != today for b in cached_bars)
        # and the cache does NOT claim coverage beyond `now` — a later
        # post-close fetch must refetch and pick up the final bar
        assert not p.cache.covers("SPY", start, end)

    def test_final_bar_served_after_close(self, tmp_path, monkeypatch):
        from datetime import date as _date
        today = _date(2024, 6, 3)
        now = datetime(2024, 6, 3, 21, 30, tzinfo=UTC)  # after the 21:00 close
        p = self._provider_with_fake_bars(tmp_path, monkeypatch, [today], now)
        bars = p.get_bars("SPY", datetime(2024, 6, 1, tzinfo=UTC),
                          datetime(2024, 6, 4, tzinfo=UTC))
        assert [b.ts.date() for b in bars] == [today]
