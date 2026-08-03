"""Backtest engine: t+1 discipline, costs, leakage probe, benchmarks."""
from datetime import datetime, timezone

import pytest

from quantfund.backtest.engine import BacktestEngine
from quantfund.core.config import Settings
from quantfund.core.instruments import Equity
from quantfund.core.snapshot import MarketSnapshot
from quantfund.data.synthetic import SyntheticProvider
from quantfund.strategies.base import (
    Signal, SignalAction, SleeveContext, Strategy,
)

UTC = timezone.utc
START = datetime(2023, 1, 2, tzinfo=UTC)
END = datetime(2024, 6, 28, tzinfo=UTC)
SYMBOLS = ["SPY", "AAPL", "MSFT"]


class BuyOnceStrategy(Strategy):
    """Buys 50% of sleeve capital in the first symbol once, then holds."""
    strategy_id = "buy_once"
    name = "Buy Once"
    warmup_bars = 10

    def __init__(self):
        self.bought = False
        self.signal_timestamps = []

    def generate_signals(self, snapshot, ctx):
        held = any(p.is_open for p in ctx.positions.values())
        if self.bought or held:
            return []
        self.bought = True
        self.signal_timestamps.append(snapshot.as_of)
        return [Signal(instrument=Equity("AAPL"),
                       action=SignalAction.TARGET_WEIGHT, target_weight=0.5,
                       rationale="test buy", strategy_id=self.strategy_id,
                       ts=snapshot.as_of)]


class LeakageProbeStrategy(Strategy):
    """Asserts inside generate_signals that no visible bar postdates as_of."""
    strategy_id = "probe"
    name = "Probe"
    warmup_bars = 5

    def __init__(self):
        self.checks = 0

    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext):
        for sym in snapshot.bars:
            bars = snapshot.get_bars(sym)
            if bars:
                assert max(b.ts for b in bars) <= snapshot.as_of, "LEAK!"
        q = snapshot.get_quote("SPY")
        assert q is None or q.ts <= snapshot.as_of
        self.checks += 1
        return []


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    return Settings(alpaca_paper=True, starting_cash=100_000,
                    data_dir=tmp_path_factory.mktemp("bt"))


@pytest.fixture(scope="module")
def result(settings):
    provider = SyntheticProvider(seed=42)
    engine = BacktestEngine(provider, settings)
    strat = BuyOnceStrategy()
    res = engine.run([strat], START, END, symbols=SYMBOLS)
    return res, strat


def test_curves_present_and_finite(result):
    res, _ = result
    assert "portfolio" in res.equity_curves
    assert "buy_and_hold" in res.equity_curves
    assert "buy_once" in res.equity_curves
    for name, curve in res.equity_curves.items():
        assert len(curve) > 200
        assert all(v == v and v > 0 for _, v in curve), f"NaN/neg in {name}"
    assert res.equity_curves["portfolio"][0][1] == pytest.approx(100_000, rel=0.01)


def test_t_plus_1_execution(result):
    res, strat = result
    assert strat.signal_timestamps, "strategy never signaled"
    signal_ts = strat.signal_timestamps[0]
    trade_fills = [t for t in res.trades if t.strategy_id == "buy_once"]
    assert trade_fills, "no fills"
    # every fill strictly after the snapshot that generated it
    assert all(f.ts > signal_ts for f in trade_fills)


def test_costs_always_positive(result):
    res, _ = result
    real = [t for t in res.trades if t.strategy_id != "expiration"]
    assert real
    assert all(t.total_friction > 0 for t in real)


def test_benchmark_bought_and_held(result):
    res, _ = result
    bh = res.equity_curves["buy_and_hold"]
    # equity changes over time → actually invested
    values = [v for _, v in bh]
    assert max(values) != min(values)


def test_leakage_probe_runs_clean(settings):
    provider = SyntheticProvider(seed=7)
    engine = BacktestEngine(provider, settings)
    probe = LeakageProbeStrategy()
    res = engine.run([probe], START, END, symbols=SYMBOLS)
    assert probe.checks > 200
    assert not any("LEAK" in w for w in res.warnings)


def test_determinism(settings):
    def go():
        return BacktestEngine(SyntheticProvider(seed=11), settings).run(
            [BuyOnceStrategy()], START, END, symbols=SYMBOLS)
    r1, r2 = go(), go()
    assert r1.equity_curves["portfolio"] == r2.equity_curves["portfolio"]


class TestExpirySettlement:
    """Expired options must NEVER settle against a fabricated 0.0 underlying."""

    def _sleeve_with_expired_put(self):
        from datetime import date
        from quantfund.core.instruments import OptionRight, make_option
        from quantfund.core.orders import Fill, OrderSide
        from quantfund.core.portfolio import Portfolio
        put = make_option("QQQ", date(2024, 2, 16), OptionRight.PUT, 105.0)
        sleeve = Portfolio(cash=10_000)
        sleeve.apply_fill(Fill(order_id="x", instrument=put, side=OrderSide.BUY,
                               qty=2, price=0.60,
                               ts=datetime(2024, 2, 1, 21, 0, tzinfo=UTC)))
        return sleeve, put

    def _engine(self):
        from quantfund.core.config import Settings
        import tempfile
        s = Settings(alpaca_paper=True, data_dir=tempfile.mkdtemp())
        return BacktestEngine(SyntheticProvider(seed=42), s)

    def test_settles_from_chain_price_when_no_bars(self):
        """Underlying absent from bars but present via the chain → settle at
        the chain's underlying price, not 0."""
        from quantfund.core.snapshot import MarketSnapshot
        from conftest import make_chain
        engine = self._engine()
        sleeve, put = self._sleeve_with_expired_put()
        as_of = datetime(2024, 2, 20, 21, 0, tzinfo=UTC)
        chain = make_chain("QQQ", spot=157.74,
                           ts=as_of, expiry=put.expiration)
        snap = MarketSnapshot(as_of=as_of, option_chains={"QQQ": chain})
        warnings, trades = [], []
        engine._settle_expired_options(sleeve, snap, trades, warnings)
        # OTM put (strike 105 vs spot 157.74) → intrinsic 0, NOT 105
        assert len(trades) == 1
        assert trades[0].price == pytest.approx(0.0)
        assert not sleeve.positions[put.key].is_open

    def test_left_open_when_no_price_at_all(self):
        from quantfund.core.snapshot import MarketSnapshot
        engine = self._engine()
        sleeve, put = self._sleeve_with_expired_put()
        snap = MarketSnapshot(as_of=datetime(2024, 2, 20, 21, 0, tzinfo=UTC))
        warnings, trades = [], []
        engine._settle_expired_options(sleeve, snap, trades, warnings)
        assert trades == []
        assert sleeve.positions[put.key].is_open
        assert any("cannot settle" in w for w in warnings)

    def test_run_includes_option_underlyings_in_snapshot(self):
        """engine.run must fetch bars for option underlyings even when they are
        not in `symbols`, so settlement always has a real price."""
        engine = self._engine()
        res = engine.run([BuyOnceStrategy()], START, END,
                         symbols=["SPY"], options_underlyings=["QQQ"])
        assert not any("cannot settle" in w for w in res.warnings)
