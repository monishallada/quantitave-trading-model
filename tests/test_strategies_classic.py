"""Momentum + mean-reversion sleeves on hand-built, known-answer data."""
import math

from quantfund.core.instruments import Equity
from quantfund.core.orders import Fill, OrderSide
from quantfund.core.portfolio import Position
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import SignalAction, SleeveContext
from quantfund.strategies.mean_reversion import MeanReversionStrategy
from quantfund.strategies.momentum import MomentumStrategy

from conftest import T0, make_bars, make_quote

N = 220  # enough history for the 200d trend filter + z-score window


def snapshot_with(closes_by_symbol, ts=T0, regime=None):
    bars = {s: make_bars(c, end_ts=ts) for s, c in closes_by_symbol.items()}
    quotes = {s: make_quote(c[-1], ts=ts) for s, c in closes_by_symbol.items()}
    return MarketSnapshot(as_of=ts, bars=bars, quotes=quotes, regime=regime)


def wiggly_series(n=N, start=100.0, drift=0.0, amp=0.001):
    """Deterministic drift + sinusoidal wiggle (nonzero vol, no randomness)."""
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] * (1.0 + drift + amp * math.sin(i / 2.0)))
    return closes


def uptrend_with_crash(n=N, drift=0.002, crash_days=5, crash=-0.025):
    """Long uptrend, sharp recent washout — the classic buy-the-dip setup:
    z-score deeply negative, price still above its 200d SMA."""
    closes = wiggly_series(n - crash_days, drift=drift)
    for _ in range(crash_days):
        closes.append(closes[-1] * (1.0 + crash))
    return closes


def downtrend_with_crash(n=N, drift=-0.002, crash_days=5, crash=-0.025):
    closes = wiggly_series(n - crash_days, drift=drift)
    for _ in range(crash_days):
        closes.append(closes[-1] * (1.0 + crash))
    return closes


def ctx_for(strategy_id, positions=None, capital=50_000):
    return SleeveContext(strategy_id=strategy_id, allocated_capital=capital,
                         positions=positions or {}, sleeve_equity=capital)


class TestMomentum:
    def test_ranks_strongest_names(self):
        snap = snapshot_with({
            "STRONG": wiggly_series(drift=0.004),
            "WEAK": wiggly_series(drift=0.001),
            "FLAT": wiggly_series(drift=0.0005),
        })
        strat = MomentumStrategy(top_n=1)
        signals = strat.generate_signals(snap, ctx_for("momentum"))
        opens = [s for s in signals if s.action == SignalAction.TARGET_WEIGHT]
        assert len(opens) == 1
        assert opens[0].instrument.symbol == "STRONG"
        assert 0 < opens[0].target_weight <= 0.25
        assert "momentum rank" in opens[0].rationale

    def test_absolute_filter_blocks_negative_momentum(self):
        """Every name falling → hold cash, don't buy the least-bad loser."""
        snap = snapshot_with({
            "DOWN1": wiggly_series(drift=-0.001),
            "DOWN2": wiggly_series(drift=-0.003),
        })
        signals = MomentumStrategy(top_n=2).generate_signals(
            snap, ctx_for("momentum"))
        assert not any(s.action == SignalAction.TARGET_WEIGHT for s in signals)

    def test_inverse_vol_sizing(self):
        """Same momentum, calmer name gets the bigger weight."""
        snap = snapshot_with({
            "CALM": wiggly_series(drift=0.003, amp=0.001),
            "WILD": wiggly_series(drift=0.003, amp=0.02),
        })
        signals = MomentumStrategy(top_n=2).generate_signals(
            snap, ctx_for("momentum"))
        w = {s.instrument.symbol: s.target_weight for s in signals
             if s.action == SignalAction.TARGET_WEIGHT}
        assert set(w) == {"CALM", "WILD"}
        assert w["CALM"] > w["WILD"]

    def test_no_signals_before_warmup(self):
        snap = snapshot_with({"AAPL": wiggly_series(n=50, drift=0.004)})
        assert MomentumStrategy().generate_signals(snap, ctx_for("momentum")) == []

    def test_close_emitted_for_dropped_name(self):
        snap = snapshot_with({
            "STRONG": wiggly_series(drift=0.004),
            "OLD": wiggly_series(drift=-0.002),
        })
        old_pos = Position(instrument=Equity("OLD"))
        old_pos.apply_fill(Fill(order_id="x", instrument=Equity("OLD"),
                                side=OrderSide.BUY, qty=10, price=100.0, ts=T0))
        strat = MomentumStrategy(top_n=1)
        signals = strat.generate_signals(
            snap, ctx_for("momentum", positions={"OLD": old_pos}))
        closes = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(closes) == 1
        assert closes[0].instrument.symbol == "OLD"


class TestMeanReversion:
    def test_entry_on_uptrend_washout(self):
        snap = snapshot_with({"DIP": uptrend_with_crash(),
                              "CALM": wiggly_series()})
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(snap, ctx_for("mean_reversion"))
        entries = [s for s in signals if s.action == SignalAction.TARGET_WEIGHT]
        assert len(entries) == 1
        assert entries[0].instrument.symbol == "DIP"
        assert 0 < entries[0].target_weight <= 0.10
        assert "z=" in entries[0].rationale

    def test_trend_filter_blocks_downtrend_knives(self):
        """A washout BELOW the 200d SMA is a falling knife — no entry."""
        snap = snapshot_with({"KNIFE": downtrend_with_crash()})
        signals = MeanReversionStrategy().generate_signals(
            snap, ctx_for("mean_reversion"))
        assert not any(s.action == SignalAction.TARGET_WEIGHT for s in signals)

    def test_exit_when_reverted(self):
        snap = snapshot_with({"HELD": wiggly_series()})
        pos = Position(instrument=Equity("HELD"))
        pos.apply_fill(Fill(order_id="x", instrument=Equity("HELD"),
                            side=OrderSide.BUY, qty=10, price=100.0, ts=T0))
        signals = MeanReversionStrategy().generate_signals(
            snap, ctx_for("mean_reversion", positions={"HELD": pos}))
        assert any(s.action == SignalAction.CLOSE for s in signals)

    def test_no_entries_in_high_vol_regime(self):
        from quantfund.core.snapshot import RegimeState
        snap = snapshot_with(
            {"DIP": uptrend_with_crash()},
            regime=RegimeState(as_of=T0, vol_regime="high", trend_regime="down"),
        )
        signals = MeanReversionStrategy().generate_signals(
            snap, ctx_for("mean_reversion"))
        assert not any(s.action == SignalAction.TARGET_WEIGHT for s in signals)
