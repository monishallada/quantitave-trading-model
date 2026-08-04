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


class TestMomentumOptionsMode:
    """Stock-replacement mode: same signal, expressed as deep-ITM calls."""

    def _snap_with_chains(self, closes_by_symbol, ts=T0):
        from conftest import make_chain
        from datetime import timedelta
        bars = {s: make_bars(c, end_ts=ts) for s, c in closes_by_symbol.items()}
        quotes = {s: make_quote(c[-1], ts=ts) for s, c in closes_by_symbol.items()}
        chains = {s: make_chain(s, spot=c[-1], ts=ts,
                                expiry=(ts + timedelta(days=60)).date())
                  for s, c in closes_by_symbol.items()}
        return MarketSnapshot(as_of=ts, bars=bars, quotes=quotes,
                              option_chains=chains)

    def test_buys_calls_not_shares(self):
        from quantfund.core.instruments import Option, OptionRight
        snap = self._snap_with_chains({"STRONG": wiggly_series(drift=0.004)})
        strat = MomentumStrategy(top_n=1, express_via="options")
        assert strat.uses_options is True
        signals = strat.generate_signals(snap, ctx_for("momentum"))
        opens = [s for s in signals if s.action == SignalAction.TARGET_WEIGHT]
        assert len(opens) == 1
        sig = opens[0]
        assert isinstance(sig.instrument, Option)
        assert sig.instrument.right == OptionRight.CALL
        assert sig.instrument.underlying_symbol == "STRONG"
        assert 0 < sig.target_weight <= 0.30
        p = sig.rationale_payload
        assert p["express_via"] == "options"
        assert p["delta"] >= 0.5          # deep ITM, not a lottery ticket
        assert p["dte"] >= 45             # long-dated => low theta bleed
        assert "stock replacement" in sig.rationale

    def test_equity_mode_still_buys_shares(self):
        snap = self._snap_with_chains({"STRONG": wiggly_series(drift=0.004)})
        strat = MomentumStrategy(top_n=1)  # default equity
        assert strat.uses_options is False
        sig = [s for s in strat.generate_signals(snap, ctx_for("momentum"))
               if s.action == SignalAction.TARGET_WEIGHT][0]
        assert isinstance(sig.instrument, Equity)

    def test_closes_option_when_name_exits_top(self):
        from quantfund.core.instruments import OptionRight, make_option
        from datetime import timedelta
        snap = self._snap_with_chains({
            "STRONG": wiggly_series(drift=0.004),
            "OLD": wiggly_series(drift=-0.002),
        })
        old_call = make_option("OLD", (T0 + timedelta(days=60)).date(),
                               OptionRight.CALL, 90.0)
        pos = Position(instrument=old_call)
        pos.apply_fill(Fill(order_id="x", instrument=old_call,
                            side=OrderSide.BUY, qty=2, price=12.0, ts=T0))
        strat = MomentumStrategy(top_n=1, express_via="options")
        signals = strat.generate_signals(
            snap, ctx_for("momentum", positions={old_call.key: pos}))
        closes = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(closes) == 1
        assert closes[0].instrument.key == old_call.key

    def test_rolls_before_expiry(self):
        from quantfund.core.instruments import OptionRight, make_option
        from datetime import timedelta
        snap = self._snap_with_chains({"STRONG": wiggly_series(drift=0.004)})
        near = make_option("STRONG", (T0 + timedelta(days=10)).date(),
                           OptionRight.CALL, 90.0)
        pos = Position(instrument=near)
        pos.apply_fill(Fill(order_id="x", instrument=near, side=OrderSide.BUY,
                            qty=1, price=12.0, ts=T0))
        strat = MomentumStrategy(top_n=1, express_via="options")
        signals = strat.generate_signals(
            snap, ctx_for("momentum", positions={near.key: pos}))
        closes = [s for s in signals if s.action == SignalAction.CLOSE]
        assert closes and "roll" in closes[0].rationale

    def test_no_duplicate_entry_for_held_contract(self):
        snap = self._snap_with_chains({"STRONG": wiggly_series(drift=0.004)})
        strat = MomentumStrategy(top_n=1, express_via="options")
        sig = [s for s in strat.generate_signals(snap, ctx_for("momentum"))
               if s.action == SignalAction.TARGET_WEIGHT][0]
        pos = Position(instrument=sig.instrument)
        pos.apply_fill(Fill(order_id="x", instrument=sig.instrument,
                            side=OrderSide.BUY, qty=1, price=10.0, ts=T0))
        again = strat.generate_signals(
            snap, ctx_for("momentum", positions={sig.instrument.key: pos}))
        assert not any(s.action == SignalAction.TARGET_WEIGHT for s in again)

    def test_falls_back_when_preferred_dte_window_empty(self):
        """Sparse chain (only short-dated): widen rather than skip the name."""
        from conftest import make_chain
        from datetime import timedelta
        from quantfund.core.instruments import Option
        closes = wiggly_series(drift=0.004)
        bars = {"STRONG": make_bars(closes, end_ts=T0)}
        chain = make_chain("STRONG", spot=closes[-1], ts=T0,
                           expiry=(T0 + timedelta(days=32)).date())  # < 45 DTE
        snap = MarketSnapshot(as_of=T0, bars=bars,
                              quotes={"STRONG": make_quote(closes[-1], ts=T0)},
                              option_chains={"STRONG": chain})
        strat = MomentumStrategy(top_n=1, express_via="options")
        opens = [s for s in strat.generate_signals(snap, ctx_for("momentum"))
                 if s.action == SignalAction.TARGET_WEIGHT]
        assert len(opens) == 1
        assert isinstance(opens[0].instrument, Option)
        assert opens[0].rationale_payload["dte"] > strat.option_roll_dte

    def test_options_mode_ranks_only_optionable_names(self):
        """A stronger name without a chain must not displace an optionable one."""
        from conftest import make_chain
        from datetime import timedelta
        from quantfund.core.instruments import Option
        strong_no_chain = wiggly_series(drift=0.006)   # best momentum, no chain
        weaker_optionable = wiggly_series(drift=0.003)
        bars = {"NOCHAIN": make_bars(strong_no_chain, end_ts=T0),
                "HASCHAIN": make_bars(weaker_optionable, end_ts=T0)}
        quotes = {"NOCHAIN": make_quote(strong_no_chain[-1], ts=T0),
                  "HASCHAIN": make_quote(weaker_optionable[-1], ts=T0)}
        chains = {"HASCHAIN": make_chain("HASCHAIN", spot=weaker_optionable[-1],
                                         ts=T0,
                                         expiry=(T0 + timedelta(days=60)).date())}
        snap = MarketSnapshot(as_of=T0, bars=bars, quotes=quotes,
                              option_chains=chains)
        strat = MomentumStrategy(top_n=1, express_via="options")
        opens = [s for s in strat.generate_signals(snap, ctx_for("momentum"))
                 if s.action == SignalAction.TARGET_WEIGHT]
        assert len(opens) == 1
        assert isinstance(opens[0].instrument, Option)
        assert opens[0].instrument.underlying_symbol == "HASCHAIN"

    def test_instance_level_strategy_id(self):
        """Equity and options expressions must be separable sleeves."""
        opts = MomentumStrategy(express_via="options")
        eq = MomentumStrategy(express_via="equity",
                              strategy_id="momentum_equity",
                              name="Momentum (equity core)")
        assert opts.strategy_id == "momentum"
        assert eq.strategy_id == "momentum_equity"
        assert eq.name == "Momentum (equity core)"
        assert MomentumStrategy.strategy_id == "momentum"  # class default intact
