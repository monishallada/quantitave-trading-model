"""RiskManager pre-trade gate + circuit breaker tests."""
from datetime import timedelta

import pytest

from quantfund.core.config import RiskLimits
from quantfund.core.instruments import Equity, OptionRight
from quantfund.core.orders import Fill, OrderLeg, OrderSide, Order, single_leg
from quantfund.core.portfolio import Portfolio
from quantfund.core.state import PlatformState, utcnow
from quantfund.risk.circuit_breakers import BREAKER_NAMES, CircuitBreakerBoard
from quantfund.risk.limits import RiskManager

from conftest import T0, make_chain, make_snapshot


def _fill(inst, side, qty, price):
    return Fill(order_id="x", instrument=inst, side=side, qty=qty, price=price, ts=T0)


@pytest.fixture
def limits():
    return RiskLimits()


@pytest.fixture
def rm(limits):
    return RiskManager(limits)


@pytest.fixture
def snap():
    chain = make_chain("AAPL", spot=200.0)
    return make_snapshot({"AAPL": 200.0, "MSFT": 400.0, "XOM": 110.0},
                        chains={"AAPL": chain})


def chain_contract(snap, right, strike):
    chain = snap.get_option_chain("AAPL")
    return next(q.instrument for q in chain.quotes
                if q.instrument.right == right and q.instrument.strike == strike)


class TestNakedShortRules:
    def test_naked_short_call_rejected(self, rm, snap):
        p = Portfolio(cash=100_000)
        call = chain_contract(snap, OptionRight.CALL, 210.0)
        d = rm.check_order(single_leg(call, OrderSide.SELL, 1), p, snap)
        assert not d.approved
        assert "undefined-risk" in d.reason

    def test_covered_call_approved(self, rm, snap):
        p = Portfolio(cash=100_000)
        p.apply_fill(_fill(Equity("AAPL"), OrderSide.BUY, 100, 200.0))
        p.mark_position("AAPL", 200.0, T0)
        call = chain_contract(snap, OptionRight.CALL, 210.0)
        d = rm.check_order(single_leg(call, OrderSide.SELL, 1), p, snap)
        assert d.approved, d.reason

    def test_cash_secured_put_approved_only_with_cash(self, rm, snap):
        put = chain_contract(snap, OptionRight.PUT, 190.0)
        rich = Portfolio(cash=100_000)
        d = rm.check_order(single_leg(put, OrderSide.SELL, 1), rich, snap)
        assert d.approved, d.reason
        poor = Portfolio(cash=5_000)
        d = rm.check_order(single_leg(put, OrderSide.SELL, 1), poor, snap)
        assert not d.approved
        assert "cash-secured" in d.reason

    def test_vertical_spread_approved(self, rm, snap):
        p = Portfolio(cash=100_000)
        long_call = chain_contract(snap, OptionRight.CALL, 200.0)
        short_call = chain_contract(snap, OptionRight.CALL, 210.0)
        order = Order(legs=[
            OrderLeg(instrument=long_call, side=OrderSide.BUY, qty=1),
            OrderLeg(instrument=short_call, side=OrderSide.SELL, qty=1),
        ])
        d = rm.check_order(order, p, snap)
        assert d.approved, d.reason

    def test_naked_allowed_when_enabled(self, snap):
        lim = RiskLimits(allow_naked_short_options=True)
        rm = RiskManager(lim)
        p = Portfolio(cash=100_000)
        call = chain_contract(snap, OptionRight.CALL, 210.0)
        d = rm.check_order(single_leg(call, OrderSide.SELL, 1), p, snap)
        assert d.approved, d.reason


class TestLimitChecks:
    def test_short_equity_rejected(self, rm, snap):
        p = Portfolio(cash=100_000)
        d = rm.check_order(single_leg(Equity("AAPL"), OrderSide.SELL, 10), p, snap)
        assert not d.approved and "short equity" in d.reason

    def test_order_notional_downsized(self, rm, snap):
        p = Portfolio(cash=1_000_000, starting_equity=1_000_000)
        # 100 shares * 400 = 40k > 10k cap → downsize to 25 shares
        d = rm.check_order(single_leg(Equity("MSFT"), OrderSide.BUY, 100), p, snap)
        assert d.approved
        assert d.adjusted_qty == 25

    def test_position_pct_cap(self, snap):
        lim = RiskLimits(max_order_notional=1e9, max_position_notional=1e9,
                         max_position_pct=0.10)
        rm = RiskManager(lim)
        p = Portfolio(cash=100_000)
        # 60 shares * 200 = 12k > 10% of 100k
        d = rm.check_order(single_leg(Equity("AAPL"), OrderSide.BUY, 60), p, snap)
        assert not d.approved and "underlying exposure" in d.reason

    def test_sector_cap(self, snap):
        lim = RiskLimits(max_order_notional=1e9, max_position_notional=1e9,
                         max_position_pct=0.30, max_sector_pct=0.25,
                         min_cash_buffer_pct=0.0)
        rm = RiskManager(lim)
        p = Portfolio(cash=100_000)
        p.apply_fill(_fill(Equity("MSFT"), OrderSide.BUY, 40, 400.0))  # 16k TECH
        p.mark_position("MSFT", 400.0, T0)
        # +14k AAPL puts TECH at 30k > 25% of ~100k equity
        d = rm.check_order(single_leg(Equity("AAPL"), OrderSide.BUY, 70), p, snap)
        assert not d.approved and "sector" in d.reason

    def test_cash_buffer(self, snap):
        lim = RiskLimits(max_order_notional=1e9, max_position_notional=1e9,
                         max_position_pct=1.0, max_sector_pct=1.0,
                         max_gross_leverage=5.0, max_net_delta_pct=5.0)
        rm = RiskManager(lim)
        p = Portfolio(cash=10_000)
        d = rm.check_order(single_leg(Equity("AAPL"), OrderSide.BUY, 49), p, snap)
        assert not d.approved and "cash buffer" in d.reason

    def test_no_market_data_rejected(self, rm, snap):
        p = Portfolio(cash=100_000)
        d = rm.check_order(single_leg(Equity("ZZZZ"), OrderSide.BUY, 10), p, snap)
        assert not d.approved and "no market data" in d.reason


class TestBreakers:
    @pytest.fixture
    def board(self, tmp_path):
        state = PlatformState(tmp_path / "s.db")
        return CircuitBreakerBoard(RiskLimits(), state), state

    def test_all_names_registered(self, board):
        b, state = board
        for name in BREAKER_NAMES:
            assert name in state.breakers

    def test_daily_loss_trips_and_clears(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        p.start_new_day()
        p.cash -= 3_000  # -3% day
        trips = b.check_all(p, snap, None)
        assert any(t.name == "daily_loss" for t in trips)
        p.cash += 3_000
        trips = b.check_all(p, snap, None)
        assert not any(t.name == "daily_loss" for t in trips)

    def test_drawdown_trips(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        p.peak_equity = 120_000  # 16.7% dd > 10%
        trips = b.check_all(p, snap, None)
        assert any(t.name == "drawdown" for t in trips)

    def test_trade_frequency(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        for _ in range(31):
            b.record_trade()
        trips = b.check_all(p, snap, None)
        assert any(t.name == "trade_frequency" for t in trips)

    def test_old_trades_age_out(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        old = utcnow() - timedelta(hours=2)
        for _ in range(40):
            b.record_trade(old)
        trips = b.check_all(p, snap, None)
        assert not any(t.name == "trade_frequency" for t in trips)

    def test_error_burst_and_reset(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        for _ in range(5):
            b.record_error("test", "boom")
        assert any(t.name == "error_burst" for t in b.check_all(p, snap, None))
        b.record_success()
        assert not any(t.name == "error_burst" for t in b.check_all(p, snap, None))

    def test_staleness(self, tmp_path):
        # tight 15-minute limit + 2h-old data → must trip
        state = PlatformState(tmp_path / "stale.db")
        b = CircuitBreakerBoard(RiskLimits(data_staleness_halt_sec=900.0), state)
        p = Portfolio(cash=100_000)
        stale_ts = utcnow() - timedelta(hours=2)
        stale = make_snapshot({"AAPL": 100.0}, ts=stale_ts)
        from quantfund.core.snapshot import MarketSnapshot
        stale_now = MarketSnapshot(as_of=utcnow(), bars=stale.bars,
                                   quotes=stale.quotes)
        trips = b.check_all(p, stale_now, None)
        assert any(t.name == "data_staleness" for t in trips)
        # fresh data under the default (96h) limit does not trip
        b2 = CircuitBreakerBoard(RiskLimits(),
                                 PlatformState(tmp_path / "fresh.db"))
        fresh = make_snapshot({"AAPL": 100.0}, ts=utcnow() - timedelta(hours=12))
        fresh_now = MarketSnapshot(as_of=utcnow(), bars=fresh.bars,
                                   quotes=fresh.quotes)
        trips = b2.check_all(p, fresh_now, None)
        assert not any(t.name == "data_staleness" for t in trips)

    def test_reconciliation(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        p.apply_fill(_fill(Equity("AAPL"), OrderSide.BUY, 100, 200.0))
        from quantfund.execution.broker import BrokerPosition
        broker_pos = [BrokerPosition(instrument=Equity("AAPL"), qty=90,
                                     avg_entry_price=200.0, market_value=18_000)]
        trips = b.check_all(p, snap, broker_pos)
        assert any(t.name == "reconciliation" for t in trips)
        broker_pos = [BrokerPosition(instrument=Equity("AAPL"), qty=100,
                                     avg_entry_price=200.0, market_value=20_000)]
        trips = b.check_all(p, snap, broker_pos)
        assert not any(t.name == "reconciliation" for t in trips)

    def test_order_bounds_latch(self, board, snap):
        b, _ = board
        p = Portfolio(cash=100_000)
        for _ in range(3):
            b.record_order_bounds_violation("too big")
        trips = b.check_all(p, snap, None)
        assert any(t.name == "order_bounds" for t in trips)


class TestNakedShortNetCoverage:
    """Regressions for the net-coverage rewrite: one long covers exactly one
    short; a BUY that merely closes an existing short is not cover."""

    def test_ratio_spread_rejected(self, rm, snap):
        p = Portfolio(cash=100_000)
        c200 = chain_contract(snap, OptionRight.CALL, 200.0)
        c210 = chain_contract(snap, OptionRight.CALL, 210.0)
        c220 = chain_contract(snap, OptionRight.CALL, 220.0)
        order = Order(legs=[
            OrderLeg(instrument=c200, side=OrderSide.BUY, qty=1),
            OrderLeg(instrument=c210, side=OrderSide.SELL, qty=1),
            OrderLeg(instrument=c220, side=OrderSide.SELL, qty=1),
        ])
        d = rm.check_order(order, p, snap)
        assert not d.approved and "undefined-risk" in d.reason

    def test_one_long_two_short_same_strike_rejected(self, rm, snap):
        p = Portfolio(cash=100_000)
        c200 = chain_contract(snap, OptionRight.CALL, 200.0)
        c210 = chain_contract(snap, OptionRight.CALL, 210.0)
        order = Order(legs=[
            OrderLeg(instrument=c200, side=OrderSide.BUY, qty=1),
            OrderLeg(instrument=c210, side=OrderSide.SELL, qty=2),
        ])
        d = rm.check_order(order, p, snap)
        assert not d.approved and "undefined-risk" in d.reason

    def test_closing_buy_is_not_cover(self, rm, snap):
        p = Portfolio(cash=100_000)
        c200 = chain_contract(snap, OptionRight.CALL, 200.0)
        c210 = chain_contract(snap, OptionRight.CALL, 210.0)
        # portfolio is already short 1 x 200C
        p.apply_fill(_fill(c200, OrderSide.SELL, 1, 5.0))
        order = Order(legs=[
            OrderLeg(instrument=c200, side=OrderSide.BUY, qty=1),   # just closes
            OrderLeg(instrument=c210, side=OrderSide.SELL, qty=1),  # naked!
        ])
        d = rm.check_order(order, p, snap)
        assert not d.approved and "undefined-risk" in d.reason

    def test_two_by_two_vertical_approved(self, rm, snap):
        lim = RiskLimits(max_order_notional=1e9)
        rm2 = RiskManager(lim)
        p = Portfolio(cash=100_000)
        c200 = chain_contract(snap, OptionRight.CALL, 200.0)
        c210 = chain_contract(snap, OptionRight.CALL, 210.0)
        order = Order(legs=[
            OrderLeg(instrument=c200, side=OrderSide.BUY, qty=2),
            OrderLeg(instrument=c210, side=OrderSide.SELL, qty=2),
        ])
        d = rm2.check_order(order, p, snap)
        assert d.approved, d.reason

    def test_shares_not_double_counted_across_expiries(self, rm, snap):
        # 100 shares can cover ONE short call, not one per expiry
        p = Portfolio(cash=100_000)
        p.apply_fill(_fill(Equity("AAPL"), OrderSide.BUY, 100, 200.0))
        p.mark_position("AAPL", 200.0, T0)
        c210 = chain_contract(snap, OptionRight.CALL, 210.0)
        c220 = chain_contract(snap, OptionRight.CALL, 220.0)
        order = Order(legs=[
            OrderLeg(instrument=c210, side=OrderSide.SELL, qty=1),
            OrderLeg(instrument=c220, side=OrderSide.SELL, qty=1),
        ])
        d = rm.check_order(order, p, snap)
        assert not d.approved and "undefined-risk" in d.reason


class TestTransientErrors:
    def test_network_blips_need_3x_to_trip(self, tmp_path, snap):
        state = PlatformState(tmp_path / "t.db")
        b = CircuitBreakerBoard(RiskLimits(), state)
        p = Portfolio(cash=100_000)
        limit = RiskLimits().max_consecutive_errors
        for _ in range(limit):          # would trip if counted as real errors
            b.record_error("runner", "ConnectionError(...)", transient=True)
        assert not any(t.name == "error_burst" for t in b.check_all(p, snap, None))
        for _ in range(2 * limit):      # sustained outage still trips
            b.record_error("runner", "ConnectionError(...)", transient=True)
        assert any(t.name == "error_burst" for t in b.check_all(p, snap, None))

    def test_real_errors_still_trip_fast(self, tmp_path, snap):
        state = PlatformState(tmp_path / "t2.db")
        b = CircuitBreakerBoard(RiskLimits(), state)
        p = Portfolio(cash=100_000)
        for _ in range(RiskLimits().max_consecutive_errors):
            b.record_error("strategy", "ValueError: bad math")
        assert any(t.name == "error_burst" for t in b.check_all(p, snap, None))

    def test_transient_classifier(self):
        from quantfund.live.runner import _is_transient
        assert _is_transient(ConnectionError("Max retries exceeded"))
        assert _is_transient(RuntimeError("Remote end closed connection"))
        assert not _is_transient(ValueError("target_weight must be within [-1, 1]"))
        assert not _is_transient(KeyError("momentum"))


class TestLongOptionPremiumCap:
    """Loss on a long option is capped at premium — but only per position.
    The portfolio cap is what stops every contract expiring worthless at once."""

    def test_blocks_premium_beyond_cap(self, snap):
        # every other cap wide open so the PREMIUM cap is the binding one
        lim = RiskLimits(max_long_option_premium_pct=0.05,
                         max_order_notional=1e9, max_position_notional=1e9,
                         max_position_pct=50.0, max_sector_pct=50.0,
                         max_gross_leverage=50.0, max_net_delta_pct=50.0,
                         max_theta_pct_per_day=10.0, max_vega_pct=10.0,
                         min_cash_buffer_pct=0.0)
        rm = RiskManager(lim)
        p = Portfolio(cash=100_000)
        call = chain_contract(snap, OptionRight.CALL, 200.0)
        # cap = 5% of $100k = $5,000; ATM call mids ~$5 => $500 of premium
        # per contract
        d1 = rm.check_order(single_leg(call, OrderSide.BUY, 4), p, snap)   # $2k
        assert d1.approved, d1.reason
        d2 = rm.check_order(single_leg(call, OrderSide.BUY, 20), p, snap)  # $10k
        assert not d2.approved
        assert "premium at risk" in d2.reason

    def test_existing_premium_counts_toward_cap(self, snap):
        # every other cap wide open so the PREMIUM cap is the binding one
        lim = RiskLimits(max_long_option_premium_pct=0.05,
                         max_order_notional=1e9, max_position_notional=1e9,
                         max_position_pct=50.0, max_sector_pct=50.0,
                         max_gross_leverage=50.0, max_net_delta_pct=50.0,
                         max_theta_pct_per_day=10.0, max_vega_pct=10.0,
                         min_cash_buffer_pct=0.0)
        rm = RiskManager(lim)
        p = Portfolio(cash=100_000)
        call = chain_contract(snap, OptionRight.CALL, 200.0)
        p.apply_fill(_fill(call, OrderSide.BUY, 8, 5.0))    # $4,000 already at risk
        p.mark_position(call.key, 5.0, T0)
        d = rm.check_order(single_leg(call, OrderSide.BUY, 6), p, snap)  # +$3,000
        assert not d.approved and "premium at risk" in d.reason

    def test_explosive_cap_reasoned_against_drawdown_halt(self):
        """Two different loss shapes have to stay survivable.

        DEEP-ITM longs retain intrinsic value: a 0.75-delta call needs the
        underlying through its strike before premium goes to zero, so a severe
        adverse move costs roughly HALF the premium. That severe case must
        land inside the drawdown halt, or the halt is reacting to damage
        already done rather than preventing it.

        CAPPED-LOSS longs (0DTE, far OTM) go to zero routinely, so the premium
        cap alone does not bound them — the DAILY loss halt does, and it must
        fire before a wipeout day can approach the drawdown halt.
        """
        lim = RiskLimits.explosive()
        severe_itm_loss = 0.5           # fraction of premium lost in a bad move
        assert (lim.max_long_option_premium_pct * severe_itm_loss
                <= lim.max_drawdown_halt_pct), (
            "a severe ITM move alone would blow through the drawdown halt")
        # derived, not hardcoded: cap * severe_loss <= drawdown_halt
        assert lim.max_long_option_premium_pct <= 2.0 * lim.max_drawdown_halt_pct
        # a wipeout day must trip the DAILY halt well before the drawdown halt,
        # otherwise the account can be gutted inside one session
        assert lim.daily_loss_halt_pct < lim.max_drawdown_halt_pct
        assert lim.daily_loss_halt_pct <= 0.30, (
            "beyond -30% in a single day the account cannot recover in any "
            "reasonable number of sessions")
        # undefined risk stays banned at every aggression level
        assert lim.allow_naked_short_options is False


class TestPreexistingRiskDoesNotBlockNewOrders:
    """A pre-existing short put that drifted under-collateralized must not
    reject unrelated option orders — especially long calls, which carry no
    undefined risk at all (2026-08-05: this froze options trading entirely)."""

    def _portfolio_with_undercollateralized_put(self, snap):
        p = Portfolio(cash=100_000)
        put = chain_contract(snap, OptionRight.PUT, 190.0)
        p.apply_fill(_fill(put, OrderSide.SELL, 2, 3.0))   # needs $38k secured
        p.mark_position(put.key, 3.0, T0)
        p.cash = 5_000                                     # cash drained away
        return p, put

    def test_long_call_not_blocked_by_unrelated_short_put(self, rm, snap):
        p, _ = self._portfolio_with_undercollateralized_put(snap)
        call = chain_contract(snap, OptionRight.CALL, 200.0)
        d = rm.check_order(single_leg(call, OrderSide.BUY, 1), p, snap)
        assert "undefined-risk" not in d.reason

    def test_closing_the_bad_put_is_allowed(self, rm, snap):
        p, put = self._portfolio_with_undercollateralized_put(snap)
        d = rm.check_order(single_leg(put, OrderSide.BUY, 2), p, snap)
        assert "undefined-risk" not in d.reason

    def test_adding_more_uncovered_shorts_still_rejected(self, rm, snap):
        p, put = self._portfolio_with_undercollateralized_put(snap)
        d = rm.check_order(single_leg(put, OrderSide.SELL, 2), p, snap)
        assert not d.approved and "undefined-risk" in d.reason


class TestPutCollateralIsNotSpendable:
    def test_equity_buy_cannot_consume_put_collateral(self, snap):
        lim = RiskLimits(max_order_notional=1e9, max_position_notional=1e9,
                         max_position_pct=5.0, max_sector_pct=5.0,
                         max_gross_leverage=5.0, max_net_delta_pct=5.0,
                         max_theta_pct_per_day=1.0, max_vega_pct=1.0,
                         min_cash_buffer_pct=0.05)
        rm = RiskManager(lim)
        p = Portfolio(cash=100_000)
        put = chain_contract(snap, OptionRight.PUT, 190.0)
        p.apply_fill(_fill(put, OrderSide.SELL, 4, 3.0))   # $76k of collateral
        p.mark_position(put.key, 3.0, T0)
        # only ~$24k is genuinely free; a $50k stock buy must be refused
        d = rm.check_order(single_leg(Equity("AAPL"), OrderSide.BUY, 250), p, snap)
        assert not d.approved
        assert "securing short puts" in d.reason


def test_gate_and_book_measure_option_exposure_the_same_way():
    """The gate must authorize using the yardstick the book reports.

    Alpaca serves no greeks on 0DTE contracts. The gate used to fall back to
    premium while Portfolio fell back to Black-Scholes, so a 21-lot QQQ 0DTE
    call was authorized as "$1,911 of exposure" and landed on the book as
    $510,170 — net dollar delta 6.39x equity (observed live 2026-08-07).
    Whatever rule is chosen, both sides must apply the SAME one.
    """
    from datetime import datetime, timezone

    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide, single_leg
    from quantfund.core.portfolio import Portfolio, Position, option_exposure
    from quantfund.core.snapshot import (
        MarketSnapshot, OptionChain, OptionQuote, Quote,
    )
    from quantfund.risk.limits import _leg_greeks

    now = datetime(2026, 8, 7, 15, 0, tzinfo=timezone.utc)
    spot = 721.0
    inst = make_option("QQQ", now.date(), OptionRight.CALL, 723.0)   # 0DTE
    # NO greeks and NO iv — exactly what the indicative feed returns on 0DTE
    oq = OptionQuote(instrument=inst,
                     quote=Quote(ts=now, bid=0.89, ask=0.95),
                     iv=None, greeks=None)
    snap = MarketSnapshot(
        as_of=now,
        quotes={"QQQ": Quote(ts=now, bid=spot - 0.01, ask=spot + 0.01)},
        option_chains={"QQQ": OptionChain(underlying="QQQ", ts=now,
                                          underlying_price=spot,
                                          quotes=(oq,))},
    )

    order = single_leg(inst, OrderSide.BUY, 21.0, strategy_id="zero_dte")
    g = _leg_greeks(order.legs[0], snap)
    assert g is not None, "gate must derive greeks when the feed omits them"
    assert 0.05 < g.delta < 0.95

    gate_expo = option_exposure(
        qty=21.0, multiplier=100, delta=g.delta,
        underlying_price=spot, premium=oq.quote.mid)

    pf = Portfolio(cash=100_000.0, starting_equity=100_000.0)
    pf.apply_fill(Fill(order_id="o", instrument=inst, side=OrderSide.BUY,
                       qty=21, price=oq.quote.mid, ts=now))
    pf.mark_from_snapshot(snap)
    book_expo = pf.exposure_by_underlying()["QQQ"]

    assert gate_expo == pytest.approx(book_expo, rel=1e-6), (
        f"gate says ${gate_expo:,.0f}, book says ${book_expo:,.0f}")
    # this contract is a capped-loss ticket, so both measure it by PREMIUM
    assert book_expo == pytest.approx(oq.quote.mid * 21 * 100, rel=1e-6)


def test_deep_itm_long_still_measured_by_delta_notional():
    """The premium rule must NOT leak to stock-replacement calls. A 0.85-delta
    call tracks the underlying nearly 1:1; pricing it at premium would hide
    real leverage behind a small number."""
    from datetime import date, datetime, timedelta, timezone

    from quantfund.core.greeks import Greeks
    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide
    from quantfund.core.portfolio import Portfolio

    now = datetime(2026, 8, 7, 15, 0, tzinfo=timezone.utc)
    spot = 772.0
    itm = make_option("SPY", now.date() + timedelta(days=14),
                      OptionRight.CALL, 758.0)
    pf = Portfolio(cash=100_000.0, starting_equity=100_000.0)
    pf.apply_fill(Fill(order_id="o", instrument=itm, side=OrderSide.BUY,
                       qty=1, price=17.60, ts=now))
    pos = pf.positions[itm.key]
    pos.mark, pos.underlying_mark = 17.60, spot
    pos.greeks = Greeks(delta=0.85, gamma=0.01, theta=-0.5, vega=0.2, rho=0.1)

    expo = pf.exposure_by_underlying()["SPY"]
    assert expo == pytest.approx(0.85 * 100 * spot), (
        "deep-ITM long must stay on delta notional")
    assert expo > 30 * (17.60 * 100)


def test_closing_a_long_option_is_never_blocked_by_exposure_caps():
    """A stop-loss must always be able to execute.

    option_exposure treats qty < 0 as a SHORT POSITION, but a negative ORDER
    LEG is a SELL — which for a long holding is a CLOSE. Measuring the leg in
    isolation priced a 21-lot 0DTE stop-loss as opening a $422k short, and the
    gate rejected it three passes running while the call sat open past its
    -35% trigger (2026-08-07 16:24-16:27). Exposure change must be measured as
    post-position minus pre-position.
    """
    from datetime import datetime, timezone

    from quantfund.core.config import RiskLimits
    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide, single_leg
    from quantfund.core.portfolio import Portfolio
    from quantfund.core.snapshot import (
        MarketSnapshot, OptionChain, OptionQuote, Quote,
    )
    from quantfund.risk.limits import RiskManager

    now = datetime(2026, 8, 7, 16, 25, tzinfo=timezone.utc)
    spot = 719.0
    call = make_option("QQQ", now.date(), OptionRight.CALL, 723.0)
    oq = OptionQuote(instrument=call,
                     quote=Quote(ts=now, bid=0.55, ask=0.58),
                     iv=None, greeks=None)
    snap = MarketSnapshot(
        as_of=now,
        quotes={"QQQ": Quote(ts=now, bid=spot - 0.01, ask=spot + 0.01)},
        option_chains={"QQQ": OptionChain(underlying="QQQ", ts=now,
                                          underlying_price=spot,
                                          quotes=(oq,))},
    )
    pf = Portfolio(cash=100_000.0, starting_equity=100_000.0)
    pf.apply_fill(Fill(order_id="entry", instrument=call, side=OrderSide.BUY,
                       qty=21, price=0.92, ts=now))
    pf.mark_from_snapshot(snap)

    rm = RiskManager(RiskLimits.explosive())
    stop = single_leg(call, OrderSide.SELL, 21.0, strategy_id="zero_dte")
    decision = rm.check_order(stop, pf, snap)
    assert decision.approved, f"stop-loss rejected: {decision.reason}"

    # and a partial close is fine too
    half = single_leg(call, OrderSide.SELL, 10.0, strategy_id="zero_dte")
    assert rm.check_order(half, pf, snap).approved


def test_opening_a_genuine_short_option_still_measured_as_short():
    """The close fix must not make real shorts look free."""
    from datetime import datetime, timedelta, timezone

    from quantfund.core.config import RiskLimits
    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import OrderSide, single_leg
    from quantfund.core.portfolio import Portfolio
    from quantfund.core.snapshot import (
        MarketSnapshot, OptionChain, OptionQuote, Quote,
    )
    from quantfund.risk.limits import RiskManager

    now = datetime(2026, 8, 7, 16, 25, tzinfo=timezone.utc)
    spot = 719.0
    call = make_option("QQQ", now.date() + timedelta(days=14),
                       OptionRight.CALL, 700.0)
    oq = OptionQuote(instrument=call,
                     quote=Quote(ts=now, bid=25.0, ask=25.4),
                     iv=None, greeks=None)
    snap = MarketSnapshot(
        as_of=now,
        quotes={"QQQ": Quote(ts=now, bid=spot - 0.01, ask=spot + 0.01)},
        option_chains={"QQQ": OptionChain(underlying="QQQ", ts=now,
                                          underlying_price=spot,
                                          quotes=(oq,))},
    )
    pf = Portfolio(cash=100_000.0, starting_equity=100_000.0)
    rm = RiskManager(RiskLimits.explosive())
    naked = single_leg(call, OrderSide.SELL, 20.0, strategy_id="x")
    d = rm.check_order(naked, pf, snap)
    assert not d.approved, "writing 20 naked calls must not pass the gate"


def test_defined_risk_spread_legs_offset_instead_of_reading_as_naked():
    """The moneyness rule is not additive across spread legs.

    Measuring a long leg by premium while its covering short leg uses full
    delta notional made a defined-risk vertical read like a large naked short
    and the gate rejected it. Within a (underlying, right, expiration) group
    that holds shorts, every leg must use the SAME basis so they offset.
    """
    from datetime import datetime, timedelta, timezone

    from quantfund.core.config import RiskLimits
    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Order, OrderLeg, OrderSide
    from quantfund.core.portfolio import Portfolio
    from quantfund.core.snapshot import (
        MarketSnapshot, OptionChain, OptionQuote, Quote,
    )
    from quantfund.risk.limits import RiskManager

    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    exp = now.date() + timedelta(days=14)
    spot = 200.0
    lo = make_option("AAPL", exp, OptionRight.CALL, 200.0)
    hi = make_option("AAPL", exp, OptionRight.CALL, 210.0)
    chain = OptionChain(
        underlying="AAPL", ts=now, underlying_price=spot,
        quotes=(
            OptionQuote(instrument=lo, quote=Quote(ts=now, bid=8.0, ask=8.2)),
            OptionQuote(instrument=hi, quote=Quote(ts=now, bid=3.0, ask=3.2)),
        ))
    snap = MarketSnapshot(
        as_of=now, quotes={"AAPL": Quote(ts=now, bid=spot - .01, ask=spot + .01)},
        option_chains={"AAPL": chain})

    pf = Portfolio(cash=100_000.0, starting_equity=100_000.0)
    rm = RiskManager(RiskLimits(max_order_notional=1e9))   # conservative caps
    spread = Order(legs=[
        OrderLeg(instrument=lo, side=OrderSide.BUY, qty=2),
        OrderLeg(instrument=hi, side=OrderSide.SELL, qty=2),
    ])
    d = rm.check_order(spread, pf, snap)
    assert d.approved, f"defined-risk vertical rejected: {d.reason}"

    # a NAKED short of the same size must still be refused
    naked = Order(legs=[OrderLeg(instrument=hi, side=OrderSide.SELL, qty=2)])
    assert not rm.check_order(naked, pf, snap).approved
