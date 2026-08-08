"""Portfolio accounting guards: cash, cost basis, realized/unrealized P&L,
net Greeks, drawdown. Costs must always reduce equity."""
from datetime import date, datetime, timezone

import pytest

from quantfund.core.greeks import Greeks
from quantfund.core.instruments import Equity, OptionRight, make_option
from quantfund.core.orders import Fill, OrderSide
from quantfund.core.portfolio import Portfolio

UTC = timezone.utc
TS = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
AAPL = Equity("AAPL")
CALL = make_option("AAPL", date(2024, 7, 19), OptionRight.CALL, 200.0)


def fill(inst, side, qty, price, commission=0.0, fees=0.0, slip=0.0):
    return Fill(order_id="o1", instrument=inst, side=side, qty=qty, price=price,
                ts=TS, commission=commission, fees=fees, slippage_cost=slip)


class TestEquityAccounting:
    def test_buy_moves_cash_and_basis(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 150.0, commission=1.0))
        assert p.cash == pytest.approx(100_000 - 15_000 - 1.0)
        pos = p.positions["AAPL"]
        assert pos.qty == 100
        assert pos.avg_cost == pytest.approx(150.0)
        # equity = cash + mv; only the commission is lost at the fill mark
        assert p.equity() == pytest.approx(100_000 - 1.0)

    def test_averaging_up(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0))
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 110.0))
        assert p.positions["AAPL"].avg_cost == pytest.approx(105.0)
        assert p.positions["AAPL"].qty == 200

    def test_partial_close_realizes_pnl(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0))
        realized = p.apply_fill(fill(AAPL, OrderSide.SELL, 40, 120.0))
        assert realized == pytest.approx(40 * 20.0)
        assert p.positions["AAPL"].qty == 60
        assert p.positions["AAPL"].avg_cost == pytest.approx(100.0)
        assert p.realized_pnl() == pytest.approx(800.0)

    def test_flip_through_zero(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0))
        realized = p.apply_fill(fill(AAPL, OrderSide.SELL, 150, 110.0))
        assert realized == pytest.approx(100 * 10.0)
        pos = p.positions["AAPL"]
        assert pos.qty == -50
        assert pos.avg_cost == pytest.approx(110.0)

    def test_commissions_reduce_equity_not_realized(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0, commission=5.0))
        p.apply_fill(fill(AAPL, OrderSide.SELL, 100, 100.0, commission=5.0, fees=1.0))
        assert p.realized_pnl() == pytest.approx(0.0)  # price-based
        assert p.total_pnl() == pytest.approx(-11.0)   # equity is net of costs
        assert p.total_commissions_fees == pytest.approx(11.0)


class TestOptionAccounting:
    def test_option_notional_uses_multiplier(self):
        p = Portfolio(cash=50_000)
        p.apply_fill(fill(CALL, OrderSide.BUY, 5, 2.50, commission=3.25))
        # 5 contracts * $2.50 * 100 = $1,250 premium
        assert p.cash == pytest.approx(50_000 - 1_250 - 3.25)
        pos = p.positions[CALL.key]
        assert pos.market_value == pytest.approx(1_250)

    def test_option_pnl_scales_by_100(self):
        p = Portfolio(cash=50_000)
        p.apply_fill(fill(CALL, OrderSide.BUY, 2, 2.00))
        realized = p.apply_fill(fill(CALL, OrderSide.SELL, 2, 3.00))
        assert realized == pytest.approx(2 * 1.00 * 100)


class TestNetGreeks:
    def test_aggregation_across_equity_and_options(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 200.0))
        p.apply_fill(fill(CALL, OrderSide.BUY, 2, 5.0))
        p.mark_position("AAPL", 200.0, TS)
        p.mark_position(
            CALL.key, 5.0, TS,
            greeks=Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.10),
            underlying_price=200.0,
        )
        g = p.net_greeks()
        # equity: 100 sh * $200 = $20k delta; option: 0.5 * 200 * $200 = $20k
        assert g.dollar_delta == pytest.approx(20_000 + 20_000)
        assert g.gamma_shares == pytest.approx(0.02 * 200)
        assert g.theta_per_day == pytest.approx(-0.05 * 200)
        assert g.vega == pytest.approx(0.10 * 200)

    def test_short_option_flips_greek_signs(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(CALL, OrderSide.SELL, 1, 5.0))
        p.mark_position(
            CALL.key, 5.0, TS,
            greeks=Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.10),
            underlying_price=200.0,
        )
        g = p.net_greeks()
        assert g.dollar_delta == pytest.approx(-0.5 * 100 * 200.0)
        assert g.theta_per_day == pytest.approx(0.05 * 100)  # short option collects theta


class TestRiskMetrics:
    def test_drawdown_tracks_peak(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0))
        p.mark_position("AAPL", 130.0, TS)
        p.peak_equity = max(p.peak_equity, p.equity())
        p.mark_position("AAPL", 110.0, TS)
        assert p.drawdown() == pytest.approx((13_000 - 11_000) / 103_000, rel=1e-3)

    def test_day_pnl_anchor(self):
        p = Portfolio(cash=100_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0))
        p.start_new_day()
        p.mark_position("AAPL", 105.0, TS)
        assert p.day_pnl() == pytest.approx(500.0)

    def test_buying_power_never_negative(self):
        p = Portfolio(cash=1_000)
        p.apply_fill(fill(AAPL, OrderSide.BUY, 100, 100.0))  # overdraws in sim
        assert p.buying_power == 0.0


def test_option_premium_and_economic_exposure_are_reported_separately():
    """A deep-ITM call's premium is a small fraction of the stock it controls.
    Reporting only premium made a ~1x-delta book look 19% deployed, which is
    how "the portfolio isn't deployed" gets said about a fully-working book.
    """
    from datetime import date, timedelta

    from quantfund.core.greeks import Greeks
    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide
    from quantfund.core.portfolio import Portfolio
    from quantfund.core.state import utcnow

    pf = Portfolio(cash=100_000.0)
    call = make_option("SPY", date.today() + timedelta(days=14),
                       OptionRight.CALL, 758.0)
    pf.apply_fill(Fill(order_id="o1", instrument=call, side=OrderSide.BUY,
                       qty=1, price=17.60, ts=utcnow()))
    pos = pf.positions[call.key]
    pos.mark = 17.60
    pos.greeks = Greeks(delta=0.85, gamma=0.01, theta=-0.5, vega=0.2, rho=0.1)
    pos.underlying_mark = 772.0

    premium = pf.gross_exposure()
    economic = pf.gross_delta_exposure()
    assert premium == pytest.approx(1760.0)              # what can go to zero
    assert economic == pytest.approx(0.85 * 100 * 772.0)  # what actually moves
    assert economic > 30 * premium, (
        "economic exposure must reflect delta notional, not premium")


def test_long_option_theta_cannot_exceed_its_premium():
    """BS theta is an instantaneous rate that explodes into expiry. A 21-lot
    QQQ 0DTE call holding $1,932 of premium reported -$7,930/day of decay —
    a loss it cannot possibly realise, which made theta the binding risk limit
    for a position whose true worst case is the premium (2026-08-07)."""
    from quantfund.core.portfolio import option_theta_per_day

    premium, qty, mult = 0.92, 21.0, 100
    max_loss = premium * qty * mult                     # $1,932

    bounded = option_theta_per_day(qty=qty, multiplier=mult,
                                   theta=-3.78, premium=premium)
    assert abs(bounded) <= max_loss + 1e-9, (
        f"theta {bounded:,.0f} exceeds the ${max_loss:,.0f} that can decay")
    assert bounded == pytest.approx(-max_loss)

    # a normal, far-from-expiry long is untouched
    mild = option_theta_per_day(qty=1, multiplier=100, theta=-0.05,
                                premium=17.60)
    assert mild == pytest.approx(-5.0)

    # SHORT options are not premium-bounded — their risk is unbounded elsewhere
    short = option_theta_per_day(qty=-6, multiplier=100, theta=-0.02,
                                 premium=0.91)
    assert short == pytest.approx(-0.02 * -6 * 100)
