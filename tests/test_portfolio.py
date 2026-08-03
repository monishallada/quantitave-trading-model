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
