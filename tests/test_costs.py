"""Realistic-accounting guards: a zero-cost fill is a bug."""
from datetime import date, datetime, timezone

import pytest

from quantfund.core.costs import CostConfig, CostModel
from quantfund.core.instruments import Equity, OptionRight, make_option
from quantfund.core.orders import OrderSide
from quantfund.core.snapshot import Quote

UTC = timezone.utc
TS = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)


@pytest.fixture
def model():
    return CostModel(CostConfig())


class TestEquityFills:
    def test_buy_never_better_than_ask(self, model):
        q = Quote(ts=TS, bid=99.95, ask=100.05)
        px = model.execution_price(Equity("AAPL"), OrderSide.BUY, q)
        assert px >= q.ask

    def test_sell_never_better_than_bid(self, model):
        q = Quote(ts=TS, bid=99.95, ask=100.05)
        px = model.execution_price(Equity("AAPL"), OrderSide.SELL, q)
        assert px <= q.bid

    def test_zero_cost_fill_is_a_bug(self, model):
        """Round trip at the same quote must lose money before any price move."""
        q = Quote(ts=TS, bid=99.95, ask=100.05)
        buy = model.simulate_fill(Equity("AAPL"), OrderSide.BUY, 100, q)
        sell = model.simulate_fill(Equity("AAPL"), OrderSide.SELL, 100, q)
        round_trip_loss = (buy.price - sell.price) * 100 + buy.commission + buy.fees \
            + sell.commission + sell.fees
        assert round_trip_loss > 0
        assert buy.slippage_cost > 0
        assert sell.slippage_cost > 0

    def test_sell_charges_regulatory_fees(self, model):
        q = Quote(ts=TS, bid=99.95, ask=100.05)
        f = model.simulate_fill(Equity("AAPL"), OrderSide.SELL, 1000, q)
        assert f.fees > 0

    def test_one_sided_quote_rejected(self, model):
        q = Quote(ts=TS, bid=0.0, ask=100.05)
        with pytest.raises(ValueError):
            model.execution_price(Equity("AAPL"), OrderSide.BUY, q)


class TestOptionFills:
    def setup_method(self):
        self.opt = make_option("AAPL", date(2024, 7, 19), OptionRight.CALL, 200.0)

    def test_option_buys_at_ask_not_mid(self, model):
        q = Quote(ts=TS, bid=2.40, ask=2.60)
        px = model.execution_price(self.opt, OrderSide.BUY, q)
        assert px == pytest.approx(2.60)
        assert px > q.mid

    def test_option_sells_at_bid_not_mid(self, model):
        q = Quote(ts=TS, bid=2.40, ask=2.60)
        px = model.execution_price(self.opt, OrderSide.SELL, q)
        assert px == pytest.approx(2.40)
        assert px < q.mid

    def test_option_commission_per_contract(self, model):
        q = Quote(ts=TS, bid=2.40, ask=2.60)
        f = model.simulate_fill(self.opt, OrderSide.BUY, 5, q)
        assert f.commission == pytest.approx(0.65 * 5)
        # slippage cost = half spread * qty * 100
        assert f.slippage_cost == pytest.approx(0.10 * 5 * 100)

    def test_price_improvement_never_reaches_mid(self):
        cfg = CostConfig(option_price_improvement=0.5)
        model = CostModel(cfg)
        q = Quote(ts=TS, bid=2.40, ask=2.60)
        px = model.execution_price(self.opt, OrderSide.BUY, q)
        assert px >= q.mid  # even max improvement pays at least mid
        with pytest.raises(ValueError):
            CostConfig(option_price_improvement=0.6)
