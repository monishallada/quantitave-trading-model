"""SimBroker: cost-model fills, rejects, working limit orders, flatten."""
from datetime import timedelta

import pytest

from quantfund.core.costs import CostConfig, CostModel
from quantfund.core.instruments import Equity, OptionRight
from quantfund.core.orders import OrderSide, OrderStatus, OrderType, single_leg
from quantfund.core.snapshot import MarketSnapshot, Quote
from quantfund.execution.sim_broker import SimBroker

from conftest import T0, make_chain, make_snapshot


@pytest.fixture
def broker():
    b = SimBroker(CostModel(CostConfig()), starting_cash=100_000)
    chain = make_chain("AAPL", spot=200.0)
    b.set_snapshot(make_snapshot({"AAPL": 200.0, "MSFT": 400.0},
                                chains={"AAPL": chain}))
    return b


def test_market_buy_pays_friction(broker):
    order = broker.submit_order(single_leg(Equity("AAPL"), OrderSide.BUY, 10,
                                           strategy_id="t"))
    assert order.status == OrderStatus.FILLED
    fill = broker.fills[-1]
    quote = broker._snapshot.get_quote("AAPL")
    assert fill.price >= quote.ask          # never better than the touch
    assert fill.total_friction > 0          # zero-cost fill is a bug
    assert broker.portfolio.cash == pytest.approx(100_000 + fill.cash_flow)


def test_option_fill_from_chain(broker):
    chain = broker._snapshot.get_option_chain("AAPL")
    call = next(q for q in chain.quotes
                if q.instrument.right == OptionRight.CALL
                and q.instrument.strike == 200.0)
    order = broker.submit_order(single_leg(call.instrument, OrderSide.BUY, 2))
    assert order.status == OrderStatus.FILLED
    fill = broker.fills[-1]
    assert fill.price == pytest.approx(call.quote.ask)   # full spread cross
    assert fill.commission == pytest.approx(0.65 * 2)


def test_unknown_contract_rejected(broker):
    from quantfund.core.instruments import make_option
    from datetime import date
    ghost = make_option("AAPL", date(2030, 1, 17), OptionRight.CALL, 500.0)
    order = broker.submit_order(single_leg(ghost, OrderSide.BUY, 1))
    assert order.status == OrderStatus.REJECTED
    assert "no quote" in order.reject_reason


def test_one_sided_quote_rejected():
    b = SimBroker(CostModel(CostConfig()), 100_000)
    snap = MarketSnapshot(
        as_of=T0,
        quotes={"XYZ": Quote(ts=T0, bid=0.0, ask=10.0)},
    )
    b.set_snapshot(snap)
    order = b.submit_order(single_leg(Equity("XYZ"), OrderSide.BUY, 1))
    assert order.status == OrderStatus.REJECTED
    assert "one-sided" in order.reject_reason


def test_limit_order_waits_then_fills(broker):
    # buy limit well below market → stays working
    order = broker.submit_order(single_leg(
        Equity("AAPL"), OrderSide.BUY, 10, order_type=OrderType.LIMIT,
        limit_price=150.0))
    assert order.status == OrderStatus.SUBMITTED
    assert len(broker._working) == 1
    # price falls: new snapshot at 140 → limit crosses, fills
    later = T0 + timedelta(days=1)
    broker.set_snapshot(make_snapshot({"AAPL": 140.0, "MSFT": 400.0}, ts=later))
    assert order.status == OrderStatus.FILLED
    assert not broker._working


def test_cancel_working_order(broker):
    order = broker.submit_order(single_leg(
        Equity("AAPL"), OrderSide.BUY, 10, order_type=OrderType.LIMIT,
        limit_price=100.0))
    assert broker.cancel_order(order.id)
    assert order.status == OrderStatus.CANCELED
    assert not broker._working


def test_flatten_all_closes_everything(broker):
    broker.submit_order(single_leg(Equity("AAPL"), OrderSide.BUY, 10))
    broker.submit_order(single_leg(Equity("MSFT"), OrderSide.BUY, 5))
    assert len(broker.portfolio.open_positions()) == 2
    orders = broker.flatten_all("test")
    assert len(orders) == 2
    assert not broker.portfolio.open_positions()
    # idempotent
    assert broker.flatten_all("again") == []


def test_account_consistent(broker):
    broker.submit_order(single_leg(Equity("AAPL"), OrderSide.BUY, 10))
    acct = broker.get_account()
    assert acct.equity == pytest.approx(broker.portfolio.equity())
    assert acct.cash == pytest.approx(broker.portfolio.cash)
    assert len(broker.get_positions()) == 1


def test_multi_leg_limit_weights_leg_quantities(broker):
    """Net-debit limit on a 1x2 ratio must weight leg prices by quantity."""
    from quantfund.core.orders import Order, OrderLeg
    chain = broker._snapshot.get_option_chain("AAPL")
    c200 = next(q for q in chain.quotes
                if q.instrument.right == OptionRight.CALL
                and q.instrument.strike == 200.0)
    c210 = next(q for q in chain.quotes
                if q.instrument.right == OptionRight.CALL
                and q.instrument.strike == 210.0)
    # costed prices: buy 1 @ ask(200C), sell 2 @ bid(210C)
    net = c200.quote.ask - 2 * c210.quote.bid
    def mk(limit):
        return Order(legs=[
            OrderLeg(instrument=c200.instrument, side=OrderSide.BUY, qty=1),
            OrderLeg(instrument=c210.instrument, side=OrderSide.SELL, qty=2),
        ], order_type=OrderType.LIMIT, limit_price=limit)
    too_tight = broker.submit_order(mk(net - 0.05))
    assert too_tight.status == OrderStatus.SUBMITTED  # working, not filled
    broker.cancel_order(too_tight.id)
    fillable = broker.submit_order(mk(net + 0.05))
    assert fillable.status == OrderStatus.FILLED
