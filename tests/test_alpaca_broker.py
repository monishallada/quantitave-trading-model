"""AlpacaPaperBroker with a fake TradingClient (no network)."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

import quantfund.execution.alpaca_broker as ab
from quantfund.core.config import Settings
from quantfund.core.instruments import Equity, Option, OptionRight, make_option
from quantfund.core.orders import Order, OrderLeg, OrderSide, OrderStatus, single_leg
from quantfund.execution.broker import BrokerError

UTC = timezone.utc


class FakeTradingClient:
    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.submitted = []
        self.canceled = []
        self.closed_all = False
        self.fail_submit = False

    def submit_order(self, req):
        if self.fail_submit:
            raise ab.APIError("simulated reject")
        self.submitted.append(req)
        return SimpleNamespace(id="ord-123", status=SimpleNamespace(value="accepted"),
                               created_at=datetime(2024, 6, 3, tzinfo=UTC))

    def cancel_orders(self):
        self.canceled.append("ALL")

    def cancel_order_by_id(self, oid):
        self.canceled.append(oid)

    def close_all_positions(self, cancel_orders=False):
        self.closed_all = True

    def get_all_positions(self):
        return [
            SimpleNamespace(symbol="AAPL", qty="100",
                            side=SimpleNamespace(value="long"),
                            avg_entry_price="190.5", market_value="19500",
                            unrealized_pl="450"),
            SimpleNamespace(symbol="AAPL240719C00200000", qty="2",
                            side=SimpleNamespace(value="long"),
                            avg_entry_price="5.10", market_value="1080",
                            unrealized_pl="60"),
        ]

    def get_account(self):
        return SimpleNamespace(equity="100000", cash="60000",
                               buying_power="60000", options_approved_level=2)

    def get_orders(self, req):
        # one filled option buy + one filled equity sell, in Alpaca's shape
        # (no commission/fee fields — that absence is the point)
        return [
            SimpleNamespace(
                id="ord-opt", status=SimpleNamespace(value="filled"),
                symbol="AAPL240719C00200000", legs=None,
                side=SimpleNamespace(value="buy"),
                filled_qty="21", filled_avg_price="0.92",
                filled_at=datetime(2024, 6, 3, 15, 8, tzinfo=UTC),
                updated_at=datetime(2024, 6, 3, 15, 8, tzinfo=UTC)),
            SimpleNamespace(
                id="ord-eq", status=SimpleNamespace(value="filled"),
                symbol="AAPL", legs=None,
                side=SimpleNamespace(value="sell"),
                filled_qty="100", filled_avg_price="190.50",
                filled_at=datetime(2024, 6, 3, 15, 9, tzinfo=UTC),
                updated_at=datetime(2024, 6, 3, 15, 9, tzinfo=UTC)),
        ]


@pytest.fixture
def broker(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "TradingClient", FakeTradingClient)
    s = Settings(alpaca_paper=True, data_dir=tmp_path)
    s.alpaca_api_key, s.alpaca_secret_key = "k", "s"
    return ab.AlpacaPaperBroker(s)


def test_refuses_live_account(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "TradingClient", FakeTradingClient)
    s = Settings(alpaca_paper=True, data_dir=tmp_path)
    s.alpaca_api_key, s.alpaca_secret_key = "k", "s"
    s.alpaca_paper = False
    with pytest.raises(ValueError, match="paper"):
        ab.AlpacaPaperBroker(s)


def test_client_constructed_paper_true(broker):
    assert broker._client.init_kwargs.get("paper") is True


def test_equity_market_order_mapping(broker):
    order = single_leg(Equity("AAPL"), OrderSide.BUY, 10, strategy_id="momentum")
    out = broker.submit_order(order)
    req = broker._client.submitted[0]
    assert req.symbol == "AAPL"
    assert float(req.qty) == 10
    assert req.side == ab.AlpacaSide.BUY
    assert out.broker_order_id == "ord-123"
    assert out.status == OrderStatus.SUBMITTED
    assert broker.strategy_for_order("ord-123")[0] == "momentum"


def test_option_single_leg_uses_occ(broker):
    opt = make_option("AAPL", date(2024, 7, 19), OptionRight.CALL, 200.0)
    broker.submit_order(single_leg(opt, OrderSide.BUY, 1))
    req = broker._client.submitted[0]
    assert req.symbol == "AAPL240719C00200000"


def test_multi_leg_ratio(broker):
    long_c = make_option("AAPL", date(2024, 7, 19), OptionRight.CALL, 200.0)
    short_c = make_option("AAPL", date(2024, 7, 19), OptionRight.CALL, 210.0)
    order = Order(legs=[
        OrderLeg(instrument=long_c, side=OrderSide.BUY, qty=2),
        OrderLeg(instrument=short_c, side=OrderSide.SELL, qty=2),
    ])
    broker.submit_order(order)
    req = broker._client.submitted[0]
    assert req.order_class == ab.AlpacaOrderClass.MLEG
    assert float(req.qty) == 2
    assert [l.ratio_qty for l in req.legs] == [1, 1]
    assert req.legs[1].side == ab.AlpacaSide.SELL


def test_positions_parse_occ(broker):
    positions = broker.get_positions()
    eq = next(p for p in positions if isinstance(p.instrument, Equity))
    opt = next(p for p in positions if isinstance(p.instrument, Option))
    assert eq.qty == 100 and eq.avg_entry_price == 190.5
    assert opt.instrument.strike == 200.0
    assert opt.instrument.right == OptionRight.CALL
    assert opt.instrument.expiration == date(2024, 7, 19)


def test_account_mapping(broker):
    acct = broker.get_account()
    assert acct.equity == 100_000
    assert acct.options_approved_level == 2


def test_flatten_all(broker):
    broker.flatten_all("test")
    assert "ALL" in broker._client.canceled
    assert broker._client.closed_all


def test_api_error_wrapped(broker):
    broker._client.fail_submit = True
    with pytest.raises(BrokerError):
        broker.submit_order(single_leg(Equity("AAPL"), OrderSide.BUY, 1))


def test_alpaca_fills_are_never_recorded_as_free(broker):
    """Alpaca paper reports no commission/fee fields. Recording its fills at
    zero cost makes a high-turnover sleeve look free: 0DTE runs ~480
    contract-legs/day, which at $0.65/contract is ~$328/day of friction that
    would otherwise never appear in the blotter or the equity curve's cost
    column. The platform's "a zero-cost fill is a bug" rule applies here too.
    """
    from quantfund.core.costs import CostConfig

    fills = broker.get_fills()
    assert fills, "fixture must produce at least one fill for this to mean anything"
    for f in fills:
        total = f.commission + f.fees
        assert total > 0, f"{f.instrument.key} recorded as a free fill"
        if isinstance(f.instrument, Option):
            assert f.commission == pytest.approx(
                CostConfig().option_fee_per_contract * f.qty)
        # the fill PRICE stays the broker's — we model fees, never execution
        assert f.slippage_cost == 0.0
