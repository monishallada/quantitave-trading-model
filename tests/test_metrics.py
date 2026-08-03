"""Known-answer metric tests."""
from datetime import datetime, timedelta, timezone

import pytest

from quantfund.backtest.metrics import compute_metrics
from quantfund.core.instruments import Equity
from quantfund.core.orders import Fill, OrderSide

UTC = timezone.utc
D0 = datetime(2024, 1, 1, 21, 0, tzinfo=UTC)


def curve(values, start=D0):
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


def test_constant_growth():
    # 0.1% per day for exactly one year (366 points spanning 365 days)
    vals = [100_000 * 1.001 ** i for i in range(366)]
    m = compute_metrics(curve(vals), [])
    expected_cagr = 1.001 ** 365 * (365.25 / 365) ** 0  # ≈ e^{365 ln 1.001}
    assert m.cagr == pytest.approx(1.001 ** 365 - 1, rel=0.02)
    assert m.max_drawdown == 0.0
    assert m.sharpe > 10  # zero-vol constant growth → huge sharpe


def test_drawdown_measured():
    vals = [100, 110, 121, 96.8, 100, 105]  # 20% dip from 121
    m = compute_metrics(curve([v * 1000 for v in vals]), [])
    assert m.max_drawdown == pytest.approx(0.20, abs=1e-9)


def test_hit_rate_from_round_trips():
    a, b = Equity("A"), Equity("B")
    t = D0
    fills = [
        Fill(order_id="1", instrument=a, side=OrderSide.BUY, qty=10, price=100, ts=t),
        Fill(order_id="2", instrument=a, side=OrderSide.SELL, qty=10, price=110,
             ts=t + timedelta(days=1)),   # win
        Fill(order_id="3", instrument=b, side=OrderSide.BUY, qty=10, price=100,
             ts=t + timedelta(days=2)),
        Fill(order_id="4", instrument=b, side=OrderSide.SELL, qty=10, price=95,
             ts=t + timedelta(days=3)),   # loss
    ]
    m = compute_metrics(curve([100_000, 100_100, 100_050, 100_075]), fills)
    assert m.hit_rate == pytest.approx(0.5)
    assert m.n_trades == 4


def test_turnover_annualized():
    # $200k traded over one year on $100k average equity → turnover 2.0
    a = Equity("A")
    fills = [
        Fill(order_id="1", instrument=a, side=OrderSide.BUY, qty=1000, price=100, ts=D0),
        Fill(order_id="2", instrument=a, side=OrderSide.SELL, qty=1000, price=100,
             ts=D0 + timedelta(days=200)),
    ]
    vals = [100_000] * 366
    m = compute_metrics(curve(vals), fills)
    assert m.turnover == pytest.approx(2.0, rel=0.02)


def test_empty_curve_safe():
    m = compute_metrics([], [])
    assert m.cagr == 0 and m.n_trades == 0


def test_same_day_curve_does_not_overflow():
    from datetime import timedelta as _td
    pts = [(D0, 100_000.0), (D0 + _td(hours=1), 100_500.0)]
    m = compute_metrics(pts, [])
    assert m.total_return == pytest.approx(0.005)
    assert m.cagr == m.cagr  # not NaN, no exception
