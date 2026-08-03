"""Walk-forward windows respect purge/embargo; overfitting flag logic."""
from datetime import datetime, timedelta, timezone

from quantfund.backtest.walkforward import (
    make_windows, overfitting_check,
)

UTC = timezone.utc
START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)


def test_windows_respect_purge():
    windows = make_windows(START, END, train_days=252, test_days=63, purge_days=5)
    assert windows
    for w in windows:
        assert w.test_start >= w.train_end + timedelta(days=5)
        assert w.train_end > w.train_start
        assert w.test_end > w.test_start
        assert w.test_end <= END


def test_test_periods_do_not_overlap():
    windows = make_windows(START, END, train_days=252, test_days=63, purge_days=5)
    for a, b in zip(windows, windows[1:]):
        assert b.test_start >= a.test_start + timedelta(days=63)


def test_short_range_gives_no_windows():
    assert make_windows(START, START + timedelta(days=100)) == []


def test_overfitting_flag_logic():
    assert overfitting_check(is_sharpe=2.0, oos_sharpe=-0.5)      # great IS, neg OOS
    assert overfitting_check(is_sharpe=2.0, oos_sharpe=0.6)       # ratio > 2
    assert overfitting_check(is_sharpe=3.0, oos_sharpe=1.2)       # high IS, < half
    assert not overfitting_check(is_sharpe=1.2, oos_sharpe=0.9)   # healthy
    assert not overfitting_check(is_sharpe=0.3, oos_sharpe=0.1)   # both weak — not "overfit"
    assert not overfitting_check(is_sharpe=-0.5, oos_sharpe=-0.7)


def test_test_windows_share_no_days():
    windows = make_windows(START, END, train_days=252, test_days=63, purge_days=5)
    assert len(windows) >= 2
    for a, b in zip(windows, windows[1:]):
        # inclusive date filters downstream: consecutive test windows must not
        # contain any common day
        assert b.test_start > a.test_end
