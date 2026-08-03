"""Walk-forward validation with purge/embargo, plus a loud overfitting flag.
Never judge a strategy on in-sample numbers: the report always separates IS
from OOS, and flags when IS >> OOS."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from quantfund.backtest.metrics import PerfMetrics
from quantfund.strategies.base import Strategy


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def make_windows(start: datetime, end: datetime, train_days: int = 252,
                 test_days: int = 63, purge_days: int = 5) -> list[WalkForwardWindow]:
    """Rolling windows. The test period starts purge_days AFTER train_end
    (embargo) so signals fitted on train data cannot bleed into test fills."""
    windows: list[WalkForwardWindow] = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=purge_days)
        # end one day short so consecutive test windows tile without sharing a
        # boundary day (engine date filters are inclusive on both ends)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > end:
            break
        windows.append(WalkForwardWindow(train_start, train_end, test_start, test_end))
        cursor = cursor + timedelta(days=test_days)
    return windows


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    in_sample: dict[str, PerfMetrics]
    out_of_sample: dict[str, PerfMetrics]
    overfitting_flag: bool
    per_window: list[dict] = field(default_factory=list)


def overfitting_check(is_sharpe: float, oos_sharpe: float) -> bool:
    """IS >> OOS heuristic: flags when in-sample looks great and out-of-sample
    doesn't back it up."""
    if is_sharpe > 1.0 and oos_sharpe < 0.0:
        return True
    if is_sharpe > 0.5 and oos_sharpe > 0 and is_sharpe / max(oos_sharpe, 1e-9) > 2.0:
        return True
    if is_sharpe > 2.0 and oos_sharpe < 0.5 * is_sharpe:
        return True
    return False


def _avg_metrics(metric_sets: list[dict[str, PerfMetrics]]) -> dict[str, PerfMetrics]:
    keys: set[str] = set()
    for m in metric_sets:
        keys |= set(m)
    out: dict[str, PerfMetrics] = {}
    for k in keys:
        ms = [m[k] for m in metric_sets if k in m]
        if not ms:
            continue
        n = len(ms)
        out[k] = PerfMetrics(
            cagr=sum(x.cagr for x in ms) / n,
            sharpe=sum(x.sharpe for x in ms) / n,
            sortino=sum(x.sortino for x in ms) / n,
            max_drawdown=sum(x.max_drawdown for x in ms) / n,
            hit_rate=sum(x.hit_rate for x in ms) / n,
            turnover=sum(x.turnover for x in ms) / n,
            total_return=sum(x.total_return for x in ms) / n,
            n_trades=sum(x.n_trades for x in ms),
        )
    return out


def run_walkforward(engine, strategies_factory: Callable[[], list[Strategy]],
                    start: datetime, end: datetime, train_days: int = 252,
                    test_days: int = 63, purge_days: int = 5,
                    **run_kwargs) -> WalkForwardResult:
    windows = make_windows(start, end, train_days, test_days, purge_days)
    is_all: list[dict[str, PerfMetrics]] = []
    oos_all: list[dict[str, PerfMetrics]] = []
    per_window: list[dict] = []
    for w in windows:
        # fresh strategy instances per segment: no state bleed across windows
        is_res = engine.run(strategies_factory(), w.train_start, w.train_end, **run_kwargs)
        oos_res = engine.run(strategies_factory(), w.test_start, w.test_end, **run_kwargs)
        is_all.append(is_res.metrics)
        oos_all.append(oos_res.metrics)
        per_window.append({
            "train": (w.train_start.date().isoformat(), w.train_end.date().isoformat()),
            "test": (w.test_start.date().isoformat(), w.test_end.date().isoformat()),
            "is_portfolio_sharpe": is_res.metrics.get("portfolio",
                                                      PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0)).sharpe,
            "oos_portfolio_sharpe": oos_res.metrics.get("portfolio",
                                                        PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0)).sharpe,
        })
    in_sample = _avg_metrics(is_all)
    out_of_sample = _avg_metrics(oos_all)
    flag = overfitting_check(
        in_sample.get("portfolio", PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0)).sharpe,
        out_of_sample.get("portfolio", PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0)).sharpe,
    )
    return WalkForwardResult(windows=windows, in_sample=in_sample,
                             out_of_sample=out_of_sample, overfitting_flag=flag,
                             per_window=per_window)
