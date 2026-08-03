"""Performance metrics, always computed net of costs (the equity curves they
consume already include every commission/fee/slippage dollar)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from quantfund.core.orders import Fill
from quantfund.core.portfolio import Position


@dataclass
class PerfMetrics:
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    total_return: float
    n_trades: int

    def as_dict(self) -> dict:
        return {
            "cagr": self.cagr, "sharpe": self.sharpe, "sortino": self.sortino,
            "max_drawdown": self.max_drawdown, "hit_rate": self.hit_rate,
            "turnover": self.turnover, "total_return": self.total_return,
            "n_trades": self.n_trades,
        }


def _returns(curve: list[tuple[datetime, float]]) -> list[float]:
    out = []
    for i in range(1, len(curve)):
        prev = curve[i - 1][1]
        if prev > 0:
            out.append(curve[i][1] / prev - 1.0)
    return out


def _hit_rate(trades: list[Fill]) -> float:
    """Replays fills per instrument through core Position accounting and counts
    the sign of realized P&L on each closing fill."""
    positions: dict[str, Position] = {}
    wins = 0
    closes = 0
    for f in sorted(trades, key=lambda f: f.ts):
        pos = positions.setdefault(f.instrument.key, Position(instrument=f.instrument))
        was_open = pos.is_open
        prev_qty = pos.qty
        realized = pos.apply_fill(f)
        closed_something = was_open and (
            (prev_qty > 0 and f.side.value == "sell") or
            (prev_qty < 0 and f.side.value == "buy")
        )
        if closed_something:
            closes += 1
            if realized > 0:
                wins += 1
    return wins / closes if closes else 0.0


def compute_metrics(equity_curve: list[tuple[datetime, float]],
                    trades: list[Fill], periods_per_year: int = 252) -> PerfMetrics:
    if len(equity_curve) < 2:
        return PerfMetrics(0, 0, 0, 0, 0, 0, 0, len(trades))
    start_eq, end_eq = equity_curve[0][1], equity_curve[-1][1]
    total_return = end_eq / start_eq - 1.0 if start_eq > 0 else 0.0
    # floor the span at one day: timedelta.days truncates to 0 for same-day
    # curves, and annualizing over ~1e-9 years overflows (ratio ** 1e9)
    span_sec = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds()
    years = max(span_sec / (365.25 * 86400.0), 1.0 / 365.25)
    if start_eq > 0 and end_eq > 0:
        try:
            cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
        except OverflowError:
            cagr = float("inf") if end_eq > start_eq else -1.0
    else:
        cagr = -1.0

    rets = _returns(equity_curve)
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        if std > 1e-12:
            sharpe = (mean / std) * math.sqrt(periods_per_year)
        else:
            # zero-vol stream: sign of the mean decides (constant growth is
            # "infinite" risk-adjusted return; cap for sanity)
            sharpe = math.copysign(100.0, mean) if abs(mean) > 1e-15 else 0.0
        downside = [r for r in rets if r < 0]
        if downside:
            dvar = sum(r * r for r in downside) / len(downside)
            dstd = math.sqrt(dvar)
            sortino = (mean / dstd) * math.sqrt(periods_per_year) if dstd > 1e-12 else 0.0
        else:
            sortino = 0.0 if mean <= 0 else float("inf")
    else:
        sharpe = sortino = 0.0

    peak = -float("inf")
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - eq / peak)

    avg_eq = sum(e for _, e in equity_curve) / len(equity_curve)
    gross_traded = sum(f.notional for f in trades)
    turnover = gross_traded / avg_eq / years if avg_eq > 0 else 0.0

    return PerfMetrics(
        cagr=cagr, sharpe=sharpe, sortino=sortino, max_drawdown=max_dd,
        hit_rate=_hit_rate(trades), turnover=turnover,
        total_return=total_return, n_trades=len(trades),
    )
