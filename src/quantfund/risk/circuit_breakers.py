"""Circuit breakers: automatic halts that protect the experiment (and the API
bill), not real money. Seven breakers, evaluated every loop; each writes its
status into PlatformState so the dashboard shows a live light per breaker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from quantfund.core.config import RiskLimits
from quantfund.core.portfolio import Portfolio
from quantfund.core.snapshot import MarketSnapshot
from quantfund.core.state import PlatformState, utcnow
from quantfund.execution.broker import BrokerPosition

BREAKER_NAMES = [
    "daily_loss", "drawdown", "trade_frequency", "data_staleness",
    "error_burst", "reconciliation", "order_bounds",
]


@dataclass(frozen=True)
class BreakerTrip:
    name: str
    reason: str


class CircuitBreakerBoard:
    def __init__(self, limits: RiskLimits, state: PlatformState):
        self.limits = limits
        self.state = state
        self._trade_times: list[datetime] = []
        self._bounds_violations: list[datetime] = []
        self._consecutive_errors = 0
        self._consecutive_transient = 0
        self._last_error = ""
        for name in BREAKER_NAMES:
            state.set_breaker(name, False)

    # ── event recorders (called by the runner) ───────────────────────────

    def record_trade(self, ts: Optional[datetime] = None) -> None:
        self._trade_times.append(ts or utcnow())

    def record_error(self, source: str, msg: str, transient: bool = False) -> None:
        """``transient`` marks connectivity blips (DNS/timeout/reset). Those
        must not halt trading as fast as logic errors — a dropped wifi packet
        is not a broken strategy — so they need 3x as many consecutive
        failures to trip the breaker."""
        self._last_error = f"{source}: {msg}"
        if transient:
            self._consecutive_transient += 1
        else:
            self._consecutive_errors += 1
        self.state.log_event("error", source, msg)

    def record_success(self) -> None:
        self._consecutive_errors = 0
        self._consecutive_transient = 0

    def record_order_bounds_violation(self, reason: str) -> None:
        self._bounds_violations.append(utcnow())
        self.state.log_event("warning", "risk", f"order bounds violation: {reason}")

    # ── evaluation ───────────────────────────────────────────────────────

    def check_all(self, portfolio: Portfolio, snapshot: Optional[MarketSnapshot],
                  broker_positions: Optional[list[BrokerPosition]],
                  local_qty: Optional[dict[str, float]] = None) -> list[BreakerTrip]:
        """``local_qty`` is an independent, fill-derived book {instrument_key:
        qty}. When provided, reconciliation compares it against
        ``broker_positions``; comparing a mirror that was itself rebuilt from
        broker positions would be vacuous."""
        lim = self.limits
        now = utcnow()
        hour_ago = now - timedelta(hours=1)
        trips: list[BreakerTrip] = []

        def judge(name: str, tripped: bool, reason: str) -> None:
            self.state.set_breaker(name, tripped, reason if tripped else "")
            if tripped:
                trips.append(BreakerTrip(name, reason))

        # daily loss
        anchor = portfolio.day_anchor_equity
        if anchor and anchor > 0:
            day_pnl = portfolio.day_pnl()
            limit_val = -lim.daily_loss_halt_pct * anchor
            judge("daily_loss", day_pnl <= limit_val,
                  f"day P&L ${day_pnl:,.0f} breached {lim.daily_loss_halt_pct:.1%} "
                  f"daily loss limit (${limit_val:,.0f})")
        else:
            judge("daily_loss", False, "")

        # drawdown
        dd = portfolio.drawdown()
        judge("drawdown", dd >= lim.max_drawdown_halt_pct,
              f"drawdown {dd:.1%} >= halt limit {lim.max_drawdown_halt_pct:.0%}")

        # trade frequency (runaway loop guard)
        self._trade_times = [t for t in self._trade_times if t >= hour_ago]
        judge("trade_frequency", len(self._trade_times) > lim.max_trades_per_hour,
              f"{len(self._trade_times)} trades in the last hour > "
              f"{lim.max_trades_per_hour}")

        # data staleness
        if snapshot is not None:
            worst: Optional[float] = None
            for sym in snapshot.bars.keys() | snapshot.quotes.keys():
                s = snapshot.staleness_seconds(sym)
                if s is not None:
                    worst = s if worst is None else max(worst, s)
            # weekends/overnight: compare against as_of age too — the snapshot
            # itself being old means the loop is running on stale data
            judge("data_staleness",
                  worst is not None and worst > lim.data_staleness_halt_sec,
                  f"freshest data is {worst:,.0f}s old > "
                  f"{lim.data_staleness_halt_sec:,.0f}s limit" if worst else "")
        else:
            judge("data_staleness", False, "")

        # error burst (transient network failures need 3x the count to trip)
        burst = (self._consecutive_errors >= lim.max_consecutive_errors
                 or self._consecutive_transient >= 3 * lim.max_consecutive_errors)
        judge("error_burst", burst,
              f"{self._consecutive_errors} errors / {self._consecutive_transient} "
              f"transient (last: {self._last_error})")

        # reconciliation: local book vs broker truth
        if broker_positions is not None:
            broker_qty = {bp.instrument.key: bp.qty for bp in broker_positions}
            if local_qty is None:
                local_qty = {k: p.qty for k, p in portfolio.positions.items()
                             if p.is_open}
            mismatches = []
            for key in set(broker_qty) | set(local_qty):
                b, l = broker_qty.get(key, 0.0), local_qty.get(key, 0.0)
                if abs(b - l) > lim.reconciliation_qty_tolerance:
                    mismatches.append(f"{key}: broker={b:g} local={l:g}")
            judge("reconciliation", bool(mismatches),
                  "position mismatch — " + "; ".join(mismatches[:5]))
        else:
            judge("reconciliation", False, "")

        # order bounds (3 size-rejections in an hour = something is sizing wrong)
        self._bounds_violations = [t for t in self._bounds_violations if t >= hour_ago]
        judge("order_bounds", len(self._bounds_violations) >= 3,
              f"{len(self._bounds_violations)} order-size rejections in the last hour")

        return trips
