"""Thread-safe platform state shared by the live runner and the dashboard.

The runner (background thread) writes; FastAPI request handlers read. Trades,
equity curve, rationales, breaker trips, and events are also persisted to
SQLite so history survives restarts and the dashboard can show it.

The kill switch is dual-persisted: a DB flag AND a sentinel file
(data/KILL_SWITCH) so a human can trip it from a terminal with `touch`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL,
    fees REAL NOT NULL,
    slippage REAL NOT NULL,
    rationale_id TEXT,
    order_id TEXT
);
CREATE TABLE IF NOT EXISTS rationales (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    buying_power REAL NOT NULL,
    gross_exposure REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    source TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TradeRecord:
    ts: datetime
    strategy_id: str
    instrument_key: str
    asset_class: str
    side: str
    qty: float
    price: float
    commission: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    rationale_id: str = ""
    order_id: str = ""


@dataclass
class BreakerStatus:
    name: str
    tripped: bool = False
    reason: str = ""
    tripped_at: Optional[datetime] = None


@dataclass
class StrategyStatus:
    strategy_id: str
    name: str = ""
    allocated_capital: float = 0.0
    weight: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    n_positions: int = 0
    last_signal_ts: Optional[str] = None
    detail: dict = field(default_factory=dict)


class PlatformState:
    """All mutation goes through methods that hold ``self._lock``."""

    def __init__(self, db_path: Path | str = "data/state.db"):
        self._lock = threading.RLock()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.kill_file = self.db_path.parent / "KILL_SWITCH"
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

        # in-memory live view (dashboard reads these)
        self.portfolio_summary: dict[str, Any] = {}
        self.positions: list[dict[str, Any]] = []
        self.strategies: dict[str, StrategyStatus] = {}
        self.breakers: dict[str, BreakerStatus] = {}
        self.halted: bool = False
        self.halt_reason: str = ""
        self.mode: str = "starting"  # starting | live | halted | killed | backtest
        self.last_loop_ts: Optional[datetime] = None
        self.llm_spend_today: float = 0.0

    # ── kill switch ──────────────────────────────────────────────────────

    def kill_switch_engaged(self) -> bool:
        with self._lock:
            if self.kill_file.exists():
                return True
            row = self._db.execute("SELECT value FROM kv WHERE key='kill_switch'").fetchone()
            return bool(row and row[0] == "1")

    def engage_kill_switch(self, reason: str = "manual") -> None:
        with self._lock:
            self.kill_file.write_text(f"{utcnow().isoformat()} {reason}\n")
            self._db.execute(
                "INSERT INTO kv(key, value) VALUES('kill_switch','1') "
                "ON CONFLICT(key) DO UPDATE SET value='1'"
            )
            self._db.commit()
            self.mode = "killed"
            self.halted = True
            self.halt_reason = f"KILL SWITCH: {reason}"
        self.log_event("critical", "kill_switch", f"engaged: {reason}")

    def release_kill_switch(self) -> None:
        with self._lock:
            if self.kill_file.exists():
                self.kill_file.unlink()
            self._db.execute(
                "INSERT INTO kv(key, value) VALUES('kill_switch','0') "
                "ON CONFLICT(key) DO UPDATE SET value='0'"
            )
            self._db.commit()
            self.halted = False
            self.halt_reason = ""
            self.mode = "live"
        self.log_event("warning", "kill_switch", "released")

    # ── recording ────────────────────────────────────────────────────────

    def record_trade(self, t: TradeRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO trades (ts, strategy_id, instrument_key, asset_class, side,"
                " qty, price, commission, fees, slippage, rationale_id, order_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.ts.isoformat(), t.strategy_id, t.instrument_key, t.asset_class,
                 t.side, t.qty, t.price, t.commission, t.fees, t.slippage,
                 t.rationale_id, t.order_id),
            )
            self._db.commit()

    def record_rationale(self, rationale_id: str, strategy_id: str, payload: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO rationales (id, ts, strategy_id, payload)"
                " VALUES (?,?,?,?)",
                (rationale_id, utcnow().isoformat(), strategy_id,
                 json.dumps(payload, default=str)),
            )
            self._db.commit()

    def record_equity(self, equity: float, cash: float, buying_power: float,
                      gross_exposure: float, ts: Optional[datetime] = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO equity_curve (ts, equity, cash, buying_power,"
                " gross_exposure) VALUES (?,?,?,?,?)",
                ((ts or utcnow()).isoformat(), equity, cash, buying_power, gross_exposure),
            )
            self._db.commit()

    def log_event(self, level: str, source: str, message: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, level, source, message) VALUES (?,?,?,?)",
                (utcnow().isoformat(), level, source, message),
            )
            self._db.commit()

    # ── live view updates ────────────────────────────────────────────────

    def update_portfolio(self, summary: dict, positions: list[dict]) -> None:
        with self._lock:
            self.portfolio_summary = summary
            self.positions = positions
            self.last_loop_ts = utcnow()

    def update_strategy(self, status: StrategyStatus) -> None:
        with self._lock:
            self.strategies[status.strategy_id] = status

    def set_breaker(self, name: str, tripped: bool, reason: str = "") -> None:
        with self._lock:
            prev = self.breakers.get(name)
            newly_tripped = tripped and (prev is None or not prev.tripped)
            self.breakers[name] = BreakerStatus(
                name=name, tripped=tripped, reason=reason,
                tripped_at=utcnow() if tripped else (prev.tripped_at if prev else None),
            )
        if newly_tripped:
            self.log_event("critical", "circuit_breaker", f"{name} TRIPPED: {reason}")

    def set_halted(self, halted: bool, reason: str = "") -> None:
        with self._lock:
            self.halted = halted
            self.halt_reason = reason
            if halted and self.mode != "killed":
                self.mode = "halted"
            elif not halted and self.mode in ("halted", "starting"):
                self.mode = "live"

    def add_llm_spend(self, usd: float) -> float:
        with self._lock:
            self.llm_spend_today += usd
            return self.llm_spend_today

    def reset_llm_spend(self) -> None:
        with self._lock:
            self.llm_spend_today = 0.0

    # ── queries (dashboard) ──────────────────────────────────────────────

    def get_trades(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, strategy_id, instrument_key, asset_class, side, qty, price,"
                " commission, fees, slippage, rationale_id, order_id, id"
                " FROM trades ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        cols = ["ts", "strategy_id", "instrument_key", "asset_class", "side", "qty",
                "price", "commission", "fees", "slippage", "rationale_id", "order_id", "id"]
        return [dict(zip(cols, r)) for r in rows]

    def get_rationale(self, rationale_id: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT id, ts, strategy_id, payload FROM rationales WHERE id=?",
                (rationale_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": row[0], "ts": row[1], "strategy_id": row[2],
                "payload": json.loads(row[3])}

    def total_costs_recorded(self) -> tuple[float, float]:
        """(commissions+fees, total friction) across every trade ever recorded.

        The live Portfolio accumulates costs only from fills ingested in the
        CURRENT process, and a restart marks prior fills as already-seen — so
        the dashboard's "costs paid" tile reset to $0.00 on every restart while
        real money had been spent ($62.01 on 2026-08-07 showed as zero). The
        trades table is the durable record; seed from it.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(commission + fees), 0.0),"
                " COALESCE(SUM(commission + fees + slippage), 0.0) FROM trades"
            ).fetchone()
        return float(row[0]), float(row[1])

    def get_equity_curve(self, limit: int = 5000) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, equity, cash, buying_power, gross_exposure FROM equity_curve"
                " ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        cols = ["ts", "equity", "cash", "buying_power", "gross_exposure"]
        return [dict(zip(cols, r)) for r in reversed(rows)]

    def get_events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, level, source, message FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(zip(["ts", "level", "source", "message"], r)) for r in rows]

    def loop_age_seconds(self) -> Optional[float]:
        with self._lock:
            if self.last_loop_ts is None:
                return None
            return (utcnow() - self.last_loop_ts).total_seconds()

    def snapshot_view(self, stall_after_sec: float = 600.0) -> dict:
        """One JSON-safe dict with everything the dashboard needs at a glance.

        Reports mode="STALLED" when the trading loop has not completed an
        iteration recently — a dashboard that says "live" while the loop is
        dead is worse than no dashboard (2026-08-06)."""
        age = self.loop_age_seconds()
        stalled = age is not None and age > stall_after_sec
        with self._lock:
            return {
                "mode": "STALLED" if stalled else self.mode,
                "stalled": stalled,
                "loop_age_sec": age,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "kill_switch": self.kill_switch_engaged(),
                "last_loop_ts": self.last_loop_ts.isoformat() if self.last_loop_ts else None,
                "portfolio": dict(self.portfolio_summary),
                "positions": list(self.positions),
                "strategies": {k: vars(v) for k, v in self.strategies.items()},
                "breakers": {
                    k: {"name": b.name, "tripped": b.tripped, "reason": b.reason,
                        "tripped_at": b.tripped_at.isoformat() if b.tripped_at else None}
                    for k, b in self.breakers.items()
                },
                "llm_spend_today": self.llm_spend_today,
            }

    def close(self) -> None:
        with self._lock:
            self._db.close()
