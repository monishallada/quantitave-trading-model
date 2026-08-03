"""Global kill switch: flatten everything and halt. Engageable from the
dashboard button, code, or by touching data/KILL_SWITCH in a terminal.
"""
from __future__ import annotations

from quantfund.core.orders import Order
from quantfund.core.state import PlatformState
from quantfund.execution.broker import Broker


class KillSwitch:
    def __init__(self, state: PlatformState, broker: Broker):
        self.state = state
        self.broker = broker

    def engage(self, reason: str = "manual") -> list[Order]:
        # Halt FIRST so that even if flattening fails we stop trading.
        self.state.engage_kill_switch(reason)
        try:
            orders = self.broker.flatten_all(reason=reason)
            self.state.log_event(
                "critical", "kill_switch",
                f"flatten_all submitted {len(orders)} closing orders",
            )
            return orders
        except Exception as e:  # noqa: BLE001 — must never raise past here
            self.state.log_event(
                "critical", "kill_switch",
                f"flatten_all FAILED: {e!r} — positions may remain; halt stays engaged",
            )
            return []

    def is_engaged(self) -> bool:
        return self.state.kill_switch_engaged()

    def release(self) -> None:
        self.state.release_kill_switch()
