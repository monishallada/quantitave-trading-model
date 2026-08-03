"""Hard spend cap for the LLM sleeve. Every API response's usage is priced and
accumulated in PlatformState; when the daily or per-run budget is hit, the
sleeve stops calling — trading continues on the classical sleeves."""
from __future__ import annotations

import logging

from quantfund.core.config import LLMConfig
from quantfund.core.state import PlatformState

log = logging.getLogger(__name__)

# $/MTok (input, output). Cache reads bill at 0.1x input; cache writes 1.25x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}
_FALLBACK_PRICE = (10.0, 50.0)  # unknown model → assume expensive


class BudgetExhausted(Exception):
    pass


class CostGuard:
    def __init__(self, llm_cfg: LLMConfig, state: PlatformState):
        self.cfg = llm_cfg
        self.state = state
        self.run_spend = 0.0

    def start_run(self) -> None:
        self.run_spend = 0.0

    @staticmethod
    def cost_of(model: str, usage) -> float:
        inp, outp = PRICES.get(model, _FALLBACK_PRICE)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return (
            input_tokens * inp / 1e6
            + output_tokens * outp / 1e6
            + cache_read * inp * 0.1 / 1e6
            + cache_write * inp * 1.25 / 1e6
        )

    def check(self) -> None:
        """Raises BudgetExhausted when either cap is already reached."""
        today = self.state.llm_spend_today
        if today >= self.cfg.daily_budget_usd:
            raise BudgetExhausted(
                f"daily LLM budget ${self.cfg.daily_budget_usd:.2f} reached "
                f"(spent ${today:.2f})")
        if self.run_spend >= self.cfg.per_run_budget_usd:
            raise BudgetExhausted(
                f"per-run LLM budget ${self.cfg.per_run_budget_usd:.2f} reached "
                f"(spent ${self.run_spend:.2f} this run)")

    def record(self, model: str, usage) -> float:
        cost = self.cost_of(model, usage)
        self.run_spend += cost
        total = self.state.add_llm_spend(cost)
        log.debug("LLM call cost $%.4f (today: $%.2f)", cost, total)
        return total
