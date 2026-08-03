"""Central configuration. Every risk limit defaults conservative; raise them
explicitly in code or env if you want more rope.

Loads .env via python-dotenv. The platform refuses to construct a live-trading
configuration: ALPACA_PAPER must be true.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "V", "UNH", "XOM", "HD", "PG", "COST", "AVGO", "LLY", "AMD", "NFLX",
]

# Options are only traded on this (liquid) subset by default.
DEFAULT_OPTIONS_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]

SECTOR_MAP: dict[str, str] = {
    "SPY": "INDEX", "QQQ": "INDEX",
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "AVGO": "TECH", "AMD": "TECH",
    "GOOGL": "COMM", "META": "COMM", "NFLX": "COMM",
    "AMZN": "CONS_DISC", "TSLA": "CONS_DISC", "HD": "CONS_DISC",
    "JPM": "FINANCE", "V": "FINANCE",
    "UNH": "HEALTH", "LLY": "HEALTH",
    "XOM": "ENERGY",
    "PG": "CONS_STAPLE", "COST": "CONS_STAPLE",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "UNKNOWN")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


@dataclass
class RiskLimits:
    """All limits are fractions of current portfolio equity unless noted."""

    max_position_pct: float = 0.10          # per-underlying directional exposure
    max_sector_pct: float = 0.25            # per-sector directional exposure
    max_strategy_pct: float = 0.40          # allocator cap per sleeve
    max_gross_leverage: float = 1.0         # gross exposure / equity
    max_net_delta_pct: float = 0.80         # |net $delta| / equity
    max_theta_pct_per_day: float = 0.002    # |theta| per day / equity
    max_vega_pct: float = 0.01              # |vega per vol pt| / equity
    max_order_notional: float = 10_000.0    # $ per single order leg
    max_position_notional: float = 25_000.0  # $ per instrument
    daily_loss_halt_pct: float = 0.02       # halt if day P&L < -2% equity
    max_drawdown_halt_pct: float = 0.10     # halt if drawdown from peak > 10%
    max_trades_per_hour: int = 30           # runaway-loop guard
    # Daily-bar cadence: over a long weekend the freshest bar is legitimately
    # ~64h old, so the dead-feed threshold defaults to 96h. Live quotes from
    # Alpaca refresh this to seconds during market hours; tighten if you run
    # intraday.
    data_staleness_halt_sec: float = 345_600.0
    max_consecutive_errors: int = 5
    reconciliation_qty_tolerance: float = 1e-6
    allow_naked_short_options: bool = False  # undefined-risk positions off by default
    allow_short_equity: bool = False
    min_cash_buffer_pct: float = 0.05       # keep >= 5% equity in cash
    target_portfolio_vol: float = 0.12      # allocator scales deployment to this

    @classmethod
    def explosive(cls) -> "RiskLimits":
        """HIGH-VARIANCE paper-testing profile (QF_RISK_PROFILE=explosive).

        Widens every limit so concentrated, short-dated-options books can run.
        Undefined-risk (naked short options) stays BANNED — the convexity here
        is long options, where max loss is the premium paid. Expect violent
        swings in both directions; this profile optimizes for variance, not
        expected value. Never use these numbers with real money.
        """
        return cls(
            max_position_pct=0.35,
            max_sector_pct=0.60,
            max_strategy_pct=0.60,
            max_gross_leverage=2.0,
            max_net_delta_pct=3.0,          # options delta may exceed equity
            max_theta_pct_per_day=0.02,     # short-dated longs burn theta fast
            max_vega_pct=0.05,
            max_order_notional=40_000.0,
            max_position_notional=60_000.0,
            daily_loss_halt_pct=0.15,       # still flattens at -15%/day
            max_drawdown_halt_pct=0.40,     # hard floor: -40% from peak
            max_trades_per_hour=60,
            allow_naked_short_options=False,  # undefined risk stays banned
            allow_short_equity=True,
            min_cash_buffer_pct=0.02,
            target_portfolio_vol=0.60,      # vol targeting mostly stands aside
        )


@dataclass
class LLMConfig:
    model: str = "claude-opus-5"
    max_tokens: int = 4096
    effort: str = "low"                     # low|medium|high — cost lever
    daily_budget_usd: float = 5.0           # hard cap: sleeve stops calling when hit
    per_run_budget_usd: float = 1.0
    max_symbols_per_run: int = 3
    memory_max_lessons: int = 200


@dataclass
class Settings:
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    anthropic_api_key: str = ""

    starting_cash: float = 100_000.0
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    options_universe: list[str] = field(default_factory=lambda: list(DEFAULT_OPTIONS_UNIVERSE))

    risk: RiskLimits = field(default_factory=RiskLimits)
    risk_profile: str = "conservative"      # "conservative" | "explosive"
    llm: LLMConfig = field(default_factory=LLMConfig)

    dashboard_port: int = 8000
    poll_interval_sec: float = 60.0         # live loop cadence
    rebalance_interval_sec: float = 900.0   # how often strategies re-evaluate
    data_dir: Path = field(default_factory=lambda: Path("data"))
    risk_free_rate: float = 0.04

    def __post_init__(self) -> None:
        if not self.alpaca_paper:
            raise ValueError(
                "This platform is paper-trading only. ALPACA_PAPER must be true."
            )
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings(env_file: str | None = ".env") -> Settings:
    if env_file:
        load_dotenv(env_file, override=False)
    paper_raw = os.getenv("ALPACA_PAPER", "true").strip().lower()
    settings = Settings(
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        alpaca_paper=paper_raw in ("true", "1", "yes"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        starting_cash=_env_float("QF_STARTING_CASH", 100_000.0),
        dashboard_port=_env_int("QF_DASHBOARD_PORT", 8000),
        data_dir=Path(os.getenv("QF_DATA_DIR", "data")),
    )
    profile = os.getenv("QF_RISK_PROFILE", "conservative").strip().lower()
    if profile == "explosive":
        settings.risk_profile = "explosive"
        settings.risk = RiskLimits.explosive()
    settings.llm.model = os.getenv("QF_LLM_MODEL", settings.llm.model)
    settings.llm.daily_budget_usd = _env_float(
        "QF_LLM_DAILY_BUDGET_USD", settings.llm.daily_budget_usd
    )
    settings.llm.per_run_budget_usd = _env_float(
        "QF_LLM_PER_RUN_BUDGET_USD", settings.llm.per_run_budget_usd
    )
    return settings
