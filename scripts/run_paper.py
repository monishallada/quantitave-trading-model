#!/usr/bin/env python3
"""Start the paper-trading loop + live dashboard.

With Alpaca keys in .env: trades your Alpaca PAPER account with live data.
Without keys: OFFLINE DEMO MODE — synthetic data + simulated broker, so you
can see the whole platform work end-to-end before wiring credentials.

Usage:
  python scripts/run_paper.py            # loop + dashboard at :8000
  python scripts/run_paper.py --once     # single iteration, print JSON, exit
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading

from quantfund.allocation.allocator import CapitalAllocator
from quantfund.core.config import load_settings
from quantfund.core.costs import CostConfig, CostModel
from quantfund.core.state import PlatformState
from quantfund.live.runner import LiveRunner
from quantfund.risk.circuit_breakers import CircuitBreakerBoard
from quantfund.risk.kill_switch import KillSwitch
from quantfund.risk.limits import RiskManager
from quantfund.strategies.convexity import ConvexMomentumStrategy
from quantfund.strategies.mean_reversion import MeanReversionStrategy
from quantfund.strategies.momentum import MomentumStrategy
from quantfund.strategies.put_income import PutIncomeStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("run_paper")


def build_platform():
    settings = load_settings()
    state = PlatformState(settings.data_dir / "state.db")

    if settings.has_alpaca:
        from quantfund.data.alpaca_data import AlpacaDataProvider
        from quantfund.execution.alpaca_broker import AlpacaPaperBroker
        provider = AlpacaDataProvider(settings)
        broker = AlpacaPaperBroker(settings)
        acct = broker.get_account()
        log.info(f"Connected to Alpaca PAPER account: equity=${acct.equity:,.2f} "
                 f"cash=${acct.cash:,.2f} buying_power=${acct.buying_power:,.2f} "
                 f"options_level={acct.options_approved_level}")
        if acct.options_approved_level < 2:
            log.warning("Options approval level %d < 2 — enable options for the "
                        "paper account at app.alpaca.markets or option orders "
                        "will be rejected", acct.options_approved_level)
    else:
        from quantfund.data.synthetic import SyntheticProvider
        from quantfund.execution.sim_broker import SimBroker
        print("=" * 72, file=sys.stderr)
        print("  OFFLINE DEMO MODE — no Alpaca keys found in .env", file=sys.stderr)
        print("  Using synthetic market data + a simulated broker.", file=sys.stderr)
        print("  Add ALPACA_API_KEY / ALPACA_SECRET_KEY to trade on Alpaca paper.",
              file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        provider = SyntheticProvider(seed=42)
        broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)

        # the sim broker needs a snapshot before the first orders
        from quantfund.core.state import utcnow
        broker.set_snapshot(provider.build_snapshot(
            settings.universe, as_of=utcnow(),
            options_underlyings=settings.options_universe))

    if settings.risk_profile == "explosive":
        print("=" * 72, file=sys.stderr)
        print("  ⚡ EXPLOSIVE RISK PROFILE — high-variance paper testing mode.", file=sys.stderr)
        print("  Concentrated short-dated options, 2-way book, wide limits.", file=sys.stderr)
        print("  Large losses are as likely as large gains. Paper only.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        state.log_event("warning", "config",
                        "EXPLOSIVE risk profile active — high-variance mode")
        # OPTIONS-FIRST ordering: the options sleeves claim capital, cash, and
        # exposure budget BEFORE any equity sleeve can consume it (2026-08-03
        # incident: equity momentum ate the whole gross-exposure and cash
        # budget, so every option order was rejected all day).
        # mean_reversion is intentionally absent here — 2023-25 backtest:
        # -0.9% full period / +0.1% OOS, i.e. it earned nothing while
        # consuming equity budget the options sleeves need.
        strategies = [
            PutIncomeStrategy(min_dte=7, max_dte=21, exit_dte=3),
            ConvexMomentumStrategy(calls_enabled=False),  # crash convexity only
            # top_n=3 + 0.65 delta: concentrate the sleeve's capital so it can
            # always afford whole contracts (5 names x deep-ITM premium
            # exceeded the sleeve's budget and produced near-zero trades)
            # Heavy deep-ITM deployment: 0.65-delta calls behave like leveraged
            # stock with a capped downside, and they are the ONLY long-option
            # expression the 2023-25 evidence supports. Portfolio premium at
            # risk is bounded by RiskLimits.max_long_option_premium_pct.
            MomentumStrategy(top_n=5, deploy_fraction=1.0, max_name_weight=0.45,
                             express_via="options", option_delta=0.80,
                             option_leverage=5.0,
                             option_max_premium_weight=0.70,
                             option_min_dte=7, option_max_dte=21,
                             option_roll_dte=4),
            # Equity momentum LAST: it is the single best OOS-validated engine
            # in the platform (2023-25 walk-forward Sharpe 1.47) and lifts
            # portfolio OOS from ~-0.3 to ~+1.1. Deliberately constrained so it
            # can never repeat the 2026-08-03 budget grab: it is last in the
            # options-first queue, holds fewer names, and deploys ~half of its
            # sleeve — the options sleeves always eat first.
            # NON-optionable names only: keeps the equity core from taking
            # duplicate exposure in the same names as the options sleeves and
            # from eating their per-name / per-sector budget.
            MomentumStrategy(top_n=3, deploy_fraction=0.55,
                             max_name_weight=0.20, express_via="equity",
                             exclude=settings.options_universe,
                             strategy_id="momentum_equity",
                             name="Momentum (equity core)"),
        ]
    else:
        strategies = [MomentumStrategy(), MeanReversionStrategy(),
                      PutIncomeStrategy()]
    if settings.has_anthropic:
        from quantfund.strategies.llm_agents.pipeline import LLMAgentStrategy
        strategies.append(LLMAgentStrategy(settings, state))
        log.info("LLM sleeve ACTIVE: model=%s daily_budget=$%.2f per_run=$%.2f",
                 settings.llm.model, settings.llm.daily_budget_usd,
                 settings.llm.per_run_budget_usd)
    else:
        log.info("LLM sleeve disabled (no ANTHROPIC_API_KEY) — running "
                 "momentum + mean_reversion only")
    log.info("Active sleeves: %s", ", ".join(s.strategy_id for s in strategies))

    kill = KillSwitch(state, broker)
    # explosive: shorter lookback adapts allocations 2x faster; lower floor
    # starves losing sleeves instead of force-feeding them 5%
    # min_weight 0.08: an options sleeve starved below one contract's premium
    # can never trade again (2023-25 validation: a 2% floor left momentum with
    # $2k against ~$2.5k deep-ITM calls → 6 trades in 3 years, dead sleeve).
    # Explosive: capital is deliberately concentrated in the ITM-options
    # engine so ~50% of the portfolio can sit in option premium. Equal
    # weighting a 4-sleeve book would cap that engine at 25% of equity.
    # Performance scoring still moves capital away from any sleeve that
    # stops earning.
    allocator = (CapitalAllocator(
                     settings.risk, lookback=30, min_weight=0.08,
                     base_weights={"momentum": 0.50,        # ITM calls
                                   "put_income": 0.22,      # collateral-based
                                   "momentum_equity": 0.18,
                                   "convexity": 0.10})      # crash hedge
                 if settings.risk_profile == "explosive"
                 else CapitalAllocator(settings.risk))
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=strategies, state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=allocator,
        kill_switch=kill,
    )
    return settings, state, runner, kill


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="run one loop iteration, print JSON, exit")
    args = parser.parse_args()

    settings, state, runner, kill = build_platform()

    if args.once:
        result = runner.run_once()
        print(json.dumps(result, default=str))
        state.close()
        return 0

    stop = threading.Event()

    def handle_sigint(*_):
        log.info("shutting down…")
        stop.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    t = threading.Thread(target=runner.run_forever, args=(stop,), daemon=True,
                         name="trading-loop")
    t.start()

    from quantfund.dashboard.app import create_app
    import uvicorn
    app = create_app(state, kill=kill.engage, release=kill.release)
    log.info("Dashboard: http://127.0.0.1:%d", settings.dashboard_port)
    try:
        uvicorn.run(app, host="127.0.0.1", port=settings.dashboard_port,
                    log_level="warning")
    finally:
        stop.set()
        t.join(timeout=5)
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
