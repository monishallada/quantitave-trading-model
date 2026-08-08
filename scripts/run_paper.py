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
from datetime import time as dtime

from quantfund.allocation.allocator import CapitalAllocator
from quantfund.core.config import load_settings
from quantfund.core.costs import CostConfig, CostModel
from quantfund.core.state import PlatformState
from quantfund.live.runner import LiveRunner
from quantfund.risk.circuit_breakers import CircuitBreakerBoard
from quantfund.risk.kill_switch import KillSwitch
from quantfund.risk.limits import RiskManager
from quantfund.strategies.mean_reversion import MeanReversionStrategy
from quantfund.strategies.momentum import MomentumStrategy
from quantfund.strategies.put_income import PutIncomeStrategy
from quantfund.strategies.zero_dte import ZeroDTEStrategy

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
            # 0DTE FIRST in the options-first queue: its positions live and die
            # inside the session, so if it queues behind a sleeve that eats the
            # cash/exposure budget it simply never trades.
            # ⚠️ UNVALIDATED: the daily t+1 backtest engine cannot test a
            # same-session instrument. Hard-capped on purpose — see zero_dte.py.
            # MAX AGGRESSION: 13% of sleeve per trade, 6 concurrent, 40
            # entries/day. Sized against the 40x net-delta budget (~$15.7k of
            # live 0DTE premium on a $100k book at ~255x delta per premium
            # dollar). This is the rapid engine: up to 80 option fills a day.
            ZeroDTEStrategy(underlyings=["SPY", "QQQ", "IWM"],
                            premium_per_trade_pct=0.13,
                            max_concurrent=6, max_entries_per_day=40,
                            take_profit=0.60, stop_loss=0.35,
                            no_entry_after_utc=dtime(19, 15)),
            PutIncomeStrategy(min_dte=7, max_dte=21, exit_dte=3, max_underlyings=6),
            # top_n=3 + 0.65 delta: concentrate the sleeve's capital so it can
            # always afford whole contracts (5 names x deep-ITM premium
            # exceeded the sleeve's budget and produced near-zero trades)
            # Heavy deep-ITM deployment: 0.65-delta calls behave like leveraged
            # stock with a capped downside, and they are the ONLY long-option
            # expression the 2023-25 evidence supports. Portfolio premium at
            # risk is bounded by RiskLimits.max_long_option_premium_pct.
            # Excludes the 0DTE underlyings. Option exposure is measured as
            # DELTA-ADJUSTED NOTIONAL, so a single 0.80-delta SPY call is
            # ~$65k of exposure against a $70k per-underlying cap — one
            # contract saturates the name. Sharing SPY/QQQ/IWM with zero_dte
            # meant every 0DTE order on them was rejected before it could be
            # placed (measured live 2026-08-07: SPY $122k vs a $70.5k cap).
            # The two options sleeves now trade disjoint universes.
            MomentumStrategy(top_n=6, deploy_fraction=1.0, max_name_weight=0.60,
                             express_via="options", option_delta=0.80,
                             option_leverage=7.0,
                             option_max_premium_weight=0.95,
                             option_min_dte=7, option_max_dte=21,
                             option_roll_dte=4,
                             exclude=["SPY", "QQQ", "IWM"]),
            # Equity momentum LAST: it is the single best OOS-validated engine
            # in the platform (2023-25 walk-forward Sharpe 1.47) and lifts
            # portfolio OOS from ~-0.3 to ~+1.1. Deliberately constrained so it
            # can never repeat the 2026-08-03 budget grab: it is last in the
            # options-first queue, holds fewer names, and deploys ~half of its
            # sleeve — the options sleeves always eat first.
            # NON-optionable names only: keeps the equity core from taking
            # duplicate exposure in the same names as the options sleeves and
            # from eating their per-name / per-sector budget.
            MomentumStrategy(top_n=4, deploy_fraction=0.95,
                             max_name_weight=0.35, express_via="equity",
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
                     base_weights={"zero_dte": 0.35,        # rapid engine
                                   "momentum": 0.30,        # ITM 1-3wk calls
                                   "momentum_equity": 0.20,
                                   "put_income": 0.15})     # short puts
                     # 78% of capital to the three OPTIONS sleeves.
                     # convexity is GONE: it lost money in all six
                     # out-of-sample validations, and zero_dte now occupies
                     # the same "long short-dated options" niche with a
                     # tighter bracket. momentum_equity keeps 22% because it
                     # is the strongest OOS sleeve in the platform and is
                     # the only thing here with a positive validated Sharpe
                     # backing the book when the options sleeves bleed.
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

    def supervised_loop():
        """Never let the trading thread die silently. On 2026-08-06 a stalled
        Alpaca socket killed this thread mid-session; uvicorn kept serving a
        dashboard that said "live" for 37 hours while nothing traded."""
        while not stop.is_set():
            try:
                runner.run_forever(stop)
                return                      # clean exit (stop_event set)
            except BaseException as e:      # noqa: BLE001
                log.critical("TRADING LOOP CRASHED (%r) — restarting in 15s", e)
                state.log_event("critical", "supervisor",
                                f"trading loop crashed: {e!r} — restarting")
                stop.wait(15)

    t = threading.Thread(target=supervised_loop, daemon=True,
                         name="trading-loop")
    t.start()

    def watchdog():
        """Restart the loop thread if it stops completing iterations."""
        nonlocal t
        stale_limit = max(300.0, settings.poll_interval_sec * 5)
        while not stop.is_set():
            stop.wait(60)
            if stop.is_set():
                return
            age = state.loop_age_seconds()
            if not t.is_alive():
                log.critical("WATCHDOG: trading thread is dead — restarting")
                state.log_event("critical", "watchdog",
                                "trading thread dead — restarted")
                t = threading.Thread(target=supervised_loop, daemon=True,
                                     name="trading-loop")
                t.start()
            elif age is not None and age > stale_limit:
                log.critical("WATCHDOG: no completed loop for %.0fs "
                             "(limit %.0fs) — loop is stalled", age, stale_limit)
                state.log_event("critical", "watchdog",
                                f"no completed trading loop for {age:.0f}s")

    threading.Thread(target=watchdog, daemon=True, name="watchdog").start()

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
