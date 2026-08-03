#!/usr/bin/env python3
"""Run backtests with realistic costs, t+1 execution, and (optionally)
walk-forward purge/embargo validation. Performance is NEVER reported without
costs and out-of-sample separation.

Usage:
  python scripts/run_backtest.py --synthetic --walkforward
  python scripts/run_backtest.py --start 2024-01-01 --end 2025-06-30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from quantfund.allocation.allocator import CapitalAllocator
from quantfund.backtest.engine import BacktestEngine
from quantfund.backtest.walkforward import run_walkforward
from quantfund.core.config import load_settings
from quantfund.core.state import utcnow
from quantfund.risk.limits import RiskManager
from quantfund.strategies.convexity import ConvexMomentumStrategy
from quantfund.strategies.mean_reversion import MeanReversionStrategy
from quantfund.strategies.momentum import MomentumStrategy
from quantfund.strategies.put_income import PutIncomeStrategy


def strategies_factory(settings, state=None, include_llm=False):
    def factory():
        if settings.risk_profile == "explosive":
            out = [MomentumStrategy(deploy_fraction=1.0, max_name_weight=0.35),
                   MeanReversionStrategy(), ConvexMomentumStrategy(),
                   PutIncomeStrategy()]
        else:
            out = [MomentumStrategy(), MeanReversionStrategy(),
                   PutIncomeStrategy()]
        if include_llm:
            from quantfund.core.state import PlatformState
            from quantfund.strategies.llm_agents.pipeline import LLMAgentStrategy
            st = state or PlatformState(settings.data_dir / "state.db")
            out.append(LLMAgentStrategy(settings, st))
        return out
    return factory


def print_metrics_table(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("-" * 96)
    header = (f"{'strategy':<18}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}"
              f"{'MaxDD':>9}{'HitRate':>9}{'Turnover':>10}{'TotRet':>9}{'Trades':>8}")
    print(header)
    for name in sorted(metrics, key=lambda k: (k in ("portfolio", "buy_and_hold"), k)):
        m = metrics[name]
        sortino = f"{m.sortino:>9.2f}" if m.sortino != float("inf") else f"{'inf':>9}"
        print(f"{name:<18}{m.cagr:>8.1%}{m.sharpe:>9.2f}{sortino}"
              f"{m.max_drawdown:>8.1%}{m.hit_rate:>8.1%}{m.turnover:>10.2f}"
              f"{m.total_return:>8.1%}{m.n_trades:>8}")
    print("-" * 96)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="use deterministic synthetic data (no keys needed)")
    parser.add_argument("--walkforward", action="store_true",
                        help="also run walk-forward with purge/embargo")
    parser.add_argument("--llm", action="store_true",
                        help="include the LLM sleeve (spends API budget!)")
    args = parser.parse_args()

    settings = load_settings()
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else utcnow() - timedelta(days=2))
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
             if args.start else end - timedelta(days=548))

    if args.synthetic or not settings.has_alpaca:
        if not args.synthetic:
            print("WARNING: no Alpaca keys — falling back to synthetic data",
                  file=sys.stderr)
        from quantfund.data.synthetic import SyntheticProvider
        provider = SyntheticProvider(seed=42)
        data_label = "SYNTHETIC (seeded GBM)"
    else:
        from quantfund.data.alpaca_data import AlpacaDataProvider
        from quantfund.data.hybrid import SyntheticChainProvider
        provider = SyntheticChainProvider(AlpacaDataProvider(settings))
        data_label = ("Alpaca daily bars (IEX) + BS-synthesized option chains "
                      "(no real historical chains on free tier)")
        print("NOTE: option chains in this backtest are Black-Scholes-priced "
              "off realized vol — directional option P&L is representative, "
              "short-vol premium (put_income) is ~zero BY CONSTRUCTION.",
              file=sys.stderr)

    if args.llm and not settings.has_anthropic:
        print("--llm requested but no ANTHROPIC_API_KEY; skipping LLM sleeve",
              file=sys.stderr)
        args.llm = False
    if args.llm:
        print("NOTE: --llm makes one full agent-chain API call per symbol per "
              "rebalance day. The cost guard caps daily spend at "
              f"${settings.llm.daily_budget_usd:.2f}; the backtest stops calling "
              "when the budget is hit.", file=sys.stderr)

    allocator = (CapitalAllocator(settings.risk, lookback=30, min_weight=0.02)
                 if settings.risk_profile == "explosive"
                 else CapitalAllocator(settings.risk))
    engine = BacktestEngine(
        provider, settings,
        allocator=allocator,
        risk_manager=RiskManager(settings.risk),
    )
    factory = strategies_factory(settings, include_llm=args.llm)

    print(f"\nBacktest {start.date()} → {end.date()}   data: {data_label}")
    print("Costs: spread crossing + slippage + fees on every fill "
          "(zero-cost fills are a bug). Execution: t+1 at next open.")

    result = engine.run(factory(), start, end,
                        options_underlyings=settings.options_universe)
    print_metrics_table("FULL-PERIOD RESULTS (in-sample — do not trust alone)",
                        result.metrics)
    if result.warnings:
        shown = result.warnings[:10]
        print(f"\nEngine warnings ({len(result.warnings)} total, first {len(shown)}):")
        for w in shown:
            print(f"  ! {w}")

    payload = {
        "start": start.isoformat(), "end": end.isoformat(), "data": data_label,
        "metrics": {k: m.as_dict() for k, m in result.metrics.items()},
        "warnings": result.warnings,
    }

    if args.walkforward:
        wf = run_walkforward(engine, factory, start, end,
                             options_underlyings=settings.options_universe)
        if not wf.windows:
            print("\nWalk-forward: range too short for a full train+purge+test "
                  "window (need ~320 days).")
        else:
            print_metrics_table("WALK-FORWARD IN-SAMPLE (train windows)", wf.in_sample)
            print_metrics_table("WALK-FORWARD OUT-OF-SAMPLE (embargoed test windows) "
                                "← judge on THIS", wf.out_of_sample)
            print("\nPer-window portfolio Sharpe (IS → OOS):")
            for w in wf.per_window:
                print(f"  train {w['train'][0]}..{w['train'][1]}  "
                      f"test {w['test'][0]}..{w['test'][1]}   "
                      f"{w['is_portfolio_sharpe']:>6.2f} → "
                      f"{w['oos_portfolio_sharpe']:>6.2f}")
            if wf.overfitting_flag:
                print("\n" + "!" * 72)
                print("!! OVERFITTING FLAG: in-sample >> out-of-sample.")
                print("!! Do NOT trust the full-period numbers above.")
                print("!" * 72)
            payload["walkforward"] = {
                "in_sample": {k: m.as_dict() for k, m in wf.in_sample.items()},
                "out_of_sample": {k: m.as_dict() for k, m in wf.out_of_sample.items()},
                "overfitting_flag": wf.overfitting_flag,
                "per_window": wf.per_window,
            }

    out_path = settings.data_dir / f"backtest_{utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nFull report saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
