# quantfund — multi-strategy stocks + options paper trading

A multi-strategy, multi-agent trading platform that paper-trades US stocks and
equity options through **Alpaca**, runs several strategy sleeves in parallel
(an **LLM multi-agent** sleeve on the Anthropic API, plus **momentum** and
**mean-reversion** baselines), allocates simulated capital across them by
risk-adjusted performance and correlation, enforces hard risk limits and
circuit breakers, and shows everything in a **live dashboard** with a kill
switch. A walk-forward backtester (purge/embargo) reports performance **net of
costs** against buy-and-hold.

> ## ⚠️ PAPER TRADING ONLY
> This is a research toy for evaluating strategies on a paper account — not
> investment advice, not production trading software. Expect bugs. The
> platform **hard-refuses live accounts** (`ALPACA_PAPER` must be `true`;
> the broker constructor raises otherwise) and every risk limit defaults
> conservative. Nothing here guarantees returns; the correctness guards exist
> precisely so you can *trust the losses*.

---
# money
## Quickstart

```bash
# 1. install (Python 3.11+; 3.12 recommended)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
#   (no uv?  python3 -m venv .venv && .venv/bin/pip install -e ".[dev]")

# 2. configure
cp .env.example .env          # then edit .env — see "Keys" below

# 3. verify everything (offline, no keys needed — all 200 tests must pass)
.venv/bin/python -m pytest

# 4. run the paper-trading loop + dashboard
.venv/bin/python scripts/run_paper.py
# → open http://127.0.0.1:8000

# single smoke iteration (prints JSON and exits):
.venv/bin/python scripts/run_paper.py --once

# 5. backtests (synthetic = deterministic, no keys):
.venv/bin/python scripts/run_backtest.py --synthetic --walkforward
# real Alpaca daily bars:
.venv/bin/python scripts/run_backtest.py --start 2024-01-01 --end 2025-06-30 --walkforward
```

### Keys (.env)

| Variable | Where to get it | Needed for |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | [app.alpaca.markets](https://app.alpaca.markets) → switch to **Paper** account → API Keys. Enable **options trading (Level ≥ 2)** in the paper account settings or option orders will be rejected. | live paper trading + real market data |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | the LLM multi-agent sleeve (optional — without it the classical sleeves still run) |

**No Alpaca keys?** `run_paper.py` starts in **offline demo mode**
(deterministic synthetic data + simulated broker with full cost modeling) so
you can watch the entire platform work before wiring credentials.

---

## How to read the dashboard

| Panel | What you're looking at |
|---|---|
| **Header tiles** | Total equity (hero), cash, buying power, day P&L, total P&L (both net of all costs), gross exposure, drawdown, cumulative costs paid. |
| **Equity & drawdown** | Portfolio equity line with crosshair tooltip; drawdown-from-peak area below it (same time axis). |
| **Positions** | Every stock and option: qty, avg cost, mark, notional, unrealized P&L; options also show delta/theta and expiry. |
| **Strategies** | Per sleeve: allocated capital, weight of equity, live P&L, open position count. Colors are fixed per sleeve. |
| **Risk & safety** | One light per circuit breaker (green ok / red TRIPPED with the reason), portfolio net Greeks ($ delta, theta/day, vega), and the LLM daily spend meter. |
| **Trade blotter** | Every trade: time, sleeve, side, instrument, qty, price, costs. **Click a row** to open the full rationale — for LLM trades that's the entire agent debate (analysts → bull/bear → thesis → trader → risk officer) plus the exact market brief the agents saw. |
| **⛔ KILL SWITCH** | Top-right, always visible. Confirms, then flattens every position and halts trading. Also engageable from a terminal: `touch data/KILL_SWITCH`, or `curl -X POST localhost:8000/api/kill -H 'Content-Type: application/json' -d '{"reason":"cli"}'`. Release from the banner button or `POST /api/release`. |

---

## Architecture

| Path | What it is |
|---|---|
| `core/snapshot.py` | **MarketSnapshot** — the no-look-ahead guard. Immutable, timestamp-pinned view; construction *and* accessors reject any datum stamped after `as_of`. Strategies/risk/allocator only ever see snapshots. |
| `core/costs.py` | Cost model: spread crossing (buy ≥ ask, sell ≤ bid), slippage bps, per-contract option fees, SEC/TAF on equity sells. **A zero-cost fill is a bug** — tests enforce it. |
| `core/portfolio.py` | Unified equity+options accounting: cash, cost basis, realized/unrealized P&L, exposures, **net Greeks** ($ delta, gamma, theta/day, vega). |
| `core/greeks.py` | Black-Scholes pricing/Greeks/IV (fallback when the chain has no Greeks). |
| `data/` | `DataProvider` interface; `AlpacaDataProvider` (daily bars IEX + quotes + option chain w/ Greeks + date-pinned news, on-disk bar cache); `SyntheticProvider` (seeded GBM, offline). News that can't be date-pinned is **excluded** from snapshots by construction. |
| `strategies/` | `Strategy` interface. **`momentum`** — 12-1 cross-sectional, inverse-vol sized, absolute-momentum filtered; expresses via shares OR **deep-ITM calls** (`express_via="options"`, stock replacement). **`put_income`** — volatility-risk-premium: cash-secured ~30-delta puts, min 8% annualized yield, 50% profit-take / time exit / 2.5x loss stop. **`convexity`** — long options for crash/breakout convexity (puts-only by default). **`mean_reversion`** — 5d z-score dip-buying above the 200d SMA. `llm_agents/` (see below). |
| `allocation/allocator.py` | Risk-adjusted (Sharpe, shrunk toward equal weight) with pairwise-correlation penalty; per-sleeve floor 5% / cap 40%; **portfolio vol targeting** — deployment scales down when realized vol exceeds the 12% annual target (floor 30%). |
| `execution/` | `Broker` interface; `SimBroker` (fills through the cost model); `AlpacaPaperBroker` (alpaca-py, paper-only, single- and multi-leg MLEG option orders, fill/position sync, flatten-all). |
| `risk/` | `RiskManager` pre-trade gate, `CircuitBreakerBoard` (7 breakers), `KillSwitch`. |
| `backtest/` | Event-driven daily engine (**t+1 execution** — signals from day *d*'s close fill at day *d+1*'s open through the cost model, never same-bar), walk-forward with **purge/embargo**, metrics (CAGR/Sharpe/Sortino/maxDD/hit rate/turnover). |
| `live/runner.py` | The paper loop: snapshot → broker sync → breakers → allocate → signals → risk gate → orders → fills → state. One iteration can never crash the loop. |
| `dashboard/` | FastAPI + one self-contained HTML file (no CDN). |

### Risk profiles (`QF_RISK_PROFILE` in .env)

`conservative` (default) or `explosive`. Explosive is a **high-variance paper-testing**
configuration: wide limits, options-first sleeve ordering, and up to **50% of equity in
long-option premium**. Undefined-risk (naked short) options stay banned in both.

| Limit | conservative | explosive |
|---|---|---|
| Per-underlying exposure | 10% | 55% |
| Per-sector | 25% | 60% |
| Gross leverage | 1.0x | 2.0x |
| \|Net $ delta\| | 80% | 300% |
| **Long-option premium at risk** | **10%** | **50%** |
| Daily loss halt (flattens) | -2% | -15% |
| Max drawdown halt | -10% | -40% |
| Short equity | banned | allowed |

The premium cap is the one that bounds the worst case: loss on a long option is capped at
its premium **per position**, but premium that all expires worthless is a 100% loss of every
dollar deployed. 50% is sized for **deep-ITM** contracts (0.80 delta), which retain intrinsic
value when wrong — it would be reckless for OTM.

### What the evidence actually says (2023-2025 walk-forward, real Alpaca bars)

Judge sleeves on **out-of-sample** columns only. Six validation runs to date:

| Sleeve | Instrument | OOS Sharpe | Verdict |
|---|---|---|---|
| `put_income` | 1-3wk cash-secured puts | +0.58 to +1.60, positive every run | validated |
| `momentum_equity` | shares | +0.65 to +1.47 | validated |
| `momentum` (options) | 1-3wk deep-ITM calls | +0.36 (was negative in longer-dated variants) | promising, not proven |
| `convexity` | short-dated long options | negative in **all six** runs | disproven; kept small as a crash hedge |

Caveats that matter: option chains in historical backtests are Black-Scholes-priced off
realized vol (real historical chains aren't on the free data tier), so short-vol premium is
~zero **by construction** and put_income's live edge may differ. And 2023-2025 is a single,
predominantly bullish regime — nothing here has been tested through a crash.

### Honesty rules baked in

- **Never report performance without costs** — every fill everywhere goes
  through the cost model (or real Alpaca paper fills); metrics are computed on
  cost-inclusive equity curves.
- **Never trust in-sample** — `run_backtest.py` labels full-period results
  "in-sample — do not trust alone"; `--walkforward` prints IS vs OOS tables and
  a **loud OVERFITTING flag** when IS Sharpe ≫ OOS Sharpe.
- **Leakage tests fail the build** — `tests/test_leakage.py` injects future
  bars/quotes/news and non-pinnable news, and probes the backtest engine with a
  strategy that asserts it can never see past `as_of`.

### Conservative default limits (raise them explicitly in `core/config.py` → `RiskLimits`)

| Limit | Default |
|---|---|
| Per-underlying exposure | 10% of equity |
| Per-sector exposure | 25% |
| Per-sleeve allocation | 40% |
| Gross leverage | 1.0× (no margin) |
| \|Net $ delta\| | 80% of equity |
| \|Theta\|/day | 0.2% of equity |
| \|Vega\|/vol-pt | 1% of equity |
| Per-order notional | $10,000 (auto-downsized) |
| Per-instrument notional | $25,000 |
| Daily loss halt (flattens) | −2% |
| Max drawdown halt (flattens) | 10% |
| Trades/hour (halts) | 30 |
| Naked short options | **disallowed** (`allow_naked_short_options=True` to opt in; covered calls, cash-secured puts, and defined-risk verticals are allowed) |
| Short equity | disallowed |
| Data staleness halt | 96 h (daily-bar cadence; tighten for intraday) |

The risk gate rejects orders that *worsen* a breach but always allows
risk-reducing trades, so an over-limit book can trade back into compliance.

---

## The LLM multi-agent sleeve

Per selected symbol (top movers, max 3/run), one full chain of **8 structured
API calls**: technical analyst → fundamental analyst → sentiment analyst
(headlines from the snapshot only) → bull vs bear debate → research-manager
thesis → trader (sizes ≤ 15% of sleeve) → risk officer (approve / downsize /
veto) → local reflection written to lesson memory
(`data/llm_memory.jsonl`, retrieved by symbol + regime next time).

- **Structured JSON out**: every stage uses the API's JSON-schema structured
  outputs (`additionalProperties: false`) — no regex on prose, ever.
- **Cost guard**: hard caps — $5/day and $1/run by default
  (`QF_LLM_DAILY_BUDGET_USD`, `QF_LLM_PER_RUN_BUDGET_USD`); spend is priced
  from actual token usage and shown on the dashboard. When a cap hits, the
  sleeve stops calling; classical sleeves keep trading.
- **Model**: `claude-opus-5` by default; switch with `QF_LLM_MODEL`
  (e.g. `claude-haiku-4-5` for cheap runs). Note: temperature/seeds are **not
  supported** on Opus-5-class models — determinism relies on strict schemas,
  terse prompts, and a fixed `effort` level (`low` by default for cost).
- **Audit trail**: the complete stage-by-stage transcript is stored as the
  trade's rationale — click the trade in the blotter to read the debate.
- If the trader asks for an option, the sleeve picks a ~30-delta 30–60 DTE
  contract off the live chain (two-sided quotes required), else falls back to
  the equity.

---

## What is real vs. stubbed / known limitations

Be aware of these before believing any number:

1. **Backtest options are synthetically priced** — chains in backtests come
   from Black-Scholes off realized vol (SyntheticProvider), not historical
   chains. Live paper trading uses the real Alpaca chain with real bid/ask.
2. **Exercise/assignment is simplified** to cash settlement at intrinsic value
   on expiry (engine records a warning per settlement). No early assignment.
3. **Alpaca paper fills report zero explicit commission/fees** — friction on
   live-paper trades is whatever the paper engine embeds in fill prices. The
   explicit cost model applies to SimBroker/backtests.
4. **"Single-LLM-agent" comparison baseline is not wired** — comparing the
   multi-agent chain to one monolithic prompt would multiply API spend; run it
   manually by setting `max_symbols_per_run` and a one-stage variant if you
   need it. (Stub.)
5. **The fundamental analyst has no fundamentals feed** — it reasons from
   price/vol structure and headlines only, and its prompt says so.
6. **LLM sleeve in backtests is off by default** (`--llm` to enable; budget
   guard still applies). Historical news is not replayed to it.
7. **Per-order risk checks are static within a batch** — orders in the same
   loop iteration are each checked against the pre-batch portfolio (the
   reconciliation breaker catches drift). Backtests apply caps per-sleeve,
   which is stricter than live (aggregate).
8. **Intraday data is not used** — daily bars + latest quotes; the loop
   trades at its polling cadence (60 s loop, 15 min strategy rebalance).
9. **Regime signal is simple** — realized-vol bands + SMA trend on SPY.

---

## Safety model

- **Kill switch** (dashboard button / `POST /api/kill` /
  `touch data/KILL_SWITCH`): engages the halt *first*, then flattens every
  position and cancels working orders. Idempotent; survives flatten failures
  (stays halted, logs).
- **Circuit breakers**, evaluated every loop: `daily_loss` and `drawdown`
  (both **flatten + halt**), `trade_frequency`, `data_staleness`,
  `error_burst`, `reconciliation` (local book vs broker truth),
  `order_bounds` (3 size-rejections/hour = runaway sizing) — these four
  **halt new orders** but keep positions. Recovery auto-clears the halt;
  the kill switch never auto-releases.

## Repo layout

```
├── pyproject.toml
├── .env.example
├── docs/ARCHITECTURE.md        # module contracts
├── scripts/
│   ├── run_paper.py            # paper loop + dashboard
│   └── run_backtest.py         # backtest + walk-forward reports
├── src/quantfund/
│   ├── core/                   # snapshot, costs, portfolio, greeks, config, state
│   ├── data/                   # provider interface, alpaca, synthetic, cache, news
│   ├── strategies/             # base, momentum, mean_reversion, llm_agents/
│   ├── allocation/             # capital allocator
│   ├── execution/              # broker interface, sim, alpaca paper
│   ├── risk/                   # limits, circuit breakers, kill switch
│   ├── backtest/               # engine, walkforward, metrics
│   ├── live/                   # runner
│   └── dashboard/              # fastapi app + static frontend
└── tests/                      # 200 offline tests incl. leakage, kill-switch, and regression tests
```
