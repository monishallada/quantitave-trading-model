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
| **Header tiles** | Total equity (hero), cash, buying power, day P&L, total P&L (both net of all costs), **premium at risk**, **economic exposure**, drawdown, cumulative costs paid. |
| **Premium at risk vs economic exposure** | Two different questions, both shown, because for options they differ by ~35x. *Premium at risk* is what can go to zero (a long option's max loss). *Economic exposure* is delta-adjusted notional — how much market the book actually controls, and what the per-underlying/per-sector risk limits enforce. A single 0.80-delta SPY call is ~$1.8k of premium but ~$65k of exposure; judging deployment by premium alone makes a fully-invested book look 19% deployed. |
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
| `strategies/` | `Strategy` interface. **`momentum`** — 12-1 cross-sectional, inverse-vol sized, absolute-momentum filtered; expresses via shares OR **deep-ITM calls** (`express_via="options"`, stock replacement). **`put_income`** — volatility-risk-premium: cash-secured ~30-delta puts, min 8% annualized yield, 50% profit-take / time exit / 2.5x loss stop. **`zero_dte`** — same-session index-ETF options; intraday 1-min signals (15m momentum + session VWAP side + relative-volume surge vs a 30m trailing median), contracts picked by **moneyness** (Alpaca's indicative feed carries no greeks on 0DTE), hard bracket of +60% take-profit / -35% stop / **forced flat at 19:30 UTC** so nothing is ever held into expiry. **Unvalidated — see below.** **`convexity`** — long options for crash/breakout convexity (puts-only; disproven, no longer in the explosive lineup). **`mean_reversion`** — 5d z-score dip-buying above the 200d SMA. `llm_agents/` (see below). |
| `allocation/allocator.py` | Risk-adjusted (Sharpe, shrunk toward equal weight) with pairwise-correlation penalty; per-sleeve floor 5% / cap 40%; **portfolio vol targeting** — deployment scales down when realized vol exceeds the 12% annual target (floor 30%). |
| `execution/` | `Broker` interface; `SimBroker` (fills through the cost model); `AlpacaPaperBroker` (alpaca-py, paper-only, single- and multi-leg MLEG option orders, fill/position sync, flatten-all). |
| `risk/` | `RiskManager` pre-trade gate, `CircuitBreakerBoard` (7 breakers), `KillSwitch`. |
| `backtest/` | Event-driven daily engine (**t+1 execution** — signals from day *d*'s close fill at day *d+1*'s open through the cost model, never same-bar), walk-forward with **purge/embargo**, metrics (CAGR/Sharpe/Sortino/maxDD/hit rate/turnover). |
| `live/runner.py` | The paper loop: snapshot → broker sync → breakers → allocate → signals → risk gate → orders → fills → state. One iteration can never crash the loop. |
| `dashboard/` | FastAPI + one self-contained HTML file (no CDN). |

### Risk profiles (`QF_RISK_PROFILE` in .env)

`conservative` (default) or `explosive`. Explosive is a **high-variance paper-testing**
configuration: wide limits, options-first sleeve ordering, and up to **65% of equity in
long-option premium**. Undefined-risk (naked short) options stay banned in both.

| Limit | conservative | explosive |
|---|---|---|
| Per-underlying exposure | 10% | 70% |
| Per-sector | 25% | 80% |
| Gross leverage *(premium basis)* | 1.0x | 2.0x |
| \|Net $ delta\| | 80% | 300% |
| **Long-option premium at risk** | **10%** | **65%** |
| Daily loss halt (flattens) | -2% | -15% |
| Max drawdown halt | -10% | -40% |
| Short equity | banned | allowed |
| Naked short options | banned | **banned** |

### How option exposure is measured (and why it is not one rule)

Every risk check that talks about "exposure" goes through a single function,
`portfolio.option_exposure` — used by **both** `Portfolio.exposure_by_underlying`
and the pre-trade risk gate. They must agree: when the gate sized 0DTE by premium
while the book marked it by Black-Scholes delta, a 21-lot QQQ call was authorized as
$1,911 of exposure and landed as **$510,170** (2026-08-07).

| position | measured as | why |
|---|---|---|
| Short options | delta notional | loss is unbounded; nothing about premium bounds it |
| Long, \|delta\| ≥ 0.5 | delta notional | stock replacement — a 0.85-delta call tracks the underlying ~1:1 and its premium is large |
| Long, \|delta\| < 0.5 | **premium** | capped-loss ticket — a 0DTE 0.33-delta call shows $510k of delta against $1,911 of premium and cannot lose more than the premium |

`theta` follows the same principle: a long option's daily decay is bounded by its
premium. Black-Scholes theta is an instantaneous rate that explodes into expiry — a
21-lot 0DTE call holding $1,932 of premium reports **−$7,930/day**, a loss it cannot
physically realise, and it made theta the binding limit for a position whose true
worst case is the premium.

### The 0DTE leverage arithmetic — read this before sizing the sleeve

An at-the-money 0DTE contract carries **~255x delta per premium dollar** ($92 of
premium = $23,446 of delta). So the net-delta cap, not premium and not cash, is what
bounds this sleeve:

| net-delta cap | delta budget (on $100k) | live 0DTE premium it supports |
|---|---|---|
| 3x | $300,000 | $1,177 |
| **8x** (current explosive) | $800,000 | **$3,139** |
| 20x | $2,000,000 | $7,848 |
| 40x | $4,000,000 | $15,696 |
| 80x | $8,000,000 | $31,392 |

**Holding $30k of live 0DTE premium needs roughly a 76x net-delta cap** — i.e. no
meaningful delta limit at all. That is a property of the instrument, not a limit that
can be tuned around, and it is why the 0DTE sleeve is sized at 3.2% premium per trade
(so all four concurrent positions fit the 8x budget) rather than the 8% a premium-only
view would suggest. Above ~40x the delta limit has stopped bounding anything and the
real protection is `max_long_option_premium_pct` (65%) — which, for a book that is
purely long options with capped loss, is arguably the correct bound anyway.

**Which limit actually bounds leverage.** The gross-leverage row is computed on *premium*,
so on an options book it effectively never binds — $19.5k of premium against a $201k cap
while the book carried **1.35x equity** of real delta exposure (measured live 2026-08-07).
Delta leverage is bounded instead by the three checks that use delta-adjusted notional:
per-underlying (70%), per-sector (80%), and net dollar delta (300%). Those are the numbers
to watch; gross-leverage-on-premium is a premium-at-risk guard wearing a leverage label.

The premium cap is the one that bounds the worst case: loss on a long option is capped at
its premium **per position**, but premium that all expires worthless is a 100% loss of every
dollar deployed. 50% is sized for **deep-ITM** contracts (0.80 delta), which retain intrinsic
value when wrong — it would be reckless for OTM. The `zero_dte` sleeve buys *near-the-money*
contracts that decay to zero the same day, which is why its own per-trade and per-day caps
(8% of sleeve capital per trade, 12 entries/day, 4 concurrent) bind long before this one does.

### What the evidence actually says (2023-2025 walk-forward, real Alpaca bars)

Judge sleeves on **out-of-sample** columns only. Six validation runs to date:

| Sleeve | Instrument | OOS Sharpe | Verdict |
|---|---|---|---|
| `put_income` | 1-3wk cash-secured puts | +0.58 to +1.60, positive every run | validated |
| `momentum_equity` | shares | +0.65 to +1.47 | validated |
| `momentum` (options) | 1-3wk deep-ITM calls | +0.36 (was negative in longer-dated variants) | promising, not proven |
| `convexity` | short-dated long options | negative in **all six** runs | **disproven — removed from the explosive lineup** |
| `zero_dte` | same-session options | **never measured** | **unvalidated — see below** |

#### `zero_dte` cannot be validated by this harness

The backtest engine is daily with t+1 execution. A 0DTE option is born and dies inside one
session, so there is no way for it to appear in a walk-forward run — its OOS Sharpe is not
"bad", it is *undefined*. Measuring it needs historical **intraday option quotes**
(ThetaData/ORATS class), which the free tier does not carry.

What is known: `convexity` — long short-dated options, the nearest relative — lost money in
all six out-of-sample runs at a 5-17% hit rate. 0DTE is that trade with 100% of the premium
as time value and no time to recover. It is in the lineup because it was explicitly asked
for, and it is capped so a bad run is survivable, not because evidence supports it.

What *has* been measured is **signal frequency** (not profit): replaying 15 trading days of
real 1-min SPY/QQQ/IWM bars through the live signal code fires a mean of 26 signals/day,
range 7-44, with zero signal-free days. The sleeve's own 12-entry/day cap binds, not the
signal supply.

The other measured number is the **friction hurdle**. Round-trip option fees are ~1.4% of
premium, and the sleeve crosses the full spread each way. Against its +60% / -35% bracket
that sets breakeven win rate at:

| contract spread | total friction | breakeven win rate |
|---|---|---|
| 1.6% (IWM median) | 3.0% | 40.0% |
| 2.8% (SPY median) | 4.2% | 41.3% |
| 5.2% (QQQ median) | 6.6% | 43.8% |
| 10% (sleeve's reject threshold) | 11.4% | 48.9% |

Zero-friction breakeven would be 36.8%. `convexity` — the only comparable sleeve that has
actually been measured on this platform — ran a **5-17% hit rate**. This sleeve has to be
a fundamentally better signal than that one, not a marginally better one.

#### Measured bracket execution (first live session, 2026-08-07)

Brackets are evaluated on the MID but executed at the BID, on a 60-second loop, so both
sides slip. Three clean stop-losses (excluding one that a since-fixed bug delayed):

| contract | trigger | filled | vs -35% |
|---|---|---|---|
| IWM 300P | 0.163 | 0.15 (-40.0%) | -5.0% |
| SPY 770P | 0.319 | 0.29 (-40.8%) | -5.8% |
| QQQ 719P | 0.637 | 0.66 (-32.7%) | **+2.3%** |

Mean **-2.8%**, range -5.8% to +2.3% — so realized brackets are roughly +57% / -38% and
breakeven lands near **40%**, consistent with the estimate above. Note the third fill was
*better* than its trigger: with n=3 the variance dwarfs the mean, and an early read off the
first two samples alone would have wrongly suggested a systematic -5.4% drag. Do not size
this sleeve off a handful of fills.

Session result: **0 wins / 4 losses, -$1,414 realized** on 8 fills. That is a count, not a
result — and roughly $416 of it was a risk-gate bug rejecting a stop-loss, not the strategy.

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
