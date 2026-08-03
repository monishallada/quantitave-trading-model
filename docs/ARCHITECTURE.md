# quantfund — Architecture & Cross-Module Contracts

This file is the source of truth for module boundaries. The core contracts
(instruments, snapshot, orders, costs, portfolio, config, state, provider,
strategy, broker) are **already implemented** under `src/quantfund/core`,
`src/quantfund/data/provider.py`, `src/quantfund/strategies/base.py`,
`src/quantfund/execution/broker.py`. Read those files before implementing
anything — do not redefine or duplicate their types.

## Price/qty conventions (everywhere)
- Prices are per share (option premiums too). Notional = price × qty × multiplier.
- Position qty is signed. Order/Fill qty is positive; direction comes from OrderSide.
- All datetimes timezone-aware UTC. Naive datetimes raise.

## Module map and owners

| Path | Contents |
|---|---|
| `quantfund/core/*` | DONE — instruments, greeks, snapshot, orders, costs, portfolio, config, state |
| `quantfund/data/provider.py` | DONE — DataProvider ABC + build_snapshot + compute_regime |
| `quantfund/data/synthetic.py` | DONE — deterministic offline provider |
| `quantfund/data/alpaca_data.py` | AlpacaDataProvider(DataProvider) |
| `quantfund/data/cache.py` | on-disk bar cache used by AlpacaDataProvider |
| `quantfund/data/news.py` | date-pinned news via Alpaca news API (pinnable flag) |
| `quantfund/strategies/momentum.py` | MomentumStrategy(Strategy) |
| `quantfund/strategies/mean_reversion.py` | MeanReversionStrategy(Strategy) |
| `quantfund/strategies/llm_agents/*` | LLMAgentStrategy(Strategy) + pipeline/schemas/memory/cost guard |
| `quantfund/allocation/allocator.py` | CapitalAllocator |
| `quantfund/execution/sim_broker.py` | SimBroker(Broker) |
| `quantfund/execution/alpaca_broker.py` | AlpacaPaperBroker(Broker) |
| `quantfund/risk/limits.py` | RiskManager (pre-trade checks) |
| `quantfund/risk/circuit_breakers.py` | CircuitBreakerBoard |
| `quantfund/risk/kill_switch.py` | KillSwitch |
| `quantfund/backtest/engine.py` | BacktestEngine |
| `quantfund/backtest/walkforward.py` | walk-forward with purge/embargo |
| `quantfund/backtest/metrics.py` | CAGR/Sharpe/Sortino/maxDD/hit rate/turnover |
| `quantfund/live/runner.py` | LiveRunner (the paper-trading loop) |
| `quantfund/dashboard/app.py` | FastAPI app factory + static frontend |

## Required signatures (implement EXACTLY these — the runner and tests call them)

### risk/limits.py
```python
@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    adjusted_qty: float | None = None   # set when the order was downsized, else None

class RiskManager:
    def __init__(self, limits: RiskLimits, sector_of: Callable[[str], str] = sector_of): ...
    def check_order(self, order: Order, portfolio: Portfolio,
                    snapshot: MarketSnapshot) -> RiskDecision:
        """Pre-trade gate. Enforces, in order:
        - naked short option ban (unless limits.allow_naked_short_options):
          a SELL leg on an option, where the resulting net position would be
          short and not covered by underlying shares (calls) — reject. Long
          puts/calls and covered/cash-secured positions are fine. Multi-leg
          orders where every short leg is paired with a long leg further OTM or
          equal qty long leg (defined-risk spread) are allowed.
        - short equity ban (unless limits.allow_short_equity)
        - per-order notional <= max_order_notional (downsize allowed via adjusted_qty)
        - resulting per-instrument notional <= max_position_notional
        - resulting per-underlying |exposure| <= max_position_pct * equity
        - resulting per-sector |exposure| <= max_sector_pct * equity
        - resulting gross leverage <= max_gross_leverage
        - resulting |net dollar delta| <= max_net_delta_pct * equity
        - resulting |theta/day| <= max_theta_pct_per_day * equity,
          |vega| <= max_vega_pct * equity (options)
        - cash buffer: post-trade cash >= min_cash_buffer_pct * equity for BUYs
        Estimate post-trade exposure using snapshot prices/greeks; be
        conservative (treat missing greeks for a short option as unbounded →
        reject)."""
```

### risk/circuit_breakers.py
```python
@dataclass
class BreakerTrip:
    name: str
    reason: str

class CircuitBreakerBoard:
    """Names (exact strings, used by dashboard):
    daily_loss, drawdown, trade_frequency, data_staleness, error_burst,
    reconciliation, order_bounds.
    """
    def __init__(self, limits: RiskLimits, state: PlatformState): ...
    def record_trade(self, ts: datetime) -> None: ...
    def record_error(self, source: str, msg: str) -> None: ...
    def record_success(self) -> None:       # resets consecutive error counter
    def check_all(self, portfolio: Portfolio, snapshot: MarketSnapshot | None,
                  broker_positions: list[BrokerPosition] | None) -> list[BreakerTrip]:
        """Evaluates every breaker, updates state.set_breaker(...) for each,
        and returns the list of currently tripped breakers. Reconciliation
        compares portfolio.positions vs broker_positions qty per instrument
        (tolerance limits.reconciliation_qty_tolerance); skip if
        broker_positions is None."""
```

### risk/kill_switch.py
```python
class KillSwitch:
    def __init__(self, state: PlatformState, broker: Broker): ...
    def engage(self, reason: str) -> list[Order]:
        """state.engage_kill_switch(reason), then broker.flatten_all(reason).
        Returns the flatten orders. Never raises: logs errors to state."""
    def is_engaged(self) -> bool: ...
    def release(self) -> None: ...
```

### allocation/allocator.py
```python
@dataclass
class SleevePerf:
    strategy_id: str
    daily_returns: list[float]     # recent sleeve daily returns, oldest first

class CapitalAllocator:
    def __init__(self, limits: RiskLimits, lookback: int = 60,
                 min_weight: float = 0.05): ...
    def allocate(self, total_equity: float, sleeves: list[SleevePerf]) -> dict[str, float]:
        """Returns {strategy_id: dollar_allocation}, sum <= total_equity.
        Method: score each sleeve by risk-adjusted performance (Sharpe-like on
        daily_returns, shrunk toward equal weight when history < lookback);
        penalize pairwise correlation (average correlation to other sleeves
        reduces score); floor at min_weight, cap at limits.max_strategy_pct;
        renormalize. With no history at all → equal weight. Deterministic."""
```

### backtest/metrics.py
```python
@dataclass
class PerfMetrics:
    cagr: float; sharpe: float; sortino: float; max_drawdown: float
    hit_rate: float; turnover: float; total_return: float; n_trades: int

def compute_metrics(equity_curve: list[tuple[datetime, float]],
                    trades: list[Fill], periods_per_year: int = 252) -> PerfMetrics: ...
```

### backtest/engine.py
```python
@dataclass
class BacktestResult:
    equity_curves: dict[str, list[tuple[datetime, float]]]  # per strategy_id + "portfolio" + "buy_and_hold"
    metrics: dict[str, PerfMetrics]
    trades: list[Fill]
    warnings: list[str]

class BacktestEngine:
    # allocator/risk_manager are duck-typed optional injections so the engine
    # never imports risk/allocation/execution modules (None → equal weights /
    # engine-internal sanity checks only)
    def __init__(self, provider: DataProvider, settings: Settings,
                 allocator: object | None = None,
                 risk_manager: object | None = None,
                 spread_bps: float = 4.0): ...
    def run(self, strategies: list[Strategy], start: datetime, end: datetime,
            symbols: list[str] | None = None, options_underlyings: list[str] | None = None,
            rebalance_every_n_days: int = 1) -> BacktestResult:
        """Daily event loop. For each trading day d (in order):
        1. snapshot = provider.build_snapshot(symbols, as_of=close_of(d), ...)
           — strategies only ever see this snapshot.
        2. per-sleeve: signals = strat.generate_signals(snapshot, ctx)
        3. risk-check each order via RiskManager, execute on SimBroker at the
           NEXT day's data (t+1 execution: use next day's quote built from next
           day's bar) — never same-bar fill on the signal bar's close.
        4. mark portfolios, roll equity curves, allocator rebalances weekly.
        Buy-and-hold benchmark: SPY (or first symbol) bought on day 1, held.
        Expire ITM options via exercise-at-intrinsic on expiration day; OTM
        expire worthless."""
```

### backtest/walkforward.py
```python
@dataclass
class WalkForwardWindow:
    train_start: datetime; train_end: datetime
    test_start: datetime; test_end: datetime

def make_windows(start: datetime, end: datetime, train_days: int = 252,
                 test_days: int = 63, purge_days: int = 5) -> list[WalkForwardWindow]:
    """Rolling windows; test period starts purge_days AFTER train_end (embargo).
    Windows advance by test_days."""

@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    in_sample: dict[str, PerfMetrics]      # aggregated
    out_of_sample: dict[str, PerfMetrics]
    overfitting_flag: bool                 # True if IS Sharpe >> OOS Sharpe (ratio > 2 or OOS < 0 while IS > 1)
    per_window: list[dict]

def run_walkforward(engine: BacktestEngine, strategies_factory: Callable[[], list[Strategy]],
                    start: datetime, end: datetime, **kwargs) -> WalkForwardResult: ...
```

### live/runner.py
```python
class LiveRunner:
    def __init__(self, settings: Settings, broker: Broker, provider: DataProvider,
                 strategies: list[Strategy], state: PlatformState): ...
    def run_forever(self, stop_event: threading.Event) -> None:
        """Loop every settings.poll_interval_sec:
        0. if state.kill_switch_engaged(): flatten once (via KillSwitch), set
           mode=killed, then idle (keep updating dashboard state).
        1. build snapshot (provider.build_snapshot)
        2. sync portfolio mirror from broker (positions/account), mark to market
        3. breakers.check_all(...); if any tripped → halt (no new orders;
           existing positions kept unless breaker is daily_loss/drawdown →
           flatten via KillSwitch)
        4. every rebalance_interval_sec: allocator.allocate(...), run each
           strategy's generate_signals, convert Signals→Orders (weight × sleeve
           capital / price), RiskManager.check_order each, submit approved via
           broker, record fills + trades + rationales into state.
        5. state.update_portfolio(...), state.record_equity(...), per-strategy
           StrategyStatus updates.
        All exceptions inside one iteration are caught → breakers.record_error;
        never crash the loop."""
    def run_once(self) -> dict:  # single iteration, used by tests/smoke
```

### dashboard/app.py
```python
def create_app(state: PlatformState, kill: Callable[[str], Any] | None = None,
               release: Callable[[], Any] | None = None) -> FastAPI:
```
Endpoints (all JSON):
- `GET /` → static/index.html
- `GET /api/summary` → state.snapshot_view()
- `GET /api/trades?limit=200` → state.get_trades()
- `GET /api/rationale/{rationale_id}` → state.get_rationale() (404 if missing)
- `GET /api/equity` → state.get_equity_curve()
- `GET /api/events` → state.get_events()
- `POST /api/kill` body `{"reason": "..."}` → calls kill(reason); returns {"ok": true}
- `POST /api/release` → calls release()
Frontend: single self-contained `static/index.html` (no CDN), polls /api/* every
3s: portfolio header (equity, cash, BP, day/total P&L), equity-curve + drawdown
chart (inline SVG/canvas), positions table w/ greeks, per-strategy panel,
trade blotter with click-through rationale modal, risk panel with breaker
lights and a big red KILL SWITCH button (POST /api/kill w/ confirm dialog).

### strategies/llm_agents/ (LLM sleeve)
- `cost_guard.py`: `CostGuard(llm_cfg, state)` — `.check(est_usd) -> bool`,
  `.record(model, usage) -> float` using price table
  {claude-opus-5: (5,25), claude-opus-4-8: (5,25), claude-sonnet-5: (3,15),
   claude-haiku-4-5: (1,5), claude-fable-5: (10,50)} $/MTok (input, output) +
  cache reads billed at 0.1× input. Persists daily spend via state.add_llm_spend.
  Hard stop when daily budget reached.
- `schemas.py`: JSON Schemas for each agent stage output (analyst, debate,
  thesis, trade decision, risk check). All object schemas use
  additionalProperties: false + required.
- `pipeline.py`: `LLMAgentStrategy(Strategy)` — strategy_id "llm_agents".
  generate_signals: pick top max_symbols_per_run symbols by |20d return|,
  for each run the agent chain via Anthropic client with
  output_config={"format": {"type": "json_schema", "schema": ...}} structured
  outputs (NO temperature param — removed on Opus 5-class models; check
  stop_reason == "refusal" before reading content; use output_config
  {"effort": llm_cfg.effort}). Stages: technical analyst → fundamental analyst
  → sentiment analyst (news from snapshot only) → bull vs bear debate (1 round
  each) → research manager thesis → trader (sizes position as target_weight
  ≤ 0.15) → risk officer (can veto/downsize). Full transcript of every stage
  goes into Signal.rationale_payload. Retrieve up to 5 relevant lessons from
  memory and inject; after each decision store a lesson stub for later
  reflection. If no API key or budget exhausted → return [] and log once.
- `memory.py`: JSONL lesson store under ctx.data_dir/llm_memory.jsonl with
  `add(lesson: dict)`, `retrieve(symbol, regime, k=5) -> list[dict]`.
- Client: `anthropic.Anthropic(api_key=...)`; all calls synchronous,
  max_tokens from config, wrapped in try/except (APIError → skip symbol).

### execution/sim_broker.py
```python
class SimBroker(Broker):
    def __init__(self, cost_model: CostModel, starting_cash: float): ...
    def set_snapshot(self, snapshot: MarketSnapshot) -> None:  # quotes for fills
    # submit_order fills market orders immediately at cost-model prices from
    # the CURRENT snapshot quotes (equity: symbol quote; option: chain quote —
    # reject if the contract isn't on the chain or quote is one-sided).
    # Limit orders: fill only if limit crosses the costed execution price.
    # Maintains its own Portfolio; get_positions/get_account reflect it.
```

### execution/alpaca_broker.py
- `AlpacaPaperBroker(Broker)` built on alpaca-py `TradingClient(paper=True)`.
- Constructor MUST raise unless settings.alpaca_paper is True.
- Map Order→alpaca requests: equities via MarketOrderRequest/LimitOrderRequest;
  single-leg options via symbol=OCC; multi-leg via OptionLegRequest list +
  order_class=OrderClass.MLEG. TimeInForce.DAY.
- get_fills: poll `get_orders(GetOrdersRequest(status=CLOSED))` → map filled
  orders to Fill (commission 0, fees 0 — Alpaca paper doesn't report; slippage 0;
  note in docstring that live-paper friction shows up in fill prices).
- get_positions: map alpaca positions; OCC symbols → parse_occ_symbol.
- flatten_all: cancel all open orders then close_all_positions(cancel_orders=True);
  also handles option positions.
- Every network call wrapped: raise BrokerError with context on APIError.

## Runner order-sizing rule (shared by backtest + live)
weight_to_qty: dollars = target_weight × sleeve_capital;
equity qty = floor(dollars / price) shares (min 1 share else skip);
option qty = floor(dollars / (premium × 100)) contracts (min 1 else skip);
CLOSE → qty = current sleeve position qty for that instrument.

## Testing rules (all builders)
- Tests must run OFFLINE: no network, no API keys. Mock alpaca-py and anthropic
  clients (monkeypatch/unittest.mock). Use SyntheticProvider for data.
- Use `tmp_path` for any PlatformState/db/memory files.
- Never sleep > 0.1s in tests.
