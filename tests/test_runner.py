"""LiveRunner smoke tests: full offline loop with synthetic data + sim broker."""
import pytest

from quantfund.allocation.allocator import CapitalAllocator
from quantfund.core.config import Settings
from quantfund.core.costs import CostConfig, CostModel
from quantfund.core.state import PlatformState, utcnow
from quantfund.data.synthetic import SyntheticProvider
from quantfund.execution.sim_broker import SimBroker
from quantfund.live.runner import LiveRunner
from quantfund.risk.circuit_breakers import CircuitBreakerBoard
from quantfund.risk.kill_switch import KillSwitch
from quantfund.risk.limits import RiskManager
from quantfund.strategies.base import Strategy
from quantfund.strategies.mean_reversion import MeanReversionStrategy
from quantfund.strategies.momentum import MomentumStrategy


class ExplodingStrategy(Strategy):
    strategy_id = "boom"
    name = "Boom"
    warmup_bars = 1

    def generate_signals(self, snapshot, ctx):
        raise RuntimeError("intentional test explosion")


@pytest.fixture
def platform(tmp_path):
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0  # rebalance every loop in tests
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))
    strategies = [MomentumStrategy(), MeanReversionStrategy()]
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=strategies, state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )
    return runner, state, broker


def test_run_once_trades_and_records(platform):
    runner, state, broker = platform
    result = runner.run_once()
    assert set(result) >= {"snapshot_ts", "halted", "orders_submitted",
                           "fills", "equity"}
    assert not result["halted"]
    assert result["orders_submitted"] > 0
    assert result["fills"] > 0
    trades = state.get_trades()
    assert trades
    t = trades[0]
    assert t["strategy_id"] in ("momentum", "mean_reversion")
    assert t["rationale_id"]
    assert state.get_rationale(t["rationale_id"]) is not None
    assert state.get_equity_curve()


def test_kill_switch_halts_loop(platform):
    runner, state, broker = platform
    runner.run_once()
    assert broker.portfolio.open_positions()
    state.engage_kill_switch("test")
    result = runner.run_once()
    assert result["halted"]
    assert result["orders_submitted"] == 0
    assert not broker.portfolio.open_positions()  # flattened


def test_exploding_strategy_does_not_kill_loop(tmp_path):
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(["SPY"], as_of=utcnow()))
    runner = LiveRunner(settings=settings, broker=broker, provider=provider,
                        strategies=[ExplodingStrategy()], state=state)
    result = runner.run_once()
    assert result is not None  # loop survived
    events = state.get_events()
    assert any("intentional test explosion" in e["message"] for e in events)


def test_state_published_for_dashboard(platform):
    runner, state, _ = platform
    runner.run_once()
    view = state.snapshot_view()
    assert view["portfolio"].get("equity") is not None
    assert "momentum" in view["strategies"]
    assert "daily_loss" in view["breakers"]


def test_restart_anchors_to_real_account_equity(tmp_path):
    """A restart when the account equity differs from config starting_cash must
    NOT trip the daily-loss breaker: day P&L anchors to post-sync equity."""
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 1e9  # no trading — isolate the anchor
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    # simulate prior losses persisted at the broker: account is at 96k
    broker = SimBroker(CostModel(CostConfig()), 96_000)
    broker.set_snapshot(provider.build_snapshot(["SPY"], as_of=utcnow()))
    runner = LiveRunner(settings=settings, broker=broker, provider=provider,
                        strategies=[], state=state)
    result = runner.run_once()
    assert not result["halted"], state.breakers.get("daily_loss")
    assert runner.portfolio.day_anchor_equity == pytest.approx(96_000, rel=1e-3)
    assert abs(runner.portfolio.day_pnl()) < 100
    assert not state.breakers["daily_loss"].tripped
    assert not state.breakers["drawdown"].tripped


class KillMidPassStrategy(Strategy):
    """Engages the kill switch from 'another thread' mid-strategy-pass, then
    emits a signal — which must be discarded, not submitted."""
    strategy_id = "kill_mid"
    name = "KillMid"
    warmup_bars = 1

    def __init__(self, state, kill):
        self._state = state
        self._kill = kill

    def generate_signals(self, snapshot, ctx):
        from quantfund.core.instruments import Equity
        from quantfund.strategies.base import Signal, SignalAction
        self._kill.engage("operator pressed the button")
        return [Signal(instrument=Equity("SPY"), action=SignalAction.TARGET_WEIGHT,
                       target_weight=0.2, rationale="should never trade",
                       strategy_id=self.strategy_id, ts=snapshot.as_of)]


def test_kill_mid_iteration_discards_pending_signals(tmp_path):
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(["SPY"], as_of=utcnow()))
    kill = KillSwitch(state, broker)
    runner = LiveRunner(settings=settings, broker=broker, provider=provider,
                        strategies=[KillMidPassStrategy(state, kill)],
                        state=state, kill_switch=kill)
    result = runner.run_once()
    assert result["halted"]
    assert result["orders_submitted"] == 0
    assert not broker.portfolio.open_positions()


def test_reconciliation_detects_broker_drift(platform):
    """Mutating the broker's book behind the runner's back must trip the
    reconciliation breaker (fill-derived local book vs broker truth)."""
    runner, state, broker = platform
    runner.run_once()
    open_pos = broker.portfolio.open_positions()
    assert open_pos
    # simulate a manual trade / missed fill at the broker: +7 shares appear
    open_pos[0].qty += 7
    result = runner.run_once()
    assert result["halted"]
    assert state.breakers["reconciliation"].tripped
    assert "mismatch" in state.breakers["reconciliation"].reason
