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


def test_restart_into_existing_positions_does_not_double_count(tmp_path):
    """Restarting against an account that already holds positions must seed the
    local book from the broker WITHOUT re-applying the fills that created it —
    otherwise reconciliation trips on every restart."""
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 1e9      # no new trading this test
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    snap = provider.build_snapshot(["SPY"], as_of=utcnow())
    broker.set_snapshot(snap)

    # pre-existing position + its fill history, as a real account would have
    from quantfund.core.instruments import Equity
    from quantfund.core.orders import OrderSide, single_leg
    broker.submit_order(single_leg(Equity("SPY"), OrderSide.BUY, 10,
                                   strategy_id="momentum"))
    assert broker.get_fills()          # the fill exists in broker history

    runner = LiveRunner(settings=settings, broker=broker, provider=provider,
                        strategies=[], state=state)
    result = runner.run_once()
    broker_qty = {p.instrument.key: p.qty for p in broker.get_positions()}
    assert runner._local_qty == broker_qty      # seeded, not doubled
    assert not state.breakers["reconciliation"].tripped
    assert not result["halted"]


def test_state_reports_stalled_when_loop_stops(tmp_path):
    """A dashboard that says "live" while the trading loop is dead is worse
    than no dashboard (2026-08-06: 37h of silent death)."""
    from datetime import timedelta
    state = PlatformState(tmp_path / "s.db")
    state.mode = "live"
    state.update_portfolio({"equity": 100.0}, [])
    fresh = state.snapshot_view(stall_after_sec=600)
    assert fresh["mode"] == "live" and fresh["stalled"] is False
    state.last_loop_ts = utcnow() - timedelta(hours=37)
    stale = state.snapshot_view(stall_after_sec=600)
    assert stale["mode"] == "STALLED"
    assert stale["stalled"] is True
    assert stale["loop_age_sec"] > 36 * 3600


def test_http_timeout_applied_to_alpaca_clients():
    """alpaca-py ships no HTTP timeout; a stalled socket hangs the loop."""
    from types import SimpleNamespace
    from quantfund.execution.alpaca_broker import apply_http_timeout
    captured = {}

    class FakeSession:
        def request(self, method, url, **kw):
            captured.update(kw)
            return "ok"

    client = SimpleNamespace(_session=FakeSession())
    apply_http_timeout(client, timeout=(5.0, 20.0))
    client._session.request("GET", "/account")
    assert captured["timeout"] == (5.0, 20.0)


class _Recorder(Strategy):
    """Records every pass in which the runner asked it for signals."""

    def __init__(self, sid, fast):
        self.strategy_id = sid
        self.name = sid
        self.warmup_bars = 1
        self.fast_cadence = fast
        self.calls = 0

    def generate_signals(self, snapshot, ctx):
        self.calls += 1
        return []


def test_fast_sleeves_run_every_loop_slow_ones_only_on_rebalance(tmp_path):
    """A fast_cadence sleeve is useless at the rebalance interval — its
    positions live and die inside the session. But the converse matters just as
    much: a fast-only pass must NOT drag the slow sleeves along, or the
    rebalance throttle silently stops existing and turnover explodes.
    """
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 10_000      # effectively never re-due
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))
    fast, slow = _Recorder("fast", True), _Recorder("slow", False)
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[fast, slow], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )
    for _ in range(3):
        runner.run_once()
    assert fast.calls == 3, f"fast sleeve skipped a loop: {fast.calls} of 3"
    assert slow.calls == 1, (
        f"slow sleeve ran {slow.calls}x — the rebalance throttle is not "
        "being applied on fast-only passes")


def test_fast_sleeve_underlyings_get_option_chains(tmp_path):
    """A fast sleeve's underlyings must have chains fetched even when they are
    not in the configured options universe, or it silently sees no contracts."""
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.options_universe = ["SPY"]
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(), options_underlyings=["SPY"]))

    fast = _Recorder("fast", True)
    fast.underlyings = ["SPY", "QQQ"]           # QQQ is NOT in the universe
    seen = {}
    orig = provider.build_snapshot

    def spy(*a, **kw):
        seen.update(kw)
        return orig(*a, **kw)

    provider.build_snapshot = spy
    LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[fast], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    ).run_once()
    assert "QQQ" in seen["options_underlyings"]
    assert set(seen["intraday_symbols"]) == {"SPY", "QQQ"}


def test_expiry_guard_flattens_0dte_even_when_its_sleeve_is_broken(tmp_path,
                                                                   monkeypatch):
    """The nightmare case: the loop is alive but the sleeve that owns a 0DTE
    contract throws every pass. Nobody emits the forced-flat, the contract
    expires ITM, Alpaca auto-exercises, and 100 shares/contract land in a
    paper account that never authorised that leverage. The guard must fire
    independently of any sleeve.
    """
    from datetime import date, datetime, time as dtime, timezone

    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide

    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))

    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[ExplodingStrategy()], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )
    runner.run_once()   # establish the book

    today = utcnow().date()
    expiring = make_option("SPY", today, OptionRight.CALL, 400.0)
    later = make_option("SPY", date(today.year + 1, 1, 15),
                        OptionRight.CALL, 400.0)
    for inst in (expiring, later):
        runner.portfolio.apply_fill(Fill(
            order_id=f"seed-{inst.symbol}", instrument=inst,
            side=OrderSide.BUY, qty=3, price=2.0, ts=utcnow()))

    submitted = []
    monkeypatch.setattr(broker, "submit_order",
                        lambda o: submitted.append(o))

    # before the cutoff: leave it alone, the sleeve owns the exit
    monkeypatch.setattr("quantfund.live.runner.utcnow",
                        lambda: datetime.combine(today, dtime(17, 0),
                                                 tzinfo=timezone.utc))
    assert runner._close_expiring_options(None) == 0
    assert not submitted

    # after the cutoff: flatten the expiring contract, and ONLY that one
    monkeypatch.setattr("quantfund.live.runner.utcnow",
                        lambda: datetime.combine(today, dtime(19, 50),
                                                 tzinfo=timezone.utc))
    assert runner._close_expiring_options(None) == 1
    assert len(submitted) == 1
    leg = submitted[0].legs[0]
    assert leg.instrument.symbol == expiring.symbol
    assert leg.side == OrderSide.SELL and leg.qty == 3
    assert any("expiry_guard" == e["source"] for e in state.get_events())


def test_restart_restores_sleeve_ownership_of_open_positions(tmp_path):
    """Ownership lives in memory and dies with the process. If a restart
    leaves the master book populated but the sleeves empty, every open
    position is orphaned: no exits run, and each sleeve sees current=0 for a
    name it already holds and re-buys it. Ownership must survive a restart.
    """
    from quantfund.core.instruments import Equity

    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0
    db = tmp_path / "s.db"

    def make_runner(st):
        provider = SyntheticProvider(seed=42)
        broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
        broker.set_snapshot(provider.build_snapshot(
            settings.universe, as_of=utcnow(),
            options_underlyings=settings.options_universe))
        return LiveRunner(
            settings=settings, broker=broker, provider=provider,
            strategies=[MomentumStrategy(), MeanReversionStrategy()], state=st,
            risk_manager=RiskManager(settings.risk),
            breakers=CircuitBreakerBoard(settings.risk, st),
            allocator=CapitalAllocator(settings.risk),
            kill_switch=KillSwitch(st, broker),
        ), broker

    state1 = PlatformState(db)
    r1, broker1 = make_runner(state1)
    r1.run_once()
    before = {sid: {k for k, p in s.positions.items() if p.is_open}
              for sid, s in r1.sleeves.items()}
    assert any(before.values()), "first run took no positions — test is vacuous"
    state1.close()

    # restart: same DB, same broker account, brand-new runner objects
    state2 = PlatformState(db)
    r2, _ = make_runner(state2)
    r2.broker = broker1                    # the account persists
    r2.run_once()

    after = {sid: {k for k, p in s.positions.items() if p.is_open}
             for sid, s in r2.sleeves.items()}
    for sid, keys in before.items():
        assert keys <= after.get(sid, set()), (
            f"sleeve {sid} lost ownership of {keys - after.get(sid, set())} "
            "across a restart — nothing will manage those exits")
    owned = {k for keys in after.values() for k in keys}
    broker_keys = {p.instrument.key for p in r2.broker.get_positions()}
    assert broker_keys <= owned, f"orphaned at broker: {broker_keys - owned}"


def test_expiry_guard_never_double_sells_into_a_naked_short(tmp_path, monkeypatch):
    """The broker position does not clear the instant an order is sent. If the
    guard re-flattens whatever is still showing, it sells the same 21 contracts
    twice and the account ends up SHORT 21 naked 0DTE calls — the one exposure
    this platform bans outright. Submit once per contract per day.
    """
    from datetime import datetime, time as dtime, timezone

    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide

    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[MomentumStrategy()], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )
    today = utcnow().date()
    opt = make_option("QQQ", today, OptionRight.CALL, 723.0)
    runner.portfolio.apply_fill(Fill(order_id="seed", instrument=opt,
                                     side=OrderSide.BUY, qty=21, price=0.92,
                                     ts=utcnow()))
    submitted = []
    monkeypatch.setattr(broker, "submit_order", lambda o: submitted.append(o))
    monkeypatch.setattr("quantfund.live.runner.utcnow",
                        lambda: datetime.combine(today, dtime(19, 50),
                                                 tzinfo=timezone.utc))

    # the position is STILL showing on every subsequent loop (order working)
    assert runner._close_expiring_options(None) == 1
    for _ in range(5):
        runner._close_expiring_options(None)

    assert len(submitted) == 1, (
        f"guard sent {len(submitted)} flatten orders for one position — "
        "that is a naked short")
    total_sold = sum(l.qty for o in submitted for l in o.legs)
    assert total_sold == 21, f"sold {total_sold} of a 21-contract position"


def test_expiry_guard_retries_when_the_broker_rejected(tmp_path, monkeypatch):
    """Dedup must key off broker ACCEPTANCE. A submit that raised was never
    placed, so abandoning it would leave a 0DTE contract to auto-exercise."""
    from datetime import datetime, time as dtime, timezone

    from quantfund.core.instruments import OptionRight, make_option
    from quantfund.core.orders import Fill, OrderSide
    from quantfund.execution.broker import BrokerError

    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[MomentumStrategy()], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )
    today = utcnow().date()
    opt = make_option("QQQ", today, OptionRight.CALL, 723.0)
    runner.portfolio.apply_fill(Fill(order_id="seed", instrument=opt,
                                     side=OrderSide.BUY, qty=21, price=0.92,
                                     ts=utcnow()))
    monkeypatch.setattr("quantfund.live.runner.utcnow",
                        lambda: datetime.combine(today, dtime(19, 50),
                                                 tzinfo=timezone.utc))

    def boom(o):
        raise BrokerError("venue rejected")

    monkeypatch.setattr(broker, "submit_order", boom)
    assert runner._close_expiring_options(None) == 0

    ok = []
    monkeypatch.setattr(broker, "submit_order", lambda o: ok.append(o))
    assert runner._close_expiring_options(None) == 1, (
        "a rejected flatten must be retried, not abandoned")
    assert len(ok) == 1


def test_sleeve_marks_track_the_market_every_loop(tmp_path):
    """Sleeve positions are only touched by apply_fill, which stamps
    mark = fill price. Without an explicit re-mark each loop the sleeve's view
    is frozen at entry forever, and EVERY mark-based exit in the platform is
    dead code — 0DTE take-profit/stop-loss and put_income's profit-take and
    loss-stop all compare mark to entry. Observed live 2026-08-07: a QQQ 0DTE
    call hit +60.3% on the master book while its sleeve still saw the entry.
    """
    from quantfund.core.instruments import Equity
    from quantfund.core.orders import Fill, OrderSide

    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[MomentumStrategy()], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )
    runner.run_once()
    sleeve = runner.sleeves["momentum"]
    held = [p for p in sleeve.positions.values() if p.is_open]
    assert held, "sleeve took no position — test would be vacuous"

    # freeze every sleeve mark at an absurd value, as a stale mark would be
    for p in held:
        p.mark = 0.01
    runner.run_once()

    for p in (q for q in sleeve.positions.values() if q.is_open):
        assert p.mark != 0.01, (
            f"{p.instrument.key} sleeve mark never refreshed — mark-based "
            "exits cannot fire")
        assert p.mark_ts is not None


def test_sleeve_returns_are_daily_not_per_loop(tmp_path, monkeypatch):
    """The allocator's field is named daily_returns and it annualizes with
    sqrt(252). Appending a sample every loop fed it ~390 intraday returns per
    day, so Sharpe was computed on 60-second returns and annualized as daily,
    and vol targeting (std * sqrt(252)) understated realized vol by ~sqrt(390)
    — silently disabling itself. One sample per trading day, at the boundary.
    """
    from datetime import datetime, timedelta, timezone

    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0
    state = PlatformState(tmp_path / "s.db")
    provider = SyntheticProvider(seed=42)
    broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
    broker.set_snapshot(provider.build_snapshot(
        settings.universe, as_of=utcnow(),
        options_underlyings=settings.options_universe))
    runner = LiveRunner(
        settings=settings, broker=broker, provider=provider,
        strategies=[MomentumStrategy()], state=state,
        risk_manager=RiskManager(settings.risk),
        breakers=CircuitBreakerBoard(settings.risk, state),
        allocator=CapitalAllocator(settings.risk),
        kill_switch=KillSwitch(state, broker),
    )

    day0 = datetime(2026, 8, 7, 15, 0, tzinfo=timezone.utc)
    clock = {"now": day0}
    monkeypatch.setattr("quantfund.live.runner.utcnow", lambda: clock["now"])

    # Force sleeve equity to MOVE on every loop, so per-loop sampling would
    # definitely append and the test can actually tell the two apart.
    loop = {"n": 0}
    for sleeve in runner.sleeves.values():
        sleeve.equity = lambda: 1000.0 + loop["n"] * 10.0   # noqa: B023

    for i in range(6):                      # six loops, all inside one day
        loop["n"] = i
        clock["now"] = day0 + timedelta(minutes=i)
        runner.run_once()
    assert all(not r for r in runner.sleeve_returns.values()), (
        f"intraday loops leaked into daily returns: {runner.sleeve_returns}")

    # crossing the day boundary produces exactly ONE sample per sleeve
    loop["n"] = 10
    clock["now"] = day0 + timedelta(days=1)
    runner.run_once()
    counts = {sid: len(r) for sid, r in runner.sleeve_returns.items()}
    assert counts and all(c == 1 for c in counts.values()), counts

    loop["n"] = 11                          # same day again -> no new sample
    clock["now"] = day0 + timedelta(days=1, minutes=5)
    runner.run_once()
    assert {sid: len(r) for sid, r in runner.sleeve_returns.items()} == counts, (
        "a second loop on the same day added another sample")


def test_cumulative_costs_survive_a_restart(tmp_path):
    """Portfolio.total_costs accumulates only from fills ingested in the
    CURRENT process, and a restart marks prior fills as already-seen. So the
    dashboard's "costs paid" tile reset to $0.00 on every restart while real
    money had been spent — $62.01 of recorded fees showed as zero on
    2026-08-07. Seed from the durable trade log instead.
    """
    settings = Settings(alpaca_paper=True, starting_cash=100_000,
                        data_dir=tmp_path)
    settings.rebalance_interval_sec = 0
    db = tmp_path / "s.db"

    def make(st):
        provider = SyntheticProvider(seed=42)
        broker = SimBroker(CostModel(CostConfig()), settings.starting_cash)
        broker.set_snapshot(provider.build_snapshot(
            settings.universe, as_of=utcnow(),
            options_underlyings=settings.options_universe))
        return LiveRunner(
            settings=settings, broker=broker, provider=provider,
            strategies=[MomentumStrategy()], state=st,
            risk_manager=RiskManager(settings.risk),
            breakers=CircuitBreakerBoard(settings.risk, st),
            allocator=CapitalAllocator(settings.risk),
            kill_switch=KillSwitch(st, broker),
        ), broker

    state1 = PlatformState(db)
    r1, broker1 = make(state1)
    r1.run_once()                      # trades recorded to the DB
    # equity BUYS carry no commission and no SEC/TAF (sells only) — the
    # friction is slippage, so check total friction, not commission+fees
    _, recorded = state1.total_costs_recorded()
    assert recorded > 0, "a zero-cost fill is a bug — fixture produced none"
    r1.run_once()                      # next sync must reflect them
    spent = r1.portfolio.total_costs
    assert spent > 0, (
        "master portfolio never accumulates costs: _ingest_fills applies "
        "fills to sleeves only, so the dashboard shows $0.00 forever")
    state1.close()

    state2 = PlatformState(db)
    r2, _ = make(state2)
    r2.broker = broker1                    # same account persists
    r2.run_once()
    assert r2.portfolio.total_costs >= spent, (
        f"restart reset costs from {spent:.2f} to "
        f"{r2.portfolio.total_costs:.2f} — the dashboard would show money "
        "as never spent")
