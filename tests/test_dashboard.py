"""Dashboard API tests via TestClient (offline)."""
import pytest
from fastapi.testclient import TestClient

from quantfund.core.state import PlatformState, StrategyStatus, TradeRecord, utcnow
from quantfund.dashboard.app import create_app


@pytest.fixture
def state(tmp_path):
    s = PlatformState(tmp_path / "s.db")
    s.record_trade(TradeRecord(
        ts=utcnow(), strategy_id="momentum", instrument_key="AAPL",
        asset_class="equity", side="buy", qty=10, price=200.0,
        commission=0.0, fees=0.1, slippage=0.4, rationale_id="r1",
        order_id="o1"))
    s.record_rationale("r1", "momentum", {"rank": 1, "score": 0.25})
    s.record_equity(100_000, 80_000, 80_000, 20_000)
    s.record_equity(100_500, 80_000, 80_000, 20_500)
    s.set_breaker("daily_loss", False)
    s.update_strategy(StrategyStatus(strategy_id="momentum", name="Momentum",
                                     allocated_capital=40_000, weight=0.4))
    s.update_portfolio({"equity": 100_500, "cash": 80_000}, [])
    return s


@pytest.fixture
def client(state):
    return TestClient(create_app(state))


def test_index_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "KILL SWITCH" in r.text


def test_summary_shape(client):
    s = client.get("/api/summary").json()
    for key in ("mode", "halted", "kill_switch", "portfolio", "positions",
                "strategies", "breakers", "llm_spend_today"):
        assert key in s
    assert s["strategies"]["momentum"]["allocated_capital"] == 40_000


def test_trades_and_rationale_roundtrip(client):
    trades = client.get("/api/trades").json()
    assert len(trades) == 1
    assert trades[0]["rationale_id"] == "r1"
    r = client.get("/api/rationale/r1").json()
    assert r["payload"]["rank"] == 1
    assert client.get("/api/rationale/nope").status_code == 404


def test_equity_ascending(client):
    eq = client.get("/api/equity").json()
    assert len(eq) == 2
    assert eq[0]["ts"] <= eq[1]["ts"]


def test_kill_with_callback(state):
    calls = []
    app = create_app(state, kill=lambda reason: calls.append(reason))
    client = TestClient(app)
    r = client.post("/api/kill", json={"reason": "unit test"})
    assert r.json() == {"ok": True}
    assert calls == ["unit test"]


def test_kill_without_callback_engages_state(client, state):
    client.post("/api/kill", json={})
    assert state.kill_switch_engaged()
    client.post("/api/release")
    assert not state.kill_switch_engaged()


def test_health(client):
    r = client.get("/api/health").json()
    assert r["ok"] is True
