"""Kill switch: flattens, halts, survives broker failure, honors the file trigger."""
import pytest

from quantfund.core.state import PlatformState
from quantfund.risk.kill_switch import KillSwitch


class FakeBroker:
    def __init__(self, fail=False):
        self.flatten_calls = 0
        self.fail = fail

    def flatten_all(self, reason=""):
        self.flatten_calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return []

    # unused Broker interface bits
    def submit_order(self, o): ...
    def cancel_order(self, i): ...
    def get_order(self, i): ...
    def get_fills(self, since=None): return []
    def get_positions(self): return []
    def get_account(self): ...


@pytest.fixture
def state(tmp_path):
    return PlatformState(tmp_path / "s.db")


def test_engage_flattens_and_halts(state):
    broker = FakeBroker()
    ks = KillSwitch(state, broker)
    ks.engage("test reason")
    assert broker.flatten_calls == 1
    assert ks.is_engaged()
    assert state.halted
    assert state.mode == "killed"
    assert "test reason" in state.halt_reason


def test_flatten_failure_still_halts(state):
    broker = FakeBroker(fail=True)
    ks = KillSwitch(state, broker)
    ks.engage("bad day")
    assert ks.is_engaged()
    assert state.halted
    events = state.get_events()
    assert any("FAILED" in e["message"] for e in events)


def test_release(state):
    ks = KillSwitch(state, FakeBroker())
    ks.engage("x")
    ks.release()
    assert not ks.is_engaged()
    assert not state.halted


def test_file_trigger_honored(state):
    ks = KillSwitch(state, FakeBroker())
    assert not ks.is_engaged()
    state.kill_file.write_text("manual touch\n")
    assert ks.is_engaged()
    state.kill_file.unlink()
    assert not ks.is_engaged()
