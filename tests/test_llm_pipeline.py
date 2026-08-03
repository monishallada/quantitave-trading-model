"""LLM sleeve tests — no network. A fake Anthropic client returns canned
schema-valid JSON per stage and records every request."""
import json
from types import SimpleNamespace

import pytest

from quantfund.core.config import Settings
from quantfund.core.state import PlatformState
from quantfund.strategies.base import SleeveContext
from quantfund.strategies.llm_agents.cost_guard import BudgetExhausted, CostGuard
from quantfund.strategies.llm_agents.memory import LessonMemory
from quantfund.strategies.llm_agents.pipeline import LLMAgentStrategy

from conftest import make_chain, make_snapshot

STAGE_PAYLOADS = {
    "technical": {"stance": "bullish", "confidence": 0.7,
                  "key_points": ["uptrend", "rsi ok"], "summary": "constructive"},
    "fundamental": {"stance": "neutral", "confidence": 0.5,
                    "key_points": ["no data feed", "price steady"],
                    "summary": "cannot assess deeply"},
    "sentiment": {"stance": "neutral", "confidence": 0.4,
                  "key_points": ["no news", "quiet tape"], "summary": "quiet"},
    "bull": {"argument": "momentum continues", "strongest_point": "trend",
             "rebuttal_of_opponent": "pullbacks shallow", "conviction": 0.7},
    "bear": {"argument": "extended", "strongest_point": "mean reversion risk",
             "rebuttal_of_opponent": "trend can end", "conviction": 0.4},
    "thesis": {"direction": "long", "thesis": "trend + no bear catalyst",
               "time_horizon_days": 20, "conviction": 0.65,
               "key_risks": ["regime flip"]},
    "trade": {"action": "open_long", "instrument_type": "equity",
              "target_weight": 0.10, "entry_rationale": "buy strength",
              "stop_thesis": "close below 20d low"},
    "risk": {"verdict": "approve", "adjusted_weight": 0.10, "concerns": [],
             "reasoning": "size ok"},
}
STAGE_ORDER = ["technical", "fundamental", "sentiment", "bull", "bear",
               "thesis", "trade", "risk"]


class FakeAnthropicClient:
    def __init__(self, overrides=None, refuse_stages=(),
                 usage_tokens=(800, 200)):
        self.overrides = overrides or {}
        self.refuse_stages = set(refuse_stages)
        self.usage_tokens = usage_tokens
        self.requests = []
        self._call_idx = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        stage = STAGE_ORDER[self._call_idx % len(STAGE_ORDER)]
        # infer stage from the schema enum rather than call order where possible
        schema = kwargs.get("output_config", {}).get("format", {}).get("schema", {})
        props = schema.get("properties", {})
        if "stance" in props:
            # analyst — count which one by call order
            pass
        if "direction" in props:
            stage = "thesis"
        elif "action" in props:
            stage = "trade"
        elif "verdict" in props:
            stage = "risk"
        elif "argument" in props:
            stage = "bull" if not any(
                "argument" in json.dumps(r.get("messages", ""))
                for r in self.requests[:-1][-1:]) else "bear"
            # simpler: alternate bull then bear
            prior = sum(1 for r in self.requests[:-1]
                        if "argument" in str(r.get("output_config", "")))
            stage = "bull" if prior == 0 else "bear"
        elif "stance" in props:
            prior = sum(1 for r in self.requests[:-1]
                        if "stance" in str(r.get("output_config", "")))
            stage = ["technical", "fundamental", "sentiment"][min(prior, 2)]
        self._call_idx += 1
        if stage in self.refuse_stages:
            return SimpleNamespace(
                stop_reason="refusal", content=[],
                usage=SimpleNamespace(input_tokens=100, output_tokens=0,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0))
        payload = self.overrides.get(stage, STAGE_PAYLOADS[stage])
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            usage=SimpleNamespace(
                input_tokens=self.usage_tokens[0],
                output_tokens=self.usage_tokens[1],
                cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


@pytest.fixture
def settings(tmp_path):
    s = Settings(alpaca_paper=True, data_dir=tmp_path)
    s.anthropic_api_key = "test-key"
    s.llm.max_symbols_per_run = 1
    return s


@pytest.fixture
def state(tmp_path):
    return PlatformState(tmp_path / "s.db")


def snap():
    return make_snapshot({"AAPL": 200.0, "MSFT": 400.0},
                        chains={"AAPL": make_chain("AAPL", spot=200.0)})


def ctx(cap=50_000):
    return SleeveContext(strategy_id="llm_agents", allocated_capital=cap,
                         positions={}, sleeve_equity=cap)


class TestPipeline:
    def test_full_chain_produces_signal_with_transcript(self, settings, state):
        client = FakeAnthropicClient()
        strat = LLMAgentStrategy(settings, state, client=client)
        signals = strat.generate_signals(snap(), ctx())
        assert len(signals) == 1
        sig = signals[0]
        assert sig.target_weight == pytest.approx(0.10)
        assert sig.confidence == pytest.approx(0.65)
        payload = sig.rationale_payload
        for k in STAGE_ORDER:
            assert k in payload, f"missing stage {k}"
        assert "brief" in payload
        # 8 API calls for a full chain
        assert len(client.requests) == 8

    def test_requests_use_structured_outputs_no_temperature(self, settings, state):
        client = FakeAnthropicClient()
        LLMAgentStrategy(settings, state, client=client).generate_signals(snap(), ctx())
        for req in client.requests:
            assert "temperature" not in req
            fmt = req["output_config"]["format"]
            assert fmt["type"] == "json_schema"
            assert fmt["schema"]["additionalProperties"] is False
            assert req["output_config"]["effort"] == settings.llm.effort
            assert req["model"] == settings.llm.model

    def test_risk_downsize_respected(self, settings, state):
        client = FakeAnthropicClient(overrides={
            "risk": {"verdict": "downsize", "adjusted_weight": 0.04,
                     "concerns": ["size"], "reasoning": "trim"}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals[0].target_weight == pytest.approx(0.04)

    def test_veto_produces_no_signal(self, settings, state):
        client = FakeAnthropicClient(overrides={
            "risk": {"verdict": "veto", "adjusted_weight": 0.0,
                     "concerns": ["bad"], "reasoning": "no"}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals == []

    def test_no_trade_thesis_stops_early(self, settings, state):
        client = FakeAnthropicClient(overrides={
            "thesis": {"direction": "no_trade", "thesis": "nothing here",
                       "time_horizon_days": 5, "conviction": 0.2,
                       "key_risks": ["chop"]}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals == []
        assert len(client.requests) == 6  # stops after thesis

    def test_refusal_skips_symbol(self, settings, state):
        client = FakeAnthropicClient(refuse_stages={"technical"})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals == []  # skipped, no crash

    def test_budget_stops_pipeline(self, settings, state):
        settings.llm.per_run_budget_usd = 0.001  # one call blows the budget
        client = FakeAnthropicClient()
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals == []
        assert len(client.requests) <= 2

    def test_no_key_disables_sleeve(self, settings, state):
        settings.anthropic_api_key = ""
        strat = LLMAgentStrategy(settings, state)  # no client injected
        assert strat.generate_signals(snap(), ctx()) == []

    def test_option_selected_when_requested(self, settings, state):
        client = FakeAnthropicClient(overrides={
            "trade": {"action": "open_long", "instrument_type": "call_option",
                      "target_weight": 0.05, "entry_rationale": "convexity",
                      "stop_thesis": "thesis break"}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert len(signals) == 1
        from quantfund.core.instruments import Option
        assert isinstance(signals[0].instrument, Option)


class TestCostGuard:
    def test_cost_math(self, settings, state):
        usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                                cache_read_input_tokens=500_000,
                                cache_creation_input_tokens=0)
        cost = CostGuard.cost_of("claude-opus-5", usage)
        assert cost == pytest.approx(5.0 + 2.5 + 0.25)

    def test_daily_budget_hard_stop(self, settings, state):
        guard = CostGuard(settings.llm, state)
        state.llm_spend_today = settings.llm.daily_budget_usd + 0.01
        with pytest.raises(BudgetExhausted):
            guard.check()

    def test_unknown_model_priced_conservatively(self):
        usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=0,
                                cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        assert CostGuard.cost_of("mystery-model", usage) == pytest.approx(10.0)


class TestMemory:
    def test_add_retrieve_ranking(self, tmp_path):
        mem = LessonMemory(tmp_path / "m.jsonl")
        mem.add({"symbol": "AAPL", "regime": "high", "lesson": "a"})
        mem.add({"symbol": "MSFT", "regime": "low", "lesson": "b"})
        mem.add({"symbol": "AAPL", "regime": "low", "lesson": "c"})
        out = mem.retrieve("AAPL", "low", k=2)
        assert out[0]["lesson"] == "c"          # symbol+regime match wins
        assert {o["lesson"] for o in out} == {"a", "c"}

    def test_pruning(self, tmp_path):
        mem = LessonMemory(tmp_path / "m.jsonl", max_lessons=5)
        for i in range(10):
            mem.add({"symbol": "X", "lesson": str(i)})
        assert len(mem._load()) == 5
        assert mem._load()[-1]["lesson"] == "9"

    def test_corrupt_lines_skipped(self, tmp_path):
        p = tmp_path / "m.jsonl"
        p.write_text('{"symbol": "A", "lesson": "ok"}\nNOT JSON\n')
        mem = LessonMemory(p)
        assert len(mem._load()) == 1


class TestBearishDirection:
    def test_short_thesis_with_put_buys_the_put(self, settings, state):
        """A bearish thesis via put_option must be a LONG PUT (positive weight),
        never a short put — selling puts is the opposite exposure."""
        client = FakeAnthropicClient(overrides={
            "thesis": {"direction": "short", "thesis": "breaking down",
                       "time_horizon_days": 15, "conviction": 0.6,
                       "key_risks": ["squeeze"]},
            "trade": {"action": "open_short", "instrument_type": "put_option",
                      "target_weight": 0.05, "entry_rationale": "buy puts",
                      "stop_thesis": "reclaim of level"}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert len(signals) == 1
        sig = signals[0]
        from quantfund.core.instruments import Option, OptionRight
        from quantfund.strategies.base import SignalAction
        assert isinstance(sig.instrument, Option)
        assert sig.instrument.right == OptionRight.PUT
        assert sig.action == SignalAction.OPEN_LONG
        assert sig.target_weight > 0          # BUY the put

    def test_short_thesis_with_call_refused(self, settings, state):
        client = FakeAnthropicClient(overrides={
            "thesis": {"direction": "short", "thesis": "down",
                       "time_horizon_days": 15, "conviction": 0.6,
                       "key_risks": ["squeeze"]},
            "trade": {"action": "open_short", "instrument_type": "call_option",
                      "target_weight": 0.05, "entry_rationale": "??",
                      "stop_thesis": "x"}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals == []  # incoherent combo refused

    def test_short_thesis_equity_blocked_by_default(self, settings, state):
        client = FakeAnthropicClient(overrides={
            "thesis": {"direction": "short", "thesis": "down",
                       "time_horizon_days": 15, "conviction": 0.6,
                       "key_risks": ["squeeze"]},
            "trade": {"action": "open_short", "instrument_type": "equity",
                      "target_weight": 0.05, "entry_rationale": "short it",
                      "stop_thesis": "x"}})
        signals = LLMAgentStrategy(settings, state, client=client) \
            .generate_signals(snap(), ctx())
        assert signals == []  # allow_short_equity defaults to False
