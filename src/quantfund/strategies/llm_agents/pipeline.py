"""LLM multi-agent strategy sleeve.

Chain per symbol: technical / fundamental / sentiment analysts → bull-vs-bear
debate → research-manager thesis → trader (sizing) → risk officer (veto power)
→ local reflection into lesson memory. Every stage is one Anthropic API call
with a strict JSON schema (structured outputs) — no regex on reasoning.

Determinism note: Opus-5-class models do not accept temperature/top_p/seeds
(the API rejects them); repeatability comes from strict schemas, terse prompts,
and a fixed effort level. Every decision's full transcript is stored as the
trade's rationale for the dashboard audit trail.

All market inputs come from the MarketSnapshot ONLY — the Anthropic API is a
reasoning engine here, not a data source.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

from quantfund.core.config import Settings
from quantfund.core.instruments import Equity, OptionRight
from quantfund.core.snapshot import MarketSnapshot
from quantfund.core.state import PlatformState
from quantfund.strategies.base import (
    Signal, SignalAction, SleeveContext, Strategy,
)
from quantfund.strategies.llm_agents.cost_guard import BudgetExhausted, CostGuard
from quantfund.strategies.llm_agents.memory import LessonMemory
from quantfund.strategies.llm_agents import schemas

log = logging.getLogger(__name__)

_ROLES = {
    "technical": (
        "You are a technical analyst at a systematic fund. Judge the setup from "
        "price/volume structure only. Be honest about uncertainty; neutral is a "
        "valid stance."),
    "fundamental": (
        "You are a fundamental analyst. NOTE: you have no fundamentals data feed "
        "— only price behavior, volatility and headlines. Reason about what the "
        "price action implies and say clearly when you cannot know something."),
    "sentiment": (
        "You are a news-sentiment analyst. Use ONLY the headlines provided (with "
        "their ages). If there is no news, say so and stay neutral."),
    "bull": (
        "You argue the BULL case as strongly as the evidence honestly allows. "
        "You will be rebutted by a bear — pre-empt their best argument."),
    "bear": (
        "You argue the BEAR case as strongly as the evidence honestly allows. "
        "Rebut the bull's argument directly."),
    "thesis": (
        "You are the research manager. Weigh the analysts and the debate, apply "
        "the lessons from memory, and commit to long, short, or no_trade. "
        "no_trade is a respectable answer."),
    "trade": (
        "You are the trader. Size the position for a paper account; target_weight "
        "is a fraction of this sleeve's capital, max 0.15. Prefer equity; use "
        "call_option/put_option only when convexity clearly helps the thesis."),
    "risk": (
        "You are the risk officer with veto power. Consider position size, "
        "regime, concentration and thesis quality. Downsize or veto freely."),
}


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


class LLMAgentStrategy(Strategy):
    strategy_id = "llm_agents"
    name = "LLM Multi-Agent"
    warmup_bars = 70
    uses_options = True

    def __init__(self, settings: Settings, state: PlatformState, client=None):
        self.settings = settings
        self.state = state
        self.guard = CostGuard(settings.llm, state)
        self.memory = LessonMemory(
            settings.data_dir / "llm_memory.jsonl",
            max_lessons=settings.llm.memory_max_lessons,
        )
        self._client = client
        self._warned_no_key = False

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    # ── one API stage ────────────────────────────────────────────────────

    def _call_stage(self, stage: str, system: str, user_content: str) -> dict:
        """One structured-output call. Raises BudgetExhausted / RuntimeError."""
        self.guard.check()
        model = self.settings.llm.model
        response = self.client.messages.create(
            model=model,
            max_tokens=self.settings.llm.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {"type": "json_schema", "schema": schemas.STAGES[stage]},
                "effort": self.settings.llm.effort,
            },
        )
        self.guard.record(model, response.usage)
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(f"stage {stage}: model refused")
        text = next(
            (b.text for b in response.content if getattr(b, "type", "") == "text"),
            None)
        if text is None:
            raise RuntimeError(f"stage {stage}: no text block in response")
        return json.loads(text)

    # ── market brief (snapshot-only) ─────────────────────────────────────

    def _brief(self, symbol: str, snapshot: MarketSnapshot, ctx: SleeveContext) -> str:
        bars = snapshot.get_bars(symbol)
        closes = [b.close for b in bars]
        last = closes[-1]

        def ret(n: int) -> Optional[float]:
            return last / closes[-n - 1] - 1.0 if len(closes) > n else None

        r5, r20, r60 = ret(5), ret(20), ret(60)
        rsi = _rsi(closes)
        vol = None
        if len(closes) > 21:
            rets = [closes[i] / closes[i - 1] - 1 for i in range(-20, 0)]
            m = sum(rets) / len(rets)
            vol = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) \
                * math.sqrt(252)
        regime = snapshot.get_regime()
        news = snapshot.get_news(symbol, limit=5)
        pos = ctx.positions.get(symbol)
        lines = [
            f"SYMBOL: {symbol}   as-of: {snapshot.as_of.isoformat()}",
            f"last price: {last:.2f}",
            f"returns: 5d {r5:+.2%} | 20d {r20:+.2%} | 60d "
            f"{'n/a' if r60 is None else f'{r60:+.2%}'}"
            if r5 is not None and r20 is not None else "returns: insufficient history",
            f"RSI(14): {'n/a' if rsi is None else f'{rsi:.0f}'}",
            f"realized vol (20d, ann.): {'n/a' if vol is None else f'{vol:.0%}'}",
            (f"regime: vol={regime.vol_regime} trend={regime.trend_regime}"
             if regime else "regime: unavailable"),
            (f"current sleeve position: {pos.qty:g} @ {pos.avg_cost:.2f}"
             if pos and pos.is_open else "current sleeve position: none"),
            f"sleeve capital: ${ctx.allocated_capital:,.0f}",
        ]
        if news:
            lines.append("recent headlines (age):")
            for n in news:
                age_h = (snapshot.as_of - n.published_at).total_seconds() / 3600
                lines.append(f"  - [{age_h:.0f}h ago] {n.headline}")
        else:
            lines.append("recent headlines: none available")
        return "\n".join(lines)

    # ── full chain for one symbol ────────────────────────────────────────

    def run_symbol(self, symbol: str, snapshot: MarketSnapshot,
                   ctx: SleeveContext) -> tuple[Optional[Signal], dict]:
        brief = self._brief(symbol, snapshot, ctx)
        regime = snapshot.get_regime()
        regime_key = regime.vol_regime if regime else "unknown"
        lessons = self.memory.retrieve(symbol, regime_key, k=5)
        lessons_txt = "\n".join(f"- {l.get('lesson', '')}" for l in lessons) \
            or "(no relevant lessons yet)"
        t: dict = {"symbol": symbol, "model": self.settings.llm.model,
                   "brief": brief, "lessons_used": lessons}

        t["technical"] = self._call_stage("technical", _ROLES["technical"], brief)
        t["fundamental"] = self._call_stage("fundamental", _ROLES["fundamental"], brief)
        t["sentiment"] = self._call_stage("sentiment", _ROLES["sentiment"], brief)

        analysts = json.dumps({k: t[k] for k in ("technical", "fundamental",
                                                 "sentiment")}, indent=1)
        t["bull"] = self._call_stage(
            "bull", _ROLES["bull"], f"{brief}\n\nANALYST VIEWS:\n{analysts}")
        t["bear"] = self._call_stage(
            "bear", _ROLES["bear"],
            f"{brief}\n\nANALYST VIEWS:\n{analysts}\n\nBULL ARGUMENT:\n"
            f"{json.dumps(t['bull'])}")
        t["thesis"] = self._call_stage(
            "thesis", _ROLES["thesis"],
            f"{brief}\n\nANALYSTS:\n{analysts}\n\nBULL:\n{json.dumps(t['bull'])}"
            f"\n\nBEAR:\n{json.dumps(t['bear'])}\n\nLESSONS FROM MEMORY:\n{lessons_txt}")

        if t["thesis"]["direction"] == "no_trade":
            return None, t

        t["trade"] = self._call_stage(
            "trade", _ROLES["trade"],
            f"{brief}\n\nAPPROVED THESIS:\n{json.dumps(t['thesis'])}")
        if t["trade"]["action"] == "no_trade":
            return None, t

        t["risk"] = self._call_stage(
            "risk", _ROLES["risk"],
            f"{brief}\n\nTHESIS:\n{json.dumps(t['thesis'])}\n\nPROPOSED TRADE:\n"
            f"{json.dumps(t['trade'])}")
        if t["risk"]["verdict"] == "veto":
            return None, t

        weight = t["trade"]["target_weight"]
        if t["risk"]["verdict"] == "downsize":
            weight = min(weight, t["risk"]["adjusted_weight"])
        weight = max(0.0, min(weight, 0.15))
        if weight <= 0:
            return None, t

        direction = t["thesis"]["direction"]

        # instrument selection: options when the trader asked AND the chain has
        # a liquid ~30-delta 30-60 DTE contract; else the underlying equity.
        # A bearish thesis is expressed by BUYING a put (defined risk, positive
        # weight) — never by selling the option: a negative weight on a put
        # would make the runner SELL puts, the OPPOSITE exposure.
        instrument = Equity(symbol)
        itype = t["trade"]["instrument_type"]
        if direction == "short" and itype == "call_option":
            # incoherent combo (bearish via long call) — refuse to guess
            t["note"] = "no_trade: thesis short but trader chose call_option"
            return None, t
        if itype in ("call_option", "put_option"):
            chain = snapshot.get_option_chain(symbol)
            if chain is not None:
                right = OptionRight.CALL if itype == "call_option" else OptionRight.PUT
                oq = chain.nearest_delta(right, 0.30, min_dte=30, max_dte=60)
                if oq is not None and oq.quote.bid > 0 and oq.quote.ask > 0:
                    instrument = oq.instrument
                else:
                    t["note"] = f"{itype} requested but no liquid contract; using equity"
            else:
                t["note"] = f"{itype} requested but no chain in snapshot; using equity"

        if isinstance(instrument, Equity) and direction == "short":
            # equity fallback for a bearish thesis = an actual short sale
            if not self.settings.risk.allow_short_equity:
                t["note"] = ("no_trade: bearish thesis needs a put (none liquid) "
                             "and short equity is disabled in RiskLimits")
                return None, t
            action = SignalAction.OPEN_SHORT
            signed_weight = -weight
        else:
            # long equity, long call, or LONG PUT (bearishness is in the right)
            action = SignalAction.OPEN_LONG
            signed_weight = weight

        signal = Signal(
            instrument=instrument,
            action=action,
            target_weight=signed_weight,
            rationale=(f"LLM {direction} {symbol} w={weight:.2f}: "
                       f"{t['thesis']['thesis'][:140]}"),
            strategy_id=self.strategy_id,
            ts=snapshot.as_of,
            confidence=max(0.0, min(1.0, t["thesis"]["conviction"])),
            rationale_payload=t,
        )
        # local reflection (no extra API spend): store the decision as a lesson
        self.memory.add({
            "symbol": symbol,
            "regime": regime_key,
            "lesson": (f"{snapshot.as_of.date()}: went {direction} w={weight:.2f}; "
                       f"thesis: {t['thesis']['thesis'][:120]}; "
                       f"risks: {', '.join(t['thesis']['key_risks'][:2])}"),
            "applies_when": f"{symbol} in {regime_key} vol regime",
            "tags": [direction, regime_key],
            "outcome": None,
        })
        return signal, t

    # ── Strategy interface ───────────────────────────────────────────────

    def generate_signals(self, snapshot: MarketSnapshot, ctx: SleeveContext) -> list[Signal]:
        if not self.settings.has_anthropic and self._client is None:
            if not self._warned_no_key:
                self.state.log_event("warning", "llm_agents",
                                     "no ANTHROPIC_API_KEY — LLM sleeve disabled")
                self._warned_no_key = True
            return []
        self.guard.start_run()

        # rank by absolute 20-bar move: where something is happening
        movers: list[tuple[str, float]] = []
        for sym in snapshot.bars.keys():
            bars = snapshot.get_bars(sym)
            if len(bars) < self.warmup_bars:
                continue
            closes = [b.close for b in bars]
            if closes[-21] > 0:
                movers.append((sym, abs(closes[-1] / closes[-21] - 1.0)))
        movers.sort(key=lambda kv: kv[1], reverse=True)
        picks = [s for s, _ in movers[: self.settings.llm.max_symbols_per_run]]

        signals: list[Signal] = []
        for sym in picks:
            try:
                sig, transcript = self.run_symbol(sym, snapshot, ctx)
            except BudgetExhausted as e:
                self.state.log_event("warning", "llm_agents", f"budget stop: {e}")
                break
            except Exception as e:  # noqa: BLE001 — one bad symbol must not kill the run
                log.warning("LLM pipeline failed for %s: %r", sym, e)
                self.state.log_event("error", "llm_agents", f"{sym}: {e!r}")
                continue
            if sig is not None:
                signals.append(sig)
        return signals
