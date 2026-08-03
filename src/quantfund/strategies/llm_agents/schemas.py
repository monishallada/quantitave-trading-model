"""JSON Schemas for every LLM agent stage. Structured outputs guarantee the
model's reply parses as valid JSON matching these schemas — no regex on prose,
ever. Every object level sets additionalProperties: false + required."""
from __future__ import annotations


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


_CONF = {"type": "number", "minimum": 0, "maximum": 1}

ANALYST_SCHEMA = _obj({
    "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
    "confidence": _CONF,
    "key_points": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                   "maxItems": 5},
    "summary": {"type": "string"},
})

DEBATE_SCHEMA = _obj({
    "argument": {"type": "string"},
    "strongest_point": {"type": "string"},
    "rebuttal_of_opponent": {"type": "string"},
    "conviction": _CONF,
})

THESIS_SCHEMA = _obj({
    "direction": {"type": "string", "enum": ["long", "short", "no_trade"]},
    "thesis": {"type": "string"},
    "time_horizon_days": {"type": "integer", "minimum": 1, "maximum": 90},
    "conviction": _CONF,
    "key_risks": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                  "maxItems": 5},
})

TRADE_SCHEMA = _obj({
    "action": {"type": "string", "enum": ["open_long", "open_short", "no_trade"]},
    "instrument_type": {"type": "string",
                        "enum": ["equity", "call_option", "put_option"]},
    "target_weight": {"type": "number", "minimum": 0, "maximum": 0.15},
    "entry_rationale": {"type": "string"},
    "stop_thesis": {"type": "string"},
})

RISK_SCHEMA = _obj({
    "verdict": {"type": "string", "enum": ["approve", "downsize", "veto"]},
    "adjusted_weight": {"type": "number", "minimum": 0, "maximum": 0.15},
    "concerns": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "reasoning": {"type": "string"},
})

STAGES = {
    "technical": ANALYST_SCHEMA,
    "fundamental": ANALYST_SCHEMA,
    "sentiment": ANALYST_SCHEMA,
    "bull": DEBATE_SCHEMA,
    "bear": DEBATE_SCHEMA,
    "thesis": THESIS_SCHEMA,
    "trade": TRADE_SCHEMA,
    "risk": RISK_SCHEMA,
}
