"""Reflection/lesson memory for the LLM sleeve: JSONL store, retrieved by
symbol + regime relevance, injected into future decisions."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class LessonMemory:
    def __init__(self, path: Path | str, max_lessons: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_lessons = max_lessons

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        lessons = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lessons.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping corrupt lesson line")
        return lessons

    def _save(self, lessons: list[dict]) -> None:
        lessons = lessons[-self.max_lessons:]
        self.path.write_text(
            "\n".join(json.dumps(l, default=str) for l in lessons) + "\n"
            if lessons else ""
        )

    def add(self, lesson: dict) -> None:
        lesson.setdefault("ts", datetime.now(timezone.utc).isoformat())
        lessons = self._load()
        lessons.append(lesson)
        self._save(lessons)

    def retrieve(self, symbol: str, regime: str | None = None, k: int = 5) -> list[dict]:
        lessons = self._load()
        scored = []
        for i, l in enumerate(lessons):
            score = 0.0
            if l.get("symbol") == symbol:
                score += 2.0
            if regime and l.get("regime") == regime:
                score += 1.0
            score += i * 1e-6  # recency tiebreak (later lines are newer)
            if score > 1e-5:
                scored.append((score, l))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [l for _, l in scored[:k]]
