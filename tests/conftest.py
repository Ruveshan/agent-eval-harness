"""Shared test fixtures. No network, no API keys — the agent under test
here is the harness itself, so agents are scripted fakes."""

from __future__ import annotations

import json
from typing import Optional

from harness.models import AgentOutput, AgentResponse, Category, RunRecord, TestCase


class ScriptedAgent:
    """An Agent whose responses are a fixed script.

    Each entry is either a string (returned as the response text) or an
    Exception instance (raised, to simulate a transport failure). The
    script loops if the harness asks for more runs than entries.
    """

    name = "scripted"
    model = "test-model"

    def __init__(self, script: list) -> None:
        self._script = script
        self._i = 0
        self.calls = 0

    async def run(self, input_text: str) -> AgentResponse:
        self.calls += 1
        item = self._script[self._i % len(self._script)]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return AgentResponse(text=item, input_tokens=100, output_tokens=30)


def output_json(category: str, merchant: Optional[str] = None, confidence: float = 0.9) -> str:
    return json.dumps(
        {"category": category, "merchant": merchant, "confidence": confidence}
    )


def make_run(
    category: Optional[str],
    raw: Optional[str] = "…",
    error: Optional[str] = None,
    merchant: Optional[str] = None,
) -> RunRecord:
    """RunRecord factory: category=None means the run produced no parse."""
    parsed = (
        AgentOutput(category=Category(category), merchant=merchant, confidence=0.9)
        if category is not None
        else None
    )
    return RunRecord(raw_output=raw, parsed=parsed, error=error, latency_ms=10.0)


def make_case(**overrides) -> TestCase:
    base = {"id": "case", "input": "SOME MERCHANT", "tags": []}
    base.update(overrides)
    return TestCase.model_validate(base)
