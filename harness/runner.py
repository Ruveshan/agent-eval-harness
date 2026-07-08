"""Async suite runner.

Responsibilities: fan out N runs per case under a concurrency cap, retry
transport-level failures with exponential backoff, parse agent output
without crashing on garbage, and hand fully-populated `CaseResult`s to
the reporting layer.

Two failure planes are kept strictly separate:

* **Infrastructure failures** (the agent raised: rate limit, network,
  5xx) are *retried* here, because they say nothing about agent quality.
  If retries are exhausted, the run is recorded with an `error` -- never
  raised out of the harness.
* **Content failures** (the agent answered, but with malformed JSON or
  the wrong category) are *never* retried -- retrying until the model
  gets it right would be grading with the answer key open. They are
  recorded and left for the validators to judge.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol, runtime_checkable

from pydantic import ValidationError

from .models import AgentOutput, AgentResponse, CaseResult, RunRecord, TestCase

# USD per million tokens (input, output) at each provider's paid tier.
# Used for cost attribution; an unknown model simply reports $0 rather
# than guessing. Note for free-tier runs (e.g. Gemini's free API tier):
# the figure shows what the run *would* cost at paid rates — actual spend
# is $0, but the equivalent cost is the number you'd budget for scale.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-sonnet-4-6": (3.00, 15.00),
}


class TransientAgentError(Exception):
    """Marker for provider failures that a retry can plausibly fix.

    Exists because the harness's retry contract is type-based, and not
    every SDK cooperates: google-genai raises one `APIError` family for
    both a retryable 429 and a fatal 401, distinguishable only by code.
    An agent facing such an SDK catches the provider error and re-raises
    the transient ones as this type; the harness never needs to know
    provider status codes.

    `retry_after_s` carries the provider's own "retry in N seconds" hint
    when one exists. Quota windows are the provider's clock, not ours —
    honoring the hint beats guessing with exponential backoff, which
    burns attempts faster than a per-minute window refills.
    """

    def __init__(self, message: str, retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@runtime_checkable
class Agent(Protocol):
    """The minimal contract an agent must satisfy to be evaluated.

    Any object with a `name`, a `model` id (for pricing lookup), and an
    async `run` returning raw text + token usage plugs in -- the harness
    has no knowledge of *how* the agent produces its answer.

    An agent may additionally expose `retryable_exceptions` (a tuple of
    exception types) to tell the runner which failures are transient.
    Only the agent knows its transport: for the Anthropic SDK that's
    rate limits, 5xx, and connection errors -- while a 401 will never
    succeed on retry and should fail fast instead of burning backoff.
    Agents that don't declare it get retry-everything as a safe default.
    """

    name: str
    model: str

    async def run(self, input_text: str) -> AgentResponse: ...


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Jitter matters under concurrency: without it, every coroutine that
    hit the same 429 sleeps the same duration and stampedes the API
    again in lockstep. The defaults are sized so the cumulative backoff
    (~60s worst case) can ride out a per-minute quota window — the shape
    of free-tier rate limits like Gemini's.
    """

    max_attempts: int = 6
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0

    def delay(self, attempt: int) -> float:
        capped = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return capped * (0.5 + random.random() / 2)


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_agent_output(text: str) -> tuple[Optional[AgentOutput], Optional[str]]:
    """Parse raw agent text into a validated AgentOutput.

    Tolerates exactly one cosmetic deviation -- a markdown code fence
    around the JSON -- because it's a formatting artifact, not a content
    error. Anything else (prose preamble, multiple objects, wrong types)
    is reported as a schema failure: silently salvaging JSON out of chat
    would hide exactly the failure mode this harness exists to measure.
    """
    candidate = text.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} (pos {exc.pos})"
    try:
        return AgentOutput.model_validate(data), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        return None, f"schema violation at '{loc}': {first['msg']}"


def _flakiness(runs: list[RunRecord]) -> float:
    """Share of runs disagreeing with the modal answer (0.0 = stable)."""
    if not runs:
        return 0.0
    buckets: dict[str, int] = {}
    for r in runs:
        key = r.parsed.category.value if r.parsed else "<invalid>"
        buckets[key] = buckets.get(key, 0) + 1
    return round(1.0 - max(buckets.values()) / len(runs), 4)


class Runner:
    """Executes a suite: N runs per case, concurrently, then validates."""

    def __init__(
        self,
        agent: Agent,
        validators: list,
        runs_per_case: int = 3,
        concurrency: int = 5,
        retry_policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        self.agent = agent
        self.validators = validators
        self.runs_per_case = runs_per_case
        self.retry_policy = retry_policy
        # One shared semaphore caps *total* in-flight API calls across all
        # cases -- the unit that rate limits actually care about.
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _run_once(self, case: TestCase) -> RunRecord:
        policy = self.retry_policy
        retryable: tuple = getattr(self.agent, "retryable_exceptions", (Exception,))
        async with self._semaphore:
            for attempt in range(policy.max_attempts):
                start = time.perf_counter()
                try:
                    response = await self.agent.run(case.input)
                except Exception as exc:
                    exhausted = attempt >= policy.max_attempts - 1
                    if exhausted or not isinstance(exc, retryable):
                        # Fatal (auth, bad request) or out of budget:
                        # record, don't raise -- one broken case must not
                        # take down the rest of the suite.
                        return RunRecord(
                            raw_output=None,
                            parsed=None,
                            error=f"{type(exc).__name__}: {exc}",
                            retries=attempt,
                        )
                    # Prefer the provider's own retry-after hint (plus a
                    # little jitter so concurrent waiters don't stampede
                    # the freshly-opened window); fall back to blind
                    # exponential backoff when there is no hint.
                    hint = getattr(exc, "retry_after_s", None)
                    delay = (
                        min(hint + random.random(), policy.max_delay_s)
                        if hint is not None
                        else policy.delay(attempt)
                    )
                    await asyncio.sleep(delay)
                    continue
                latency_ms = (time.perf_counter() - start) * 1000
                parsed, parse_error = parse_agent_output(response.text)
                in_price, out_price = PRICING_USD_PER_MTOK.get(
                    self.agent.model, (0.0, 0.0)
                )
                cost = (
                    response.input_tokens * in_price
                    + response.output_tokens * out_price
                ) / 1_000_000
                return RunRecord(
                    raw_output=response.text,
                    parsed=parsed,
                    error=parse_error,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=round(cost, 6),
                    retries=attempt,
                )
        raise AssertionError("unreachable: loop always returns")

    async def _run_case(self, case: TestCase) -> CaseResult:
        runs = await asyncio.gather(
            *(self._run_once(case) for _ in range(self.runs_per_case))
        )
        runs = list(runs)
        validations = [
            v for v in (val.validate(case, runs) for val in self.validators)
            if v is not None
        ]
        return CaseResult(
            case=case,
            runs=runs,
            validations=validations,
            passed=all(v.passed for v in validations),
            flakiness=_flakiness(runs),
        )

    async def run_suite(
        self,
        cases: list[TestCase],
        on_case_complete: Optional[Callable[[CaseResult], None]] = None,
    ) -> tuple[list[CaseResult], str, str]:
        """Run every case; returns (results, started_at, finished_at).

        Cases run concurrently, but results are returned in suite order so
        reports are stable across runs.
        """
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        async def run_and_notify(case: TestCase) -> CaseResult:
            result = await self._run_case(case)
            if on_case_complete is not None:
                on_case_complete(result)
            return result

        results = list(await asyncio.gather(*(run_and_notify(c) for c in cases)))
        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return results, started_at, finished_at
