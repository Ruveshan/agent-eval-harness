"""Default agent under test: transaction classifier on Google Gemini.

Chosen as the default because Gemini has a genuinely free API tier — get a
key at https://aistudio.google.com/apikey (no card required) and export it
as GEMINI_API_KEY. The trade-off: free-tier rate limits are low (roughly
10–15 requests/min), so run the suite with `--concurrency 1` and let the
harness's backoff absorb the 429s.

Provider-mapping note worth defending: google-genai raises one exception
family (`APIError`) for everything, distinguishing a retryable 429 from a
fatal 401 only by `.code`. The harness's retry contract is type-based, so
this agent re-raises transient codes as `TransientAgentError` — the agent
owns the knowledge of what's transient on its transport, the harness stays
provider-agnostic.

Like the Claude agent, JSON is requested by prompt rather than forced via
Gemini's JSON mode (`response_mime_type`), because format discipline is one
of the behaviors under test. Thinking is disabled: a classifier doesn't
need it, and thought tokens would eat the small output budget.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from harness.models import AgentResponse
from harness.runner import TransientAgentError

from .prompt import SYSTEM_PROMPT

# gemini-2.5-flash-lite has the friendliest free-tier limits; override via
# env for a stronger model, e.g. GEMINI_MODEL=gemini-2.5-flash.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# HTTP statuses that a retry can plausibly fix. Everything else (401 bad
# key, 400 malformed request) fails fast.
_TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


class GeminiTransactionAgent:
    """Classifies raw bank transaction strings into structured JSON."""

    name = "transaction-classifier"
    model = MODEL

    retryable_exceptions = (TransientAgentError, ConnectionError)

    def __init__(self, client: genai.Client | None = None) -> None:
        # genai.Client() reads GEMINI_API_KEY / GOOGLE_API_KEY from the
        # environment and raises at construction if neither is set.
        self._client = client or genai.Client()

    async def run(self, input_text: str) -> AgentResponse:
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=f"<transaction>{input_text}</transaction>",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except genai_errors.APIError as exc:
            if exc.code in _TRANSIENT_CODES:
                raise TransientAgentError(f"HTTP {exc.code}: {exc.message}") from exc
            raise
        usage = response.usage_metadata
        return AgentResponse(
            text=response.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
