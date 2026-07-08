"""Alternative agent under test: transaction classifier on the Anthropic API.

Select with `python -m harness run --agent claude` (requires a paid
ANTHROPIC_API_KEY). The default free-tier agent is `gemini_agent.py`; both
share the same prompt so results are comparable across providers.

Design decisions worth defending:

* **The harness owns retries**, so the SDK's built-in retry is disabled
  (`max_retries=0`). Two retry layers stacked make backoff timing and
  retry counts unobservable; the harness wants to measure them.
* **JSON is requested by prompt, not forced by the API.** Structured
  outputs could guarantee the schema server-side, but format discipline
  is one of the behaviors this harness evaluates — forcing it would blind
  the SchemaValidator to a real production failure mode. Flipping to
  structured outputs later turns that validator into a cheap invariant
  check rather than making it useless.
* **Injection resistance lives in the system prompt**: the transaction
  is wrapped in explicit data delimiters and the prompt states that its
  content is never to be treated as instructions. The adversarial suite
  cases measure how well this actually holds.
"""

from __future__ import annotations

import anthropic
from anthropic import AsyncAnthropic

from harness.models import AgentResponse

from .prompt import SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


class TransactionAgent:
    """Classifies raw bank transaction strings into structured JSON."""

    name = "transaction-classifier"
    model = MODEL

    # Transient failures the harness may retry with backoff. Everything
    # else (401 auth, 400 bad request) fails fast: retrying can't fix it
    # and only hammers the API. InternalServerError covers all >=500,
    # including 529 overloaded.
    retryable_exceptions = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        # Reads ANTHROPIC_API_KEY from the environment by default.
        # max_retries=0: the harness implements (and measures) backoff itself.
        self._client = client or AsyncAnthropic(max_retries=0)

    async def run(self, input_text: str) -> AgentResponse:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"<transaction>{input_text}</transaction>",
                }
            ],
        )
        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        return AgentResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
