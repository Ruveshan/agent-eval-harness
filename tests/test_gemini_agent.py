"""Provider-mapping tests for the Gemini agent — no network, no key.

The interesting logic in the agent is the transient/fatal split: google-genai
raises one APIError family for both retryable (429/5xx) and fatal (401/400)
statuses, and the agent must translate that into the harness's type-based
retry contract.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from agents.gemini_agent import GeminiTransactionAgent
from harness.runner import TransientAgentError


class _FakeAPIError(genai_errors.APIError):
    """Real APIError type without the real constructor (which wants an
    httpx response object); only the type and `.code` matter here."""

    def __init__(self, code: int) -> None:
        Exception.__init__(self, f"HTTP {code}")
        self.code = code
        self.message = "boom"


def _stub_client(exc: Exception | None = None, response=None):
    async def generate_content(**kwargs):
        if exc is not None:
            raise exc
        return response

    return SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )


def test_rate_limit_is_reraised_as_transient():
    agent = GeminiTransactionAgent(client=_stub_client(exc=_FakeAPIError(429)))
    with pytest.raises(TransientAgentError):
        asyncio.run(agent.run("UBER TRIP"))


def test_server_error_is_reraised_as_transient():
    agent = GeminiTransactionAgent(client=_stub_client(exc=_FakeAPIError(503)))
    with pytest.raises(TransientAgentError):
        asyncio.run(agent.run("UBER TRIP"))


def test_auth_error_propagates_untranslated():
    # A 401 must NOT become transient — the harness would retry it.
    agent = GeminiTransactionAgent(client=_stub_client(exc=_FakeAPIError(401)))
    with pytest.raises(genai_errors.APIError) as excinfo:
        asyncio.run(agent.run("UBER TRIP"))
    assert not isinstance(excinfo.value, TransientAgentError)


def test_success_maps_text_and_usage():
    response = SimpleNamespace(
        text='{"category": "transport", "merchant": "Uber", "confidence": 0.95}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=450, candidates_token_count=30
        ),
    )
    agent = GeminiTransactionAgent(client=_stub_client(response=response))
    result = asyncio.run(agent.run("UBER TRIP"))
    assert '"transport"' in result.text
    assert result.input_tokens == 450 and result.output_tokens == 30


def test_empty_response_text_becomes_empty_string():
    # response.text is None when the model returns no candidates — the
    # agent must hand the harness a string, never None.
    response = SimpleNamespace(text=None, usage_metadata=None)
    agent = GeminiTransactionAgent(client=_stub_client(response=response))
    result = asyncio.run(agent.run("UBER TRIP"))
    assert result.text == "" and result.input_tokens == 0
