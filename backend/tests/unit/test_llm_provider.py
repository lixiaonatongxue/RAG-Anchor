"""Offline tests for the OpenAI-compatible provider.

Every exchange uses an injected fake transport; no socket is opened and no
model service is contacted.
"""

from __future__ import annotations

import json

import pytest

from rag_anchor.config import (
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_ENABLED,
    ENV_LLM_MAX_OUTPUT_TOKENS,
    ENV_LLM_MODEL,
    ENV_LLM_REASONING_EFFORT,
    ENV_LLM_TIMEOUT_SECONDS,
    LLMSettings,
    Settings,
)
from rag_anchor.infra.llm import (
    DisabledLLMProvider,
    HTTPResponse,
    HTTPTimeoutError,
    HTTPTransportError,
    LLMDisabledError,
    LLMMessage,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    OpenAICompatibleLLMProvider,
    build_llm_provider,
)


class RecordingTransport:
    """A fake HTTP transport that records the request instead of sending it."""

    def __init__(
        self,
        response: HTTPResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def post(self, url, *, headers, body, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def get(self, url, *, headers, timeout):
        raise AssertionError("The LLM provider must not issue GET requests")


def _provider(transport: RecordingTransport) -> OpenAICompatibleLLMProvider:
    return OpenAICompatibleLLMProvider(
        base_url="http://127.0.0.1:65535/v1",
        model="qwen-test",
        api_key="test-secret-key",
        timeout_seconds=7.5,
        max_output_tokens=321,
        transport=transport,
    )


def _assistant_body(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode(
        "utf-8"
    )


def _messages() -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content="system instructions"),
        LLMMessage(role="user", content='{"question": "q"}'),
    ]


def test_provider_sends_expected_request_and_returns_content() -> None:
    transport = RecordingTransport(
        HTTPResponse(status=200, body=_assistant_body('{"answer": "ok", "citation_ids": ["E1"]}'))
    )

    result = _provider(transport).complete(_messages())

    assert result == '{"answer": "ok", "citation_ids": ["E1"]}'
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "http://127.0.0.1:65535/v1/chat/completions"
    assert call["timeout"] == 7.5
    headers = call["headers"]
    assert headers["Authorization"] == "Bearer test-secret-key"
    payload = json.loads(call["body"])
    assert payload["model"] == "qwen-test"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 321
    assert payload["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in payload
    assert payload["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": '{"question": "q"}'},
    ]


def test_provider_sends_configured_reasoning_effort() -> None:
    transport = RecordingTransport(HTTPResponse(status=200, body=_assistant_body("ok")))
    provider = OpenAICompatibleLLMProvider(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        reasoning_effort="none",
        transport=transport,
    )

    provider.complete(_messages())

    payload = json.loads(transport.calls[0]["body"])
    assert payload["reasoning_effort"] == "none"


def test_provider_omits_authorization_when_key_is_empty() -> None:
    transport = RecordingTransport(HTTPResponse(status=200, body=_assistant_body("ok")))
    provider = OpenAICompatibleLLMProvider(
        base_url="http://llm.test/v1/",
        model="qwen-test",
        transport=transport,
    )

    provider.complete(_messages())

    assert transport.calls[0]["url"] == "http://llm.test/v1/chat/completions"
    assert "Authorization" not in transport.calls[0]["headers"]


def test_provider_http_error_does_not_leak_body_or_key() -> None:
    transport = RecordingTransport(
        HTTPResponse(status=500, body=b"upstream detail test-secret-key")
    )

    with pytest.raises(LLMServiceError) as excinfo:
        _provider(transport).complete(_messages())

    message = str(excinfo.value)
    assert "500" in message
    assert "upstream detail" not in message
    assert "test-secret-key" not in message


def test_provider_maps_transport_timeout_to_timeout_error() -> None:
    transport = RecordingTransport(error=HTTPTimeoutError("socket timed out"))

    with pytest.raises(LLMTimeoutError):
        _provider(transport).complete(_messages())


def test_provider_maps_transport_failure_to_service_error() -> None:
    transport = RecordingTransport(error=HTTPTransportError("connection refused"))

    with pytest.raises(LLMServiceError) as excinfo:
        _provider(transport).complete(_messages())
    assert not isinstance(excinfo.value, LLMTimeoutError)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"choices": []}',
        b'{"choices": [{}]}',
        b'{"choices": [{"message": {}}]}',
        b'{"choices": [{"message": {"content": "   "}}]}',
    ],
)
def test_provider_rejects_unusable_response_envelopes(body: bytes) -> None:
    transport = RecordingTransport(HTTPResponse(status=200, body=body))

    with pytest.raises(LLMResponseError):
        _provider(transport).complete(_messages())


def test_disabled_provider_never_pretends_to_answer() -> None:
    provider = DisabledLLMProvider("not configured")

    with pytest.raises(LLMDisabledError, match="not configured"):
        provider.complete(_messages())


def test_factory_requires_explicit_enable_and_complete_configuration() -> None:
    disabled = build_llm_provider(
        LLMSettings(
            enabled=False,
            base_url="http://llm.test/v1",
            model="qwen-test",
        )
    )
    incomplete = build_llm_provider(LLMSettings(enabled=True, base_url="", model=""))
    enabled = build_llm_provider(
        LLMSettings(enabled=True, base_url="http://llm.test/v1", model="qwen-test")
    )

    assert isinstance(disabled, DisabledLLMProvider)
    assert isinstance(incomplete, DisabledLLMProvider)
    assert isinstance(enabled, OpenAICompatibleLLMProvider)


def test_settings_from_env_parses_llm_options_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LLM_ENABLED, "true")
    monkeypatch.setenv(ENV_LLM_BASE_URL, "http://127.0.0.1:9000/v1")
    monkeypatch.setenv(ENV_LLM_MODEL, "Qwen3.5-27B-GPTQ-Int4")
    monkeypatch.setenv(ENV_LLM_API_KEY, "local-secret")
    monkeypatch.setenv(ENV_LLM_TIMEOUT_SECONDS, "12.5")
    monkeypatch.setenv(ENV_LLM_MAX_OUTPUT_TOKENS, "256")
    monkeypatch.setenv(ENV_LLM_REASONING_EFFORT, "NONE")

    settings = Settings.from_env()

    assert settings.llm.enabled is True
    assert settings.llm.base_url == "http://127.0.0.1:9000/v1"
    assert settings.llm.model == "Qwen3.5-27B-GPTQ-Int4"
    assert settings.llm.api_key == "local-secret"
    assert settings.llm.timeout_seconds == 12.5
    assert settings.llm.max_output_tokens == 256
    assert settings.llm.reasoning_effort == "none"
    assert "local-secret" not in repr(settings.llm)


def test_llm_settings_reject_a_model_filesystem_path_as_base_url() -> None:
    with pytest.raises(ValueError, match="http"):
        LLMSettings(
            enabled=True,
            base_url="/models/Qwen3.5-27B-GPTQ-Int4",
            model="qwen",
        )


def test_llm_settings_reject_invalid_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LLM_ENABLED, "sometimes")
    with pytest.raises(ValueError, match=ENV_LLM_ENABLED):
        LLMSettings.from_env()

    monkeypatch.setenv(ENV_LLM_ENABLED, "false")
    monkeypatch.setenv(ENV_LLM_TIMEOUT_SECONDS, "soon")
    with pytest.raises(ValueError, match=ENV_LLM_TIMEOUT_SECONDS):
        LLMSettings.from_env()

    monkeypatch.delenv(ENV_LLM_TIMEOUT_SECONDS)
    monkeypatch.setenv(ENV_LLM_MAX_OUTPUT_TOKENS, "many")
    with pytest.raises(ValueError, match=ENV_LLM_MAX_OUTPUT_TOKENS):
        LLMSettings.from_env()

    monkeypatch.delenv(ENV_LLM_MAX_OUTPUT_TOKENS)
    monkeypatch.setenv(ENV_LLM_REASONING_EFFORT, "maximum")
    with pytest.raises(ValueError, match=ENV_LLM_REASONING_EFFORT):
        LLMSettings.from_env()
