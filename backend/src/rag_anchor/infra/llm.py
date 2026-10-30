"""Minimal OpenAI-compatible chat completion transport.

No request is made at import time, at application startup, or by merely
configuring an address. A request is only sent when grounded generation is
explicitly enabled *and* requested, and both the endpoint and model name are
present.

The built-in transport uses only the standard library. It disables ambient
proxy configuration and HTTP redirects so a local endpoint cannot be silently
redirected, and it never logs or embeds the API key in exception messages.
Response bodies are not copied into errors, so upstream detail cannot leak
into HTTP responses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

from rag_anchor.config import (
    ALLOWED_LLM_REASONING_EFFORTS,
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    LLMSettings,
)

MessageRole = Literal["system", "user"]


class LLMError(RuntimeError):
    """Base class for model-provider failures."""


class LLMDisabledError(LLMError):
    """Grounded generation is disabled or not configured (HTTP 503)."""


class LLMServiceError(LLMError):
    """The model service could not be reached or returned an error (HTTP 502)."""


class LLMTimeoutError(LLMServiceError):
    """The model request exceeded the configured timeout (HTTP 504)."""


class LLMResponseError(LLMError):
    """The model service returned an unusable response envelope (HTTP 502)."""


class HTTPTransportError(RuntimeError):
    """The transport could not complete an HTTP exchange."""


class HTTPTimeoutError(HTTPTransportError):
    """The transport timed out."""


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    """The status and body returned by an injected HTTP transport."""

    status: int
    body: bytes


class HTTPTransport(Protocol):
    """The tiny HTTP surface needed by the providers, injectable for tests."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse:
        """GET ``url`` and return the response status and bytes."""
        ...

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HTTPResponse:
        """POST ``body`` and return the response status and bytes."""
        ...


class _NoRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Refuse redirects so credentials cannot be forwarded to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibHTTPTransport:
    """Standard-library HTTP transport with proxies disabled."""

    def __init__(self) -> None:
        self._opener = urlrequest.build_opener(
            urlrequest.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse:
        request = urlrequest.Request(
            url,
            headers=dict(headers),
            method="GET",
        )
        return self._perform(request, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HTTPResponse:
        request = urlrequest.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        return self._perform(request, timeout=timeout)

    def _perform(self, request: urlrequest.Request, *, timeout: float) -> HTTPResponse:
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return HTTPResponse(status=response.getcode(), body=response.read())
        except urlerror.HTTPError as exc:
            status = exc.code
            exc.close()
            return HTTPResponse(status=status, body=b"")
        except TimeoutError as exc:
            raise HTTPTimeoutError("HTTP request timed out") from exc
        except urlerror.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise HTTPTimeoutError("HTTP request timed out") from exc
            raise HTTPTransportError("HTTP request failed") from exc
        except (OSError, ValueError) as exc:
            raise HTTPTransportError("HTTP request failed") from exc


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One chat message. Application instructions use the ``system`` role."""

    role: MessageRole
    content: str


class LLMProvider(Protocol):
    """The single capability the QA service needs from a model backend."""

    def complete(self, messages: Sequence[LLMMessage]) -> str:
        """Return the raw assistant text for ``messages``."""
        ...


class DisabledLLMProvider:
    """A provider that fails loudly instead of pretending a model answered."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def complete(self, messages: Sequence[LLMMessage]) -> str:
        raise LLMDisabledError(self._reason)


class OpenAICompatibleLLMProvider:
    """Call ``POST {base_url}/chat/completions`` with an explicit timeout.

    ``transport`` exists for offline tests via a fake transport. The production
    path uses :class:`UrllibHTTPTransport`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        reasoning_effort: str = "",
        transport: HTTPTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if reasoning_effort and reasoning_effort not in ALLOWED_LLM_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(ALLOWED_LLM_REASONING_EFFORTS)}"
            )

        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._transport: HTTPTransport = transport or UrllibHTTPTransport()

    def complete(self, messages: Sequence[LLMMessage]) -> str:
        if not messages:
            raise ValueError("messages must not be empty")

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            # The generation contract is one JSON object; constrain decoding to it.
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            response = self._transport.post(
                self._endpoint,
                headers=headers,
                body=body,
                timeout=self._timeout_seconds,
            )
        except HTTPTimeoutError as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except HTTPTransportError as exc:
            raise LLMServiceError("LLM service request failed") from exc

        if not 200 <= response.status < 300:
            raise LLMServiceError(f"LLM service returned HTTP {response.status}")

        try:
            parsed = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError("LLM response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise LLMResponseError("LLM response was not a JSON object")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("LLM response did not contain any choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseError("LLM response contained an invalid choice")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("LLM response contained no assistant message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM response contained no text")
        return content


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    """Build the configured provider without probing the network."""
    if not settings.enabled:
        return DisabledLLMProvider(
            "Grounded generation is disabled; set RAG_ANCHOR_LLM_ENABLED=true "
            "after the model service is verified"
        )
    if not settings.configured:
        return DisabledLLMProvider(
            "Grounded generation requires RAG_ANCHOR_LLM_BASE_URL and RAG_ANCHOR_LLM_MODEL"
        )
    return OpenAICompatibleLLMProvider(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_output_tokens=settings.max_output_tokens,
        reasoning_effort=settings.reasoning_effort,
    )


__all__ = [
    "DisabledLLMProvider",
    "HTTPResponse",
    "HTTPTimeoutError",
    "HTTPTransport",
    "HTTPTransportError",
    "LLMDisabledError",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMResponseError",
    "LLMServiceError",
    "LLMTimeoutError",
    "OpenAICompatibleLLMProvider",
    "UrllibHTTPTransport",
    "build_llm_provider",
]
