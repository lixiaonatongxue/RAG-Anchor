"""Runtime configuration for the backend API.

The default data directory is anchored to the ``backend/`` project directory
rather than the current working directory, so starting uvicorn from a different
directory still uses the same on-disk store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
"""Maximum accepted upload size (20 MiB) unless overridden by the environment."""

UPLOAD_CHUNK_SIZE = 1024 * 1024
"""Number of bytes read from the upload stream per iteration."""

ENV_DATA_DIR = "RAG_ANCHOR_DATA_DIR"
ENV_MAX_UPLOAD_BYTES = "RAG_ANCHOR_MAX_UPLOAD_BYTES"

DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 512

ENV_LLM_ENABLED = "RAG_ANCHOR_LLM_ENABLED"
ENV_LLM_BASE_URL = "RAG_ANCHOR_LLM_BASE_URL"
ENV_LLM_MODEL = "RAG_ANCHOR_LLM_MODEL"
ENV_LLM_API_KEY = "RAG_ANCHOR_LLM_API_KEY"
ENV_LLM_TIMEOUT_SECONDS = "RAG_ANCHOR_LLM_TIMEOUT_SECONDS"
ENV_LLM_MAX_OUTPUT_TOKENS = "RAG_ANCHOR_LLM_MAX_OUTPUT_TOKENS"
ENV_LLM_REASONING_EFFORT = "RAG_ANCHOR_LLM_REASONING_EFFORT"

ALLOWED_LLM_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high"})
"""Values accepted for the optional ``reasoning_effort`` request field."""


def default_data_dir() -> Path:
    """Return the default runtime data directory, independent of the CWD."""
    return Path(__file__).resolve().parents[2] / ".data"


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Connection settings for an OpenAI-compatible model service.

    ``enabled`` defaults to ``False`` so merely providing an address never
    causes an implicit network call. The API key is optional (local inference
    servers often do not require one) and is excluded from ``repr``.
    """

    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS
    reasoning_effort: str = ""

    def __post_init__(self) -> None:
        base_url = self.base_url.strip()
        if base_url:
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{ENV_LLM_BASE_URL} must be an http(s) URL")
        if self.timeout_seconds <= 0:
            raise ValueError(f"{ENV_LLM_TIMEOUT_SECONDS} must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError(f"{ENV_LLM_MAX_OUTPUT_TOKENS} must be a positive integer")
        if self.reasoning_effort and self.reasoning_effort not in ALLOWED_LLM_REASONING_EFFORTS:
            raise ValueError(
                f"{ENV_LLM_REASONING_EFFORT} must be one of {sorted(ALLOWED_LLM_REASONING_EFFORTS)}"
            )

    @property
    def configured(self) -> bool:
        """Whether both the endpoint URL and model name are present."""
        return bool(self.base_url.strip() and self.model.strip())

    @classmethod
    def from_env(cls) -> LLMSettings:
        """Read model connection settings without contacting the service."""
        timeout_raw = os.environ.get(ENV_LLM_TIMEOUT_SECONDS, "").strip()
        try:
            timeout_seconds = float(timeout_raw) if timeout_raw else DEFAULT_LLM_TIMEOUT_SECONDS
        except ValueError as exc:
            raise ValueError(f"{ENV_LLM_TIMEOUT_SECONDS} must be a number") from exc

        tokens_raw = os.environ.get(ENV_LLM_MAX_OUTPUT_TOKENS, "").strip()
        try:
            max_output_tokens = int(tokens_raw) if tokens_raw else DEFAULT_LLM_MAX_OUTPUT_TOKENS
        except ValueError as exc:
            raise ValueError(f"{ENV_LLM_MAX_OUTPUT_TOKENS} must be an integer") from exc

        return cls(
            enabled=_parse_bool(ENV_LLM_ENABLED, os.environ.get(ENV_LLM_ENABLED, "")),
            base_url=os.environ.get(ENV_LLM_BASE_URL, "").strip(),
            model=os.environ.get(ENV_LLM_MODEL, "").strip(),
            api_key=os.environ.get(ENV_LLM_API_KEY, "").strip(),
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            reasoning_effort=os.environ.get(ENV_LLM_REASONING_EFFORT, "").strip().lower(),
        )


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class Settings:
    """Filesystem, upload and optional model service settings."""

    data_dir: Path
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    llm: LLMSettings = LLMSettings()

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be a positive integer")

    @property
    def documents_dir(self) -> Path:
        """Directory holding validated PDF files named after their document ID."""
        return self.data_dir / "documents"

    @property
    def temp_dir(self) -> Path:
        """Directory for in-progress uploads before they are validated."""
        return self.data_dir / "tmp"

    @property
    def database_path(self) -> Path:
        """SQLite database holding document metadata."""
        return self.data_dir / "metadata.sqlite3"

    def ensure_directories(self) -> None:
        """Create the runtime directories if they do not exist yet."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables."""
        raw_data_dir = os.environ.get(ENV_DATA_DIR, "").strip()
        data_dir = Path(raw_data_dir).expanduser() if raw_data_dir else default_data_dir()

        raw_max_bytes = os.environ.get(ENV_MAX_UPLOAD_BYTES, "").strip()
        max_upload_bytes = DEFAULT_MAX_UPLOAD_BYTES
        if raw_max_bytes:
            try:
                max_upload_bytes = int(raw_max_bytes)
            except ValueError as exc:
                raise ValueError(f"{ENV_MAX_UPLOAD_BYTES} must be an integer") from exc

        return cls(
            data_dir=data_dir,
            max_upload_bytes=max_upload_bytes,
            llm=LLMSettings.from_env(),
        )
