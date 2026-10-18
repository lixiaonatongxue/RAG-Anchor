"""Runtime configuration for the backend API.

The default data directory is anchored to the ``backend/`` project directory
rather than the current working directory, so starting uvicorn from a different
directory still uses the same on-disk store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
"""Maximum accepted upload size (20 MiB) unless overridden by the environment."""

UPLOAD_CHUNK_SIZE = 1024 * 1024
"""Number of bytes read from the upload stream per iteration."""

ENV_DATA_DIR = "RAG_ANCHOR_DATA_DIR"
ENV_MAX_UPLOAD_BYTES = "RAG_ANCHOR_MAX_UPLOAD_BYTES"


def default_data_dir() -> Path:
    """Return the default runtime data directory, independent of the CWD."""
    return Path(__file__).resolve().parents[2] / ".data"


@dataclass(frozen=True, slots=True)
class Settings:
    """Filesystem locations and upload limits for the document API."""

    data_dir: Path
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

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

        return cls(data_dir=data_dir, max_upload_bytes=max_upload_bytes)
