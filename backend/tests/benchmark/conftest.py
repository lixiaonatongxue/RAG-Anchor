"""Import path setup for the benchmark harness tests."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent

for entry in (BACKEND_ROOT, TEST_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
