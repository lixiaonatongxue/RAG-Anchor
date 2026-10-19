"""Deterministic PDF evidence anchoring benchmark (M0-06B1).

The package is executed as ``python -m benchmarks.cli`` from ``backend/``.
Truth generation (``truth.py``), fixture generation (``generate.py``),
matching (``match.py``), metrics (``metrics.py``), and reporting
(``report.py``) must never import ``pypdfium2`` or ``rag_anchor``. Only
``runner.py`` adapts the production ``search_anchors()`` service.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
