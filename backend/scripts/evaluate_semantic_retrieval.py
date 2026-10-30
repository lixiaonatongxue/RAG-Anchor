#!/usr/bin/env python3
"""Repeatable M2-01 retrieval evaluation over a labelled synthetic PDF.

The harness never downloads a model and never invents metrics. ``--provider
auto`` runs only when the embedding service is configured and enabled through
the documented environment variables; otherwise it reports ``unexecuted``.
``--provider fake`` is an engineering-only smoke run with deterministic
keyword vectors and must not be presented as embedding quality.

Measured per query: literal Recall@K, semantic Recall@K, top-1 relevance,
semantic citation locatability, retrieval latency. The fixed dataset must hold
at least 30 labelled queries, at least 10 of them unanswerable, and cover every
required failure category. Unanswerable queries are excluded from Recall@K and
reported separately.

The threshold sweep is derived offline from one low-floor semantic search per
query; it adds no embeddings and never changes the configured production
threshold used by QA. Measured once: semantic index build time and
vector/database size.

``--answer-mode`` defaults to ``evidence_only``, preserving the existing
no-model behavior. ``grounded_generation`` additionally requires an enabled
local OpenAI-compatible model service; it records model calls, generated
answers, and independently re-checked citation IDs/spans. Citation validity and
PDF locatability prove provenance, not factual support.

Run from ``backend/``::

    uv run python scripts/evaluate_semantic_retrieval.py --provider fake
    RAG_ANCHOR_EMBEDDING_ENABLED=1 \
    RAG_ANCHOR_EMBEDDING_BASE_URL=http://127.0.0.1:11434 \
    RAG_ANCHOR_EMBEDDING_MODEL=bge-m3 \
    uv run python scripts/evaluate_semantic_retrieval.py --provider auto
    RAG_ANCHOR_LLM_ENABLED=1 \
    RAG_ANCHOR_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
    RAG_ANCHOR_LLM_MODEL=qwen3:8b \
    RAG_ANCHOR_LLM_REASONING_EFFORT=none \
    ... --provider auto --answer-mode grounded_generation
"""

# User-facing evaluation notes use full-width Chinese punctuation on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from rag_anchor.config import EmbeddingSettings, LLMSettings, Settings
from rag_anchor.domain.text import TextChunk
from rag_anchor.infra.embedding import EmbeddingProvider
from rag_anchor.infra.llm import LLMMessage, LLMProvider, build_llm_provider
from rag_anchor.main import create_app
from rag_anchor.services.qa_service import (
    _SYSTEM_PROMPT,
    ANSWER_MODE_EVIDENCE_ONLY,
    ANSWER_MODE_GROUNDED_GENERATION,
)
from rag_anchor.services.semantic_service import (
    MAX_TOP_K,
    SemanticIndexService,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = BACKEND_ROOT / "evaluation" / "semantic_dataset.json"
DOCUMENTS_URL = "/api/v1/documents"

MIN_QUERY_COUNT = 30
MIN_UNANSWERABLE_COUNT = 10
REQUIRED_CATEGORIES = frozenset(
    {
        "zh_to_en",
        "en_paraphrase",
        "cross_page",
        "partial_candidates",
        "keyword_distractor",
        "different_conclusion",
        "unanswerable",
        "recall_vs_top1",
    }
)
THRESHOLD_SWEEP = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
SEMANTIC_SCORE_FLOOR = -1.0
"""Lowest accepted cosine score; used once per query so thresholds are free."""

ANSWER_MODES = (ANSWER_MODE_EVIDENCE_ONLY, ANSWER_MODE_GROUNDED_GENERATION)

PROMPT_VARIANT_A = "A"
PROMPT_VARIANT_B = "B"
PROMPT_VARIANTS = (PROMPT_VARIANT_A, PROMPT_VARIANT_B)
PROMPT_RULES_CHANGED_IN_B = (7, 12)

_B_RULE_7 = (
    "7. Answer only what the question actually asks. Put only evidence that directly "
    "supports the asked content into a claim. If supplied evidence is true but not "
    "required by the question, put it in unused_evidence instead of a claim; never "
    "create an extra claim merely to complete evidence accounting. Evidence not used "
    "by any claim must still be accounted for under rule 12. If the question asks for "
    "several parts, cover each asked part that the evidence directly supports, and "
    "list any uncovered required part in limitations. Keep entities, objects, actions, "
    "quantities, time periods, and negation exactly as written; never swap or merge "
    "them. If the evidence is insufficient to answer the question, refuse.\n"
)

_B_RULE_12 = (
    "12. Output exactly one JSON object, no Markdown fence and no extra text. "
    "For an answer use shape "
    '{"answerable": true, "claims": [{"text": "...", "citation_ids": ["E1"]}], '
    '"unused_evidence": [{"citation_id": "E2", "reason": "..."}], '
    '"limitations": ["..."]}. '
    "Every claim must directly answer the question and be a substantive statement "
    "supported by its non-empty citation_ids. Every supplied evidence id must appear "
    "exactly once, either in a claim's citation_ids or in unused_evidence with a short "
    "reason; evidence that does not answer the question belongs in unused_evidence, "
    "never in a claim merely to complete accounting, and no supplied evidence may be "
    "silently dropped. limitations lists question parts the evidence does not cover "
    "and may be empty. For a refusal use exactly "
    '{"answerable": false, "claims": [], "unused_evidence": [], "limitations": []}. '
    "Do not put prose outside claims or limitations.\n"
)


class _CountingLLMProvider:
    """Count model calls and their latency without changing provider behavior."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.calls = 0
        self.durations: list[float] = []

    def complete(self, messages: Sequence[LLMMessage]) -> str:
        started = time.perf_counter()
        try:
            return self._provider.complete(messages)
        finally:
            self.calls += 1
            self.durations.append(time.perf_counter() - started)


def _replace_prompt_rule(prompt: str, number: int, replacement: str) -> str:
    """Replace exactly one numbered prompt rule, leaving every other line intact."""
    prefix = f"{number}. "
    lines = prompt.splitlines(keepends=True)
    indices = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(indices) != 1:
        raise ValueError(f"system prompt must contain exactly one rule {number}")
    if not replacement.startswith(prefix):
        raise ValueError(f"replacement for rule {number} must keep its number")
    lines[indices[0]] = replacement if replacement.endswith("\n") else replacement + "\n"
    return "".join(lines)


def _prompt_for_variant(variant: str) -> str:
    """Return the production prompt (A) or the evaluation-only reworded prompt (B).

    B is derived from A by replacing only rules 7 and 12, so no other system
    instruction, output contract, or server validation changes.
    """
    if variant == PROMPT_VARIANT_A:
        return _SYSTEM_PROMPT
    if variant == PROMPT_VARIANT_B:
        prompt = _replace_prompt_rule(_SYSTEM_PROMPT, 7, _B_RULE_7)
        return _replace_prompt_rule(prompt, 12, _B_RULE_12)
    raise ValueError(f"prompt_variant must be one of {list(PROMPT_VARIANTS)}")


def _prompt_metadata(variant: str) -> dict:
    prompt = _prompt_for_variant(variant)
    return {
        "variant": variant,
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "chars": len(prompt),
        "changed_rules": list(PROMPT_RULES_CHANGED_IN_B) if variant == PROMPT_VARIANT_B else [],
        "system_prompt": prompt,
    }


class _PromptVariantProvider:
    """Evaluation-only wrapper that swaps the system prompt and records raw output.

    The wrapped provider still performs the real request and the server still
    validates the response exactly as in production. Recording the raw text is
    diagnostic only; it never changes what is returned to the QA service.
    """

    def __init__(self, provider: LLMProvider, system_prompt: str) -> None:
        self._provider = provider
        self.system_prompt = system_prompt
        self.outputs: list[str] = []

    def complete(self, messages: Sequence[LLMMessage]) -> str:
        replaced = 0
        updated: list[LLMMessage] = []
        for message in messages:
            if message.role == "system":
                replaced += 1
                updated.append(LLMMessage(role="system", content=self.system_prompt))
            else:
                updated.append(message)
        if replaced != 1:
            raise ValueError("expected exactly one system message to replace")
        output = self._provider.complete(updated)
        self.outputs.append(output)
        return output


def _inspect_model_output(outputs: list[str], allowed_ids: set[str]) -> dict | None:
    """Summarize the last raw model response for diagnosis, never for scoring.

    Parsing here mirrors the server's envelope only loosely. The server remains
    the sole authority for accepting or rejecting a response. Since the server
    now computes unused evidence itself, ``unaccounted_ids`` describes only the
    raw model text (typically a legacy ``unused_evidence`` field) and does not
    imply rejection; these counts are diagnostic, never used for scoring.
    """
    if not outputs:
        return None
    raw = outputs[-1].strip()
    summary: dict = {
        "calls": len(outputs),
        "json_ok": False,
        "answerable": None,
        "claims": None,
        "unused_evidence": None,
        "limitations": None,
        "cited_ids": [],
        "unused_ids": [],
        "unknown_cited_ids": [],
        "unaccounted_ids": [],
    }
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(raw)
    except ValueError:
        return summary
    if not isinstance(payload, dict):
        return summary
    summary["json_ok"] = True

    answerable = payload.get("answerable")
    summary["answerable"] = answerable if isinstance(answerable, bool) else None

    claims = payload.get("claims")
    if isinstance(claims, list):
        summary["claims"] = len(claims)
        cited: list[str] = []
        for claim in claims:
            if isinstance(claim, dict):
                citation_ids = claim.get("citation_ids")
                if isinstance(citation_ids, list):
                    cited.extend(item for item in citation_ids if isinstance(item, str))
        summary["cited_ids"] = cited

    unused = payload.get("unused_evidence")
    if isinstance(unused, list):
        summary["unused_evidence"] = len(unused)
        summary["unused_ids"] = [
            item["citation_id"]
            for item in unused
            if isinstance(item, dict) and isinstance(item.get("citation_id"), str)
        ]

    limitations = payload.get("limitations")
    if isinstance(limitations, list):
        summary["limitations"] = len(limitations)

    summary["unknown_cited_ids"] = sorted(set(summary["cited_ids"]) - allowed_ids)
    summary["unaccounted_ids"] = sorted(
        allowed_ids - set(summary["cited_ids"]) - set(summary["unused_ids"])
    )
    return summary


def _verify_citations(
    citations: list[dict],
    *,
    allowed_ids: set[str],
    page_texts: dict[int, str],
    chunks: dict[str, TextChunk],
    document_id: UUID,
    relevant_keys: set[tuple[int, int, int]],
) -> dict:
    """Independently re-check the server-owned citation fields.

    The API already rebuilds citations from stored rows, so a failure here is a
    contract regression. ID validity and span locatability are reported apart
    from label alignment: a locatable span can still fail to answer the
    question, and only ``citations_on_relevant_span`` is a label proxy.
    """
    seen: set[str] = set()
    invalid_ids: list[str] = []
    duplicate_ids: list[str] = []
    content_invalid_ids: list[str] = []
    on_relevant = 0

    for citation in citations:
        citation_id = str(citation.get("citation_id", ""))
        if citation_id not in allowed_ids:
            invalid_ids.append(citation_id)
        if citation_id in seen:
            duplicate_ids.append(citation_id)
        seen.add(citation_id)

        page_index = citation.get("page_index")
        start = citation.get("support_start")
        end = citation.get("support_end")
        support_text = citation.get("support_text")
        page_text = page_texts.get(page_index) if isinstance(page_index, int) else None
        chunk = chunks.get(str(citation.get("chunk_id", "")))

        content_ok = (
            page_text is not None
            and isinstance(start, int)
            and isinstance(end, int)
            and end > start
            and page_text[start:end] == support_text
            and citation.get("hit_text") == support_text
            and chunk is not None
            and chunk.document_id == str(document_id)
            and chunk.page_index == page_index
            and chunk.page_number == citation.get("page_number")
            and chunk.start_offset <= start
            and end <= chunk.end_offset
            and chunk.text[start - chunk.start_offset : end - chunk.start_offset] == support_text
        )
        if not content_ok:
            content_invalid_ids.append(citation_id)
        elif (page_index, start, end) in relevant_keys:
            on_relevant += 1

    return {
        "ids_checked": len(citations),
        "ids_valid": len(citations) - len(invalid_ids),
        "invalid_ids": invalid_ids,
        "duplicate_ids": duplicate_ids,
        "content_checked": len(citations),
        "content_verified": len(citations) - len(content_invalid_ids),
        "content_invalid_ids": content_invalid_ids,
        "citations_on_relevant_span": on_relevant,
    }


def _generation_metrics(rows: list[dict], answer_mode: str) -> dict:
    """Separate retrieval candidates, generated answers and label alignment.

    Answerable and unanswerable rows have opposite expectations: answering an
    unanswerable query is a false positive, while refusing an answerable one is
    a miss. ``citations_on_relevant_span`` only means the cited sentence is one
    of the hand-labelled relevant sentences; it is not entailment.
    """

    def summarize(subset: list[dict]) -> dict:
        total = len(subset)

        def rate(count: int) -> float | None:
            return round(count / total, 4) if total else None

        answered = sum(1 for row in subset if row["qa"]["status"] == "answered")
        insufficient = sum(1 for row in subset if row["qa"]["status"] == "insufficient_evidence")
        evidence_found = sum(1 for row in subset if row["qa"]["status"] == "evidence_found")
        errors = sum(1 for row in subset if row["qa"]["status_code"] != 200)
        answered_with_body = sum(
            1
            for row in subset
            if row["qa"]["status"] == "answered" and (row["qa"]["answer"] or "").strip()
        )
        answers_with_citations = sum(
            1 for row in subset if row["qa"]["status"] == "answered" and row["qa"]["citations"] > 0
        )
        answers_with_valid_citations = sum(
            1
            for row in subset
            if row["qa"]["status"] == "answered"
            and row["qa"]["citations"] > 0
            and not row["qa"]["invalid_ids"]
            and not row["qa"]["duplicate_ids"]
        )
        answers_citing_relevant = sum(
            1
            for row in subset
            if row["qa"]["status"] == "answered" and row["qa"]["citations_on_relevant_span"] > 0
        )
        citations = sum(row["qa"]["citations"] for row in subset)
        on_relevant = sum(row["qa"]["citations_on_relevant_span"] for row in subset)

        return {
            "count": total,
            "answered": answered,
            "answered_rate": rate(answered),
            "insufficient_evidence": insufficient,
            "insufficient_evidence_rate": rate(insufficient),
            "evidence_found": evidence_found,
            "errors": errors,
            "answered_with_body": answered_with_body,
            "server_notice_answers": sum(
                1
                for row in subset
                if row["qa"]["status"] == "insufficient_evidence"
                and (row["qa"]["answer"] or "").strip()
            ),
            "answers_with_citations": answers_with_citations,
            "answers_with_valid_citations": answers_with_valid_citations,
            "answers_citing_labelled_relevant_span": answers_citing_relevant,
            "citations": citations,
            "citations_on_labelled_relevant_span": on_relevant,
            "citations_off_labelled_relevant_span": citations - on_relevant,
        }

    answerable = [row for row in rows if row["relevant_count"] > 0]
    unanswerable = [row for row in rows if row["relevant_count"] == 0]
    ids_checked = sum(row["qa"]["ids_checked"] for row in rows)
    ids_valid = sum(row["qa"]["ids_valid"] for row in rows)
    content_checked = sum(row["qa"]["content_checked"] for row in rows)
    content_verified = sum(row["qa"]["content_verified"] for row in rows)
    return {
        "answer_mode": answer_mode,
        "answerable": summarize(answerable),
        "unanswerable": summarize(unanswerable),
        "citation_ids": {
            "checked": ids_checked,
            "valid": ids_valid,
            "invalid": ids_checked - ids_valid,
            "duplicates": sum(len(row["qa"]["duplicate_ids"]) for row in rows),
        },
        "citation_content": {
            "checked": content_checked,
            "verified": content_verified,
            "invalid": content_checked - content_verified,
        },
    }


def _load_fake_provider() -> EmbeddingProvider:
    """Load the test-only deterministic provider without importing test packages."""
    path = BACKEND_ROOT / "tests" / "unit" / "semantic_fakes.py"
    spec = importlib.util.spec_from_file_location("_semantic_fakes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Fake embedding provider is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FakeEmbeddingProvider()


def _write_pdf(pages: list[dict]) -> bytes:
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer)
    for page in pages:
        y = 740.0
        for line in page["lines"]:
            document.drawString(72.0, y, line)
            y -= 24.0
        document.showPage()
    document.save()
    return buffer.getvalue()


def _resolve_spans(
    page_texts: dict[int, str],
    labelled: list[dict],
) -> list[tuple[int, int, int, str]]:
    resolved: list[tuple[int, int, int, str]] = []
    for span in labelled:
        page_index = int(span["page_index"])
        text = str(span["text"])
        page_text = page_texts[page_index]
        start = page_text.find(text)
        if start < 0:
            raise ValueError(f"Label not found on page {page_index}: {text!r}")
        resolved.append((page_index, start, start + len(text), text))
    return resolved


def _validate_dataset(dataset: dict) -> None:
    """Reject labels that are ambiguous, incomplete or too small to compare.

    The source text and every label are fixed before a harness run, so these
    checks fail the run instead of letting a typo or a shrunken dataset
    silently change the denominators.
    """
    pages = dataset.get("pages")
    queries = dataset.get("queries")
    if not isinstance(pages, list) or not pages:
        raise ValueError("dataset.pages must be a non-empty list")
    if not isinstance(queries, list):
        raise ValueError("dataset.queries must be a list")

    legend = dataset.get("category_legend")
    if not isinstance(legend, dict):
        raise ValueError("dataset.category_legend must be an object")
    missing_legend = REQUIRED_CATEGORIES - set(legend)
    if missing_legend:
        raise ValueError(f"category_legend is missing: {sorted(missing_legend)}")

    page_texts: dict[int, str] = {}
    for position, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError("every page must be an object")
        page_index = page.get("page_index")
        if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
            raise ValueError("page_index must be a non-negative integer")
        if page_index != position:
            raise ValueError("pages must be ordered by contiguous page_index")
        if page_index in page_texts:
            raise ValueError(f"duplicate page_index: {page_index}")
        lines = page.get("lines")
        if (
            not isinstance(lines, list)
            or not lines
            or not all(isinstance(line, str) and line.strip() for line in lines)
        ):
            raise ValueError(f"page {page_index} must have non-empty text lines")
        page_texts[page_index] = "\n".join(lines)

    def validate_spans(query_id: str, kind: str, spans: object) -> set[tuple[int, str]]:
        if not isinstance(spans, list):
            raise ValueError(f"{query_id}.{kind} must be a list")
        keys: set[tuple[int, str]] = set()
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError(f"{query_id}.{kind} spans must be objects")
            page_index = span.get("page_index")
            text = span.get("text")
            if isinstance(page_index, bool) or not isinstance(page_index, int):
                raise ValueError(f"{query_id}.{kind} page_index must be an integer")
            if page_index not in page_texts:
                raise ValueError(f"{query_id}.{kind} references missing page {page_index}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{query_id}.{kind} text must be non-empty")
            if page_texts[page_index].count(text) != 1:
                raise ValueError(
                    f"{query_id}.{kind} text must occur exactly once on page {page_index}"
                )
            key = (page_index, text)
            if key in keys:
                raise ValueError(f"{query_id}.{kind} contains a duplicate span")
            keys.add(key)
        return keys

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    covered: set[str] = set()
    unanswerable = 0

    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("every query must be an object")
        query_id = query.get("id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("every query needs a non-empty id")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query id: {query_id}")
        seen_ids.add(query_id)

        category = query.get("category")
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"{query_id} has unknown category: {category!r}")
        covered.add(category)

        question = query.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{query_id} needs a non-empty question")
        if question in seen_questions:
            raise ValueError(f"duplicate question: {question!r}")
        seen_questions.add(question)

        relevant = validate_spans(query_id, "relevant", query.get("relevant"))
        irrelevant = validate_spans(query_id, "irrelevant", query.get("irrelevant", []))
        if relevant & irrelevant:
            raise ValueError(f"{query_id} labels the same span as relevant and irrelevant")

        if category == "unanswerable":
            if relevant:
                raise ValueError(f"{query_id} is unanswerable but has relevant spans")
            unanswerable += 1
        elif not relevant:
            raise ValueError(f"{query_id} is answerable but has no relevant spans")

    if len(queries) < MIN_QUERY_COUNT:
        raise ValueError(f"dataset needs at least {MIN_QUERY_COUNT} queries")
    if unanswerable < MIN_UNANSWERABLE_COUNT:
        raise ValueError(f"dataset needs at least {MIN_UNANSWERABLE_COUNT} unanswerable queries")
    missing_categories = REQUIRED_CATEGORIES - covered
    if missing_categories:
        raise ValueError(f"dataset does not cover categories: {sorted(missing_categories)}")


def _threshold_values(default_min_score: float) -> tuple[float, ...]:
    """Return the fixed sweep plus the threshold the harness actually used."""
    return tuple(sorted({*THRESHOLD_SWEEP, round(float(default_min_score), 4)}))


def _coverage_metrics(rows: list[dict]) -> dict:
    """Report span-level recall and all-relevant query coverage at ``k``.

    Query-level Recall@K only asks whether at least one labelled relevant span
    was retrieved. These two metrics keep the span denominator and the
    "every required span" condition visible instead of hiding partial coverage.
    """

    answerable = [row for row in rows if row["relevant_count"] > 0]
    total_spans = sum(row["relevant_count"] for row in answerable)
    found_spans = sum(int(row["semantic"]["relevant_found"]) for row in answerable)
    full_coverage = sum(1 for row in answerable if row["semantic"]["full_coverage"])
    return {
        "answerable_relevant_spans": total_spans,
        "relevant_spans_found": found_spans,
        "semantic_span_recall_at_k": (round(found_spans / total_spans, 4) if total_spans else None),
        "semantic_full_coverage_queries": full_coverage,
        "semantic_full_coverage_at_k": (
            round(full_coverage / len(answerable), 4) if answerable else None
        ),
    }


def _threshold_metrics(rows: list[dict], threshold: float, k: int = 5) -> dict:
    """Aggregate one threshold from already-scored candidates.

    ``rows[*].semantic.scored`` is the ranked output of one low-floor search,
    so filtering it is equivalent to querying at ``threshold`` without a new
    embedding request. Unanswerable rows never enter Recall@K.
    """

    def hits(row: dict) -> list[dict]:
        return [item for item in row["semantic"]["scored"] if item["score"] >= threshold][:k]

    answerable = [row for row in rows if row["relevant_count"] > 0]
    unanswerable = [row for row in rows if row["relevant_count"] == 0]

    def rate(count: int, total: int) -> float | None:
        return round(count / total, 4) if total else None

    recall = sum(1 for row in answerable if any(item["relevant"] for item in hits(row)))
    top1 = sum(1 for row in answerable if (row_hits := hits(row)) and row_hits[0]["relevant"])
    answerable_with_candidates = sum(1 for row in answerable if hits(row))
    unanswerable_with_candidates = sum(1 for row in unanswerable if hits(row))
    non_relevant_top1 = sum(
        1 for row in answerable if (row_hits := hits(row)) and not row_hits[0]["relevant"]
    )

    return {
        "min_score": round(float(threshold), 4),
        "answerable_queries": len(answerable),
        "answerable_with_candidates": answerable_with_candidates,
        "semantic_recall_at_k": rate(recall, len(answerable)),
        "semantic_top1_relevance": rate(top1, len(answerable)),
        "unanswerable_queries": len(unanswerable),
        "unanswerable_with_candidates": unanswerable_with_candidates,
        "unanswerable_candidate_rate": rate(unanswerable_with_candidates, len(unanswerable)),
        "non_relevant_top1_queries": non_relevant_top1,
    }


def _semantic_index_size(database_path: Path, document_id: UUID) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(LENGTH(vector)), 0), COUNT(*)
            FROM semantic_candidates WHERE document_id = ?
            """,
            (str(document_id),),
        ).fetchone()
    return {
        "vector_bytes": int(row[0]),
        "candidate_count": int(row[1]),
        "database_bytes": database_path.stat().st_size,
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "mean_ms": round(statistics.fmean(ordered) * 1000.0, 3),
        "p95_ms": round(p95 * 1000.0, 3),
    }


def _unanswerable_metrics(rows: list[dict]) -> dict[str, int | float | None]:
    """Summarize unanswerable queries separately from Recall@K.

    Unanswerable rows have no labelled relevant span and are therefore never
    part of the Recall@K denominator. Having retrieval candidates is reported
    separately from the QA state: a candidate is only a ranking result, while
    ``evidence_found``/``answered`` means the system claimed evidence existed.
    """
    unanswerable = [row for row in rows if row["relevant_count"] == 0]
    total = len(unanswerable)

    def rate(count: int) -> float | None:
        return round(count / total, 4) if total else None

    literal_with_candidates = sum(1 for row in unanswerable if row["literal"]["hits"] > 0)
    semantic_with_candidates = sum(1 for row in unanswerable if row["semantic"]["hits"] > 0)
    claimed_evidence = sum(
        1 for row in unanswerable if row["qa"]["status"] in {"evidence_found", "answered"}
    )
    citations = sum(int(row["qa"]["citations"]) for row in unanswerable)

    return {
        "count": total,
        "literal_with_candidates": literal_with_candidates,
        "literal_candidate_rate": rate(literal_with_candidates),
        "semantic_with_candidates": semantic_with_candidates,
        "semantic_candidate_rate": rate(semantic_with_candidates),
        "qa_claimed_evidence": claimed_evidence,
        "qa_claimed_evidence_rate": rate(claimed_evidence),
        "qa_citations": citations,
    }


def _select_queries(dataset: dict, query_ids: Sequence[str] | None) -> list[dict]:
    """Return the fixed dataset order, optionally restricted to requested IDs."""
    queries = dataset["queries"]
    if query_ids is None:
        return list(queries)
    requested = list(dict.fromkeys(str(query_id).strip() for query_id in query_ids))
    if not requested or not all(requested):
        raise ValueError("query_ids must contain non-empty ids")
    known = {str(query["id"]) for query in queries}
    unknown = [query_id for query_id in requested if query_id not in known]
    if unknown:
        raise ValueError(f"unknown query ids: {unknown}")
    wanted = set(requested)
    return [query for query in queries if query["id"] in wanted]


def _prior_query_rows(
    prior_result: dict | None,
    *,
    dataset_name: str,
    provider_name: str,
    answer_mode: str,
    prompt_variant: str,
    prompt_sha256: str,
    selected_ids: set[str],
) -> list[dict]:
    """Validate a merge source so a two-step run cannot mix experiment arms."""
    if prior_result is None:
        return []
    if prior_result.get("status") != "completed":
        raise ValueError("merge source must be a completed evaluation result")
    if prior_result.get("dataset") != dataset_name:
        raise ValueError("merge source dataset does not match")
    if prior_result.get("provider") != provider_name:
        raise ValueError("merge source provider does not match")
    if prior_result.get("answer_mode") != answer_mode:
        raise ValueError("merge source answer_mode does not match")
    if answer_mode == ANSWER_MODE_GROUNDED_GENERATION:
        prompt = prior_result.get("prompt")
        if not isinstance(prompt, dict) or prompt.get("variant") != prompt_variant:
            raise ValueError("merge source prompt variant does not match")
        if prompt.get("sha256") != prompt_sha256:
            raise ValueError("merge source prompt text does not match")

    rows = prior_result.get("queries")
    if not isinstance(rows, list):
        raise ValueError("merge source has no query rows")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("merge source has a malformed query row")
        query_id = row["id"]
        if query_id not in selected_ids:
            raise ValueError(f"merge source row is outside the selected queries: {query_id}")
        if query_id in seen:
            raise ValueError(f"merge source has a duplicate query row: {query_id}")
        seen.add(query_id)
    return rows


def _run(
    dataset: dict,
    provider_name: str,
    answer_mode: str = ANSWER_MODE_EVIDENCE_ONLY,
    *,
    llm_provider: LLMProvider | None = None,
    prompt_variant: str = PROMPT_VARIANT_A,
    query_ids: Sequence[str] | None = None,
    prior_result: dict | None = None,
) -> dict:
    _validate_dataset(dataset)
    if answer_mode not in ANSWER_MODES:
        raise ValueError(f"answer_mode must be one of {sorted(ANSWER_MODES)}")
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant must be one of {list(PROMPT_VARIANTS)}")
    system_prompt = _prompt_for_variant(prompt_variant)
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    selected_queries = _select_queries(dataset, query_ids)
    selected_ids = {str(query["id"]) for query in selected_queries}
    prior_rows = _prior_query_rows(
        prior_result,
        dataset_name=dataset["name"],
        provider_name=provider_name,
        answer_mode=answer_mode,
        prompt_variant=prompt_variant,
        prompt_sha256=prompt_sha256,
        selected_ids=selected_ids,
    )
    prior_ids = {row["id"] for row in prior_rows}
    pending_queries = [query for query in selected_queries if query["id"] not in prior_ids]
    embedding = EmbeddingSettings.from_env()
    default_min_score = embedding.min_score
    if provider_name == "auto" and (not embedding.enabled or not embedding.configured):
        return {
            "status": "unexecuted",
            "reason": (
                "Real embedding provider is not enabled/configured "
                "(RAG_ANCHOR_EMBEDDING_ENABLED/BASE_URL/MODEL). "
                "If bge-m3 is missing run: ollama pull bge-m3"
            ),
            "answer_mode": answer_mode,
            "real_embedding_calls": False,
            "real_generation_calls": False,
        }

    llm_settings = (
        LLMSettings.from_env() if answer_mode == ANSWER_MODE_GROUNDED_GENERATION else LLMSettings()
    )
    if (
        answer_mode == ANSWER_MODE_GROUNDED_GENERATION
        and llm_provider is None
        and not (llm_settings.enabled and llm_settings.configured)
    ):
        return {
            "status": "unexecuted",
            "reason": (
                "Grounded generation is not enabled/configured "
                "(RAG_ANCHOR_LLM_ENABLED/BASE_URL/MODEL)."
            ),
            "answer_mode": answer_mode,
            "real_embedding_calls": provider_name != "fake",
            "real_generation_calls": False,
        }

    provider = _load_fake_provider() if provider_name == "fake" else None
    prompt_provider: _PromptVariantProvider | None = None
    counter: _CountingLLMProvider | None = None
    if answer_mode == ANSWER_MODE_GROUNDED_GENERATION:
        base_llm = llm_provider if llm_provider is not None else build_llm_provider(llm_settings)
        prompt_provider = _PromptVariantProvider(base_llm, system_prompt)
        counter = _CountingLLMProvider(prompt_provider)
    with tempfile.TemporaryDirectory(prefix="rag-anchor-eval-") as directory:
        settings = Settings(
            data_dir=Path(directory) / "data",
            llm=llm_settings,
            embedding=embedding,
        )
        app = create_app(settings, embedding_provider=provider, llm_provider=counter)
        k = 5
        with TestClient(app) as client:
            pdf_bytes = _write_pdf(dataset["pages"])
            upload = client.post(
                DOCUMENTS_URL,
                files={"file": ("semantic-eval.pdf", pdf_bytes, "application/pdf")},
            )
            if upload.status_code != 201:
                return {
                    "status": "failed",
                    "reason": f"upload failed: {upload.text}",
                    "answer_mode": answer_mode,
                }
            document_id = UUID(upload.json()["document_id"])
            index = client.post(f"{DOCUMENTS_URL}/{document_id}/index")
            if index.status_code != 200:
                return {
                    "status": "failed",
                    "reason": f"text index failed: {index.text}",
                    "answer_mode": answer_mode,
                }

            started = time.perf_counter()
            build = client.post(f"{DOCUMENTS_URL}/{document_id}/semantic-index")
            index_seconds = time.perf_counter() - started
            if build.status_code != 200:
                return {
                    "status": "unexecuted" if provider_name != "fake" else "failed",
                    "reason": (
                        "semantic index build failed; check the embedding service. "
                        "If bge-m3 is missing run: ollama pull bge-m3. "
                        f"HTTP {build.status_code}: {build.text}"
                    ),
                    "real_embedding_calls": provider_name != "fake",
                    "answer_mode": answer_mode,
                    "real_generation_calls": False,
                }

            document_service = app.state.document_service
            retrieval = app.state.retrieval_service
            semantic: SemanticIndexService = app.state.semantic_service
            page_texts = {
                page.page_index: page.text
                for page in document_service.repository.get_document_page_texts(document_id)
            }
            chunks = {
                chunk.chunk_id: chunk
                for chunk in document_service.repository.get_document_chunks(document_id)
            }
            regression_hits = retrieval.search_chunks(
                "stages",
                document_id=document_id,
                match_case=True,
                limit=k,
                offset=0,
            )
            literal_regression = (
                "pass"
                if any(
                    hit.page_index == 0 and hit.hit_text == "stages"
                    for hit in regression_hits.matches
                )
                else "fail"
            )

            query_rows: list[dict] = []

            for query in pending_queries:
                question = str(query["question"])
                relevant_spans = _resolve_spans(page_texts, query.get("relevant", []))
                relevant_keys = {(page, start, end) for page, start, end, _ in relevant_spans}
                irrelevant_spans = _resolve_spans(page_texts, query.get("irrelevant", []))
                irrelevant_keys = {(page, start, end) for page, start, end, _ in irrelevant_spans}

                literal_started = time.perf_counter()
                literal_result = retrieval.search_chunks(
                    question,
                    document_id=document_id,
                    match_case=False,
                    limit=k,
                    offset=0,
                )
                literal_seconds = time.perf_counter() - literal_started

                literal_hits = literal_result.matches[:k]
                literal_top1_relevant = bool(literal_hits) and any(
                    hit.page_index == page and hit.match_start < end and hit.match_end > start
                    for hit in literal_hits[:1]
                    for page, start, end, _ in relevant_spans
                )
                literal_relevant = any(
                    hit.page_index == page and hit.match_start < end and hit.match_end > start
                    for hit in literal_hits
                    for page, start, end, _ in relevant_spans
                )
                literal_irrelevant_hits = sum(
                    1
                    for hit in literal_hits
                    if any(
                        hit.page_index == page and hit.match_start < end and hit.match_end > start
                        for page, start, end, _ in irrelevant_spans
                    )
                )

                # One floor search supplies both the configured-threshold hits
                # and the offline sweep. The production threshold is unchanged.
                semantic_started = time.perf_counter()
                semantic_result = semantic.search(
                    document_id,
                    question,
                    top_k=MAX_TOP_K,
                    min_score=SEMANTIC_SCORE_FLOOR,
                )
                semantic_seconds = time.perf_counter() - semantic_started

                semantic_hits = [
                    match for match in semantic_result.matches if match.score >= default_min_score
                ][:k]
                semantic_scored = [
                    {
                        "score": match.score,
                        "page_index": match.page_index,
                        "start_offset": match.start_offset,
                        "end_offset": match.end_offset,
                        "relevant": (
                            match.page_index,
                            match.start_offset,
                            match.end_offset,
                        )
                        in relevant_keys,
                    }
                    for match in semantic_result.matches
                ]
                semantic_irrelevant_hits = sum(
                    1
                    for match in semantic_hits
                    if (match.page_index, match.start_offset, match.end_offset) in irrelevant_keys
                )
                semantic_top1_relevant = bool(semantic_hits) and any(
                    match.page_index == page
                    and match.start_offset == start
                    and match.end_offset == end
                    for match in semantic_hits[:1]
                    for page, start, end, _ in relevant_spans
                )
                semantic_relevant = any(
                    match.page_index == page
                    and match.start_offset == start
                    and match.end_offset == end
                    for match in semantic_hits
                    for page, start, end, _ in relevant_spans
                )
                semantic_found_keys = {
                    (match.page_index, match.start_offset, match.end_offset)
                    for match in semantic_hits
                } & relevant_keys
                semantic_full_coverage = (
                    len(semantic_found_keys) == len(relevant_keys) if relevant_keys else None
                )

                calls_before = counter.calls if counter is not None else None
                outputs_before = len(prompt_provider.outputs) if prompt_provider is not None else 0
                qa_started = time.perf_counter()
                qa_response = client.post(
                    f"{DOCUMENTS_URL}/{document_id}/qa",
                    json={
                        "question": question,
                        "retrieval_mode": "semantic",
                        "max_citations": k,
                        "answer_mode": answer_mode,
                    },
                )
                qa_seconds = time.perf_counter() - qa_started
                generation_ms = (
                    round(counter.durations[-1] * 1000.0, 3)
                    if counter is not None and counter.calls > calls_before
                    else None
                )
                allowed_ids = {f"E{index}" for index in range(1, len(semantic_hits) + 1)}
                new_outputs = (
                    prompt_provider.outputs[outputs_before:] if prompt_provider is not None else []
                )
                llm_output = _inspect_model_output(new_outputs, allowed_ids)
                qa_ok = qa_response.status_code == 200
                qa_body = qa_response.json() if qa_ok else {}
                qa_error_detail = None
                if not qa_ok:
                    try:
                        error_body = qa_response.json()
                    except ValueError:
                        error_body = None
                    if isinstance(error_body, dict):
                        qa_error_detail = error_body.get("detail")
                citations = qa_body.get("citations", []) if qa_ok else []
                citation_check = _verify_citations(
                    citations,
                    allowed_ids=allowed_ids,
                    page_texts=page_texts,
                    chunks=chunks,
                    document_id=document_id,
                    relevant_keys=relevant_keys,
                )
                located = 0
                for citation in citations:
                    locate = client.post(
                        f"{DOCUMENTS_URL}/{document_id}/evidence/locate",
                        json={
                            "items": [
                                {
                                    "page_index": citation["page_index"],
                                    "match_start": citation["support_start"],
                                    "match_end": citation["support_end"],
                                    "hit_text": citation["support_text"],
                                    "chunk_id": citation["chunk_id"],
                                }
                            ]
                        },
                    )
                    if locate.status_code == 200 and locate.json()["results"][0]["rects"]:
                        located += 1

                query_rows.append(
                    {
                        "id": query["id"],
                        "category": query["category"],
                        "question": question,
                        "relevant_count": len(relevant_keys),
                        "irrelevant_count": len(irrelevant_keys),
                        "literal": {
                            "hits": len(literal_hits),
                            "irrelevant_hits": literal_irrelevant_hits,
                            "latency_ms": round(literal_seconds * 1000.0, 3),
                            "recall_at_k": literal_relevant if relevant_keys else None,
                            "top1_relevant": (literal_top1_relevant if relevant_keys else None),
                        },
                        "semantic": {
                            "hits": len(semantic_hits),
                            "irrelevant_hits": semantic_irrelevant_hits,
                            "scored_candidates": len(semantic_scored),
                            "latency_ms": round(semantic_seconds * 1000.0, 3),
                            "recall_at_k": semantic_relevant if relevant_keys else None,
                            "top1_relevant": (semantic_top1_relevant if relevant_keys else None),
                            "relevant_found": len(semantic_found_keys),
                            "relevant_total": len(relevant_keys),
                            "full_coverage": semantic_full_coverage,
                            "top_score": (
                                round(semantic_hits[0].score, 6) if semantic_hits else None
                            ),
                            "scored": semantic_scored,
                        },
                        "qa": {
                            "status_code": qa_response.status_code,
                            "error_detail": qa_error_detail,
                            "status": qa_body.get("status") if qa_ok else None,
                            "answer_mode": qa_body.get("answer_mode") if qa_ok else None,
                            "answer": qa_body.get("answer") if qa_ok else None,
                            "answer_chars": (len(qa_body.get("answer", "")) if qa_ok else 0),
                            "latency_ms": round(qa_seconds * 1000.0, 3),
                            "generation_ms": generation_ms,
                            "llm_output": llm_output,
                            "citations": len(citations),
                            "citation_ids": [
                                str(citation.get("citation_id", "")) for citation in citations
                            ],
                            **citation_check,
                            "locatable_citations": located,
                        },
                    }
                )

            new_query_count = len(query_rows)
            rows_by_id = {row["id"]: row for row in prior_rows}
            for row in query_rows:
                rows_by_id[row["id"]] = row
            missing_after_run = [
                query["id"] for query in selected_queries if query["id"] not in rows_by_id
            ]
            if missing_after_run:
                raise RuntimeError(f"evaluation rows missing after run: {missing_after_run}")
            query_rows = [rows_by_id[query["id"]] for query in selected_queries]

            answerable = [row for row in query_rows if row["relevant_count"] > 0]
            literal_recall = (
                sum(1 for row in answerable if row["literal"]["recall_at_k"]) / len(answerable)
                if answerable
                else 0.0
            )
            semantic_recall = (
                sum(1 for row in answerable if row["semantic"]["recall_at_k"]) / len(answerable)
                if answerable
                else 0.0
            )
            literal_top1 = (
                sum(1 for row in answerable if row["literal"]["top1_relevant"]) / len(answerable)
                if answerable
                else 0.0
            )
            semantic_top1 = (
                sum(1 for row in answerable if row["semantic"]["top1_relevant"]) / len(answerable)
                if answerable
                else 0.0
            )
            non_relevant_top1 = sum(
                1
                for row in query_rows
                if row["semantic"]["hits"] > 0 and row["semantic"]["top1_relevant"] is False
            )
            coverage = _coverage_metrics(query_rows)
            threshold_sweep = [
                _threshold_metrics(query_rows, threshold, k)
                for threshold in _threshold_values(default_min_score)
            ]
            generation = _generation_metrics(query_rows, answer_mode)
            category_counts = Counter(query["category"] for query in selected_queries)
            evaluated_unanswerable = sum(1 for row in query_rows if row["relevant_count"] == 0)
            llm_durations = [
                row["qa"]["generation_ms"] / 1000.0
                for row in query_rows
                if row["qa"].get("generation_ms") is not None
            ]
            citations_located = sum(int(row["qa"]["locatable_citations"]) for row in query_rows)
            citations_attempted = sum(int(row["qa"]["citations"]) for row in query_rows)

            metadata = semantic.repository.get_semantic_index_metadata(document_id)
            size = _semantic_index_size(settings.database_path, document_id)

            robustness: dict[str, str] = {"literal_keyword_regression": literal_regression}
            if provider_name == "fake":
                other = _load_fake_provider()
                other.model_version = "different-version"
                stale_service = SemanticIndexService(
                    document_service,
                    other,
                    settings=settings.embedding,
                )
                try:
                    stale_service.search(document_id, "阶段")
                except SemanticIndexStaleError:
                    robustness["changed_model_version_is_stale"] = "pass"
                else:
                    robustness["changed_model_version_is_stale"] = "fail"

                client.post(f"{DOCUMENTS_URL}/{document_id}/index")
                try:
                    semantic.search(document_id, "阶段")
                except SemanticIndexUnavailableError:
                    robustness["reindex_invalidates_vectors"] = "pass"
                else:
                    robustness["reindex_invalidates_vectors"] = "fail"

                document_service.delete_document(document_id)
                robustness["delete_cascades"] = (
                    "pass"
                    if semantic.repository.get_semantic_index_metadata(document_id) is None
                    and semantic.repository.list_semantic_candidates(document_id) == []
                    else "fail"
                )

            return {
                "status": "completed",
                "dataset": dataset["name"],
                "dataset_coverage": {
                    "dataset_queries": len(dataset["queries"]),
                    "queries": len(query_rows),
                    "answerable_queries": len(query_rows) - evaluated_unanswerable,
                    "unanswerable_queries": evaluated_unanswerable,
                    "categories": dict(sorted(category_counts.items())),
                    "is_full_dataset": len(query_rows) == len(dataset["queries"]),
                },
                "query_selection": {
                    "requested_ids": (list(query_ids) if query_ids is not None else None),
                    "evaluated_ids": [row["id"] for row in query_rows],
                    "full_dataset": len(query_rows) == len(dataset["queries"]),
                },
                "merge": (
                    {
                        "prior_generated_at": prior_result.get("generated_at"),
                        "prior_queries": len(prior_rows),
                        "new_queries": new_query_count,
                        "merged_queries": len(query_rows),
                    }
                    if prior_result is not None
                    else None
                ),
                "prompt": (
                    _prompt_metadata(prompt_variant)
                    if answer_mode == ANSWER_MODE_GROUNDED_GENERATION
                    else None
                ),
                "provider": provider_name,
                "model": (
                    {
                        "name": metadata.model_name,
                        "version": metadata.model_version,
                        "digest": metadata.model_digest,
                        "index_version": metadata.index_version,
                        "candidate_version": metadata.candidate_version,
                        "dimension": metadata.dimension,
                        "normalization": metadata.normalization,
                    }
                    if metadata
                    else None
                ),
                "real_embedding_calls": provider_name != "fake",
                "answer_mode": answer_mode,
                "real_generation_calls": (
                    answer_mode == ANSWER_MODE_GROUNDED_GENERATION
                    and provider_name != "fake"
                    and llm_provider is None
                ),
                "llm": (
                    {
                        "model": llm_settings.model,
                        "base_url": llm_settings.base_url,
                        "timeout_seconds": llm_settings.timeout_seconds,
                        "max_output_tokens": llm_settings.max_output_tokens,
                        "reasoning_effort": llm_settings.reasoning_effort or None,
                        "calls": len(llm_durations),
                        "total_seconds": round(sum(llm_durations), 3),
                        "call_latency": _latency_summary(llm_durations),
                    }
                    if answer_mode == ANSWER_MODE_GROUNDED_GENERATION
                    else None
                ),
                "k": k,
                "generated_at": datetime.now(UTC).isoformat(),
                "metrics": {
                    "answerable_queries": len(answerable),
                    "total_queries": len(query_rows),
                    "literal_recall_at_k": round(literal_recall, 4),
                    "semantic_recall_at_k": round(semantic_recall, 4),
                    "literal_top1_relevance": round(literal_top1, 4),
                    "semantic_top1_relevance": round(semantic_top1, 4),
                    "non_relevant_semantic_top1_queries": non_relevant_top1,
                    "answerable_relevant_spans": coverage["answerable_relevant_spans"],
                    "relevant_spans_found": coverage["relevant_spans_found"],
                    "semantic_span_recall_at_k": coverage["semantic_span_recall_at_k"],
                    "semantic_full_coverage_queries": coverage["semantic_full_coverage_queries"],
                    "semantic_full_coverage_at_k": coverage["semantic_full_coverage_at_k"],
                    "unanswerable": _unanswerable_metrics(query_rows),
                    "citation_locatability": (
                        round(citations_located / citations_attempted, 4)
                        if citations_attempted
                        else None
                    ),
                    "citations_located": citations_located,
                    "citations_attempted": citations_attempted,
                },
                "generation": generation,
                "threshold_sweep": {
                    "k": k,
                    "default_min_score": default_min_score,
                    "method": (
                        "Offline score filter over one min_score=-1.0 search per query; "
                        "no additional embeddings. QA and citations still use "
                        "default_min_score."
                    ),
                    "rows": threshold_sweep,
                },
                "timing": {
                    "semantic_index_seconds": round(index_seconds, 3),
                    "literal_retrieval": _latency_summary(
                        [row["literal"]["latency_ms"] / 1000.0 for row in query_rows]
                    ),
                    "semantic_retrieval": _latency_summary(
                        [row["semantic"]["latency_ms"] / 1000.0 for row in query_rows]
                    ),
                    "semantic_qa": _latency_summary(
                        [row["qa"]["latency_ms"] / 1000.0 for row in query_rows]
                    ),
                },
                "index": {
                    **size,
                    "pages": len(page_texts),
                },
                "robustness": robustness,
                "queries": query_rows,
                "real_embedding_note": (
                    "未执行真实 embedding 召回率评测：本次使用 Fake Provider，"
                    "数值只验证工程管线，不代表语义检索质量。"
                    if provider_name == "fake"
                    else "真实 embedding 评测；结果仍受小样本限制。"
                ),
                "generation_note": (
                    "未执行语言模型生成：answer_mode=evidence_only。"
                    if answer_mode == ANSWER_MODE_EVIDENCE_ONLY
                    else (
                        "真实本地语言模型生成；引用合法与可定位不等于事实支持。"
                        if provider_name != "fake" and llm_provider is None
                        else "Mock/Fake 语言模型生成；仅验证工程管线，不代表模型质量。"
                    )
                ),
            }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("auto", "fake"),
        default="auto",
        help="auto requires configured real embeddings; fake is engineering-only",
    )
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=ANSWER_MODE_EVIDENCE_ONLY,
        help=(
            "evidence_only (default) never calls a model; "
            "grounded_generation requires a configured local LLM"
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=PROMPT_VARIANT_A,
        help=(
            "A (default) is the production prompt; B is an evaluation-only reword "
            "of rules 7 and 12 and never changes server validation"
        ),
    )
    parser.add_argument(
        "--query-ids",
        default=None,
        help="comma-separated query ids for a paired precheck; default is the full dataset",
    )
    parser.add_argument(
        "--merge-from",
        type=Path,
        default=None,
        help=(
            "completed result JSON whose query rows are reused; only missing selected "
            "queries are run, so a precheck is not executed twice"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    query_ids = (
        [part.strip() for part in args.query_ids.split(",") if part.strip()]
        if args.query_ids
        else None
    )
    try:
        prior_result = (
            json.loads(args.merge_from.read_text(encoding="utf-8"))
            if args.merge_from is not None
            else None
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "invalid_merge_source",
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    try:
        result = _run(
            dataset,
            args.provider,
            args.answer_mode,
            prompt_variant=args.prompt_variant,
            query_ids=query_ids,
            prior_result=prior_result,
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "invalid_dataset",
                    "dataset": dataset.get("name"),
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "unexecuted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
