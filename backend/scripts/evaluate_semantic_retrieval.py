#!/usr/bin/env python3
"""Repeatable M2-01 retrieval evaluation over a labelled synthetic PDF.

The harness never downloads a model and never invents metrics. ``--provider
auto`` runs only when the embedding service is configured and enabled through
the documented environment variables; otherwise it reports ``unexecuted``.
``--provider fake`` is an engineering-only smoke run with deterministic
keyword vectors and must not be presented as embedding quality.

Measured per query: literal Recall@K, semantic Recall@K, top-1 relevance,
semantic citation locatability, retrieval latency. Measured once: semantic
index build time and vector/database size.

Run from ``backend/``::

    uv run python scripts/evaluate_semantic_retrieval.py --provider fake
    RAG_ANCHOR_EMBEDDING_ENABLED=1 \
    RAG_ANCHOR_EMBEDDING_BASE_URL=http://127.0.0.1:11434 \
    RAG_ANCHOR_EMBEDDING_MODEL=bge-m3 \
    uv run python scripts/evaluate_semantic_retrieval.py --provider auto
"""

# User-facing evaluation notes use full-width Chinese punctuation on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from rag_anchor.config import EmbeddingSettings, Settings
from rag_anchor.infra.embedding import EmbeddingProvider
from rag_anchor.main import create_app
from rag_anchor.services.semantic_service import (
    SemanticIndexService,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = BACKEND_ROOT / "evaluation" / "semantic_dataset.json"
DOCUMENTS_URL = "/api/v1/documents"


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


def _run(dataset: dict, provider_name: str) -> dict:
    embedding = EmbeddingSettings.from_env()
    if provider_name == "auto" and (not embedding.enabled or not embedding.configured):
        return {
            "status": "unexecuted",
            "reason": (
                "Real embedding provider is not enabled/configured "
                "(RAG_ANCHOR_EMBEDDING_ENABLED/BASE_URL/MODEL). "
                "If bge-m3 is missing run: ollama pull bge-m3"
            ),
            "real_embedding_calls": False,
        }

    provider = _load_fake_provider() if provider_name == "fake" else None
    with tempfile.TemporaryDirectory(prefix="rag-anchor-eval-") as directory:
        settings = Settings(
            data_dir=Path(directory) / "data",
            embedding=embedding,
        )
        app = create_app(settings, embedding_provider=provider)
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
                }
            document_id = UUID(upload.json()["document_id"])
            index = client.post(f"{DOCUMENTS_URL}/{document_id}/index")
            if index.status_code != 200:
                return {
                    "status": "failed",
                    "reason": f"text index failed: {index.text}",
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
                }

            document_service = app.state.document_service
            retrieval = app.state.retrieval_service
            semantic: SemanticIndexService = app.state.semantic_service
            page_texts = {
                page.page_index: page.text
                for page in document_service.repository.get_document_page_texts(document_id)
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
            literal_latencies: list[float] = []
            semantic_latencies: list[float] = []
            qa_latencies: list[float] = []
            located_total = 0
            locate_attempts = 0

            for query in dataset["queries"]:
                question = str(query["question"])
                relevant_spans = _resolve_spans(page_texts, query.get("relevant", []))
                relevant_keys = {(page, start, end) for page, start, end, _ in relevant_spans}

                literal_started = time.perf_counter()
                literal_result = retrieval.search_chunks(
                    question,
                    document_id=document_id,
                    match_case=False,
                    limit=k,
                    offset=0,
                )
                literal_latencies.append(time.perf_counter() - literal_started)

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

                semantic_started = time.perf_counter()
                semantic_result = semantic.search(document_id, question, top_k=k)
                semantic_latencies.append(time.perf_counter() - semantic_started)

                semantic_hits = semantic_result.matches[:k]
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

                qa_started = time.perf_counter()
                qa_response = client.post(
                    f"{DOCUMENTS_URL}/{document_id}/qa",
                    json={
                        "question": question,
                        "retrieval_mode": "semantic",
                        "max_citations": k,
                    },
                )
                qa_latencies.append(time.perf_counter() - qa_started)
                citations = (
                    qa_response.json().get("citations", [])
                    if qa_response.status_code == 200
                    else []
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
                    locate_attempts += 1
                    if locate.status_code == 200 and locate.json()["results"][0]["rects"]:
                        located += 1
                        located_total += 1

                query_rows.append(
                    {
                        "id": query["id"],
                        "question": question,
                        "relevant_count": len(relevant_keys),
                        "literal": {
                            "hits": len(literal_hits),
                            "recall_at_k": literal_relevant if relevant_keys else None,
                            "top1_relevant": (literal_top1_relevant if relevant_keys else None),
                        },
                        "semantic": {
                            "hits": len(semantic_hits),
                            "total_above_threshold": semantic_result.total,
                            "recall_at_k": semantic_relevant if relevant_keys else None,
                            "top1_relevant": (semantic_top1_relevant if relevant_keys else None),
                            "top_score": (
                                round(semantic_hits[0].score, 6) if semantic_hits else None
                            ),
                        },
                        "qa": {
                            "status_code": qa_response.status_code,
                            "status": (
                                qa_response.json().get("status")
                                if qa_response.status_code == 200
                                else None
                            ),
                            "citations": len(citations),
                            "locatable_citations": located,
                        },
                    }
                )

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
                    "unanswerable": _unanswerable_metrics(query_rows),
                    "citation_locatability": (
                        round(located_total / locate_attempts, 4) if locate_attempts else None
                    ),
                    "citations_located": located_total,
                    "citations_attempted": locate_attempts,
                },
                "timing": {
                    "semantic_index_seconds": round(index_seconds, 3),
                    "literal_retrieval": _latency_summary(literal_latencies),
                    "semantic_retrieval": _latency_summary(semantic_latencies),
                    "semantic_qa": _latency_summary(qa_latencies),
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
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("auto", "fake"),
        default="auto",
        help="auto requires configured real embeddings; fake is engineering-only",
    )
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    result = _run(dataset, args.provider)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "unexecuted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
