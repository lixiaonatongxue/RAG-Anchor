"""Unit tests for the aggregate metrics of the offline evaluation script."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest

from rag_anchor.domain.text import TextChunk

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_semantic_retrieval.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_semantic_retrieval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    relevant_count: int,
    literal_hits: int,
    semantic_hits: int,
    status: str | None,
    citations: int,
) -> dict:
    return {
        "relevant_count": relevant_count,
        "literal": {"hits": literal_hits},
        "semantic": {"hits": semantic_hits},
        "qa": {"status": status, "citations": citations},
    }


def _threshold_row(relevant_count: int, scored: list[tuple[float, bool]]) -> dict:
    return {
        "relevant_count": relevant_count,
        "semantic": {
            "scored": [{"score": score, "relevant": relevant} for score, relevant in scored]
        },
    }


def _dataset(metrics: ModuleType) -> dict:
    return json.loads(metrics.DATASET_PATH.read_text(encoding="utf-8"))


def _query(dataset: dict, query_id: str) -> dict:
    return next(query for query in dataset["queries"] if query["id"] == query_id)


def _span_pairs(query: dict, kind: str) -> set[tuple[int, str]]:
    return {(span["page_index"], span["text"]) for span in query.get(kind, [])}


def test_fixed_dataset_meets_size_category_and_unanswerable_requirements() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)

    metrics._validate_dataset(dataset)

    queries = dataset["queries"]
    assert len(queries) >= metrics.MIN_QUERY_COUNT
    assert sum(1 for query in queries if not query["relevant"]) >= metrics.MIN_UNANSWERABLE_COUNT
    assert {query["category"] for query in queries} == metrics.REQUIRED_CATEGORIES


def test_fixed_dataset_rejects_duplicate_ids_and_broken_labels() -> None:
    metrics = _load_script()

    duplicate = copy.deepcopy(_dataset(metrics))
    duplicate["queries"][1]["id"] = duplicate["queries"][0]["id"]
    with pytest.raises(ValueError, match="duplicate query id"):
        metrics._validate_dataset(duplicate)

    broken_label = copy.deepcopy(_dataset(metrics))
    broken_label["queries"][0]["relevant"][0]["text"] += " missing"
    with pytest.raises(ValueError, match="occur exactly once"):
        metrics._validate_dataset(broken_label)


def test_fixed_dataset_rejects_unanswerable_labels_and_missing_categories() -> None:
    metrics = _load_script()

    mislabelled = copy.deepcopy(_dataset(metrics))
    unanswerable = next(
        query for query in mislabelled["queries"] if query["category"] == "unanswerable"
    )
    unanswerable["relevant"] = [copy.deepcopy(mislabelled["queries"][0]["relevant"][0])]
    with pytest.raises(ValueError, match="unanswerable but has relevant"):
        metrics._validate_dataset(mislabelled)

    incomplete = copy.deepcopy(_dataset(metrics))
    incomplete["queries"] = [
        query for query in incomplete["queries"] if query["category"] != "different_conclusion"
    ]
    with pytest.raises(ValueError, match="does not cover categories"):
        metrics._validate_dataset(incomplete)


def test_threshold_metrics_filter_scores_without_changing_denominators() -> None:
    metrics = _load_script()
    rows = [
        # Answerable: top-1 is a distractor, two supporting spans follow.
        _threshold_row(2, [(0.80, False), (0.55, True), (0.40, True)]),
        # Answerable: the supporting span falls below the threshold.
        _threshold_row(1, [(0.70, False), (0.25, True)]),
        # Unanswerable: a candidate exists but no span supports it.
        _threshold_row(0, [(0.60, False), (0.20, False)]),
        _threshold_row(0, []),
    ]

    at_default = metrics._threshold_metrics(rows, 0.30)

    assert at_default["answerable_queries"] == 2
    assert at_default["semantic_recall_at_k"] == 0.5
    assert at_default["semantic_top1_relevance"] == 0.0
    assert at_default["unanswerable_queries"] == 2
    assert at_default["unanswerable_with_candidates"] == 1
    assert at_default["unanswerable_candidate_rate"] == 0.5
    assert at_default["non_relevant_top1_queries"] == 2

    above_candidates = metrics._threshold_metrics(rows, 0.61)

    assert above_candidates["semantic_recall_at_k"] == 0.0
    assert above_candidates["semantic_top1_relevance"] == 0.0
    assert above_candidates["unanswerable_with_candidates"] == 0
    assert above_candidates["unanswerable_candidate_rate"] == 0.0


def test_threshold_metrics_use_recall_at_k_and_handle_an_empty_group() -> None:
    metrics = _load_script()
    rows = [_threshold_row(1, [(0.90, False), (0.70, False), (0.65, True), (0.40, True)])]

    result = metrics._threshold_metrics(rows, 0.30, k=3)

    assert result["semantic_recall_at_k"] == 1.0
    assert result["semantic_top1_relevance"] == 0.0

    empty = metrics._threshold_metrics([], 0.30)
    assert empty["semantic_recall_at_k"] is None
    assert empty["semantic_top1_relevance"] is None
    assert empty["unanswerable_candidate_rate"] is None


def test_unanswerable_metrics_separate_candidates_from_claims() -> None:
    metrics = _load_script()
    rows = [
        _row(
            relevant_count=0,
            literal_hits=0,
            semantic_hits=2,
            status="evidence_found",
            citations=2,
        ),
        _row(
            relevant_count=0,
            literal_hits=1,
            semantic_hits=0,
            status="answered",
            citations=1,
        ),
        _row(
            relevant_count=0,
            literal_hits=0,
            semantic_hits=0,
            status="insufficient_evidence",
            citations=0,
        ),
        # Answerable rows must not contribute to the unanswerable aggregate.
        _row(
            relevant_count=1,
            literal_hits=5,
            semantic_hits=5,
            status="answered",
            citations=5,
        ),
    ]

    result = metrics._unanswerable_metrics(rows)

    assert result == {
        "count": 3,
        "literal_with_candidates": 1,
        "literal_candidate_rate": 0.3333,
        "semantic_with_candidates": 1,
        "semantic_candidate_rate": 0.3333,
        "qa_claimed_evidence": 2,
        "qa_claimed_evidence_rate": 0.6667,
        "qa_citations": 3,
    }


def test_unanswerable_metrics_handle_an_empty_group() -> None:
    metrics = _load_script()
    rows = [
        _row(
            relevant_count=2,
            literal_hits=1,
            semantic_hits=1,
            status="evidence_found",
            citations=1,
        )
    ]

    result = metrics._unanswerable_metrics(rows)

    assert result["count"] == 0
    assert result["literal_with_candidates"] == 0
    assert result["semantic_with_candidates"] == 0
    assert result["qa_claimed_evidence"] == 0
    assert result["qa_citations"] == 0
    assert result["literal_candidate_rate"] is None
    assert result["semantic_candidate_rate"] is None
    assert result["qa_claimed_evidence_rate"] is None


def test_answer_mode_defaults_to_evidence_only() -> None:
    metrics = _load_script()

    args = metrics._build_parser().parse_args([])

    assert args.answer_mode == "evidence_only"


def test_answer_mode_accepts_grounded_generation() -> None:
    metrics = _load_script()

    args = metrics._build_parser().parse_args(["--answer-mode", "grounded_generation"])

    assert args.answer_mode == "grounded_generation"


def test_answer_mode_rejects_an_unknown_value() -> None:
    metrics = _load_script()

    with pytest.raises(SystemExit):
        metrics._build_parser().parse_args(["--answer-mode", "guess"])


def test_prompt_variant_b_changes_only_rules_7_and_12() -> None:
    metrics = _load_script()

    prompt_a = metrics._prompt_for_variant("A")
    prompt_b = metrics._prompt_for_variant("B")

    assert prompt_a == metrics._SYSTEM_PROMPT
    lines_a = prompt_a.splitlines()
    lines_b = prompt_b.splitlines()
    assert len(lines_a) == len(lines_b)
    changed = [
        index
        for index, (left, right) in enumerate(zip(lines_a, lines_b, strict=True))
        if left != right
    ]
    assert [lines_a[index].split(".", 1)[0] for index in changed] == ["7", "12"]
    assert "unused_evidence" in lines_b[changed[0]]
    assert "answer only what the question actually asks" in lines_b[changed[0]].lower()
    assert "never create an extra claim" in lines_b[changed[0]].lower()
    assert "entities, objects, actions" in lines_b[changed[0]]
    assert "exactly once" in lines_b[changed[1]]
    assert '"answerable": false' in lines_b[changed[1]]


def test_prompt_metadata_hashes_the_effective_prompt() -> None:
    metrics = _load_script()

    metadata_a = metrics._prompt_metadata("A")
    metadata_b = metrics._prompt_metadata("B")

    assert metadata_a["variant"] == "A"
    assert metadata_a["changed_rules"] == []
    assert metadata_b["changed_rules"] == [7, 12]
    assert (
        metadata_a["sha256"]
        == hashlib.sha256(metadata_a["system_prompt"].encode("utf-8")).hexdigest()
    )
    assert (
        metadata_b["sha256"]
        == hashlib.sha256(metadata_b["system_prompt"].encode("utf-8")).hexdigest()
    )
    assert metadata_a["sha256"] != metadata_b["sha256"]


def test_prompt_variant_defaults_to_production_a() -> None:
    metrics = _load_script()

    args = metrics._build_parser().parse_args([])

    assert args.prompt_variant == "A"
    assert args.query_ids is None
    assert args.merge_from is None


def test_query_selection_preserves_dataset_order_and_rejects_unknown_ids() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)

    selected = metrics._select_queries(
        dataset,
        ["zh-upload-size", "en-weather", "zh-audit-content"],
    )

    assert [query["id"] for query in selected] == [
        "en-weather",
        "zh-audit-content",
        "zh-upload-size",
    ]
    with pytest.raises(ValueError, match="unknown query ids"):
        metrics._select_queries(dataset, ["does-not-exist"])


def test_inspect_model_output_reports_unused_evidence_and_accounting() -> None:
    metrics = _load_script()
    raw = json.dumps(
        {
            "answerable": True,
            "claims": [{"text": "依据 E1。", "citation_ids": ["E1"]}],
            "unused_evidence": [
                {"citation_id": "E2", "reason": "not asked"},
                {"citation_id": "E3", "reason": "not asked"},
            ],
            "limitations": [],
        },
        ensure_ascii=False,
    )

    summary = metrics._inspect_model_output([raw], {"E1", "E2", "E3"})

    assert summary["json_ok"] is True
    assert summary["answerable"] is True
    assert summary["claims"] == 1
    assert summary["unused_evidence"] == 2
    assert summary["cited_ids"] == ["E1"]
    assert summary["unused_ids"] == ["E2", "E3"]
    assert summary["unknown_cited_ids"] == []
    assert summary["unaccounted_ids"] == []

    unknown = metrics._inspect_model_output(
        [
            json.dumps(
                {
                    "answerable": True,
                    "claims": [{"text": "x", "citation_ids": ["E9"]}],
                    "unused_evidence": [],
                    "limitations": [],
                }
            )
        ],
        {"E1"},
    )
    assert unknown["unknown_cited_ids"] == ["E9"]
    assert unknown["unaccounted_ids"] == ["E1"]

    malformed = metrics._inspect_model_output(["not json"], {"E1"})
    assert malformed["json_ok"] is False
    assert malformed["claims"] is None
    assert metrics._inspect_model_output([], {"E1"}) is None


def test_verify_citations_flags_unknown_ids_and_misaligned_spans() -> None:
    metrics = _load_script()
    document_id = UUID("00000000-0000-0000-0000-000000000001")
    page_text = "Alpha beta."
    chunk = TextChunk(
        chunk_id="c1",
        document_id=str(document_id),
        page_index=0,
        page_number=1,
        start_offset=0,
        end_offset=len(page_text),
        text=page_text,
    )

    def citation(citation_id: str, start: int, end: int, text: str) -> dict:
        return {
            "citation_id": citation_id,
            "chunk_id": "c1",
            "page_index": 0,
            "page_number": 1,
            "support_start": start,
            "support_end": end,
            "support_text": text,
            "hit_text": text,
        }

    result = metrics._verify_citations(
        [
            citation("E1", 0, len(page_text), page_text),
            citation("E9", 0, len(page_text), page_text),
            citation("E2", 0, 4, "beta"),
        ],
        allowed_ids={"E1", "E2"},
        page_texts={0: page_text},
        chunks={"c1": chunk},
        document_id=document_id,
        relevant_keys={(0, 0, len(page_text))},
    )

    assert result["ids_valid"] == 2
    assert result["invalid_ids"] == ["E9"]
    assert result["content_verified"] == 2
    assert result["content_invalid_ids"] == ["E2"]
    assert result["citations_on_relevant_span"] == 2


def test_grounded_mode_is_unexecuted_without_llm_configuration(monkeypatch) -> None:
    metrics = _load_script()
    for name in (
        "RAG_ANCHOR_LLM_ENABLED",
        "RAG_ANCHOR_LLM_BASE_URL",
        "RAG_ANCHOR_LLM_MODEL",
        "RAG_ANCHOR_LLM_API_KEY",
        "RAG_ANCHOR_LLM_TIMEOUT_SECONDS",
        "RAG_ANCHOR_LLM_MAX_OUTPUT_TOKENS",
        "RAG_ANCHOR_LLM_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)

    result = metrics._run(_dataset(metrics), "fake", "grounded_generation")

    assert result["status"] == "unexecuted"
    assert "RAG_ANCHOR_LLM" in result["reason"]
    assert result["real_generation_calls"] is False


class _FakeGroundedLLM:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, messages) -> str:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return "not json"
        payload = json.loads(messages[-1].content)
        evidence_ids = [item["id"] for item in payload["evidence"]]
        return json.dumps(
            {
                "answerable": True,
                "claims": [{"text": "依据 E1 回答。", "citation_ids": [evidence_ids[0]]}],
                "unused_evidence": [
                    {"citation_id": citation_id, "reason": "fixture: not used"}
                    for citation_id in evidence_ids[1:]
                ],
                "limitations": [],
            },
            ensure_ascii=False,
        )


def test_evidence_only_mode_never_calls_an_injected_provider(monkeypatch) -> None:
    metrics = _load_script()
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.30")
    fake_llm = _FakeGroundedLLM()

    result = metrics._run(_dataset(metrics), "fake", "evidence_only", llm_provider=fake_llm)

    assert result["status"] == "completed"
    assert result["answer_mode"] == "evidence_only"
    assert result["llm"] is None
    assert fake_llm.calls == 0
    assert all(row["qa"]["status"] == "evidence_found" for row in result["queries"])
    assert all(row["qa"]["generation_ms"] is None for row in result["queries"])
    assert all(row["qa"]["latency_ms"] >= 0 for row in result["queries"])


def test_grounded_mode_uses_injected_provider_and_separates_outcomes(monkeypatch) -> None:
    metrics = _load_script()
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.30")
    fake_llm = _FakeGroundedLLM()

    result = metrics._run(
        _dataset(metrics),
        "fake",
        "grounded_generation",
        llm_provider=fake_llm,
    )

    assert result["status"] == "completed", result
    assert result["answer_mode"] == "grounded_generation"
    assert result["real_generation_calls"] is False
    assert result["llm"]["calls"] == 42
    assert fake_llm.calls == 42
    assert result["metrics"]["answerable_queries"] == 29
    assert result["metrics"]["unanswerable"]["count"] == 13
    assert result["metrics"]["answerable_relevant_spans"] == 40
    assert result["metrics"]["semantic_span_recall_at_k"] is not None
    assert result["metrics"]["semantic_full_coverage_at_k"] is not None
    assert len(result["queries"]) == 42
    generation = result["generation"]
    assert generation["answerable"]["answered"] == 29
    assert generation["unanswerable"]["answered"] == 13
    assert generation["citation_ids"] == {
        "checked": 42,
        "valid": 42,
        "invalid": 0,
        "duplicates": 0,
    }
    assert generation["citation_content"]["invalid"] == 0
    assert result["metrics"]["citation_locatability"] == 1.0
    assert all(row["semantic"]["latency_ms"] >= 0 for row in result["queries"])
    assert all(row["qa"]["latency_ms"] >= 0 for row in result["queries"])
    assert all(row["qa"]["generation_ms"] is not None for row in result["queries"])
    assert result["prompt"]["variant"] == "A"
    assert result["prompt"]["changed_rules"] == []
    assert result["query_selection"]["full_dataset"] is True
    assert result["merge"] is None
    assert all(row["qa"]["llm_output"]["json_ok"] for row in result["queries"])
    assert all(row["qa"]["llm_output"]["unused_evidence"] >= 0 for row in result["queries"])


def test_failed_generation_records_the_server_error_detail(monkeypatch) -> None:
    metrics = _load_script()
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.30")
    fake_llm = _FakeGroundedLLM(fail_first=True)

    result = metrics._run(
        _dataset(metrics),
        "fake",
        "grounded_generation",
        llm_provider=fake_llm,
    )

    failed = [row for row in result["queries"] if row["qa"]["status_code"] == 502]
    assert len(failed) == 1
    assert failed[0]["qa"]["error_detail"] == "LLM response could not be verified"
    assert all(
        row["qa"]["error_detail"] is None
        for row in result["queries"]
        if row["qa"]["status_code"] == 200
    )
    assert fake_llm.calls == 42


def test_subset_run_reports_partial_scope_and_uses_the_selected_prompt(
    monkeypatch,
) -> None:
    metrics = _load_script()
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.30")
    fake_llm = _FakeGroundedLLM()
    query_ids = ["zh-upload-size", "en-weather", "unans-cache-size"]

    result = metrics._run(
        _dataset(metrics),
        "fake",
        "grounded_generation",
        llm_provider=fake_llm,
        prompt_variant="B",
        query_ids=query_ids,
    )

    assert result["status"] == "completed", result
    assert [row["id"] for row in result["queries"]] == [
        "en-weather",
        "zh-upload-size",
        "unans-cache-size",
    ]
    assert result["dataset_coverage"]["dataset_queries"] == 42
    assert result["dataset_coverage"]["queries"] == 3
    assert result["dataset_coverage"]["is_full_dataset"] is False
    assert result["query_selection"]["requested_ids"] == query_ids
    assert result["query_selection"]["full_dataset"] is False
    assert result["prompt"]["variant"] == "B"
    assert result["prompt"]["changed_rules"] == [7, 12]
    assert result["merge"] is None
    assert fake_llm.calls == 3


def test_merge_reuses_precheck_rows_without_repeating_model_calls(monkeypatch) -> None:
    metrics = _load_script()
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.30")
    fake_llm = _FakeGroundedLLM()
    selected_ids = ["en-weather", "zh-upload-size", "zh-audit-content", "en-alert-queue"]
    precheck_ids = ["en-weather", "zh-upload-size"]

    precheck = metrics._run(
        _dataset(metrics),
        "fake",
        "grounded_generation",
        llm_provider=fake_llm,
        query_ids=precheck_ids,
    )
    merged = metrics._run(
        _dataset(metrics),
        "fake",
        "grounded_generation",
        llm_provider=fake_llm,
        query_ids=selected_ids,
        prior_result=precheck,
    )

    assert precheck["llm"]["calls"] == 2
    assert merged["status"] == "completed", merged
    assert merged["merge"]["prior_queries"] == 2
    assert merged["merge"]["new_queries"] == 2
    assert merged["merge"]["merged_queries"] == 4
    assert merged["merge"]["prior_generated_at"] == precheck["generated_at"]
    assert fake_llm.calls == 4
    assert merged["llm"]["calls"] == 4
    precheck_rows = {row["id"]: row for row in precheck["queries"]}
    merged_rows = {row["id"]: row for row in merged["queries"]}
    assert merged_rows["en-weather"] == precheck_rows["en-weather"]
    assert merged_rows["zh-upload-size"] == precheck_rows["zh-upload-size"]
    assert [row["id"] for row in merged["queries"]] == [
        "en-weather",
        "zh-audit-content",
        "zh-upload-size",
        "en-alert-queue",
    ]
    assert merged["metrics"]["total_queries"] == 4


def test_merge_rejects_a_different_prompt_arm(monkeypatch) -> None:
    metrics = _load_script()
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.30")
    fake_llm = _FakeGroundedLLM()
    precheck = metrics._run(
        _dataset(metrics),
        "fake",
        "grounded_generation",
        llm_provider=fake_llm,
        query_ids=["en-weather"],
    )

    wrong_variant = copy.deepcopy(precheck)
    wrong_variant["prompt"]["variant"] = "B"
    with pytest.raises(ValueError, match="prompt variant"):
        metrics._run(
            _dataset(metrics),
            "fake",
            "grounded_generation",
            llm_provider=fake_llm,
            query_ids=["en-weather"],
            prior_result=wrong_variant,
        )

    wrong_text = copy.deepcopy(precheck)
    wrong_text["prompt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt text"):
        metrics._run(
            _dataset(metrics),
            "fake",
            "grounded_generation",
            llm_provider=fake_llm,
            query_ids=["en-weather"],
            prior_result=wrong_text,
        )
    assert fake_llm.calls == 1


def test_main_passes_prompt_and_merge_options_to_the_runner(
    monkeypatch,
    tmp_path,
) -> None:
    metrics = _load_script()
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps({"status": "completed", "queries": []}),
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_run(dataset, provider, answer_mode, **kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(metrics, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_semantic_retrieval.py",
            "--prompt-variant",
            "B",
            "--query-ids",
            "en-weather, zh-upload-size",
            "--merge-from",
            str(prior_path),
        ],
    )

    assert metrics.main() == 0
    assert captured["prompt_variant"] == "B"
    assert captured["query_ids"] == ["en-weather", "zh-upload-size"]
    assert captured["prior_result"] == {"status": "completed", "queries": []}


def test_revised_dataset_records_adjudicated_counts_and_labels() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)

    metrics._validate_dataset(dataset)

    queries = dataset["queries"]
    assert dataset["name"] == "m2-01-multilingual-retrieval-labels-v2.1"
    assert len(queries) == 42
    assert sum(1 for query in queries if query["relevant"]) == 29
    assert sum(1 for query in queries if not query["relevant"]) == 13
    assert dict(Counter(query["category"] for query in queries)) == {
        "zh_to_en": 6,
        "en_paraphrase": 6,
        "cross_page": 4,
        "partial_candidates": 3,
        "keyword_distractor": 3,
        "different_conclusion": 2,
        "unanswerable": 13,
        "recall_vs_top1": 5,
    }
    assert sum(len(query["relevant"]) for query in queries) == 40
    assert sum(len(query.get("irrelevant", [])) for query in queries) == 74


def test_weather_query_is_unanswerable_and_publication_time_is_not_support() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)
    query = _query(dataset, "en-weather")

    assert query["question"] == "What does the forecast say?"
    assert query["category"] == "unanswerable"
    assert query["relevant"] == []
    irrelevant = _span_pairs(query, "irrelevant")
    assert (
        14,
        "The weather service publishes a forecast every morning.",
    ) in irrelevant
    assert (
        3,
        "The weather forecast is unrelated to this document.",
    ) not in _span_pairs(query, "relevant")


def test_latency_query_targets_record_location_not_measurement_scope() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)
    query = _query(dataset, "zh-latency")

    assert query["question"] == "检索延迟记录在哪里？"  # noqa: RUF001
    assert query["relevant"] == [
        {
            "page_index": 2,
            "text": "The evaluation report records each latency result.",
        }
    ]
    irrelevant = _span_pairs(query, "irrelevant")
    assert (2, "Retrieval latency is measured for every semantic query.") in irrelevant
    assert (3, "Retrieval latency is measured for every semantic query.") in irrelevant
    assert (
        1,
        "The preview image is generated only after the security check.",
    ) in irrelevant


def test_audit_content_labels_only_entry_fields() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)
    query = _query(dataset, "zh-audit-content")

    assert query["relevant"] == [
        {
            "page_index": 9,
            "text": "Each audit entry includes the actor and the timestamp.",
        }
    ]
    irrelevant = _span_pairs(query, "irrelevant")
    assert (9, "The audit log records every access grant change.") in irrelevant
    assert (9, "Audit entries are retained for one year.") in irrelevant


def test_alert_queue_does_not_treat_recipientless_alert_as_operator_alert() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)
    query = _query(dataset, "en-alert-queue")

    assert query["relevant"] == [
        {
            "page_index": 12,
            "text": ("Operators receive alerts when the queue depth exceeds one hundred jobs."),
        }
    ]
    irrelevant = _span_pairs(query, "irrelevant")
    cache_span = (
        5,
        "A cache hit rate below fifty percent triggers an alert.",
    )
    assert cache_span in irrelevant
    assert (12, "Operators rotate the alert routing list every quarter.") in irrelevant
    assert "operators" not in cache_span[1].lower()


def test_upload_limits_separate_page_cap_from_preview() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)
    query = _query(dataset, "cross-page-upload-limits")
    relevant = _span_pairs(query, "relevant")
    irrelevant = _span_pairs(query, "irrelevant")

    assert (
        11,
        "The maximum number of pages in a PDF is five hundred.",
    ) in relevant
    preview = (
        11,
        "PDF previews are limited to the first ten pages.",
    )
    assert preview not in relevant
    assert preview in irrelevant


def test_security_query_keeps_both_supported_mechanisms() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)
    query = _query(dataset, "zh-security")

    assert _span_pairs(query, "relevant") == {
        (
            1,
            "Photos are encrypted before upload and stored with a random identifier.",
        ),
        (
            1,
            "The preview image is generated only after the security check.",
        ),
    }


def test_login_and_export_backup_queries_keep_their_adjudicated_labels() -> None:
    metrics = _load_script()
    dataset = _dataset(metrics)

    login = _query(dataset, "keyword-login-method")
    assert _span_pairs(login, "relevant") == {
        (13, "The service does not support password-based login."),
        (13, "Authentication requires a hardware security key."),
    }
    assert _span_pairs(login, "irrelevant") == {
        (8, "Only the document owner may grant access to a shared file."),
        (8, "Access grants expire automatically after seven days."),
    }

    exports = _query(dataset, "diff-conclusion-export-backup")
    assert _span_pairs(exports, "relevant") == {
        (7, "Deleted personal exports are never kept in backups."),
    }
    assert _span_pairs(exports, "irrelevant") == {
        (6, "Backup copies of deleted uploads are kept for ninety days."),
        (7, "Personal exports are deleted immediately after download."),
    }


def _coverage_row(
    relevant_count: int,
    relevant_found: int,
    full_coverage: bool | None,
) -> dict:
    return {
        "relevant_count": relevant_count,
        "semantic": {
            "relevant_found": relevant_found,
            "full_coverage": full_coverage,
        },
    }


def test_coverage_metrics_separate_span_recall_from_full_coverage() -> None:
    metrics = _load_script()
    rows = [
        _coverage_row(3, 2, False),
        _coverage_row(1, 1, True),
        _coverage_row(0, 0, None),
    ]

    result = metrics._coverage_metrics(rows)

    assert result == {
        "answerable_relevant_spans": 4,
        "relevant_spans_found": 3,
        "semantic_span_recall_at_k": 0.75,
        "semantic_full_coverage_queries": 1,
        "semantic_full_coverage_at_k": 0.5,
    }

    empty = metrics._coverage_metrics([_coverage_row(0, 0, None)])
    assert empty["semantic_span_recall_at_k"] is None
    assert empty["semantic_full_coverage_at_k"] is None
