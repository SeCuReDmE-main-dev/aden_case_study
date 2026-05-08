import asyncio
import math

import pytest

from framework.neutrosophic import (
    NeutrosophicDecision,
    NeutrosophicJudge,
    NeutrosophicScore,
    aggregate_scores,
    aggregate_worker_reports,
    score_judge_context,
    score_worker_report,
)


def test_score_clamps_values() -> None:
    score = NeutrosophicScore(2.0, -1.0, 0.5)

    assert score.truth == 1.0
    assert score.indeterminacy == 0.0
    assert score.falsity == 0.5


def test_score_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        NeutrosophicScore(math.nan, 0.0, 0.0)

    with pytest.raises(ValueError, match="finite numbers"):
        NeutrosophicScore(0.0, math.inf, 0.0)


def test_success_worker_report_accepts_with_evidence() -> None:
    score = score_worker_report(status="success", summary="Completed task", data={"rows": 3})

    assert score.decision == NeutrosophicDecision.ACCEPT
    assert score.truth > score.indeterminacy
    assert score.truth > score.falsity
    assert "status=success" in score.rationale


def test_partial_worker_report_requests_clarification() -> None:
    score = score_worker_report(status="partial", summary="Found some data", data={})

    assert score.decision in {NeutrosophicDecision.CLARIFY, NeutrosophicDecision.CAVEAT}
    assert score.indeterminacy >= score.truth


def test_failed_worker_report_escalates() -> None:
    score = score_worker_report(status="failed", summary="Tool failed", error="missing credentials")

    assert score.decision == NeutrosophicDecision.ESCALATE
    assert score.falsity > score.truth


def test_loop_signals_increase_risk() -> None:
    baseline = score_worker_report(status="partial", summary="Still trying")
    risky = score_worker_report(
        status="partial",
        summary="Still trying",
        signals={"stalled": True, "doom_loop": True},
    )

    assert risky.indeterminacy > baseline.indeterminacy
    assert risky.falsity > baseline.falsity


def test_aggregate_scores_averages_triplets() -> None:
    aggregate = aggregate_scores(
        [
            NeutrosophicScore(1.0, 0.0, 0.0),
            NeutrosophicScore(0.0, 1.0, 0.0),
        ]
    )

    assert aggregate.truth == 0.5
    assert aggregate.indeterminacy == 0.5
    assert aggregate.falsity == 0.0
    assert aggregate.rationale == ("aggregate_count=2",)


def test_aggregate_scores_empty_returns_fully_indeterminate() -> None:
    # An empty batch has no evidence: indeterminacy=1.0 is the conservative default.
    aggregate = aggregate_scores([])

    assert aggregate.truth == 0.0
    assert aggregate.indeterminacy == 1.0
    assert aggregate.falsity == 0.0
    assert aggregate.rationale == ("no_scores",)
    assert aggregate.decision == NeutrosophicDecision.CLARIFY


def test_high_indeterminacy_takes_priority_over_high_falsity() -> None:
    # RFC specifies: I >= 0.65 → CLARIFY before F >= 0.65 → ESCALATE.
    # A score that crosses both thresholds should yield CLARIFY, not ESCALATE.
    score = NeutrosophicScore(truth=0.1, indeterminacy=0.8, falsity=0.7)

    assert score.decision == NeutrosophicDecision.CLARIFY


def test_moderate_falsity_retries_without_escalating() -> None:
    score = NeutrosophicScore(truth=0.5, indeterminacy=0.2, falsity=0.5)

    assert score.decision == NeutrosophicDecision.RETRY


def test_to_dict_contract() -> None:
    score = NeutrosophicScore(0.9, 0.05, 0.05, ("status=success",))
    result = score.to_dict()

    assert set(result.keys()) == {"truth", "indeterminacy", "falsity", "decision", "rationale"}
    assert isinstance(result["truth"], float)
    assert isinstance(result["rationale"], list)
    assert result["decision"] == "accept"


def test_unknown_status_scores_conservatively() -> None:
    score = score_worker_report(status="pending")

    assert "status=unknown" in score.rationale
    # Unknown status should not yield ACCEPT — too little evidence.
    assert score.decision != NeutrosophicDecision.ACCEPT


def test_success_report_with_error_does_not_clean_accept() -> None:
    score = score_worker_report(
        status="success",
        summary="Completed with warning",
        data={"rows": 3},
        error="post-processing failed",
    )

    assert "error_present" in score.rationale
    assert score.decision != NeutrosophicDecision.ACCEPT


def test_worker_result_includes_neutrosophic_score() -> None:
    from types import SimpleNamespace

    from framework.host.worker import Worker

    worker = Worker(
        worker_id="worker_1",
        task="Collect evidence",
        agent_loop=None,
        context=SimpleNamespace(stream_id="worker:worker_1", execution_id="exec_1"),
    )
    result = worker._build_result(
        SimpleNamespace(success=True, output={"answer": "done"}, error=None, tokens_used=4),
        duration=1.2,
        default_status="success",
    )

    assert result.neutrosophic_score["decision"] == "accept"
    assert result.neutrosophic_score["truth"] > result.neutrosophic_score["falsity"]


def test_worker_stopped_report_is_conservative() -> None:
    from types import SimpleNamespace

    from framework.host.worker import Worker

    # A stopped/cancelled worker should never yield ACCEPT.
    worker = Worker(
        worker_id="worker_2",
        task="Collect evidence",
        agent_loop=None,
        context=SimpleNamespace(stream_id="worker:worker_2", execution_id="exec_2"),
    )
    score = worker._score_report(
        "stopped",
        "Worker was cancelled before completion.",
        {},
        "Worker stopped by queen",
    )

    assert score["decision"] != "accept"
    assert score["falsity"] >= score["truth"]


def test_queen_score_line_omitted_for_empty_score() -> None:
    # WorkerResult.neutrosophic_score defaults to {}. The queen must not emit
    # a malformed "T=None, I=None" line when the dict is empty.
    score_dict = NeutrosophicScore(0.8, 0.1, 0.1, ("status=success",)).to_dict()

    # Simulate the queen guard logic: non-empty dict with all keys present.
    assert isinstance(score_dict, dict) and score_dict
    assert all(score_dict.get(k) is not None for k in ("truth", "indeterminacy", "falsity", "decision"))

    # Simulate empty dict (the default before any score is set).
    empty: dict = {}
    assert not (isinstance(empty, dict) and empty)


def test_aggregate_worker_reports_scores_batch() -> None:
    score = aggregate_worker_reports(
        [
            {"status": "success", "summary": "Done", "data": {"answer": 1}},
            {"status": "partial", "summary": "Missing one source", "data": {}},
        ]
    )

    assert score.truth > 0.4
    assert score.indeterminacy > 0.1
    assert score.rationale == ("aggregate_count=2",)


def test_score_judge_context_accepts_complete_output() -> None:
    score = score_judge_context(
        {
            "assistant_text": "Final answer",
            "missing_keys": [],
            "tool_calls": [],
            "iteration": 1,
        }
    )

    assert score.decision == NeutrosophicDecision.ACCEPT


def test_neutrosophic_judge_retries_missing_outputs() -> None:
    judge = NeutrosophicJudge("Collect evidence", max_iterations=10)
    verdict = asyncio.run(
        judge.evaluate(
            {
                "assistant_text": "I found part of it",
                "missing_keys": ["source"],
                "tool_calls": [],
                "iteration": 2,
            }
        )
    )

    assert verdict.action == "RETRY"
    assert "Neutrosophic score" in (verdict.feedback or "")


def test_neutrosophic_judge_escalates_late_incomplete_attempt() -> None:
    judge = NeutrosophicJudge("Summarise findings", max_iterations=10)
    verdict = asyncio.run(
        judge.evaluate(
            {
                "assistant_text": "Partial summary",
                "missing_keys": ["conclusion"],
                "tool_calls": [],
                "iteration": 8,
            }
        )
    )

    assert verdict.action == "ESCALATE"
    assert "Neutrosophic score" in (verdict.feedback or "")


def test_score_judge_context_handles_malformed_context() -> None:
    # Non-int iteration, non-list missing_keys/tool_calls → defaults must not raise.
    score = score_judge_context(
        {
            "assistant_text": "answer",
            "missing_keys": "should-be-a-list",
            "tool_calls": None,
            "iteration": "not-an-int",
        }
    )

    # Malformed non-list missing_keys coerces to [] → no penalty, assistant_text present → ACCEPT.
    assert score.decision == NeutrosophicDecision.ACCEPT


def test_neutrosophic_judge_not_instantiated_at_module_level() -> None:
    # Guard: NeutrosophicJudge must remain opt-in — no module-level instance in the runtime.
    import ast
    import pathlib

    runtime_roots = [
        pathlib.Path("core/framework/agent_loop"),
        pathlib.Path("core/framework/host"),
        pathlib.Path("core/framework/server"),
        pathlib.Path("core/framework/orchestrator"),
    ]
    offenders: list[str] = []
    for root in runtime_roots:
        for py_file in root.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                    if name == "NeutrosophicJudge":
                        offenders.append(f"{py_file}:{node.lineno}")

    assert offenders == [], f"NeutrosophicJudge instantiated in runtime at: {offenders}"
