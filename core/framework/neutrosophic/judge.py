"""Optional judge adapter backed by neutrosophic scoring."""

from __future__ import annotations

from typing import Any

from framework.agent_loop.internals.types import JudgeVerdict
from framework.neutrosophic.scoring import NeutrosophicDecision, NeutrosophicScore


def score_judge_context(context: dict[str, Any]) -> NeutrosophicScore:
    """Convert a judge context into a neutrosophic score."""
    assistant_text = str(context.get("assistant_text") or "").strip()
    missing_keys = context.get("missing_keys", [])
    tool_calls = context.get("tool_calls", [])
    iteration = context.get("iteration", 0)

    if not isinstance(missing_keys, list):
        missing_keys = []
    if not isinstance(tool_calls, list):
        tool_calls = []
    if not isinstance(iteration, int):
        iteration = 0

    truth = 0.2
    indeterminacy = 0.35
    falsity = 0.05
    rationale: list[str] = []

    if assistant_text:
        truth += 0.25
        indeterminacy -= 0.05
        rationale.append("assistant_text_present")
    else:
        indeterminacy += 0.15
        rationale.append("assistant_text_missing")

    if missing_keys:
        indeterminacy += min(0.35, 0.12 * len(missing_keys))
        rationale.append(f"missing_keys={len(missing_keys)}")
    else:
        truth += 0.35
        rationale.append("required_outputs_present")

    if tool_calls:
        indeterminacy += 0.08
        rationale.append("tool_calls_pending")

    if iteration >= 7 and missing_keys:
        falsity += 0.3
        rationale.append("late_iteration_incomplete")

    return NeutrosophicScore(truth, indeterminacy, falsity, tuple(rationale))


class NeutrosophicJudge:
    """Opt-in judge that maps T/I/F pressure to Hive judge verdicts."""

    def __init__(self, task: str, max_iterations: int = 10):
        self._task = task
        self._max_iterations = max_iterations

    async def evaluate(self, context: dict[str, Any]) -> JudgeVerdict:
        score = score_judge_context(context)
        missing_keys = context.get("missing_keys", [])
        if not isinstance(missing_keys, list):
            missing_keys = []

        if score.decision == NeutrosophicDecision.ACCEPT:
            return JudgeVerdict(action="ACCEPT", feedback="")

        if score.decision == NeutrosophicDecision.ESCALATE:
            return JudgeVerdict(
                action="ESCALATE",
                feedback=self._feedback("Escalating because the current attempt is incomplete late in the run.", score),
            )

        remaining = self._remaining_iterations(context)
        if missing_keys:
            return JudgeVerdict(
                action="RETRY",
                feedback=self._feedback(
                    f"Missing required outputs: {missing_keys}. Remaining iterations: {remaining}.",
                    score,
                ),
            )

        if score.indeterminacy >= 0.65:
            return JudgeVerdict(
                action="RETRY",
                feedback=self._feedback(
                    "The answer is still too indeterminate. Add evidence or clarify the result.",
                    score,
                ),
            )

        return JudgeVerdict(action="ACCEPT", feedback="")

    def _remaining_iterations(self, context: dict[str, Any]) -> int:
        iteration = context.get("iteration", 0)
        if not isinstance(iteration, int):
            iteration = 0
        return max(0, self._max_iterations - iteration - 1)

    def _feedback(self, message: str, score: NeutrosophicScore) -> str:
        score_data = score.to_dict()
        return (
            f"Your task: {self._task}\n"
            f"{message}\n"
            "Neutrosophic score: "
            f"T={score_data['truth']}, I={score_data['indeterminacy']}, "
            f"F={score_data['falsity']}, decision={score_data['decision']}."
        )
