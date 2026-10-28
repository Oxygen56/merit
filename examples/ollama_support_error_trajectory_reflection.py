"""Train-only reflection with sanitized feedback from a rejected candidate's traces."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from merit.agent_contract import AgentOptimizerAdapter, AgentSpec, evaluate_agent
from merit.agent_methods import ModelReflectionProposer
from merit.budget import Budget
from merit.multi_step_executor import MultiStepToolExecutor
from ollama_support_error_reflection import (
    LOOKUP_VALUES,
    PROPOSER_INSTRUCTIONS,
    TRAIN,
    CallCounter,
    make_model,
    validate_complete_instructions,
)


def latest_attempt(traces, task_id: str):
    rows = [row for row in traces if row["task_id"] == task_id]
    starts = [index for index, row in enumerate(rows) if row["step"] == 0]
    if not starts:
        raise ValueError(f"No trace for {task_id}")
    return sorted(rows[starts[-1]:], key=lambda row: row["step"])


def is_placeholder(argument: str) -> bool:
    return bool(re.fullmatch(r"<[^>]+>|[a-z]+(?:_[a-z]+)+", argument))


def build_structured_feedback(
    tasks, traces, results, allow_heldout: bool = False
) -> dict[str, object]:
    outcomes = {row["task_id"]: row["correct"] for row in results}
    task_feedback = []
    for task in tasks:
        if task.task_id.startswith("heldout-") and not allow_heldout:
            raise ValueError("Trajectory feedback must not include held-out tasks")
        rows = latest_attempt(traces, task.task_id)
        ticket_match = re.search(r"support ticket (\S+)\.", task.prompt)
        if ticket_match is None:
            raise ValueError(f"No ticket key in {task.task_id}")
        ticket = ticket_match.group(1)
        lookups = [row for row in rows if row["action"] == "lookup"]
        placeholders = sorted({
            row["argument"] for row in lookups if is_placeholder(row["argument"])
        })
        successful_policy_steps = {
            row["step"] for row in lookups
            if row["argument"].startswith("policy-") and row["tool_error"] is None
        }
        premature_calculation = any(
            row["action"] == "calculator" and row["step"] not in successful_policy_steps
            and not any(step < row["step"] for step in successful_policy_steps)
            for row in rows
        )
        failed_lookup_counts = {}
        for row in lookups:
            if row["tool_error"] is not None:
                failed_lookup_counts[row["argument"]] = failed_lookup_counts.get(row["argument"], 0) + 1
        repeated_failed_lookups = sum(
            count - 1 for count in failed_lookup_counts.values() if count > 1
        )
        required_failure_steps = {
            row["step"] for row in lookups
            if row["tool_error"] is not None
            and row["argument"].startswith(("ticket-", "order-", "policy-"))
        }
        post_failure_actions = sum(
            row["action"] in {"lookup", "calculator"}
            and any(step < row["step"] for step in required_failure_steps)
            for row in rows
        )
        task_feedback.append({
            "task_id": task.task_id,
            "correct": outcomes[task.task_id],
            "action_sequence": [row["action"] for row in rows],
            "first_lookup_used_input_ticket": bool(
                lookups and lookups[0]["argument"] == ticket
            ),
            "literal_placeholder_lookup_arguments": placeholders,
            "calculation_before_successful_policy_lookup": premature_calculation,
            "repeated_failed_lookup_count": repeated_failed_lookups,
            "actions_after_required_record_failure": post_failure_actions,
            "ended_with_final": bool(rows and rows[-1]["action"] == "final"),
            "tool_errors": sorted({
                row["tool_error"] for row in rows if row["tool_error"] is not None
            }),
        })
    return {
        "task_traces": task_feedback,
        "detected_issues": {
            "tasks_skipping_input_ticket_lookup": [
                row["task_id"] for row in task_feedback
                if not row["first_lookup_used_input_ticket"]
            ],
            "literal_placeholder_arguments": sorted({
                argument for row in task_feedback
                for argument in row["literal_placeholder_lookup_arguments"]
            }),
            "tasks_calculating_before_policy_lookup": [
                row["task_id"] for row in task_feedback
                if row["calculation_before_successful_policy_lookup"]
            ],
            "tasks_repeating_failed_lookups": [
                row["task_id"] for row in task_feedback
                if row["repeated_failed_lookup_count"] > 0
            ],
            "tasks_continuing_after_required_record_failure": [
                row["task_id"] for row in task_feedback
                if row["actions_after_required_record_failure"] > 0
            ],
            "tasks_without_final_answer": [
                row["task_id"] for row in task_feedback if not row["ended_with_final"]
            ],
        },
        "general_corrections": [
            "Treat placeholders in instruction examples as non-executable text.",
            "Start each task by looking up the ticket key supplied in that task input.",
            "Use returned record keys for dependent lookups; stop only after an actual required record is missing.",
            "Do not calculate until the policy lookup has succeeded and returned a numeric rate.",
            "Never repeat a failed lookup with the same key; after a required record is missing, return REFUND_UNAVAILABLE.",
            "Once a required record lookup fails, return REFUND_UNAVAILABLE immediately without further lookups or calculations.",
        ],
    }


def run(source_pilot: Path, output_root: Path) -> dict:
    source = json.loads((source_pilot / "pilot.json").read_text())
    source_traces = json.loads((source_pilot / "agent_traces.json").read_text())
    if source["heldout_tasks_evaluated"] != 0 or not source["candidate_instructions"]:
        raise ValueError("Source must be a train-only pilot with a candidate")
    feedback = build_structured_feedback(TRAIN, source_traces, source["candidate_results"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "structured_train_feedback.json").write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2) + "\n"
    )

    model = make_model(output_root / "cache")
    executor = MultiStepToolExecutor(model, LOOKUP_VALUES, max_steps=6, recover_tool_errors=True)
    start = AgentSpec(
        source["candidate_instructions"],
        {"temperature": 0, "max_tokens": 128, "seed": 101},
    )
    initial = evaluate_agent(executor, start, TRAIN)
    proposer_model = CallCounter(model)
    proposer_spec = AgentSpec(
        PROPOSER_INSTRUCTIONS + "\nUse structured_train_feedback to repair executable behavior, not merely the wording.",
        {"temperature": 0.2, "max_tokens": 1024, "seed": 10_101},
    )
    proposer = ModelReflectionProposer(
        proposer_model,
        proposer_spec,
        max_failures=5,
        instruction_validator=validate_complete_instructions,
        max_attempts=3,
        feedback_provider=lambda context: feedback,
    )
    optimizer = AgentOptimizerAdapter(
        "model_reflection_with_trajectory_feedback",
        proposer,
        proposer_source=model.name,
    )
    budget = Budget(1)

    def evaluate_candidate(candidate):
        budget.consume()
        return evaluate_agent(executor, candidate, TRAIN)

    summary = {
        "seed": 101,
        "heldout_tasks_evaluated": 0,
        "source_heldout_tasks_evaluated": source["heldout_tasks_evaluated"],
        "source_candidate_train_accuracy": source["candidate_train_accuracy"],
        "proposal_model_calls": 0,
        "initial_train_accuracy": initial.accuracy,
        "initial_results": [asdict(row) for row in initial.results],
        "candidate_train_accuracy": None,
        "candidate_accepted": None,
        "candidate_instructions": None,
        "candidate_results": [],
        "candidate_evaluations": 0,
    }
    try:
        result = optimizer.optimize(start, TRAIN, initial, budget, 101, evaluate_candidate)
    except ValueError as error:
        summary["proposal_error"] = str(error)
        if hasattr(error, "usage"):
            summary["proposer_usage"] = asdict(error.usage)
    else:
        entry = result.history[0]
        summary.update({
            "candidate_train_accuracy": entry.evaluation.accuracy,
            "candidate_accepted": entry.accepted,
            "candidate_instructions": entry.candidate.instructions,
            "candidate_results": [asdict(row) for row in entry.evaluation.results],
            "proposer_usage": asdict(entry.proposer_usage),
        })
    summary.update({
        "proposal_model_calls": proposer_model.calls,
        "candidate_evaluations": budget.used,
        "model_steps": len(executor.traces),
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "paid_api_calls": 0,
    })
    (output_root / "pilot.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (output_root / "agent_traces.json").write_text(
        json.dumps(executor.traces, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pilot",
        type=Path,
        default=Path("results/support-error-reflection-retry-pilot"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/support-error-trajectory-feedback-pilot"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.source_pilot, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
