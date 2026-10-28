"""Train-only patch with immediate-stop feedback and trajectory-gated selection."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merit.agent_contract import AgentOptimizerAdapter, AgentSpec, evaluate_agent
from merit.agent_methods import ModelReflectionProposer
from merit.budget import Budget
from merit.multi_step_executor import MultiStepToolExecutor
from ollama_support_error_recovery_patch import (
    RECOVERY_RULE_TEXT,
    validate_recovery_coverage,
)
from ollama_support_error_reflection import (
    LOOKUP_VALUES,
    PROPOSER_INSTRUCTIONS,
    TRAIN,
    CallCounter,
    make_model,
)
from ollama_support_error_structured_patch import parse_patch, render_patch
from ollama_support_error_trajectory_reflection import build_structured_feedback

STOP_RULE_TEXT = {
    **RECOVERY_RULE_TEXT,
    ("required_record_failure", "avoid", "further_lookup_or_calculation"):
        "Once a required record lookup fails, return REFUND_UNAVAILABLE immediately; do not perform any further lookups or calculations",
}
STOP_OUTPUT_REQUIREMENT = """Return only one JSON object with key "rules". Each rule must contain exactly "when", "action", and "argument_source".
Choose from these allowed combinations:
[task_start, lookup, ticket_key_from_input]
[ticket_found, lookup, order_key_from_ticket]
[order_found, lookup, policy_key_from_order]
[policy_found, calculator, order_total_times_policy_rate]
[required_record_missing, final, unavailable_status]
[calculation_complete, final, calculator_result]
[before_policy_found, avoid, premature_calculation]
[lookup_failed, avoid, same_failed_lookup_retry]
[required_record_failure, avoid, further_lookup_or_calculation]
Use each "when" at most once. Include rules that correct every detected trajectory issue while preserving correct behavior. Do not include record keys, placeholder values, prose, or held-out tasks."""
SELECTION_ISSUE_KEYS = (
    "tasks_skipping_input_ticket_lookup",
    "literal_placeholder_arguments",
    "tasks_calculating_before_policy_lookup",
    "tasks_repeating_failed_lookups",
    "tasks_continuing_after_required_record_failure",
    "tasks_without_final_answer",
)


def validate_stop_coverage(rules, feedback) -> None:
    validate_recovery_coverage(rules, feedback)
    issues = feedback["detected_issues"]
    if (
        issues["tasks_continuing_after_required_record_failure"]
        and "required_record_failure" not in {rule["when"] for rule in rules}
    ):
        raise ValueError(
            "Patch does not address every detected trajectory issue; add rules for: required_record_failure"
        )


def selection_issues(feedback) -> dict[str, object]:
    issues = feedback["detected_issues"]
    return {key: issues[key] for key in SELECTION_ISSUE_KEYS if issues[key]}


def accept_candidate(current_accuracy, candidate_accuracy, current_feedback, candidate_feedback) -> bool:
    if selection_issues(candidate_feedback):
        return False
    if candidate_accuracy > current_accuracy:
        return True
    return candidate_accuracy == current_accuracy and bool(selection_issues(current_feedback))


def run(source_pilot: Path, output_root: Path) -> dict:
    source = json.loads((source_pilot / "pilot.json").read_text())
    source_traces = json.loads((source_pilot / "agent_traces.json").read_text())
    if source["heldout_tasks_evaluated"] != 0 or not source["candidate_instructions"]:
        raise ValueError("Source must be a train-only pilot with a candidate")
    feedback = build_structured_feedback(TRAIN, source_traces, source["candidate_results"])
    initial_issues = selection_issues(feedback)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "structured_train_feedback.json").write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2) + "\n"
    )

    model = make_model(output_root / "cache")
    executor = MultiStepToolExecutor(model, LOOKUP_VALUES, max_steps=6, recover_tool_errors=True)
    start = AgentSpec(source["candidate_instructions"], {"temperature": 0, "max_tokens": 128, "seed": 101})
    initial = evaluate_agent(executor, start, TRAIN)
    accepted_patch = None
    candidate_feedback = None

    def render_and_record(current: AgentSpec, text: str) -> AgentSpec:
        nonlocal accepted_patch
        rules = parse_patch(text, STOP_RULE_TEXT)
        validate_stop_coverage(rules, feedback)
        candidate = render_patch(current, text, STOP_RULE_TEXT)
        accepted_patch = rules
        return candidate

    proposer_model = CallCounter(model)
    proposer_spec = AgentSpec(
        PROPOSER_INSTRUCTIONS + "\nReturn the requested constrained JSON rule patch, not a revised prompt.",
        {"temperature": 0.2, "max_tokens": 1024, "seed": 10_101},
    )
    proposer = ModelReflectionProposer(
        proposer_model,
        proposer_spec,
        max_failures=5,
        max_attempts=3,
        feedback_provider=lambda context: feedback,
        proposal_renderer=render_and_record,
        output_requirement=STOP_OUTPUT_REQUIREMENT,
    )
    optimizer = AgentOptimizerAdapter(
        "model_stop_patch",
        proposer,
        accept=lambda current, candidate: accept_candidate(
            current, candidate, feedback, candidate_feedback
        ),
        proposer_source=model.name,
    )
    budget = Budget(1)

    def evaluate_candidate(candidate):
        nonlocal candidate_feedback
        budget.consume()
        evaluation = evaluate_agent(executor, candidate, TRAIN)
        candidate_feedback = build_structured_feedback(
            TRAIN, executor.traces, [asdict(row) for row in evaluation.results]
        )
        return evaluation

    summary = {
        "seed": 101,
        "heldout_tasks_evaluated": 0,
        "source_heldout_tasks_evaluated": source["heldout_tasks_evaluated"],
        "source_candidate_train_accuracy": source["candidate_train_accuracy"],
        "selection_policy": "accuracy improvement or equal accuracy with valid trajectories replacing invalid ones",
        "initial_selection_issues": initial_issues,
        "proposal_model_calls": 0,
        "initial_train_accuracy": initial.accuracy,
        "initial_results": [asdict(row) for row in initial.results],
        "candidate_train_accuracy": None,
        "candidate_accepted": None,
        "candidate_trajectory_valid": None,
        "candidate_selection_issues": None,
        "candidate_instructions": None,
        "candidate_patch": None,
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
            "candidate_trajectory_valid": not selection_issues(candidate_feedback),
            "candidate_selection_issues": selection_issues(candidate_feedback),
            "candidate_instructions": entry.candidate.instructions,
            "candidate_patch": accepted_patch,
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
    if candidate_feedback is not None:
        (output_root / "candidate_train_feedback.json").write_text(
            json.dumps(candidate_feedback, ensure_ascii=False, indent=2) + "\n"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pilot", type=Path,
        default=Path("results/support-error-recovery-patch-validated-pilot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/support-error-stop-patch-pilot"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.source_pilot, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
