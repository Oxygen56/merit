"""Train-only reflection from the latest rejected trajectory with priority rendering."""
from __future__ import annotations

import argparse
import json
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
)
from ollama_support_error_stop_patch import (
    STOP_OUTPUT_REQUIREMENT,
    STOP_RULE_TEXT,
    accept_candidate,
    selection_issues,
    validate_stop_coverage,
)
from ollama_support_error_structured_patch import parse_patch, render_patch
from ollama_support_error_trajectory_reflection import build_structured_feedback

RULE_PRIORITY = (
    "required_record_failure",
    "before_policy_found",
    "lookup_failed",
    "task_start",
    "ticket_found",
    "order_found",
    "policy_found",
    "required_record_missing",
    "calculation_complete",
)


def render_priority_patch(current, text):
    rules = parse_patch(text, STOP_RULE_TEXT)
    ordered = sorted(rules, key=lambda rule: RULE_PRIORITY.index(rule["when"]))
    return render_patch(current, json.dumps({"rules": ordered}), STOP_RULE_TEXT)


def run(source_pilot: Path, output_root: Path, seed: int = 101) -> dict:
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
        {"temperature": 0, "max_tokens": 128, "seed": seed},
    )
    initial = evaluate_agent(executor, start, TRAIN)
    accepted_patch = None
    candidate_feedback = None

    def render_and_record(current: AgentSpec, text: str) -> AgentSpec:
        nonlocal accepted_patch
        rules = parse_patch(text, STOP_RULE_TEXT)
        validate_stop_coverage(rules, feedback)
        candidate = render_priority_patch(current, text)
        accepted_patch = parse_patch(candidate.instructions and text, STOP_RULE_TEXT)
        return candidate

    proposer_model = CallCounter(model)
    proposer_spec = AgentSpec(
        PROPOSER_INSTRUCTIONS + "\nReturn a constrained JSON rule patch, not a revised prompt. The latest rejected candidate still performs actions after a required record failure.",
        {"temperature": 0.2, "max_tokens": 1024, "seed": seed + 10_000},
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
        "model_dynamic_patch",
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
        "seed": seed,
        "heldout_tasks_evaluated": 0,
        "source_heldout_tasks_evaluated": source["heldout_tasks_evaluated"],
        "source_candidate_train_accuracy": source["candidate_train_accuracy"],
        "renderer": "required-record-failure-first",
        "initial_selection_issues": selection_issues(feedback),
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
        result = optimizer.optimize(start, TRAIN, initial, budget, seed, evaluate_candidate)
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
        default=Path("results/support-error-stop-patch-pilot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/support-error-dynamic-patch-pilot"),
    )
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    print(json.dumps(run(args.source_pilot, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
