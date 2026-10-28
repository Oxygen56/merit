"""Train-only constrained patch with feedback for recovery and action ordering."""
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
from ollama_support_error_structured_patch import RULE_TEXT, parse_patch, render_patch
from ollama_support_error_trajectory_reflection import build_structured_feedback

RECOVERY_RULE_TEXT = {
    **RULE_TEXT,
    ("before_policy_found", "avoid", "premature_calculation"):
        "Do not calculate before the policy lookup has succeeded and returned its numeric rate",
    ("lookup_failed", "avoid", "same_failed_lookup_retry"):
        "After a lookup fails, do not repeat the same failed lookup; if the required record is missing, return REFUND_UNAVAILABLE",
}
RECOVERY_OUTPUT_REQUIREMENT = """Return only one JSON object with key "rules". Each rule must contain exactly "when", "action", and "argument_source".
Choose from these allowed combinations:
[task_start, lookup, ticket_key_from_input]
[ticket_found, lookup, order_key_from_ticket]
[order_found, lookup, policy_key_from_order]
[policy_found, calculator, order_total_times_policy_rate]
[required_record_missing, final, unavailable_status]
[calculation_complete, final, calculator_result]
[before_policy_found, avoid, premature_calculation]
[lookup_failed, avoid, same_failed_lookup_retry]
Use each "when" at most once. Include rules that correct every detected trajectory issue while preserving correct behavior. Do not include record keys, placeholder values, prose, or held-out tasks."""


def validate_recovery_coverage(rules, feedback) -> None:
    issues = feedback["detected_issues"]
    required = set()
    if issues["tasks_calculating_before_policy_lookup"]:
        required.add("before_policy_found")
    if issues["tasks_repeating_failed_lookups"]:
        required.add("lookup_failed")
    missing = sorted(required - {rule["when"] for rule in rules})
    if missing:
        raise ValueError(
            "Patch does not address every detected trajectory issue; add rules for: "
            + ", ".join(missing)
        )


def run(source_pilot: Path, output_root: Path, require_recovery_rules: bool = False) -> dict:
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
    start = AgentSpec(source["candidate_instructions"], {"temperature": 0, "max_tokens": 128, "seed": 101})
    initial = evaluate_agent(executor, start, TRAIN)
    accepted_patch = None

    def render_and_record(current: AgentSpec, text: str) -> AgentSpec:
        nonlocal accepted_patch
        rules = parse_patch(text, RECOVERY_RULE_TEXT)
        if require_recovery_rules:
            validate_recovery_coverage(rules, feedback)
        candidate = render_patch(current, text, RECOVERY_RULE_TEXT)
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
        output_requirement=RECOVERY_OUTPUT_REQUIREMENT,
    )
    optimizer = AgentOptimizerAdapter("model_recovery_patch", proposer, proposer_source=model.name)
    budget = Budget(1)

    def evaluate_candidate(candidate):
        budget.consume()
        return evaluate_agent(executor, candidate, TRAIN)

    summary = {
        "seed": 101,
        "heldout_tasks_evaluated": 0,
        "source_heldout_tasks_evaluated": source["heldout_tasks_evaluated"],
        "source_candidate_train_accuracy": source["candidate_train_accuracy"],
        "recovery_rule_validation": require_recovery_rules,
        "proposal_model_calls": 0,
        "initial_train_accuracy": initial.accuracy,
        "initial_results": [asdict(row) for row in initial.results],
        "candidate_train_accuracy": None,
        "candidate_accepted": None,
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pilot", type=Path,
        default=Path("results/support-error-structured-patch-simple-enum-pilot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/support-error-recovery-patch-pilot"),
    )
    parser.add_argument("--require-recovery-rules", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.source_pilot,
        args.output,
        require_recovery_rules=args.require_recovery_rules,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
