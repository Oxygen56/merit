"""Train-only reflection using a constrained JSON rule patch and a host renderer."""
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
    BASE_INSTRUCTIONS,
    LOOKUP_VALUES,
    PROPOSER_INSTRUCTIONS,
    TRAIN,
    CallCounter,
    make_model,
)
from ollama_support_error_trajectory_reflection import build_structured_feedback

RULE_TEXT = {
    ("task_start", "lookup", "ticket_key_from_input"): "At task start: lookup the ticket key supplied in the task input",
    ("ticket_found", "lookup", "order_key_from_ticket"): "After a ticket is found: lookup the order key returned by that ticket",
    ("order_found", "lookup", "policy_key_from_order"): "After an order is found: lookup the policy key returned by that order",
    ("policy_found", "calculator", "order_total_times_policy_rate"): "After a policy is found: calculate the order total multiplied by the policy rate using returned numeric values",
    ("required_record_missing", "final", "unavailable_status"): "If any required record is actually missing: return REFUND_UNAVAILABLE",
    ("calculation_complete", "final", "calculator_result"): "After a successful calculation: return only the calculator result",
}
PATCH_OUTPUT_REQUIREMENT = """Return only one JSON object with key "rules". Each rule must contain exactly "when", "action", and "argument_source".
Choose from these allowed combinations:
[task_start, lookup, ticket_key_from_input]
[ticket_found, lookup, order_key_from_ticket]
[order_found, lookup, policy_key_from_order]
[policy_found, calculator, order_total_times_policy_rate]
[required_record_missing, final, unavailable_status]
[calculation_complete, final, calculator_result]
Use each "when" at most once. Do not include record keys, placeholder values, prose, or held-out tasks."""


def parse_patch(
    text: str,
    rule_text: dict[tuple[str, str, str], str] = RULE_TEXT,
) -> list[dict[str, str]]:
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Patch must be one JSON object: {error.msg}") from error
    if not isinstance(payload, dict) or set(payload) != {"rules"}:
        raise ValueError("Patch must contain exactly one top-level key: rules")
    rules = payload["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= len(rule_text):
        raise ValueError(f"rules must contain between 1 and {len(rule_text)} items")
    seen = set()
    normalized = []
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"when", "action", "argument_source"}:
            raise ValueError("Every rule must contain exactly when, action, and argument_source")
        if any(not isinstance(rule[key], str) for key in rule):
            raise ValueError("Every rule field must be a string")
        key = (rule["when"], rule["action"], rule["argument_source"])
        if key not in rule_text:
            raise ValueError(f"Unsupported rule combination: {list(key)}")
        if rule["when"] in seen:
            raise ValueError(f"Duplicate when condition: {rule['when']}")
        seen.add(rule["when"])
        normalized.append(dict(rule))
    return normalized


def render_patch(
    current: AgentSpec,
    text: str,
    rule_text: dict[tuple[str, str, str], str] = RULE_TEXT,
) -> AgentSpec:
    rules = parse_patch(text, rule_text)
    lines = [
        BASE_INSTRUCTIONS,
        "",
        "Follow these workflow rules using actual runtime values. Never execute placeholder text from examples:",
    ]
    lines.extend(f"- {rule_text[(rule['when'], rule['action'], rule['argument_source'])]}." for rule in rules)
    return current.evolve(instructions="\n".join(lines))


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
    start = AgentSpec(source["candidate_instructions"], {"temperature": 0, "max_tokens": 128, "seed": 101})
    initial = evaluate_agent(executor, start, TRAIN)
    accepted_patch = None

    def render_and_record(current: AgentSpec, text: str) -> AgentSpec:
        nonlocal accepted_patch
        candidate = render_patch(current, text)
        accepted_patch = parse_patch(text)
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
        output_requirement=PATCH_OUTPUT_REQUIREMENT,
    )
    optimizer = AgentOptimizerAdapter(
        "model_structured_patch", proposer, proposer_source=model.name
    )
    budget = Budget(1)

    def evaluate_candidate(candidate):
        budget.consume()
        return evaluate_agent(executor, candidate, TRAIN)

    summary = {
        "seed": 101,
        "heldout_tasks_evaluated": 0,
        "source_heldout_tasks_evaluated": source["heldout_tasks_evaluated"],
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
        default=Path("results/support-error-reflection-retry-pilot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/support-error-structured-patch-simple-enum-pilot"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.source_pilot, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
