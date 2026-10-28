"""Run repeated local-model checks on dependent support-workflow tool calls."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from merit.agent_contract import AgentOptimizerAdapter, AgentSpec, AgentTask, evaluate_agent
from merit.agent_methods import AgentNoChangeOptimizer, FixedInstructionsProposer, accept_always
from merit.agent_pair import AgentTaskBundle, make_agent_paired_manifest, run_agent_paired
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.evidence import validate_result_directory
from merit.multi_step_executor import MultiStepToolExecutor, safe_calculator
from merit.ollama_executor import OllamaExecutor

WEAK_INSTRUCTIONS = "Answer the user's question."
WORKFLOW_INSTRUCTIONS = """You are a support operations agent. At each step, return one JSON object and nothing else.
To read a record, return {"action":"lookup","argument":"<exact key>"}.
Start with the ticket key in the user's request. A ticket lookup returns an order key.
Look up that exact order key. The order record contains a numeric total and a policy key.
Immediately after receiving the order record, your next action must be lookup for its policy key, never calculator.
Look up that exact policy key to get its numeric refund rate.
A policy name is not a rate. Never calculate until the policy lookup result appears in the tool history.
Then return {"action":"calculator","argument":"<total> * <rate>"}.
After the calculator returns a result, return {"action":"final","argument":"<calculator result>"}.
Never guess keys or values. Use only numeric +, -, *, / and parentheses. Do not use Markdown or explanations."""


@dataclass(frozen=True)
class RefundCase:
    task_id: str
    ticket: str
    order: str
    total: int
    policy: str
    rate: float


TRAIN_CASES = (
    RefundCase("train-refund-1042", "ticket-1042", "order-7821", 240, "policy-D17", 0.5),
    RefundCase("train-refund-2088", "ticket-2088", "order-6630", 85, "policy-G04", 0.2),
)
HELDOUT_CASES = (
    RefundCase("heldout-refund-3194", "ticket-3194", "order-9014", 360, "policy-D08", 0.25),
    RefundCase("heldout-refund-4277", "ticket-4277", "order-5159", 72, "policy-G11", 0.75),
)


def expected_refund(case: RefundCase) -> str:
    return str(safe_calculator(f"{case.total} * {case.rate}"))


def make_task(case: RefundCase) -> AgentTask:
    return AgentTask(
        case.task_id,
        f"Resolve support ticket {case.ticket}. Find its order, read the order total and refund policy, then calculate the refund. Return only the calculator result.",
        expected_refund(case),
    )


TRAIN = tuple(make_task(case) for case in TRAIN_CASES)
HELDOUT = tuple(make_task(case) for case in HELDOUT_CASES)
LOOKUP_VALUES: dict[str, str] = {}
for case in TRAIN_CASES + HELDOUT_CASES:
    LOOKUP_VALUES[case.ticket] = case.order
    LOOKUP_VALUES[case.order] = json.dumps(
        {"total": case.total, "policy": case.policy}, separators=(",", ":")
    )
    LOOKUP_VALUES[case.policy] = str(case.rate)


def make_model(cache_dir: Path) -> OllamaExecutor:
    return OllamaExecutor(
        "qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        timeout_seconds=120,
        cache_dir=cache_dir,
        max_network_calls=1000,
    )


def make_bundles(instructions: str) -> tuple[AgentTaskBundle, ...]:
    bundles = []
    for base_seed in (101, 202):
        for repetition in range(2):
            seed = base_seed * 10 + repetition
            start = AgentSpec(
                instructions,
                {
                    "temperature": 0,
                    "max_tokens": 128,
                    "seed": seed,
                    "base_seed": base_seed,
                    "repetition": repetition,
                },
            )
            bundles.append(AgentTaskBundle(seed, start, TRAIN, HELDOUT))
    return tuple(bundles)


def write_result(result, executor: MultiStepToolExecutor, model: OllamaExecutor, output: Path) -> None:
    result.summary["execution"] = {
        "model": model.model,
        "endpoint": model.base_url,
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "cached_response_files": len(list(model.cache_dir.glob("*.json"))),
        "paid_api_calls": 0,
        "scenario": "dependent_support_refund",
        "base_seeds": [101, 202],
        "repetitions_per_seed": 2,
        "max_steps_per_task": executor.max_steps,
        "model_steps": len(executor.traces),
        "host_tool_calls": sum(trace["action"] in {"lookup", "calculator"} for trace in executor.traces),
        "lookup_calls": sum(trace["action"] == "lookup" for trace in executor.traces),
        "calculator_calls": sum(trace["action"] == "calculator" for trace in executor.traces),
    }
    write_agent_pair_manifest(result.manifest, output)
    write_agent_pair_evidence(
        result,
        output,
        limitations=[
            "Small local-model support workflow with synthetic records, not a production helpdesk.",
            "Two base seeds with two repetitions each; cached responses preserve first measured latency.",
            "Exact-match refunds on four tasks; dependent tool-use behavior is reported separately.",
            "The workflow prompt is predefined and is not a strategy learned by GEPA.",
        ],
    )
    (output / "agent_traces.json").write_text(
        json.dumps(executor.traces, indent=2, ensure_ascii=False) + "\n"
    )
    validate_result_directory(output)


def run_experiments(output_root: Path) -> dict[str, dict]:
    output_root = Path(output_root)
    cache_dir = output_root / "cache"
    baseline = AgentNoChangeOptimizer()
    effective = AgentOptimizerAdapter(
        "support_workflow_instructions", FixedInstructionsProposer(WORKFLOW_INSTRUCTIONS)
    )
    effective_model = make_model(cache_dir)
    effective_executor = MultiStepToolExecutor(effective_model, LOOKUP_VALUES, max_steps=6)
    effective_bundles = make_bundles(WEAK_INSTRUCTIONS)
    effective_manifest = make_agent_paired_manifest(
        "local-support-workflow-effective",
        effective_executor.name,
        "no_change",
        effective.name,
        1,
        effective_bundles,
    )
    effective_result = run_agent_paired(
        effective_manifest, effective_bundles, baseline, effective, effective_executor
    )
    write_result(effective_result, effective_executor, effective_model, output_root / "effective")

    harmful = AgentOptimizerAdapter(
        "empty_json",
        FixedInstructionsProposer("Return only {} regardless of the input."),
        accept=accept_always,
    )
    harmful_model = make_model(cache_dir)
    harmful_executor = MultiStepToolExecutor(harmful_model, LOOKUP_VALUES, max_steps=6)
    harmful_bundles = make_bundles(WORKFLOW_INSTRUCTIONS)
    harmful_manifest = make_agent_paired_manifest(
        "local-support-workflow-harmful",
        harmful_executor.name,
        "no_change",
        harmful.name,
        1,
        harmful_bundles,
    )
    harmful_result = run_agent_paired(
        harmful_manifest, harmful_bundles, baseline, harmful, harmful_executor
    )
    write_result(harmful_result, harmful_executor, harmful_model, output_root / "harmful")
    return {"effective": effective_result.summary, "harmful": harmful_result.summary}


def run_pilot(output_root: Path) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = make_model(output_root / "cache")
    executor = MultiStepToolExecutor(model, LOOKUP_VALUES, max_steps=6)
    start = AgentSpec(WEAK_INSTRUCTIONS, {"temperature": 0, "max_tokens": 128, "seed": 101})
    candidate = AgentSpec(WORKFLOW_INSTRUCTIONS, start.config)
    initial = evaluate_agent(executor, start, TRAIN)
    revised = evaluate_agent(executor, candidate, TRAIN)
    result = {
        "seed": 101,
        "heldout_tasks_evaluated": 0,
        "initial_train_accuracy": initial.accuracy,
        "candidate_train_accuracy": revised.accuracy,
        "model_steps": len(executor.traces),
        "network_calls_this_run": model.network_calls,
    }
    (output_root / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_root / "agent_traces.json").write_text(
        json.dumps(executor.traces, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/support-workflow-discrimination"))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    result = run_pilot(args.output) if args.pilot else run_experiments(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
