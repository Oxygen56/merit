"""Learn support error-handling instructions from train failures with local reflection."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merit.agent_contract import AgentOptimizerAdapter, AgentSpec, AgentTask, evaluate_agent
from merit.agent_methods import (
    AgentNoChangeOptimizer,
    ModelReflectionProposer,
    RandomPromptEditor,
    accept_always,
)
from merit.agent_pair import AgentTaskBundle, make_agent_paired_manifest, run_agent_paired
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.budget import Budget
from merit.evidence import validate_result_directory
from merit.multi_step_executor import MultiStepToolExecutor
from merit.ollama_executor import OllamaExecutor

BASE_INSTRUCTIONS = """You are a support operations agent. At each step, return one JSON object and nothing else.
To read a record, return {"action":"lookup","argument":"<exact key>"}.
To calculate, return {"action":"calculator","argument":"<numeric expression>"}.
To finish, return {"action":"final","argument":"<answer>"}.
Never guess keys or values. Use only numeric +, -, *, / and parentheses. Do not use Markdown or explanations."""
PROPOSER_INSTRUCTIONS = """Improve tool-agent system prompts using only the supplied training failures.
Return complete replacement instructions, preserving the exact action JSON syntax and unrelated behavior.
Infer general procedures for both normal refunds and missing records.
Do not include ticket, order, or policy keys and do not copy expected answers from the examples.
Do not ask for held-out tasks."""
LOOKUP_VALUES = {
    "ticket-5001": "order-5001",
    "order-5001": '{"total":120,"policy":"policy-N01"}',
    "policy-N01": "0.5",
    "ticket-5002": "order-X71",
    "ticket-5003": "order-5003",
    "order-5003": '{"total":90,"policy":"policy-X71"}',
    "ticket-6001": "order-6001",
    "order-6001": '{"total":160,"policy":"policy-N02"}',
    "policy-N02": "0.25",
    "ticket-6002": "order-X72",
    "ticket-6003": "order-6003",
    "order-6003": '{"total":75,"policy":"policy-X72"}',
}


def task(task_id: str, ticket: str, expected: str) -> AgentTask:
    return AgentTask(
        task_id,
        f"Resolve support ticket {ticket}. Find its order, read the order total and refund policy, then calculate the refund. If a required record is missing, return REFUND_UNAVAILABLE. Otherwise return only the calculator result.",
        expected,
    )


TRAIN = (
    task("train-normal-5001", "ticket-5001", "60.0"),
    task("train-missing-order-5002", "ticket-5002", "REFUND_UNAVAILABLE"),
    task("train-missing-policy-5003", "ticket-5003", "REFUND_UNAVAILABLE"),
)
HELDOUT = (
    task("heldout-normal-6001", "ticket-6001", "40.0"),
    task("heldout-missing-order-6002", "ticket-6002", "REFUND_UNAVAILABLE"),
    task("heldout-missing-policy-6003", "ticket-6003", "REFUND_UNAVAILABLE"),
)


def make_model(cache_dir: Path) -> OllamaExecutor:
    return OllamaExecutor(
        "qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        timeout_seconds=120,
        cache_dir=cache_dir,
        max_network_calls=1200,
    )


def make_bundles() -> tuple[AgentTaskBundle, ...]:
    bundles = []
    for base_seed in (101, 202):
        for repetition in range(2):
            seed = base_seed * 10 + repetition
            start = AgentSpec(
                BASE_INSTRUCTIONS,
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


def make_reflection(model: OllamaExecutor, seed: int) -> AgentOptimizerAdapter:
    proposer_spec = AgentSpec(
        PROPOSER_INSTRUCTIONS,
        {"temperature": 0.2, "max_tokens": 1024, "seed": seed + 10_000},
    )
    proposer = ModelReflectionProposer(model, proposer_spec, max_failures=5)
    return AgentOptimizerAdapter("model_reflection", proposer, proposer_source=model.name)


def write_result(result, executor: MultiStepToolExecutor, model: OllamaExecutor, output: Path) -> None:
    result.summary["execution"] = {
        "model": model.model,
        "endpoint": model.base_url,
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "cached_response_files": len(list(model.cache_dir.glob("*.json"))),
        "paid_api_calls": 0,
        "scenario": "support_error_model_reflection",
        "base_seeds": [101, 202],
        "repetitions_per_seed": 2,
        "max_steps_per_task": executor.max_steps,
        "model_steps": len(executor.traces),
        "host_tool_calls": sum(trace["action"] in {"lookup", "calculator"} for trace in executor.traces),
        "lookup_calls": sum(trace["action"] == "lookup" for trace in executor.traces),
        "calculator_calls": sum(trace["action"] == "calculator" for trace in executor.traces),
        "tool_errors": sum(trace["tool_error"] is not None for trace in executor.traces),
        "failed_lookup_calls": sum(
            trace["action"] == "lookup" and trace["tool_error"] is not None
            for trace in executor.traces
        ),
    }
    write_agent_pair_manifest(result.manifest, output)
    write_agent_pair_evidence(
        result,
        output,
        limitations=[
            "Small local-model reflection run with six synthetic support tasks.",
            "The proposer receives train failures only; final tasks are evaluated separately.",
            "Two base seeds with two repetitions each; cached responses preserve first measured latency.",
            "This run does not establish a general ranking of prompt-optimization methods.",
        ],
    )
    (output / "agent_traces.json").write_text(
        json.dumps(executor.traces, indent=2, ensure_ascii=False) + "\n"
    )
    validate_result_directory(output)


REQUIRED_INSTRUCTION_FRAGMENTS = tuple(
    line.strip() for line in BASE_INSTRUCTIONS.splitlines() if line.strip()
)


def validate_complete_instructions(instructions: str) -> str | None:
    if "```" in instructions:
        return "Remove Markdown fences and return a standalone instruction prompt."
    normalized = "".join(instructions.split())
    missing = [
        fragment for fragment in REQUIRED_INSTRUCTION_FRAGMENTS
        if "".join(fragment.split()).rstrip(".") not in normalized
    ]
    if missing:
        return "The response is not a complete replacement prompt. Preserve these required lines; whitespace and a trailing period may vary: " + " | ".join(missing)
    return None


class CallCounter:
    def __init__(self, executor: OllamaExecutor) -> None:
        self.executor = executor
        self.calls = 0

    @property
    def name(self) -> str:
        return self.executor.name

    def invoke(self, agent: AgentSpec, task: AgentTask):
        self.calls += 1
        return self.executor.invoke(agent, task)


def run_pilot(output_root: Path, retry_invalid: bool = False) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = make_model(output_root / "cache")
    executor = MultiStepToolExecutor(model, LOOKUP_VALUES, max_steps=6, recover_tool_errors=True)
    start = AgentSpec(BASE_INSTRUCTIONS, {"temperature": 0, "max_tokens": 128, "seed": 101})
    initial = evaluate_agent(executor, start, TRAIN)
    proposer_model = CallCounter(model)
    if retry_invalid:
        proposer_spec = AgentSpec(
            PROPOSER_INSTRUCTIONS,
            {"temperature": 0.2, "max_tokens": 1024, "seed": 10_101},
        )
        proposer = ModelReflectionProposer(
            proposer_model,
            proposer_spec,
            max_failures=5,
            instruction_validator=validate_complete_instructions,
            max_attempts=3,
        )
        optimizer = AgentOptimizerAdapter(
            "model_reflection_with_validation",
            proposer,
            proposer_source=model.name,
        )
    else:
        optimizer = make_reflection(proposer_model, 101)
    budget = Budget(1)

    def evaluate_candidate(candidate):
        budget.consume()
        return evaluate_agent(executor, candidate, TRAIN)

    summary = {
        "seed": 101,
        "heldout_tasks_evaluated": 0,
        "validation_retry_enabled": retry_invalid,
        "proposal_model_calls": 0,
        "initial_train_accuracy": initial.accuracy,
        "initial_results": [asdict(row) for row in initial.results],
        "candidate_train_accuracy": None,
        "candidate_accepted": None,
        "candidate_instructions": None,
        "candidate_results": [],
        "candidate_evaluations": budget.used,
        "network_calls_this_run": 0,
    }
    try:
        result = optimizer.optimize(
            start,
            TRAIN,
            initial,
            budget,
            101,
            evaluate_candidate,
        )
    except ValueError as error:
        summary["proposal_error"] = str(error)
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
    })
    (output_root / "pilot.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (output_root / "agent_traces.json").write_text(
        json.dumps(executor.traces, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/support-error-model-reflection"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        print(json.dumps(run_pilot(args.output, retry_invalid=args.retry_invalid), indent=2, ensure_ascii=False))
        return
    output = Path(args.output)
    model = make_model(output / "cache")
    executor = MultiStepToolExecutor(model, LOOKUP_VALUES, max_steps=6, recover_tool_errors=True)
    bundles = make_bundles()
    manifest = make_agent_paired_manifest(
        "local-support-error-model-reflection",
        executor.name,
        "no_change",
        "model_reflection",
        1,
        bundles,
        ("random_prompt_edit",),
    )
    result = run_agent_paired(
        manifest,
        bundles,
        AgentNoChangeOptimizer(),
        make_reflection(model, 101),
        executor,
        controls=(
            AgentOptimizerAdapter(
                "random_prompt_edit",
                RandomPromptEditor(),
                accept=accept_always,
                proposer_source="python_random",
            ),
        ),
    )
    write_result(result, executor, model, output)
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
