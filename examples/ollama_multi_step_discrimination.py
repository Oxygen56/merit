"""Run repeated local-model method checks on a multi-step host-tool agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from merit.agent_contract import AgentOptimizerAdapter, AgentSpec, AgentTask, evaluate_agent
from merit.agent_methods import AgentNoChangeOptimizer, FixedInstructionsProposer, accept_always
from merit.agent_pair import AgentTaskBundle, make_agent_paired_manifest, run_agent_paired
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.evidence import validate_result_directory
from merit.multi_step_executor import MultiStepToolExecutor
from merit.ollama_executor import OllamaExecutor

WEAK_INSTRUCTIONS = "Answer the user's question."
STRONG_INSTRUCTIONS = """You are a multi-step tool agent. At each step, return one JSON object and nothing else.
To read a stored number, return {"action":"lookup","argument":"<key>"}.
After all needed values are available, return {"action":"calculator","argument":"<arithmetic expression>"}.
After the calculator returns a result, return {"action":"final","argument":"<result>"}.
Never guess stored values. Use only numeric +, -, *, / and parentheses. Do not use Markdown or explanations."""
LOOKUP_VALUES = {
    "price-apple": "7",
    "quantity-apple": "6",
    "price-server": "20",
    "quantity-server": "3",
    "price-book": "12.5",
    "quantity-book": "4",
    "price-camera": "199",
    "quantity-camera": "2",
}


def task(task_id: str, prefix: str, expected: str) -> AgentTask:
    return AgentTask(
        task_id,
        f"Find the stored values for keys price-{prefix} and quantity-{prefix}, then multiply them. Use tools rather than guessing. Return only the numeric result.",
        expected,
    )


TRAIN = (task("train-apple-total", "apple", "42"), task("train-server-total", "server", "60"))
HELDOUT = (task("heldout-book-total", "book", "50.0"), task("heldout-camera-total", "camera", "398"))


def make_model(cache_dir: Path) -> OllamaExecutor:
    return OllamaExecutor(
        "qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        timeout_seconds=120,
        cache_dir=cache_dir,
        max_network_calls=800,
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


def write_result(name: str, result, executor: MultiStepToolExecutor, model: OllamaExecutor, output: Path) -> None:
    result.summary["execution"] = {
        "model": model.model,
        "endpoint": model.base_url,
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "cached_response_files": len(list(model.cache_dir.glob("*.json"))),
        "paid_api_calls": 0,
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
            "Small local-model multi-step tool experiment; not a general model or optimizer ranking.",
            "Two base seeds with two repetitions each; cached responses preserve first measured latency.",
            "Exact-match final answers on four synthetic tasks; tool-use behavior is reported separately.",
        ],
    )
    (output / "agent_traces.json").write_text(json.dumps(executor.traces, indent=2, ensure_ascii=False) + "\n")
    validate_result_directory(output)


def run_experiments(output_root: Path) -> dict[str, dict]:
    output_root = Path(output_root)
    cache_dir = output_root / "cache"
    baseline = AgentNoChangeOptimizer()
    effective = AgentOptimizerAdapter(
        "multi_step_instructions", FixedInstructionsProposer(STRONG_INSTRUCTIONS)
    )
    effective_model = make_model(cache_dir)
    effective_executor = MultiStepToolExecutor(effective_model, LOOKUP_VALUES)
    effective_bundles = make_bundles(WEAK_INSTRUCTIONS)
    effective_manifest = make_agent_paired_manifest(
        "local-multi-step-effective",
        effective_executor.name,
        "no_change",
        effective.name,
        1,
        effective_bundles,
    )
    effective_result = run_agent_paired(
        effective_manifest, effective_bundles, baseline, effective, effective_executor
    )
    write_result(
        "effective", effective_result, effective_executor, effective_model, output_root / "effective"
    )

    harmful = AgentOptimizerAdapter(
        "empty_json",
        FixedInstructionsProposer("Return only {} regardless of the input."),
        accept=accept_always,
    )
    harmful_model = make_model(cache_dir)
    harmful_executor = MultiStepToolExecutor(harmful_model, LOOKUP_VALUES)
    harmful_bundles = make_bundles(STRONG_INSTRUCTIONS)
    harmful_manifest = make_agent_paired_manifest(
        "local-multi-step-harmful",
        harmful_executor.name,
        "no_change",
        harmful.name,
        1,
        harmful_bundles,
    )
    harmful_result = run_agent_paired(
        harmful_manifest, harmful_bundles, baseline, harmful, harmful_executor
    )
    write_result("harmful", harmful_result, harmful_executor, harmful_model, output_root / "harmful")
    return {"effective": effective_result.summary, "harmful": harmful_result.summary}


def run_pilot(output_root: Path) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = make_model(output_root / "cache")
    executor = MultiStepToolExecutor(model, LOOKUP_VALUES)
    start = AgentSpec(WEAK_INSTRUCTIONS, {"temperature": 0, "max_tokens": 128, "seed": 101})
    candidate = AgentSpec(STRONG_INSTRUCTIONS, start.config)
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
    (output_root / "agent_traces.json").write_text(json.dumps(executor.traces, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/multi-step-method-discrimination"))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    result = run_pilot(args.output) if args.pilot else run_experiments(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
