"""M7 validation-gated final evaluation for the repeated dynamic-patch candidates."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merit.agent_contract import AgentSpec, evaluate_agent, tasks_digest
from merit.multi_step_executor import MultiStepToolExecutor
from ollama_support_error_reflection import HELDOUT, LOOKUP_VALUES, TRAIN, make_model, task
from ollama_support_error_stop_patch import accept_candidate, selection_issues
from ollama_support_error_trajectory_reflection import build_structured_feedback

FINAL = (
    task("final-normal-7001", "ticket-7001", "80.0"),
    task("final-missing-order-7002", "ticket-7002", "REFUND_UNAVAILABLE"),
    task("final-missing-policy-7003", "ticket-7003", "REFUND_UNAVAILABLE"),
)
FINAL_LOOKUP_VALUES = {
    "ticket-7001": "order-7001",
    "order-7001": '{"total":200,"policy":"policy-F01"}',
    "policy-F01": "0.4",
    "ticket-7002": "order-Z81",
    "ticket-7003": "order-7003",
    "order-7003": '{"total":110,"policy":"policy-Z81"}',
}


def evaluation_summary(evaluation):
    return {
        "accuracy": evaluation.accuracy,
        "usage": asdict(evaluation.usage),
        "results": [asdict(row) for row in evaluation.results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("results/support-error-dynamic-patch-train-repeated"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/support-error-dynamic-validation-final"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = {
        "protocol": "merit-support-validation-final-v1",
        "model": "qwen2.5:7b",
        "base_seeds": [101, 202],
        "repetitions_per_seed": 2,
        "train_task_digest": tasks_digest(TRAIN),
        "validation_task_digest": tasks_digest(HELDOUT),
        "final_task_digest": tasks_digest(FINAL),
        "validation_task_ids": [row.task_id for row in HELDOUT],
        "final_task_ids": [row.task_id for row in FINAL],
        "selection": "validation accuracy and valid validation trajectories only",
        "final_policy": "run only when the candidate is accepted on validation; no final-task feedback",
    }
    (args.output / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n"
    )

    model = make_model(args.output / "cache")
    executor = MultiStepToolExecutor(
        model,
        {**LOOKUP_VALUES, **FINAL_LOOKUP_VALUES},
        max_steps=6,
        recover_tool_errors=True,
    )
    repeated = json.loads((args.candidates / "summary.json").read_text())
    if not repeated["m6_passed"] or repeated["heldout_tasks_evaluated"] != 0:
        raise ValueError("M6 must pass without held-out evaluation before M7")
    runs = []
    for source_run in repeated["runs"]:
        seed = source_run["seed"]
        candidate_pilot = json.loads((Path(source_run["result_directory"]) / "pilot.json").read_text())
        start_pilot = json.loads((Path("results/support-error-stop-patch-pilot") / "pilot.json").read_text())
        start = AgentSpec(
            start_pilot["candidate_instructions"],
            {"temperature": 0, "max_tokens": 128, "seed": seed},
        )
        candidate = AgentSpec(
            candidate_pilot["candidate_instructions"],
            {"temperature": 0, "max_tokens": 128, "seed": seed},
        )
        initial_validation = evaluate_agent(executor, start, HELDOUT)
        initial_feedback = build_structured_feedback(
            HELDOUT,
            executor.traces,
            [asdict(row) for row in initial_validation.results],
            allow_heldout=True,
        )
        candidate_validation = evaluate_agent(executor, candidate, HELDOUT)
        candidate_feedback = build_structured_feedback(
            HELDOUT,
            executor.traces,
            [asdict(row) for row in candidate_validation.results],
            allow_heldout=True,
        )
        accepted = accept_candidate(
            initial_validation.accuracy,
            candidate_validation.accuracy,
            initial_feedback,
            candidate_feedback,
        )
        baseline_final = None
        successor_final = None
        if accepted:
            baseline_final = evaluate_agent(executor, start, FINAL)
            successor_final = evaluate_agent(executor, candidate, FINAL)
        runs.append({
            "base_seed": source_run["base_seed"],
            "repetition": source_run["repetition"],
            "seed": seed,
            "initial_validation": evaluation_summary(initial_validation),
            "initial_validation_issues": selection_issues(initial_feedback),
            "candidate_validation": evaluation_summary(candidate_validation),
            "candidate_validation_issues": selection_issues(candidate_feedback),
            "candidate_accepted": accepted,
            "successor": "candidate" if accepted else "start",
            "baseline_final": evaluation_summary(baseline_final) if baseline_final else None,
            "successor_final": evaluation_summary(successor_final) if successor_final else None,
        })

    final_runs = [row for row in runs if row["successor_final"] is not None]
    aggregate = {
        **experiment,
        "run_count": len(runs),
        "accepted_validation_runs": sum(row["candidate_accepted"] for row in runs),
        "final_runs": len(final_runs),
        "mean_baseline_final_accuracy": (
            sum(row["baseline_final"]["accuracy"] for row in final_runs) / len(final_runs)
            if final_runs else None
        ),
        "mean_successor_final_accuracy": (
            sum(row["successor_final"]["accuracy"] for row in final_runs) / len(final_runs)
            if final_runs else None
        ),
        "runs": runs,
        "execution": {
            "network_calls_this_run": model.network_calls,
            "cache_hits_this_run": model.cache_hits,
            "model_steps": len(executor.traces),
            "paid_api_calls": 0,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "agent_traces.json").write_text(
        json.dumps(executor.traces, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
