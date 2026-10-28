"""Repeat the train-only dynamic patch across two base seeds and two repetitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ollama_support_error_dynamic_patch import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pilot", type=Path,
        default=Path("results/support-error-stop-patch-pilot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/support-error-dynamic-patch-train-repeated"),
    )
    args = parser.parse_args()
    runs = []
    for base_seed in (101, 202):
        for repetition in range(2):
            seed = base_seed * 10 + repetition
            output = args.output / f"seed-{seed}"
            summary = run(args.source_pilot, output, seed)
            runs.append({
                "base_seed": base_seed,
                "repetition": repetition,
                "seed": seed,
                "candidate_train_accuracy": summary["candidate_train_accuracy"],
                "candidate_accepted": summary["candidate_accepted"],
                "candidate_trajectory_valid": summary["candidate_trajectory_valid"],
                "candidate_selection_issues": summary["candidate_selection_issues"],
                "proposal_model_calls": summary["proposal_model_calls"],
                "candidate_evaluations": summary["candidate_evaluations"],
                "proposer_usage": summary.get("proposer_usage"),
                "initial_results": summary["initial_results"],
                "candidate_results": summary["candidate_results"],
                "heldout_tasks_evaluated": summary["heldout_tasks_evaluated"],
                "result_directory": str(output),
            })
    aggregate = {
        "scope": "M6 train-only repeated dynamic-patch evaluation",
        "base_seeds": [101, 202],
        "repetitions_per_seed": 2,
        "run_count": len(runs),
        "heldout_tasks_evaluated": sum(row["heldout_tasks_evaluated"] for row in runs),
        "accepted_runs": sum(row["candidate_accepted"] is True for row in runs),
        "valid_trajectory_runs": sum(row["candidate_trajectory_valid"] is True for row in runs),
        "full_accuracy_runs": sum(row["candidate_train_accuracy"] == 1.0 for row in runs),
        "m6_passed": all(
            row["candidate_accepted"] is True
            and row["candidate_trajectory_valid"] is True
            and row["candidate_train_accuracy"] == 1.0
            for row in runs
        ),
        "runs": runs,
        "paid_api_calls": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
