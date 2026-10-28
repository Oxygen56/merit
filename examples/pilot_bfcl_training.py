"""Evaluate only the initial training agent; never load held-out tasks."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from merit.agent_contract import AgentSpec, AgentTask
from merit.experiment import load_config, load_tasks
from merit.langgraph_executor import LangGraphToolExecutor
from merit.ollama_executor import OllamaExecutor


def main() -> None:
    config_path = Path(__file__).parent / "ollama_langgraph_gepa_qwen25_7b_diagnostic.yaml"
    output_dir = Path(__file__).parents[1] / "results" / "ollama-langgraph-gepa-qwen25-7b-harder"
    config = load_config(config_path)
    dataset_path = config_path.parent / config["dataset"]
    train, _, source = load_tasks(dataset_path)
    model_config = config["model"]
    model = OllamaExecutor(
        model_config["name"],
        base_url=model_config["base_url"],
        timeout_seconds=model_config["timeout_seconds"],
        cache_dir=output_dir / "cache",
        max_network_calls=model_config["max_network_calls"],
    )
    executor = LangGraphToolExecutor(model)
    rows = []
    for seed in config["seeds"]:
        agent = AgentSpec(config["agent"]["instructions"], {**config["agent"]["config"], "seed": seed})
        for task in train:
            response = executor.invoke(agent, task)
            correct = response.output == task.expected
            row = {"seed": seed, "task_id": task.task_id, "correct": correct, "expected": task.expected, "output": response.output}
            rows.append(row)
            print(f"seed={seed} task={task.task_id} correct={correct}", flush=True)
    summary = {
        "experiment": config["experiment"] + ":training-only-pilot",
        "model": model_config["name"],
        "train_tasks": len(train),
        "seeds": config["seeds"],
        "per_seed_accuracy": {str(seed): sum(row["correct"] for row in rows if row["seed"] == seed) / len(train) for seed in config["seeds"]},
        "failure_tasks": dict(Counter(row["task_id"] for row in rows if not row["correct"])),
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "paid_api_calls": 0,
        "dataset_source": source,
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train-baseline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"results", "dataset_source"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
