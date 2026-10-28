"""Configuration-driven local model experiments for the Merit CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_contract import AgentOptimizerAdapter, AgentSpec, AgentTask
from .agent_methods import (
    AgentNoChangeOptimizer,
    FixedInstructionsProposer,
    ModelReflectionProposer,
    PromptSuffixProposer,
    RandomPromptEditor,
    accept_always,
)
from .agent_pair import AgentTaskBundle, make_agent_paired_manifest, run_agent_paired
from .agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from .evidence import validate_result_directory
from .gepa_optimizer import GepaOptimizer
from .langgraph_executor import LangGraphToolExecutor
from .ollama_executor import OllamaExecutor


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML configs need: pip install 'merit-agent-eval[yaml]'") from error
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be an object")
    return payload


def load_tasks(path: Path, limits: dict[str, int] | None = None) -> tuple[tuple[AgentTask, ...], tuple[AgentTask, ...], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    limits = limits or {}

    def tasks(name: str) -> tuple[AgentTask, ...]:
        rows = payload.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Dataset requires a non-empty {name} list")
        limit = limits.get(name)
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} limit must be a positive integer")
            rows = rows[:limit]
        result = tuple(AgentTask(str(row["id"]), str(row["prompt"]), str(row["expected"])) for row in rows)
        if len({row.task_id for row in result}) != len(result):
            raise ValueError(f"Dataset {name} task IDs must be unique")
        return result

    return tasks("train"), tasks("heldout"), payload.get("source", {})


def load_validation_tasks(path: Path, limits: dict[str, int] | None = None) -> tuple[AgentTask, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("validation")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset requires a non-empty validation list")
    limit = (limits or {}).get("validation")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("validation limit must be a positive integer")
        rows = rows[:limit]
    result = tuple(AgentTask(str(row["id"]), str(row["prompt"]), str(row["expected"])) for row in rows)
    if len({row.task_id for row in result}) != len(result):
        raise ValueError("Dataset validation task IDs must be unique")
    return result


SCHEMA_GUIDED_SUFFIX = """Return one JSON object with exactly the top-level keys name and arguments; do not use Markdown or explanations.
Copy the tool name and parameter names exactly from the supplied schema. Include every parameter needed by the question, including explicitly mentioned optional parameters and documented default values when required. Preserve arrays, nested objects, numbers, booleans, and strings; do not invent parameters or values."""


def _optimizer(
    name: str,
    model: OllamaExecutor,
    gepa_config: dict[str, Any],
    validation_tasks: tuple[AgentTask, ...] = (),
    validation_executor: Any = None,
) -> Any:
    if name == "no_change":
        return AgentNoChangeOptimizer()
    if name == "random_prompt_edit":
        return AgentOptimizerAdapter(name, RandomPromptEditor(), accept=accept_always, proposer_source="python_random")
    if name == "schema_guided_prompt":
        return AgentOptimizerAdapter(
            name, PromptSuffixProposer(SCHEMA_GUIDED_SUFFIX), proposer_source="python_heuristic"
        )
    if name == "empty_json":
        return AgentOptimizerAdapter(
            name,
            FixedInstructionsProposer("Return only {} regardless of the input."),
            accept=accept_always,
            proposer_source="python_harmful_control",
        )
    if name == "model_reflection":
        proposer = ModelReflectionProposer(
            model,
            AgentSpec(
                "Improve the supplied agent instructions using only its training failures. Return only the revised instructions.",
                config={"temperature": 0, "max_tokens": 512},
            ),
        )
        return AgentOptimizerAdapter(name, proposer, proposer_source=model.name)
    if name == "gepa":
        unknown = set(gepa_config) - {"reflection_minibatch_size", "feedback_format", "selection_data"}
        if unknown:
            raise ValueError(f"Unsupported GEPA settings: {sorted(unknown)}")
        selection_data = str(gepa_config.get("selection_data", "train"))
        if selection_data not in {"train", "validation"}:
            raise ValueError("GEPA selection_data must be train or validation")
        if selection_data == "validation" and not validation_tasks:
            raise ValueError("GEPA validation selection requires validation tasks")
        return GepaOptimizer(
            model,
            AgentSpec(
                "Revise the supplied agent instructions using only the provided training trajectories and feedback. Return only the revised instructions.",
                config={"temperature": 0, "max_tokens": 512},
            ),
            reflection_minibatch_size=int(gepa_config.get("reflection_minibatch_size", 3)),
            feedback_format=str(gepa_config.get("feedback_format", "exact_match")),
            validation_tasks=validation_tasks if selection_data == "validation" else (),
            validation_executor=validation_executor,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def run_config(config_path: Path, output_dir: Path, max_network_calls: int | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    experiment_name = str(config.get("experiment", config_path.stem))
    model_config = config.get("model", {})
    model_name = str(model_config.get("name", ""))
    if not model_name:
        raise ValueError("model.name is required")
    output_dir = Path(output_dir).resolve()
    cache_dir = output_dir / "cache"
    limit = max_network_calls if max_network_calls is not None else model_config.get("max_network_calls")
    model = OllamaExecutor(
        model_name,
        base_url=str(model_config.get("base_url", "http://127.0.0.1:11434/v1")),
        timeout_seconds=float(model_config.get("timeout_seconds", 120)),
        cache_dir=cache_dir,
        max_network_calls=limit,
    )
    use_langgraph = config.get("executor", "ollama") == "langgraph_tool"
    if config.get("executor", "ollama") not in {"ollama", "langgraph_tool"}:
        raise ValueError("executor must be ollama or langgraph_tool")
    executor = LangGraphToolExecutor(model) if use_langgraph else model
    dataset_path = (config_path.parent / config["dataset"]).resolve()
    train, heldout, source = load_tasks(dataset_path, config.get("limits"))
    gepa_config = config.get("gepa", {})
    if not isinstance(gepa_config, dict):
        raise ValueError("gepa settings must be an object")
    validation = (
        load_validation_tasks(dataset_path, config.get("limits"))
        if gepa_config.get("selection_data") == "validation"
        else ()
    )
    seeds = config.get("seeds", [101])
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    agent_config = config.get("agent", {})
    instructions = str(agent_config.get("instructions", "Answer each task precisely."))
    base_settings = agent_config.get("config", {"temperature": 0, "max_tokens": 256})
    bundles = tuple(
        AgentTaskBundle(seed, AgentSpec(instructions, {**base_settings, "seed": seed}), train, heldout, validation)
        for seed in seeds
    )
    methods = config.get("methods", {})
    baseline_name = str(methods.get("baseline", "no_change"))
    challenger_name = str(methods.get("challenger", "model_reflection"))
    control_names = tuple(methods.get("controls", ["random_prompt_edit"]))
    baseline = _optimizer(baseline_name, model, gepa_config, validation, executor)
    challenger = _optimizer(challenger_name, model, gepa_config, validation, executor)
    controls = tuple(_optimizer(name, model, gepa_config, validation, executor) for name in control_names)
    budget = int(config.get("candidate_budget", 1))
    manifest = make_agent_paired_manifest(
        experiment_name, executor.name, baseline_name, challenger_name, budget, bundles, control_names,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run-config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_agent_pair_manifest(manifest, output_dir)
    experiment = run_agent_paired(manifest, bundles, baseline, challenger, executor, controls)
    experiment.summary["execution"] = {
        "model": model_name,
        "endpoint": model.base_url,
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "cached_response_files": len(list(cache_dir.glob("*.json"))),
        "paid_api_calls": 0,
        "workflow": "langgraph_tool" if use_langgraph else "ollama_chat",
        "tool_call_records": len(executor.tool_traces) if use_langgraph else 0,
        "valid_tool_calls": sum(trace["valid_tool_call"] for trace in executor.tool_traces) if use_langgraph else 0,
        "dataset_source": source,
    }
    limitations = [
        "Small local-model experiment; do not use it as a general model or optimizer ranking.",
        "All model calls used the loopback Ollama endpoint; cached responses preserve their first measured latency.",
    ]
    if source:
        limitations.append(str(source.get("metric_notice", "See the dataset source notice in run-config.json.")))
    write_agent_pair_evidence(experiment, output_dir, limitations=limitations)
    if use_langgraph:
        (output_dir / "tool_traces.json").write_text(
            json.dumps(executor.tool_traces, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    validate_result_directory(output_dir)
    return {"summary": experiment.summary, "output": str(output_dir), "report": str(output_dir / "report.html")}
