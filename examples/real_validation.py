"""Small, bounded real-model validations; network access requires an explicit flag.

The evaluated model never receives held-out tasks during optimization. Outputs are
not cleaned to improve exact-match scores. Credentials are read only by the HTTP
executor and are never included in logs or evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from merit import (
    AgentContractManifest,
    AgentNoChangeOptimizer,
    AgentOptimizerAdapter,
    AgentResponse,
    AgentSpec,
    AgentTask,
    ModelReflectionProposer,
    OpenAICompatibleExecutor,
    RandomPromptEditor,
    TaskFamily,
    accept_always,
    run_agent_contract,
    run_transfer_evaluation,
    tasks_digest,
    validate_result_directory,
)
from merit.agent_pair import (
    AgentPairedExperiment,
    AgentTaskBundle,
    make_agent_paired_manifest,
    run_agent_paired,
)
from merit.transfer import make_transfer_manifest
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.agent_report import write_agent_evidence, write_agent_manifest
from merit.stats import bootstrap_mean_ci, sign_flip_pvalue
from merit.transfer_report import write_transfer_evidence, write_transfer_manifest


class BoundedExecutor:
    """Fail before a network call at the hard cap and persist each raw response."""

    def __init__(self, inner: Any, max_calls: int, state_dir: Path) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.inner = inner
        self.name = inner.name
        self.max_calls = max_calls
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = state_dir / "response-cache.json"
        self.call_path = state_dir / "provider-calls.jsonl"
        self.cache = self._read_cache()
        self.network_calls = 0
        self.cache_hits = 0
        self.seen_network_keys = {
            record["cache_key"]
            for record in self._read_calls()
            if isinstance(record, dict) and isinstance(record.get("cache_key"), str)
        }

    def _read_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Response cache must be a JSON object")
        return payload

    def _read_calls(self) -> list[dict[str, Any]]:
        if not self.call_path.exists():
            return []
        return [json.loads(line) for line in self.call_path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _key(agent: AgentSpec, task: AgentTask) -> str:
        import hashlib

        value = json.dumps(
            {"agent": agent.artifact_id, "task": asdict(task)},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        key = self._key(agent, task)
        if key in self.cache:
            if key in self.seen_network_keys:
                self.cache_hits += 1
                record = self.cache[key]
                return AgentResponse(record["output"], self._usage(record["usage"]))
        if self.network_calls >= self.max_calls:
            raise RuntimeError(f"External model hard call cap reached: {self.max_calls}")
        response = self.inner.invoke(agent, task)
        self.network_calls += 1
        usage = asdict(response.usage)
        self.cache[key] = {"output": response.output, "usage": usage}
        self.seen_network_keys.add(key)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.cache_path)
        record = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "sequence": len(self.seen_network_keys),
            "executor": self.name,
            "model": self.inner.model,
            "cache_key": key,
            "agent_artifact_id": agent.artifact_id,
            "task_id": task.task_id,
            "input": task.prompt,
            "raw_output": response.output,
            "usage": usage,
        }
        with self.call_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return response

    @staticmethod
    def _usage(value: dict[str, Any]) -> Any:
        from merit import Usage

        return Usage(**value)

    def audit(self) -> dict[str, Any]:
        calls = self._read_calls()
        return {
            "model": self.inner.model,
            "executor": self.name,
            "hard_call_cap": self.max_calls,
            "network_calls_this_process": self.network_calls,
            "recorded_network_calls": len(calls),
            "cache_hits_this_process": self.cache_hits,
            "provider_retries": 0,
            "empty_output_calls": sum(not call["raw_output"] for call in calls),
            "provider_input_tokens": sum(call["usage"]["input_tokens"] for call in calls),
            "provider_output_tokens": sum(call["usage"]["output_tokens"] for call in calls),
            "provider_latency_ms": sum(call["usage"]["latency_ms"] for call in calls),
            "all_recorded_usage_complete": all(call["usage"]["tokens_reported"] for call in calls),
            "all_recorded_costs_complete": all(call["usage"]["cost_reported"] for call in calls),
            "raw_call_log": "execution/provider-calls.jsonl",
        }


_LOOKUP = {
    "billing-eu": "degraded",
    "search-cn": "operational",
    "payments-us": "maintenance",
    "archive-apac": "operational",
    "inventory-au": "degraded",
    "mail-ca": "operational",
    "auth-br": "maintenance",
}
_ALLOWED_BINARY = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b, ast.Mod: lambda a, b: a % b}
_ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def safe_calculator(expression: str) -> int | float:
    """Evaluate arithmetic with an AST allow-list; names, calls and eval are forbidden."""

    if len(expression) > 80:
        raise ValueError("Expression is too long")

    def visit(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 12:
            raise ValueError("Expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1_000_000_000:
                raise ValueError("Numeric literal is outside the allowed range")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise ValueError("Division by zero")
            result = _ALLOWED_BINARY[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            result = _ALLOWED_UNARY[type(node.op)](visit(node.operand, depth + 1))
        else:
            raise ValueError("Only numeric +, -, *, /, % and parentheses are allowed")
        if not math.isfinite(result) or abs(result) > 1_000_000_000:
            raise ValueError("Result is outside the allowed range")
        return result

    return visit(ast.parse(expression, mode="eval"))


class ToolAgentExecutor:
    """A real model selects a tool; the host validates and executes that tool."""

    def __init__(self, model: BoundedExecutor) -> None:
        self.model = model
        self.name = model.name + "+host_tools_v1"
        self.traces: list[dict[str, Any]] = []

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        response = self.model.invoke(agent, task)
        match = re.fullmatch(r"TOOL\s+(calculator|lookup)\s+([^\r\n]+)", response.output.strip())
        tool_name: str | None = None
        argument: str | None = None
        tool_result: str | None = None
        error: str | None = None
        if match:
            tool_name, argument = match.group(1), match.group(2).strip()
            try:
                if tool_name == "calculator":
                    tool_result = str(safe_calculator(argument))
                else:
                    if argument not in _LOOKUP:
                        raise ValueError("Unknown lookup key")
                    tool_result = _LOOKUP[argument]
            except (ValueError, SyntaxError, TypeError) as exc:
                error = str(exc)
        output = tool_result if tool_result is not None else response.output
        # Preserve the original provider latency on both network calls and cached replay.
        usage = asdict(response.usage)
        trace = {
            "artifact_id": agent.artifact_id,
            "task_id": task.task_id,
            "input": task.prompt,
            "raw_model_output": response.output,
            "tool_name": tool_name,
            "tool_argument": argument,
            "tool_result": tool_result,
            "tool_error": error,
            "final_output": output,
            "correct": output == task.expected,
            "usage": usage,
        }
        self.traces.append(trace)
        (self.model.state_dir / "agent_traces.json").write_text(
            json.dumps(self.traces, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return AgentResponse(output, self.model._usage(usage))


def make_executor(max_calls: int, state_dir: Path) -> BoundedExecutor:
    inner = OpenAICompatibleExecutor(
        os.environ["KIMI_BASE_URL"].rstrip("/") + "/v1",
        "kimi-for-coding",
        api_key_env="KIMI_API_KEY",
        timeout_seconds=90,
        max_retries=0,
    )
    return BoundedExecutor(inner, max_calls, state_dir)


def llm_bundle(seed: int) -> AgentTaskBundle:
    names = ["Acme", "Globex", "Initech", "Umbrella"]
    issues = ["VPN", "SSO", "billing", "latency"]
    shift = seed % len(names)
    records = [
        (f"customer={names[(index + shift) % 4]};priority=P{1 + index % 3};issue={issues[index]}", f"P{1 + index % 3}|{names[(index + shift) % 4]}|{issues[index]}")
        for index in range(4)
    ]
    train_records = [
        ("customer=Northwind;priority=P3;issue=export", "P3|Northwind|export"),
        ("customer=Contoso;priority=P2;issue=SSO", "P2|Contoso|SSO"),
        ("customer=Cyberdyne;priority=P1;issue=VPN", "P1|Cyberdyne|VPN"),
    ]
    assert {text for text, _ in train_records}.isdisjoint(text for text, _ in records)
    train = tuple(AgentTask(f"{seed}-train-{index}", text, expected) for index, (text, expected) in enumerate(train_records))
    heldout = tuple(AgentTask(f"{seed}-heldout-{index}", text, expected) for index, (text, expected) in enumerate(records))
    start = AgentSpec(
        "Transform each record into the required output. If no output structure is specified, copy the input verbatim. Return only the result.",
        {"temperature": 1, "max_tokens": 256, "seed": seed},
    )
    return AgentTaskBundle(seed, start, train, heldout)


def transfer_families(base: Any, seed: int) -> tuple[TaskFamily, ...]:
    source = TaskFamily(
        "verbatim_records", "source",
        tuple(AgentTask(f"{seed}-source-{index}", text, text) for index, text in enumerate(("case=Keep;owner=Lin", "  spaces=unchanged  ", "plain legacy record"))),
    )
    target = TaskFamily("ticket_formatting", "target", base.heldout_tasks)
    transfer = TaskFamily(
        "renamed_fields", "transfer",
        (
            AgentTask(f"{seed}-transfer-0", "account=Stark;severity=P2;category=database", "P2|Stark|database"),
            AgentTask(f"{seed}-transfer-1", "account=Wayne;severity=P1;category=network", "P1|Wayne|network"),
            AgentTask(f"{seed}-transfer-2", "account=Wonka;severity=P3;category=access", "P3|Wonka|access"),
        ),
    )
    return source, target, transfer


def run_transfer(base: Any, executor: Any, output: Path, name: str, limitations: list[str], agent: bool = False) -> Any:
    families = transfer_families(base, base.manifest.seed) if not agent else tool_transfer_families(base)
    manifest = make_transfer_manifest(name, base, families)
    write_transfer_manifest(manifest, output)
    experiment = run_transfer_evaluation(manifest, base, executor, families)
    experiment.summary["execution_audit"] = executor.model.audit() if agent else executor.audit()
    if agent:
        experiment.summary["tool_invocations"] = sum(trace["tool_name"] is not None for trace in executor.traces)
    write_transfer_evidence(experiment, output, limitations=limitations)
    validate_result_directory(output)
    return experiment


def run_llm(output: Path, max_calls: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    executor = make_executor(max_calls, output / "execution")
    bundles = tuple(llm_bundle(seed) for seed in (101, 202))
    controls = ("random_prompt_edit",)
    manifest = make_agent_paired_manifest(
        "real-llm-reflection-paired", executor.name, "no_change", "model_reflection", 1, bundles, controls
    )
    write_agent_pair_manifest(manifest, output)
    runs = []
    transfers = []
    for bundle in bundles:
        seed_manifest = make_agent_paired_manifest(
            manifest.experiment, manifest.executor, manifest.baseline, manifest.challenger, 1, (bundle,), controls
        )
        proposer_spec = AgentSpec(
            "You improve system prompts using only the supplied training failures. Infer general rules, preserve unrelated behavior, and return only complete replacement instructions.",
            {"temperature": 1, "max_tokens": 768, "seed": bundle.seed + 10_000},
        )
        proposer = ModelReflectionProposer(executor, proposer_spec)
        seed_result = run_agent_paired(
            seed_manifest,
            (bundle,),
            AgentNoChangeOptimizer(),
            AgentOptimizerAdapter("model_reflection", proposer, proposer_source=executor.name),
            executor,
            controls=(AgentOptimizerAdapter("random_prompt_edit", RandomPromptEditor(), accept=accept_always, proposer_source="python_random"),),
        )
        run = seed_result.runs[0]
        runs.append(run)
        transfer = run_transfer(
            run.challenger, executor, output / f"transfer-seed-{bundle.seed}", f"real-llm-transfer-{bundle.seed}",
            [
                "Two real-model paired seeds with four held-out tasks each are exploratory, not a general model ranking.",
                "Only training failures are available to the proposer; source and renamed-field transfer tasks are independently re-evaluated.",
                "Provider token usage is recorded; no price was configured, so dollar cost is unknown rather than free.",
            ],
        )
        transfers.append(transfer)
        (output / "checkpoints").mkdir(exist_ok=True)
        (output / "checkpoints" / f"seed-{bundle.seed}.json").write_text(
            json.dumps({"paired_run": asdict(run), "transfer": transfer.to_dict()}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Completed real LLM seed {bundle.seed}: {executor.audit()['recorded_network_calls']} recorded calls", flush=True)
    if len(runs) != len(manifest.runs):
        raise RuntimeError("Not every planned seed completed")
    differences = [run.heldout_accuracy_improvement for run in runs]
    experiment = AgentPairedExperiment(
        manifest, tuple(runs),
        {
            "run_count": len(runs),
            "metric": "heldout_exact_match_accuracy",
            "improvement_definition": "challenger_accuracy - baseline_accuracy; positive favors challenger",
            "mean_baseline_heldout_accuracy": sum(run.baseline.successor_heldout.accuracy for run in runs) / len(runs),
            "mean_challenger_heldout_accuracy": sum(run.challenger.successor_heldout.accuracy for run in runs) / len(runs),
            "mean_heldout_accuracy_improvement": sum(differences) / len(differences),
            "paired_bootstrap_95_ci": list(bootstrap_mean_ci(differences)),
            "one_sided_sign_flip_p_value": sign_flip_pvalue(differences),
            "candidate_budget_per_arm": 1,
            "baseline_candidate_evaluations": sum(run.baseline.summary["candidate_evaluations"] for run in runs),
            "challenger_candidate_evaluations": sum(run.challenger.summary["candidate_evaluations"] for run in runs),
            "baseline_proposer_cost_usd": 0.0,
            "challenger_proposer_cost_usd": sum(run.challenger.summary["proposer_cost_usd"] for run in runs),
            "baseline_total_input_tokens": sum(run.baseline.summary["total_input_tokens"] for run in runs),
            "baseline_total_output_tokens": sum(run.baseline.summary["total_output_tokens"] for run in runs),
            "challenger_total_input_tokens": sum(run.challenger.summary["total_input_tokens"] for run in runs),
            "challenger_total_output_tokens": sum(run.challenger.summary["total_output_tokens"] for run in runs),
            "challenger_proposer_input_tokens": sum(run.challenger.summary["proposer_input_tokens"] for run in runs),
            "challenger_proposer_output_tokens": sum(run.challenger.summary["proposer_output_tokens"] for run in runs),
            "challenger_combined_total_latency_ms": sum(run.challenger.summary["combined_total_latency_ms"] for run in runs),
            "baseline_total_tokens_reported": all(run.baseline.summary["total_tokens_reported"] for run in runs),
            "challenger_total_tokens_reported": all(run.challenger.summary["total_tokens_reported"] for run in runs),
            "baseline_total_cost_reported": all(run.baseline.summary["total_cost_reported"] for run in runs),
            "challenger_total_cost_reported": all(run.challenger.summary["total_cost_reported"] for run in runs),
            "controls": {"random_prompt_edit": {"mean_heldout_accuracy": sum(run.controls[0].successor_heldout.accuracy for run in runs) / len(runs), "mean_candidate_evaluations": 1.0, "total_proposer_cost_usd": 0.0}},
            "execution_audit": executor.audit(),
            "transfer_checks": {str(run.seed): {"target": transfer.summary["target_improvement"], "transfer": transfer.summary["mean_transfer_improvement"], "worst_source_delta": transfer.summary["worst_source_accuracy_delta"]} for run, transfer in zip(runs, transfers)},
            "scope": "Small real-model validation only; not an open-ended RSI or general model-quality claim",
        },
    )
    limitations = [
        "This run really called kimi-for-coding; it is not a deterministic local executor.",
        "Two paired seeds, three train tasks and four held-out tasks per seed support only a narrow, exploratory format-induction claim.",
        "No-change has zero candidates; model reflection and random editing each have one candidate. The proposer sees training failures only.",
        "Provider usage is retained. No token price was configured, so cost_reported is false and zero dollar fields do not mean free.",
        "This bounded prompt-improvement experiment does not demonstrate open-ended recursive self-improvement.",
    ]
    write_agent_pair_evidence(experiment, output, limitations=limitations)
    validate_result_directory(output)
    print(json.dumps({"summary": experiment.summary, "output": str(output)}, indent=2, ensure_ascii=False), flush=True)


def tool_task(task_id: str, kind: str, argument: str, expected: str) -> AgentTask:
    if kind == "calculator":
        prompt = f"Calculate {argument}. Use the calculator tool. Return only the final numeric answer."
    elif kind == "direct":
        prompt = f"Calculate {argument} without using any tool. Return only the final numeric answer."
    else:
        prompt = f"Return the current stored status for key {argument}. Use the lookup tool; do not guess. Return only the status."
    return AgentTask(task_id, prompt, expected)


def tool_transfer_families(base: Any) -> tuple[TaskFamily, ...]:
    source = TaskFamily(
        "direct_arithmetic", "source",
        (tool_task("source-direct-0", "direct", "6*6", "36"), tool_task("source-direct-1", "direct", "20-8", "12"), tool_task("source-direct-2", "direct", "9+4", "13")),
    )
    target = TaskFamily("tool_use", "target", base.heldout_tasks)
    transfer = TaskFamily(
        "unseen_tool_arguments", "transfer",
        (tool_task("transfer-calc", "calculator", "12*5+3", "63"), tool_task("transfer-lookup", "lookup", "auth-br", "maintenance"), tool_task("transfer-lookup-2", "lookup", "mail-ca", "operational")),
    )
    return source, target, transfer


def run_agent(output: Path, max_calls: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model = make_executor(max_calls, output / "execution")
    executor = ToolAgentExecutor(model)
    start = AgentSpec(
        "You are a tool-using agent. For arithmetic, return exactly TOOL calculator <expression>. For stored statuses, return exactly TOOL lookup <key>. Never guess a stored value. Use one line with no explanation.",
        {"temperature": 1, "max_tokens": 256, "seed": 303},
    )
    train = (
        tool_task("agent-train-calc", "calculator", "7*8+5", "61"),
        tool_task("agent-train-lookup", "lookup", "billing-eu", "degraded"),
        tool_task("agent-train-calc-2", "calculator", "(19-4)*3", "45"),
    )
    heldout = (
        tool_task("agent-heldout-calc", "calculator", "13*4-7", "45"),
        tool_task("agent-heldout-lookup", "lookup", "search-cn", "operational"),
        tool_task("agent-heldout-lookup-2", "lookup", "payments-us", "maintenance"),
        tool_task("agent-heldout-calc-2", "calculator", "72/8+11", "20.0"),
    )
    proposer_spec = AgentSpec(
        "Improve tool-agent system prompts using only supplied training failures. Return complete replacement instructions, preserving the exact tool syntax and unrelated behavior.",
        {"temperature": 1, "max_tokens": 2048, "seed": 10_303},
    )
    proposer = ModelReflectionProposer(model, proposer_spec)
    optimizer = AgentOptimizerAdapter("model_reflection", proposer, proposer_source=model.name)
    manifest = AgentContractManifest(
        "real-tool-agent-two-round-reflection", executor.name, optimizer.name, 303, 2,
        start.artifact_id, tasks_digest(train), tasks_digest(heldout), proposer_source=model.name,
    )
    write_agent_manifest(manifest, output)
    experiment = run_agent_contract(manifest, start, train, heldout, optimizer, executor)
    if len(experiment.history) != 2:
        raise RuntimeError("The two-round real-agent trajectory did not complete")
    limitations = [
        "This is one real LLM-driven tool workflow, not a mock: the model selected tools and the host executed them.",
        "Three training tasks, four held-out tasks and two model-reflection candidates form one bounded lineage, not open-ended RSI.",
        "Calculator inputs use an AST allow-list; eval is forbidden. Lookup values are available only through the host tool.",
        "Provider token and latency observations are retained; no token price was configured, so dollar cost is unknown.",
        "One trajectory cannot establish stable improvement, causal Meta-Plasticity, or performance on other agents.",
    ]
    run_transfer(
        experiment, executor, output / "transfer", "real-tool-agent-transfer", limitations, agent=True
    )
    experiment.summary.update({
        "tool_invocations": sum(trace["tool_name"] is not None for trace in executor.traces),
        "tool_errors": sum(trace["tool_error"] is not None for trace in executor.traces),
        "tool_traces": executor.traces,
        "execution_audit": model.audit(),
        "candidate_lineage": [
            {"iteration": entry.iteration, "candidate": entry.candidate.artifact_id, "parent": entry.candidate.parent, "accepted": entry.accepted, "train_accuracy": entry.evaluation.accuracy, "instructions": entry.candidate.instructions}
            for entry in experiment.history
        ],
        "scope": "Real LLM-driven tool selection with a bounded two-round prompt lineage; not open-ended RSI",
    })
    (output / "agent_traces.json").write_text(json.dumps(executor.traces, indent=2, ensure_ascii=False), encoding="utf-8")
    write_agent_evidence(experiment, output, limitations=limitations)
    validate_result_directory(output)
    validate_result_directory(output / "transfer")
    print(json.dumps({"summary": {key: value for key, value in experiment.summary.items() if key != "tool_traces"}, "output": str(output)}, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("llm", "agent"))
    parser.add_argument("--accept-paid-run", action="store_true", help="Required consent for external requests")
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.accept_paid_run:
        parser.error("External requests require --accept-paid-run and an explicit --max-calls budget")
    output = (args.output or Path("results") / ("real-llm" if args.mode == "llm" else "real-agent")).resolve()
    if args.mode == "llm":
        run_llm(output, args.max_calls)
    else:
        run_agent(output, args.max_calls)


if __name__ == "__main__":
    main()
