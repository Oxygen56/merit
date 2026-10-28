"""GEPA prompt optimization with separate reflection and candidate-selection tasks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from .agent_contract import (
    AgentEvaluation,
    AgentExecutor,
    AgentHistoryEntry,
    AgentOptimizationResult,
    AgentResponse,
    AgentSpec,
    AgentTask,
    Usage,
    evaluate_agent,
)
from .budget import Budget


def _candidate_key(candidate: Mapping[str, str]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _argument_differences(expected: Any, actual: Any, path: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = [
            f"- {path}.{key}: missing argument; expected {_json_value(expected[key])}"
            for key in sorted(expected.keys() - actual.keys())
        ]
        differences.extend(
            f"- {path}.{key}: extra argument; expected absent; actual {_json_value(actual[key])}"
            for key in sorted(actual.keys() - expected.keys())
        )
        for key in sorted(expected.keys() & actual.keys()):
            differences.extend(_argument_differences(expected[key], actual[key], f"{path}.{key}"))
        return differences
    if _json_type(expected) != _json_type(actual):
        return [
            f"- {path}: type mismatch; expected {_json_type(expected)} {_json_value(expected)}; "
            f"actual {_json_type(actual)} {_json_value(actual)}"
        ]
    if expected != actual:
        return [f"- {path}: value mismatch; expected {_json_value(expected)}; actual {_json_value(actual)}"]
    return []


def structured_json_feedback(expected: str, output: str, score: float) -> str:
    """Explain JSON tool-call errors without changing the exact-match score."""
    differences: list[str] = []
    try:
        expected_call = json.loads(expected)
        actual_call = json.loads(output)
    except json.JSONDecodeError:
        differences.append("- Output or expected value is not valid JSON; return one JSON object only")
    else:
        if not isinstance(actual_call, dict):
            differences.append("- Output must be a JSON object with name and arguments")
        else:
            expected_name = expected_call.get("name") if isinstance(expected_call, dict) else None
            if actual_call.get("name") != expected_name:
                differences.append(
                    f"- name: tool mismatch; expected {_json_value(expected_name)}; "
                    f"actual {_json_value(actual_call.get('name'))}"
                )
            expected_arguments = expected_call.get("arguments", {}) if isinstance(expected_call, dict) else {}
            actual_arguments = actual_call.get("arguments", {})
            if not isinstance(actual_arguments, dict):
                differences.append(
                    f"- arguments: type mismatch; expected object; actual {_json_type(actual_arguments)}"
                )
            else:
                differences.extend(_argument_differences(expected_arguments, actual_arguments, "arguments"))
    if not differences:
        differences.append("- No structural JSON differences; preserve the exact required output format")
    return (
        f"Exact-match score: {score:g}.\n"
        "Structured differences:\n"
        + "\n".join(differences)
        + "\nConvert these failures into general rules for unseen tools and questions. "
        "Do not copy task questions, tool names, argument values, or worked examples into the revised instructions. "
        "Preserve the required output format."
    )


@dataclass
class _CandidateRecord:
    agent: AgentSpec
    evaluation: AgentEvaluation
    proposer_usage: Usage
    remaining_budget: int
    selection_evaluation: AgentEvaluation | None = None


class _MeritGEPAAdapter:
    """Evaluate each candidate once on training tasks and optional selection tasks."""

    propose_new_texts = None

    def __init__(
        self,
        start: AgentSpec,
        initial_evaluation: AgentEvaluation,
        train_tasks: tuple[AgentTask, ...],
        budget: Budget,
        evaluate_candidate: Callable[[AgentSpec], AgentEvaluation],
        feedback_format: str = "exact_match",
        validation_tasks: tuple[AgentTask, ...] = (),
        initial_selection_evaluation: AgentEvaluation | None = None,
        evaluate_selection: Callable[[AgentSpec], AgentEvaluation] | None = None,
    ) -> None:
        self.start = start
        self.train_tasks = train_tasks
        self.validation_tasks = validation_tasks
        self.budget = budget
        self.evaluate_candidate = evaluate_candidate
        self.evaluate_selection = evaluate_selection
        self.feedback_format = feedback_format
        self.train_ids = {task.task_id for task in train_tasks}
        self.validation_ids = {task.task_id for task in validation_tasks}
        self.seed_key = _candidate_key({"instructions": start.instructions})
        self.records = {
            self.seed_key: _CandidateRecord(
                start, initial_evaluation, Usage(), budget.remaining, initial_selection_evaluation,
            ),
        }
        self.order: list[str] = []
        self.parent_key = self.seed_key
        self.pending_proposer_usage = Usage()

    def add_proposer_usage(self, usage: Usage) -> None:
        self.pending_proposer_usage = self.pending_proposer_usage + usage

    def evaluate(self, batch: list[AgentTask], candidate: dict[str, str], capture_traces: bool = False) -> Any:
        from gepa.core.adapter import EvaluationBatch

        key = _candidate_key(candidate)
        if key not in self.records:
            if self.parent_key not in self.records:
                raise RuntimeError("GEPA proposed a candidate before evaluating its parent")
            parent = self.records[self.parent_key].agent
            agent = AgentSpec(candidate["instructions"], parent.config, parent=parent.artifact_id)
            evaluation = self.evaluate_candidate(agent)
            selection_evaluation = None
            if self.validation_tasks:
                if self.evaluate_selection is None:
                    raise RuntimeError("GEPA validation evaluation is not configured")
                selection_evaluation = self.evaluate_selection(agent)
            self.records[key] = _CandidateRecord(
                agent,
                evaluation,
                self.pending_proposer_usage,
                self.budget.remaining,
                selection_evaluation,
            )
            self.order.append(key)
            self.pending_proposer_usage = Usage()
        record = self.records[key]
        batch_ids = {task.task_id for task in batch}
        if batch_ids and batch_ids.issubset(self.validation_ids):
            evaluation = record.selection_evaluation
            if evaluation is None:
                raise RuntimeError("GEPA requested validation scores before validation evaluation")
        elif batch_ids and batch_ids.issubset(self.train_ids):
            evaluation = record.evaluation
        else:
            raise ValueError("GEPA batch must contain only known training or validation tasks")
        by_id = {result.task_id: result for result in evaluation.results}
        outputs = [by_id[task.task_id].output for task in batch]
        scores = [float(by_id[task.task_id].correct) for task in batch]
        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "task_id": task.task_id,
                    "input": task.prompt,
                    "expected": task.expected,
                    "output": by_id[task.task_id].output,
                }
                for task in batch
            ]
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        key = _candidate_key(candidate)
        if key not in self.records:
            raise RuntimeError("GEPA requested reflection for an unevaluated candidate")
        self.parent_key = key
        if eval_batch.trajectories is None:
            raise ValueError("GEPA reflection requires trajectories")
        if any(trajectory["task_id"] not in self.train_ids for trajectory in eval_batch.trajectories):
            raise ValueError("GEPA reflection only accepts training trajectories")
        examples = []
        for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores, strict=True):
            if self.feedback_format == "structured_json_diff":
                feedback = structured_json_feedback(trajectory["expected"], trajectory["output"], score)
            else:
                feedback = (
                    f"Exact-match score: {score:g}. Expected output: {trajectory['expected']}. "
                    "Use this training example to diagnose the instruction change; "
                    "preserve the required output format."
                )
            examples.append(
                {
                    "Inputs": {"task": trajectory["input"]},
                    "Generated Outputs": trajectory["output"],
                    "Feedback": feedback,
                }
            )
        return {component: examples for component in components_to_update}


class GepaOptimizer:
    """Run GEPA without exposing final-test tasks or validation trajectories to reflection."""

    name = "gepa"

    def __init__(
        self,
        executor: AgentExecutor,
        proposer_agent: AgentSpec,
        reflection_minibatch_size: int = 3,
        feedback_format: str = "exact_match",
        validation_tasks: tuple[AgentTask, ...] = (),
        validation_executor: AgentExecutor | None = None,
    ) -> None:
        if reflection_minibatch_size < 1:
            raise ValueError("reflection_minibatch_size must be positive")
        if feedback_format not in {"exact_match", "structured_json_diff"}:
            raise ValueError("feedback_format must be exact_match or structured_json_diff")
        self.executor = executor
        self.proposer_agent = proposer_agent
        self.reflection_minibatch_size = reflection_minibatch_size
        self.feedback_format = feedback_format
        self.validation_tasks = tuple(validation_tasks)
        self.validation_executor = validation_executor or executor

    @property
    def proposer_source(self) -> str:
        return self.executor.name

    def optimize(
        self,
        start: AgentSpec,
        train_tasks: tuple[AgentTask, ...],
        initial_evaluation: AgentEvaluation,
        budget: Budget,
        seed: int,
        evaluate: Callable[[AgentSpec], AgentEvaluation],
    ) -> AgentOptimizationResult:
        try:
            import gepa
        except ImportError as error:
            raise RuntimeError("GEPA needs: pip install 'merit-agent-eval[gepa]'") from error

        if {task.task_id for task in train_tasks}.intersection(task.task_id for task in self.validation_tasks):
            raise ValueError("GEPA training and validation task IDs must be disjoint")
        initial_selection = None
        evaluate_selection = None
        if self.validation_tasks:
            initial_selection = evaluate_agent(self.validation_executor, start, self.validation_tasks)
            evaluate_selection = lambda agent: evaluate_agent(self.validation_executor, agent, self.validation_tasks)
        adapter = _MeritGEPAAdapter(
            start,
            initial_evaluation,
            train_tasks,
            budget,
            evaluate,
            self.feedback_format,
            self.validation_tasks,
            initial_selection,
            evaluate_selection,
        )
        reflection_calls = 0

        def reflect(prompt: str) -> str:
            nonlocal reflection_calls
            response = self.executor.invoke(
                self.proposer_agent,
                AgentTask(f"gepa-proposer-{seed}-{reflection_calls}", prompt, ""),
            )
            if not isinstance(response, AgentResponse):
                raise TypeError("GEPA reflection executor must return an AgentResponse")
            reflection_calls += 1
            adapter.add_proposer_usage(response.usage)
            return response.output

        minibatch_size = min(self.reflection_minibatch_size, len(train_tasks))
        selection_tasks = self.validation_tasks or train_tasks
        # This second limit also terminates repeated identical proposals, which do not
        # consume another Merit candidate evaluation because their results are reused.
        max_metric_calls = len(selection_tasks) + budget.remaining * (2 * minibatch_size + len(selection_tasks))
        result = gepa.optimize(
            seed_candidate={"instructions": start.instructions},
            trainset=list(train_tasks),
            valset=list(self.validation_tasks) if self.validation_tasks else None,
            adapter=adapter,
            reflection_lm=reflect,
            reflection_minibatch_size=minibatch_size,
            max_metric_calls=max_metric_calls,
            stop_callbacks=lambda state: budget.remaining <= 0,
            seed=seed,
            use_merge=False,
            cache_evaluation=False,
            track_best_outputs=False,
            display_progress_bar=False,
        )
        best_candidate = result.best_candidate
        if not isinstance(best_candidate, dict) or not isinstance(best_candidate.get("instructions"), str):
            raise TypeError("GEPA returned an invalid candidate")
        best_key = _candidate_key(best_candidate)
        if best_key not in adapter.records:
            raise RuntimeError("GEPA selected a candidate without a Merit evaluation")
        accepted_keys = {_candidate_key(candidate) for candidate in result.candidates}
        history = tuple(
            AgentHistoryEntry(
                iteration=iteration,
                candidate=adapter.records[key].agent,
                evaluation=adapter.records[key].evaluation,
                accepted=key in accepted_keys,
                remaining_budget=adapter.records[key].remaining_budget,
                proposer_usage=adapter.records[key].proposer_usage,
                selection_evaluation=adapter.records[key].selection_evaluation,
            )
            for iteration, key in enumerate(adapter.order)
        )
        best = adapter.records[best_key]
        details = {
            "library": "gepa",
            "library_version": version("gepa"),
            "selection_data": "validation_only" if self.validation_tasks else "train_only",
            "validation_task_count": len(self.validation_tasks),
            "feedback_format": self.feedback_format,
            "reflection_minibatch_size": minibatch_size,
            "reflection_calls": reflection_calls,
            "gepa_total_metric_calls": result.total_metric_calls,
            "gepa_best_idx": result.best_idx,
            "gepa_candidates": result.candidates,
            "gepa_parents": result.parents,
            "gepa_train_aggregate_scores": result.val_aggregate_scores,
            "gepa_selection_aggregate_scores": result.val_aggregate_scores,
        }
        return AgentOptimizationResult(
            best.agent,
            best.evaluation,
            history,
            details=details,
            additional_proposer_usage=adapter.pending_proposer_usage,
            initial_selection_evaluation=initial_selection,
            successor_selection_evaluation=best.selection_evaluation,
        )
