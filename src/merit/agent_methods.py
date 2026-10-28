"""Prompt-improvement methods, failure feedback and agent ablation controls."""

from __future__ import annotations

import json
from collections.abc import Callable

from .agent_contract import (
    AgentEvaluation,
    AgentExecutor,
    AgentHistoryEntry,
    AgentOptimizationResult,
    AgentProposal,
    AgentProposalContext,
    AgentResponse,
    AgentSpec,
    AgentTask,
    Budget,
    Usage,
)


def accept_always(current_accuracy: float, candidate_accuracy: float) -> bool:
    del current_accuracy, candidate_accuracy
    return True


class AgentNoChangeOptimizer:
    """Preserve the start artifact without proposing or evaluating candidates."""

    name = "no_change"
    proposer_source = "none"

    def optimize(
        self,
        start: AgentSpec,
        train_tasks: tuple[AgentTask, ...],
        initial_evaluation: AgentEvaluation,
        budget: Budget,
        seed: int,
        evaluate: Callable[[AgentSpec], AgentEvaluation],
    ) -> AgentOptimizationResult:
        del train_tasks, budget, seed, evaluate
        return AgentOptimizationResult(start, initial_evaluation, ())


class RandomPromptEditor:
    """Make seeded syntactic prompt edits; this is a control, not a smart proposer."""

    additions = (
        "Answer precisely and preserve the requested format.",
        "Check the input and output before finalizing.",
        "Prefer deterministic behavior over guessing.",
        "Follow every explicit output constraint.",
    )

    def __call__(self, context: AgentProposalContext) -> AgentSpec:
        instructions = context.current.instructions
        operation = context.rng.randrange(3)
        if operation == 0:
            updated = f"{instructions}\n{context.rng.choice(self.additions)}"
        elif operation == 1 and "\n" in instructions:
            updated = instructions.rsplit("\n", 1)[0]
        elif operation == 2 and "precisely" in instructions:
            updated = instructions.replace("precisely", "carefully")
        else:
            updated = f"{instructions}\n{context.rng.choice(self.additions)}"
        return context.current.evolve(instructions=updated)


def _failure_examples(context: AgentProposalContext, limit: int) -> list[dict[str, str]]:
    tasks = {task.task_id: task for task in context.train_tasks}
    examples = []
    for result in context.current_evaluation.results:
        if result.correct:
            continue
        task = tasks[result.task_id]
        examples.append(
            {
                "task_id": task.task_id,
                "input": task.prompt,
                "expected": task.expected,
                "actual": result.output,
            }
        )
        if len(examples) >= limit:
            break
    return examples


class FailureFeedbackProposer:
    """A deterministic heuristic that appends train failures; it is not LLM reflection."""

    def __init__(self, max_failures: int = 3) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be positive")
        self.max_failures = max_failures

    def __call__(self, context: AgentProposalContext) -> AgentSpec:
        examples = _failure_examples(context, self.max_failures)
        feedback = json.dumps(examples, ensure_ascii=False, sort_keys=True)
        updated = (
            f"{context.current.instructions}\n"
            "Correct these training failures while preserving unrelated behavior:\n"
            f"{feedback}"
        )
        return context.current.evolve(instructions=updated)


class PromptSuffixProposer:
    """Append a fixed, task-independent instruction without reading hidden tasks."""

    def __init__(self, suffix: str) -> None:
        if not suffix.strip():
            raise ValueError("Prompt suffix must not be empty")
        self.suffix = suffix.strip()

    def __call__(self, context: AgentProposalContext) -> AgentSpec:
        return context.current.evolve(
            instructions=f"{context.current.instructions}\n{self.suffix}"
        )


class FixedInstructionsProposer:
    """Replace instructions with fixed text, useful as an explicit harmful control."""

    def __init__(self, instructions: str) -> None:
        if not instructions.strip():
            raise ValueError("Fixed instructions must not be empty")
        self.instructions = instructions

    def __call__(self, context: AgentProposalContext) -> AgentSpec:
        return context.current.evolve(instructions=self.instructions)


class InvalidInstructionsError(ValueError):
    """Report exhausted proposal attempts without losing their aggregate usage."""

    def __init__(self, message: str, usage: Usage) -> None:
        super().__init__(message)
        self.usage = usage


class ModelReflectionProposer:
    """Use another executor to revise instructions from current train failures."""

    def __init__(
        self,
        executor: AgentExecutor,
        proposer_agent: AgentSpec,
        max_failures: int = 5,
        instruction_validator: Callable[[str], str | None] | None = None,
        max_attempts: int = 1,
        feedback_provider: Callable[[AgentProposalContext], dict[str, object] | None] | None = None,
        proposal_renderer: Callable[[AgentSpec, str], AgentSpec] | None = None,
        output_requirement: str | None = None,
    ) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if instruction_validator is not None and not callable(instruction_validator):
            raise TypeError("instruction_validator must be callable")
        if feedback_provider is not None and not callable(feedback_provider):
            raise TypeError("feedback_provider must be callable")
        if proposal_renderer is not None and not callable(proposal_renderer):
            raise TypeError("proposal_renderer must be callable")
        self.executor = executor
        self.proposer_agent = proposer_agent
        self.max_failures = max_failures
        self.instruction_validator = instruction_validator
        self.max_attempts = max_attempts
        self.feedback_provider = feedback_provider
        self.proposal_renderer = proposal_renderer
        self.output_requirement = output_requirement or "Return only revised instructions; preserve unrelated behavior and do not ask for held-out tasks."

    @property
    def source(self) -> str:
        return self.executor.name

    def __call__(self, context: AgentProposalContext) -> AgentProposal:
        examples = _failure_examples(context, self.max_failures)
        structured_feedback = (
            self.feedback_provider(context) if self.feedback_provider is not None else None
        )
        total_usage = Usage()
        validation_error = None
        previous_instructions = None
        for attempt in range(1, self.max_attempts + 1):
            request = {
                "current_instructions": context.current.instructions,
                "current_config": context.current.config,
                "train_failures": examples,
                "requirement": self.output_requirement,
            }
            if structured_feedback is not None:
                request["structured_train_feedback"] = structured_feedback
            if validation_error is not None:
                request["previous_invalid_instructions"] = previous_instructions
                request["validation_error"] = validation_error
                request["retry_requirement"] = self.output_requirement + " Fix the validation error and return no explanation."
            prompt = json.dumps(request, ensure_ascii=False, sort_keys=True)
            response = self.executor.invoke(
                self.proposer_agent,
                AgentTask(
                    task_id=(
                        f"proposer-{context.iteration}" if attempt == 1
                        else f"proposer-{context.iteration}-attempt-{attempt}"
                    ),
                    prompt=prompt,
                    expected="",
                ),
            )
            if not isinstance(response, AgentResponse):
                raise TypeError("Proposer executor must return an AgentResponse")
            total_usage = total_usage + response.usage
            instructions = response.output.strip()
            previous_instructions = instructions
            if not instructions:
                validation_error = "Proposer returned empty instructions"
            elif self.proposal_renderer is not None:
                try:
                    artifact = self.proposal_renderer(context.current, instructions)
                except ValueError as error:
                    validation_error = str(error)
                else:
                    if not isinstance(artifact, AgentSpec):
                        raise TypeError("Proposal renderer must return an AgentSpec")
                    return AgentProposal(artifact=artifact, usage=total_usage)
            elif self.instruction_validator is not None:
                validation_error = self.instruction_validator(instructions)
            else:
                validation_error = None
            if validation_error is None:
                return AgentProposal(
                    artifact=context.current.evolve(instructions=instructions),
                    usage=total_usage,
                )
        raise InvalidInstructionsError(
            f"Proposer returned invalid instructions after {self.max_attempts} attempts: "
            f"{validation_error}",
            total_usage,
        )
