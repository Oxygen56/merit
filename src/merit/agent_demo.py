"""A deterministic local executor that validates the agent adapter contract."""

from __future__ import annotations

from .agent_contract import (
    AgentContractExperiment,
    AgentContractManifest,
    AgentOptimizerAdapter,
    AgentProposalContext,
    AgentResponse,
    AgentSpec,
    AgentTask,
    Usage,
    run_agent_contract,
    tasks_digest,
)


class LocalTextNormalizer:
    """A config-driven text agent; token counts are deterministic local proxies."""

    name = "local_text_normalizer"

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        output = task.prompt
        if agent.config.get("strip", False):
            output = output.strip()
        if agent.config.get("lowercase", False):
            output = output.lower()
        usage = Usage(
            input_tokens=len(f"{agent.instructions} {task.prompt}".split()),
            output_tokens=len(output.split()),
            cost_usd=0.0,
            latency_ms=0.0,
        )
        return AgentResponse(output=output, usage=usage)


def make_local_agent_start() -> AgentSpec:
    return AgentSpec(
        instructions="Normalize the input according to the supplied configuration.",
        config={"lowercase": False, "strip": False},
    )


def make_local_agent_tasks() -> tuple[tuple[AgentTask, ...], tuple[AgentTask, ...]]:
    train = (
        AgentTask("train-lowercase", "HELLO", "hello"),
        AgentTask("train-strip", "  world  ", "world"),
        AgentTask("train-both", "  MIXED CASE  ", "mixed case"),
        AgentTask("train-noop", "plain", "plain"),
    )
    heldout = (
        AgentTask("heldout-lowercase", "NEW TASK", "new task"),
        AgentTask("heldout-strip", "  FRESH  ", "fresh"),
        AgentTask("heldout-both", "  HELD OUT  ", "held out"),
        AgentTask("heldout-noop", "unchanged", "unchanged"),
    )
    return train, heldout


def propose_normalizer(context: AgentProposalContext) -> AgentSpec:
    flag = ("lowercase", "strip")[context.iteration % 2]
    return context.current.evolve(config_updates={flag: not context.current.config.get(flag, False)})


def make_local_agent_manifest(
    candidate_budget: int = 4,
    optimizer_name: str = "toggle_normalization",
) -> AgentContractManifest:
    start = make_local_agent_start()
    train, heldout = make_local_agent_tasks()
    return AgentContractManifest(
        experiment="local-agent-contract",
        executor=LocalTextNormalizer.name,
        optimizer=optimizer_name,
        seed=7,
        candidate_budget=candidate_budget,
        start_artifact_id=start.artifact_id,
        train_task_digest=tasks_digest(train),
        heldout_task_digest=tasks_digest(heldout),
    )


def run_local_agent_demo(candidate_budget: int = 4) -> AgentContractExperiment:
    start = make_local_agent_start()
    train, heldout = make_local_agent_tasks()
    manifest = make_local_agent_manifest(candidate_budget)
    optimizer = AgentOptimizerAdapter("toggle_normalization", propose_normalizer)
    return run_agent_contract(
        manifest=manifest,
        start=start,
        train_tasks=train,
        heldout_tasks=heldout,
        optimizer=optimizer,
        executor=LocalTextNormalizer(),
    )
