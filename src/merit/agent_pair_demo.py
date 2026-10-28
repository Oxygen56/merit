"""A deterministic multi-seed local agent improvement and ablation comparison."""

from __future__ import annotations

import random

from .agent_contract import AgentOptimizerAdapter, AgentProposalContext, AgentTask
from .agent_demo import LocalTextNormalizer, make_local_agent_start, propose_normalizer
from .agent_methods import (
    AgentNoChangeOptimizer,
    FailureFeedbackProposer,
    RandomPromptEditor,
    accept_always,
)
from .agent_pair import (
    AgentPairedExperiment,
    AgentPairedManifest,
    AgentTaskBundle,
    make_agent_paired_manifest,
    run_agent_paired,
)


CONTROL_NAMES = ("no_change", "random_prompt_edit", "failure_feedback")


def propose_lowercase_only(context: AgentProposalContext):
    return context.current.evolve(
        config_updates={"lowercase": not context.current.config.get("lowercase", False)}
    )


def make_local_agent_bundle(seed: int) -> AgentTaskBundle:
    rng = random.Random(seed)
    lower_words = ["HELLO", "WELCOME", "CONTRACT", "MERIT", "AGENT"]
    plain_words = ["world", "plain", "stable", "local", "text"]
    mixed_words = ["Mixed Case", "Held Out", "New Task", "Agent Improvement"]
    lower = rng.choice(lower_words)
    plain = rng.choice(plain_words)
    mixed = rng.choice(mixed_words)
    transfer_lower = rng.choice([word for word in lower_words if word != lower])
    transfer_plain = rng.choice([word for word in plain_words if word != plain])
    transfer_mixed = rng.choice([word for word in mixed_words if word != mixed])
    train = (
        AgentTask(f"{seed}-train-lowercase", lower, lower.lower()),
        AgentTask(f"{seed}-train-strip", f"  {plain}  ", plain),
        AgentTask(f"{seed}-train-both", f"  {mixed.upper()}  ", mixed.lower()),
        AgentTask(f"{seed}-train-noop", transfer_plain, transfer_plain),
    )
    heldout = (
        AgentTask(f"{seed}-heldout-lowercase", transfer_lower, transfer_lower.lower()),
        AgentTask(f"{seed}-heldout-strip", f"  {transfer_plain.upper()}  ", transfer_plain.lower()),
        AgentTask(f"{seed}-heldout-both", f"  {transfer_mixed.upper()}  ", transfer_mixed.lower()),
        AgentTask(f"{seed}-heldout-noop", plain, plain),
    )
    return AgentTaskBundle(seed, make_local_agent_start(), train, heldout)


def make_local_agent_bundles(seed_count: int = 8) -> tuple[AgentTaskBundle, ...]:
    return tuple(make_local_agent_bundle(seed) for seed in range(2101, 2101 + seed_count))


def make_local_agent_paired_manifest(
    seed_count: int = 8,
    candidate_budget: int = 4,
) -> AgentPairedManifest:
    return make_agent_paired_manifest(
        experiment="local-agent-paired",
        executor=LocalTextNormalizer.name,
        baseline="lowercase_only",
        challenger="toggle_normalization",
        candidate_budget=candidate_budget,
        bundles=make_local_agent_bundles(seed_count),
        controls=CONTROL_NAMES,
    )


def make_local_agent_controls() -> tuple[AgentOptimizerAdapter, ...]:
    return (
        AgentNoChangeOptimizer(),
        AgentOptimizerAdapter(
            "random_prompt_edit",
            RandomPromptEditor(),
            accept=accept_always,
            proposer_source="python_random",
        ),
        AgentOptimizerAdapter(
            "failure_feedback",
            FailureFeedbackProposer(),
            proposer_source="python_heuristic",
        ),
    )


def run_local_agent_paired_demo(
    seed_count: int = 8,
    candidate_budget: int = 4,
) -> AgentPairedExperiment:
    bundles = make_local_agent_bundles(seed_count)
    manifest = make_agent_paired_manifest(
        experiment="local-agent-paired",
        executor=LocalTextNormalizer.name,
        baseline="lowercase_only",
        challenger="toggle_normalization",
        candidate_budget=candidate_budget,
        bundles=bundles,
        controls=CONTROL_NAMES,
    )
    baseline = AgentOptimizerAdapter("lowercase_only", propose_lowercase_only)
    challenger = AgentOptimizerAdapter("toggle_normalization", propose_normalizer)
    return run_agent_paired(
        manifest,
        bundles,
        baseline,
        challenger,
        LocalTextNormalizer(),
        controls=make_local_agent_controls(),
    )
