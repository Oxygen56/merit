"""A deterministic local example separating agent strength from improver plasticity."""

from __future__ import annotations

from .agent_contract import AgentProposeCallback, AgentSpec, AgentTask
from .agent_demo import LocalTextNormalizer, make_local_agent_start, propose_normalizer
from .agent_pair import AgentTaskBundle
from .agent_pair_demo import make_local_agent_bundles, propose_lowercase_only
from .meta_plasticity import (
    ImproverSnapshot,
    ImproverTrajectory,
    MetaPlasticityExperiment,
    MetaPlasticityManifest,
    PlasticityProposer,
    make_meta_plasticity_manifest,
    run_meta_plasticity,
)
from .transfer import TaskFamily


def make_local_plasticity_trajectories(count: int = 4) -> tuple[ImproverTrajectory, ...]:
    """Construct snapshot artifacts directly; no real improver training is claimed."""

    trajectories = []
    for index in range(count):
        trajectory_id = f"local-trajectory-{index}"
        early_improver = AgentSpec(
            instructions=f"Trajectory {index}: early proposer uses staged normalization.",
            config={"followup_action": "normalize"},
        )
        late_improver = AgentSpec(
            instructions=f"Trajectory {index}: late proposer only toggles lowercase.",
            config={"followup_action": "lowercase_only"},
        )
        late_current = make_local_agent_start().evolve(config_updates={"strip": True, "lowercase": True})
        trajectories.append(
            ImproverTrajectory(
                trajectory_id=trajectory_id,
                early=ImproverSnapshot(
                    trajectory_id,
                    "early",
                    early_improver,
                    make_local_agent_start(),
                ),
                late=ImproverSnapshot(
                    trajectory_id,
                    "late",
                    late_improver,
                    late_current,
                ),
            )
        )
    return tuple(trajectories)


def make_local_plasticity_proposer() -> PlasticityProposer:
    def factory(improver: AgentSpec) -> AgentProposeCallback:
        action = improver.config.get("followup_action")
        if action == "normalize":
            return propose_normalizer
        if action == "lowercase_only":
            return propose_lowercase_only
        raise ValueError(f"Unknown local follow-up action: {action}")

    return PlasticityProposer(
        name="local_snapshot_rules",
        source="python_snapshot",
        factory=factory,
        config={"strategy": "snapshot_config", "normalization_order": ["lowercase", "strip"]},
        accept_policy="greedy_accuracy",
    )


def make_local_plasticity_families() -> tuple[TaskFamily, ...]:
    return (
        TaskFamily(
            name="plasticity_verbatim_echo",
            role="source",
            tasks=(
                AgentTask("plasticity-source-case", "Keep CASE", "Keep CASE"),
                AgentTask("plasticity-source-spaces", "  keep spaces  ", "  keep spaces  "),
                AgentTask("plasticity-source-plain", "plain", "plain"),
                AgentTask("plasticity-source-capital", "Already", "Already"),
            ),
        ),
        TaskFamily(
            name="plasticity_unicode_normalization",
            role="transfer",
            tasks=(
                AgentTask("plasticity-transfer-accent", "CAFÉ", "café"),
                AgentTask("plasticity-transfer-cjk", "  中文  ", "中文"),
                AgentTask("plasticity-transfer-both", "  NAÏVE  ", "naïve"),
                AgentTask("plasticity-transfer-noop", "déjà", "déjà"),
            ),
        ),
    )


def make_local_meta_plasticity_manifest(
    trajectory_count: int = 4,
    seed_count: int = 4,
    candidate_budget: int = 4,
    dev_threshold: float = 1.0,
) -> MetaPlasticityManifest:
    return make_meta_plasticity_manifest(
        experiment="local-meta-plasticity",
        executor=LocalTextNormalizer.name,
        proposer=make_local_plasticity_proposer(),
        candidate_budget=candidate_budget,
        dev_threshold=dev_threshold,
        trajectories=make_local_plasticity_trajectories(trajectory_count),
        bundles=make_local_agent_bundles(seed_count),
        families=make_local_plasticity_families(),
    )


def run_local_meta_plasticity_demo(
    trajectory_count: int = 4,
    seed_count: int = 4,
    candidate_budget: int = 4,
    dev_threshold: float = 1.0,
) -> MetaPlasticityExperiment:
    trajectories = make_local_plasticity_trajectories(trajectory_count)
    bundles: tuple[AgentTaskBundle, ...] = make_local_agent_bundles(seed_count)
    proposer = make_local_plasticity_proposer()
    families = make_local_plasticity_families()
    manifest = make_meta_plasticity_manifest(
        experiment="local-meta-plasticity",
        executor=LocalTextNormalizer.name,
        proposer=proposer,
        candidate_budget=candidate_budget,
        dev_threshold=dev_threshold,
        trajectories=trajectories,
        bundles=bundles,
        families=families,
    )
    return run_meta_plasticity(
        manifest,
        trajectories,
        bundles,
        proposer,
        LocalTextNormalizer(),
        families,
    )
