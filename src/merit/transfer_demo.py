"""Local task-family transfer and regression example."""

from __future__ import annotations

from .agent_contract import AgentContractExperiment, AgentTask
from .agent_demo import LocalTextNormalizer, run_local_agent_demo
from .transfer import (
    TaskFamily,
    TransferExperiment,
    TransferManifest,
    make_transfer_manifest,
    run_transfer_evaluation,
)


def make_local_task_families(base: AgentContractExperiment) -> tuple[TaskFamily, ...]:
    source = TaskFamily(
        name="verbatim_echo",
        role="source",
        tasks=(
            AgentTask("source-case", "Keep CASE", "Keep CASE"),
            AgentTask("source-spaces", "  keep spaces  ", "  keep spaces  "),
            AgentTask("source-plain", "plain", "plain"),
            AgentTask("source-capital", "Already", "Already"),
        ),
    )
    target = TaskFamily(name="normalization", role="target", tasks=base.heldout_tasks)
    transfer = TaskFamily(
        name="unicode_normalization",
        role="transfer",
        tasks=(
            AgentTask("transfer-accent", "CAFÉ", "café"),
            AgentTask("transfer-cjk", "  中文  ", "中文"),
            AgentTask("transfer-both", "  NAÏVE  ", "naïve"),
            AgentTask("transfer-noop", "déjà", "déjà"),
        ),
    )
    return source, target, transfer


def make_local_transfer_manifest(base: AgentContractExperiment) -> TransferManifest:
    return make_transfer_manifest(
        experiment="local-transfer-regression",
        base=base,
        families=make_local_task_families(base),
    )


def run_local_transfer_demo(candidate_budget: int = 4) -> TransferExperiment:
    base = run_local_agent_demo(candidate_budget=candidate_budget)
    families = make_local_task_families(base)
    manifest = make_transfer_manifest("local-transfer-regression", base, families)
    return run_transfer_evaluation(manifest, base, LocalTextNormalizer(), families)
