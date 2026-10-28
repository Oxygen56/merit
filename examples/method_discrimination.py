"""Check that Merit separates effective, neutral, and harmful improvers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from merit.agent_contract import AgentOptimizerAdapter, AgentSpec
from merit.agent_demo import LocalTextNormalizer, propose_normalizer
from merit.agent_methods import AgentNoChangeOptimizer, accept_always
from merit.agent_pair import AgentTaskBundle, make_agent_paired_manifest, run_agent_paired
from merit.agent_pair_demo import make_local_agent_bundles, make_local_agent_controls, propose_lowercase_only
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.evidence import validate_result_directory


def propose_disable_normalization(context):
    return context.current.evolve(config_updates={"lowercase": False, "strip": False})


def run_experiments(output_root: Path) -> dict[str, dict]:
    output_root = Path(output_root)
    bundles = make_local_agent_bundles(seed_count=8)
    baseline = AgentNoChangeOptimizer()
    effective = AgentOptimizerAdapter("toggle_normalization", propose_normalizer)
    effective_controls = (
        AgentOptimizerAdapter("lowercase_only", propose_lowercase_only),
        *make_local_agent_controls()[1:],
    )
    effective_manifest = make_agent_paired_manifest(
        "method-discrimination-effective",
        LocalTextNormalizer.name,
        "no_change",
        "toggle_normalization",
        4,
        bundles,
        tuple(control.name for control in effective_controls),
    )
    effective_result = run_agent_paired(
        effective_manifest,
        bundles,
        baseline,
        effective,
        LocalTextNormalizer(),
        effective_controls,
    )

    capable_start = AgentSpec(
        "Normalize the input according to the supplied configuration.",
        {"lowercase": True, "strip": True},
    )
    harmful_bundles = tuple(
        AgentTaskBundle(bundle.seed, capable_start, bundle.train_tasks, bundle.heldout_tasks)
        for bundle in bundles
    )
    harmful = AgentOptimizerAdapter(
        "disable_normalization",
        propose_disable_normalization,
        accept=accept_always,
    )
    harmful_manifest = make_agent_paired_manifest(
        "method-discrimination-harmful",
        LocalTextNormalizer.name,
        "no_change",
        "disable_normalization",
        1,
        harmful_bundles,
    )
    harmful_result = run_agent_paired(
        harmful_manifest,
        harmful_bundles,
        baseline,
        harmful,
        LocalTextNormalizer(),
    )

    assert effective_result.summary["mean_heldout_accuracy_improvement"] > 0
    assert effective_result.summary["controls"]["random_prompt_edit"]["mean_heldout_accuracy"] == effective_result.summary["mean_baseline_heldout_accuracy"]
    assert effective_result.summary["controls"]["failure_feedback"]["mean_heldout_accuracy"] == effective_result.summary["mean_baseline_heldout_accuracy"]
    assert harmful_result.summary["mean_heldout_accuracy_improvement"] < 0

    summaries = {}
    for name, result in (("effective", effective_result), ("harmful", harmful_result)):
        output = output_root / name
        write_agent_pair_manifest(result.manifest, output)
        write_agent_pair_evidence(
            result,
            output,
            limitations=["Deterministic local executor; this checks method discrimination, not LLM quality."],
        )
        validate_result_directory(output)
        summaries[name] = result.summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/method-discrimination"))
    args = parser.parse_args()
    print(json.dumps(run_experiments(args.output), indent=2))


if __name__ == "__main__":
    main()
