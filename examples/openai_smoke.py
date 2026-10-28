"""Explicit paid OpenAI-compatible smoke; not used by offline tests or demos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from merit import (
    AgentContractManifest,
    AgentOptimizerAdapter,
    AgentSpec,
    AgentTask,
    OpenAICompatibleExecutor,
    TokenPricing,
    run_agent_contract,
    tasks_digest,
)
from merit.agent_report import write_agent_evidence, write_agent_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four external model invocations with explicit cost consent")
    parser.add_argument("--accept-paid-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/openai-smoke"))
    parser.add_argument("--input-per-million", type=float)
    parser.add_argument("--output-per-million", type=float)
    args = parser.parse_args()
    if not args.accept_paid_run:
        parser.error("This example makes external paid requests; rerun with --accept-paid-run after setting a budget")
    if (args.input_per_million is None) != (args.output_per_million is None):
        parser.error("Input and output prices must be supplied together, or both omitted")
    pricing = None
    if args.input_per_million is not None:
        pricing = TokenPricing(args.input_per_million, args.output_per_million)
    executor = OpenAICompatibleExecutor(
        base_url=os.environ["OPENAI_BASE_URL"],
        model=os.environ["OPENAI_MODEL"],
        api_key_env="OPENAI_API_KEY",
        pricing=pricing,
    )
    start = AgentSpec("Return exactly the requested text.", config={"temperature": 0, "max_tokens": 16})
    train = (AgentTask("smoke-train", "Return exactly: merit", "merit"),)
    heldout = (AgentTask("smoke-heldout", "Return exactly: evaluation", "evaluation"),)
    optimizer = AgentOptimizerAdapter(
        "clarify_output",
        lambda context: context.current.evolve(
            instructions=context.current.instructions + " Do not add punctuation or explanation."
        ),
    )
    manifest = AgentContractManifest(
        experiment="openai-compatible-smoke",
        executor=executor.name,
        optimizer=optimizer.name,
        seed=7,
        candidate_budget=1,
        start_artifact_id=start.artifact_id,
        train_task_digest=tasks_digest(train),
        heldout_task_digest=tasks_digest(heldout),
        proposer_source=optimizer.proposer_source,
    )
    print("Planned external invocations: 4. Provider retries may add requests; stop if this exceeds your budget.")
    output = args.output.resolve()
    write_agent_manifest(manifest, output)
    experiment = run_agent_contract(manifest, start, train, heldout, optimizer, executor)
    evidence_path, report_path = write_agent_evidence(
        experiment,
        output,
        limitations=[
            "This is a real OpenAI-compatible executor smoke, not a local model simulation.",
            "One train task and one held-out task validate connectivity only, not model quality or statistical significance.",
            "Usage and cost are limited to provider responses and any explicitly configured token prices; retries may add requests.",
        ],
    )
    print(f"Evidence: {evidence_path}")
    print(f"Report: {report_path}")
    print(f"Cost reported: {experiment.summary['total_cost_reported']}")
    print(f"Total cost USD: {experiment.summary['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()
