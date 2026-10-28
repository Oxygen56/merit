"""Runnable offline executor example: python examples/custom_executor.py"""

from __future__ import annotations

import argparse
from pathlib import Path

from merit import (
    AgentContractManifest,
    AgentOptimizerAdapter,
    AgentResponse,
    AgentSpec,
    AgentTask,
    Usage,
    run_agent_contract,
    tasks_digest,
)
from merit.agent_report import write_agent_evidence, write_agent_manifest


class StripExecutor:
    name = "example_strip_executor_v1"

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        output = task.prompt.strip() if agent.config.get("strip") else task.prompt
        return AgentResponse(output, Usage(input_tokens=len(task.prompt.split()), output_tokens=len(output.split())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/custom-executor"))
    args = parser.parse_args()
    start = AgentSpec("Apply the configured text transformation.", config={"strip": False})
    train = (AgentTask("example-train", "  hello  ", "hello"),)
    heldout = (AgentTask("example-heldout", "  MERIT  ", "MERIT"),)
    optimizer = AgentOptimizerAdapter(
        "enable_strip",
        lambda context: context.current.evolve(config_updates={"strip": True}),
    )
    executor = StripExecutor()
    manifest = AgentContractManifest(
        experiment="custom-executor-example",
        executor=executor.name,
        optimizer=optimizer.name,
        seed=7,
        candidate_budget=1,
        start_artifact_id=start.artifact_id,
        train_task_digest=tasks_digest(train),
        heldout_task_digest=tasks_digest(heldout),
    )
    output = args.output.resolve()
    write_agent_manifest(manifest, output)
    experiment = run_agent_contract(manifest, start, train, heldout, optimizer, executor)
    evidence_path, report_path = write_agent_evidence(experiment, output)
    print(f"Evidence: {evidence_path}")
    print(f"Report: {report_path}")
    print(f"Held-out improvement: {experiment.summary['heldout_accuracy_improvement']:+.2%}")


if __name__ == "__main__":
    main()
