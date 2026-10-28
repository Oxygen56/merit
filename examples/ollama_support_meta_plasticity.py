"""M8: paired follow-up optimization from real early/late improver snapshots."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from merit.agent_contract import AgentSpec, evaluate_agent
from merit.agent_methods import ModelReflectionProposer
from merit.agent_pair import AgentTaskBundle
from merit.meta_plasticity import (
    ImproverSnapshot,
    ImproverTrajectory,
    PlasticityProposer,
    make_meta_plasticity_manifest,
    run_meta_plasticity,
)
from merit.meta_plasticity_report import write_meta_plasticity_evidence, write_meta_plasticity_manifest
from merit.multi_step_executor import MultiStepToolExecutor
from merit.transfer import TaskFamily
from ollama_support_error_dynamic_patch import render_priority_patch
from ollama_support_error_reflection import (
    BASE_INSTRUCTIONS,
    HELDOUT,
    LOOKUP_VALUES,
    PROPOSER_INSTRUCTIONS,
    TRAIN,
    CallCounter,
    make_model,
    task,
)
from ollama_support_error_stop_patch import (
    STOP_OUTPUT_REQUIREMENT,
    STOP_RULE_TEXT,
    validate_stop_coverage,
)
from ollama_support_error_structured_patch import parse_patch
from ollama_support_error_trajectory_reflection import build_structured_feedback
from ollama_support_error_validation_final import FINAL

TRANSFER = (
    task("transfer-normal-8001", "ticket-8001", "50.0"),
    task("transfer-missing-policy-8003", "ticket-8003", "REFUND_UNAVAILABLE"),
)
TRANSFER_LOOKUP_VALUES = {
    "ticket-8001": "order-8001",
    "order-8001": '{"total":125,"policy":"policy-T01"}',
    "policy-T01": "0.4",
    "ticket-8003": "order-8003",
    "order-8003": '{"total":140,"policy":"policy-T03"}',
}
EARLY_IMPROVER_INSTRUCTIONS = (
    PROPOSER_INSTRUCTIONS
    + "\nReturn the requested constrained JSON rule patch, not a revised prompt."
)
LATE_IMPROVER_SUFFIX = (
    "\nReturn a constrained JSON rule patch, not a revised prompt. "
    "The latest rejected candidate still performs actions after a required record failure."
)


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path, role: str, seed: int) -> dict:
    pilot = json.loads((path / "pilot.json").read_text())
    return {
        "role": role,
        "seed": seed,
        "pilot_path": str(path),
        "pilot_sha256": digest_file(path / "pilot.json"),
        "traces_sha256": digest_file(path / "agent_traces.json"),
        "candidate_train_accuracy": pilot["candidate_train_accuracy"],
        "candidate_accepted": pilot["candidate_accepted"],
        "candidate_trajectory_valid": pilot["candidate_trajectory_valid"],
        "heldout_tasks_evaluated": pilot["heldout_tasks_evaluated"],
        "candidate_instructions": pilot["candidate_instructions"],
    }


def make_snapshots(output: Path):
    early_path = Path("results/support-error-stop-patch-pilot")
    early = source_record(early_path, "early", 101)
    trajectories = []
    sources = []
    for seed in (1010, 2020):
        late_path = Path(f"results/support-error-dynamic-patch-train-repeated/seed-{seed}")
        late = source_record(late_path, "late", seed)
        early_agent = AgentSpec(
            early["candidate_instructions"],
            {"temperature": 0, "max_tokens": 128, "seed": seed},
        )
        late_agent = AgentSpec(
            late["candidate_instructions"],
            {"temperature": 0, "max_tokens": 128, "seed": seed},
        )
        early_improver = AgentSpec(
            EARLY_IMPROVER_INSTRUCTIONS,
            {"temperature": 0.2, "max_tokens": 1024, "seed": 10101},
        )
        late_improver = AgentSpec(
            PROPOSER_INSTRUCTIONS + LATE_IMPROVER_SUFFIX,
            {"temperature": 0.2, "max_tokens": 1024, "seed": seed + 10000},
        )
        trajectory_id = f"support-{seed}"
        trajectories.append(ImproverTrajectory(
            trajectory_id,
            ImproverSnapshot(trajectory_id, "early", early_improver, early_agent),
            ImproverSnapshot(trajectory_id, "late", late_improver, late_agent),
        ))
        sources.append({"trajectory_id": trajectory_id, "early": early, "late": late})
    payload = {
        "snapshot_origin": "Actual train-only stop-patch and dynamic-patch runs; no manually assigned quality labels",
        "sources": sources,
    }
    (output / "snapshot-sources.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return tuple(trajectories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/support-meta-plasticity-real"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trajectories = make_snapshots(args.output)
    bundles = tuple(
        AgentTaskBundle(
            seed,
            AgentSpec(BASE_INSTRUCTIONS, {"temperature": 0, "max_tokens": 128, "seed": seed}),
            TRAIN,
            HELDOUT,
        )
        for seed in (3030, 4040)
    )
    families = (
        TaskFamily("support_final_source", "source", (FINAL[0], FINAL[2])),
        TaskFamily("support_transfer", "transfer", TRANSFER),
    )
    model = make_model(args.output / "cache")
    executor = MultiStepToolExecutor(
        model,
        {**LOOKUP_VALUES, **TRANSFER_LOOKUP_VALUES},
        max_steps=6,
        recover_tool_errors=True,
    )

    # Feedback is computed independently for each common start, before follow-up optimization.
    feedback_by_start = {}
    initial_by_start = {}
    for bundle in bundles:
        initial = evaluate_agent(executor, bundle.start, TRAIN)
        feedback_by_start[bundle.start.artifact_id] = build_structured_feedback(
            TRAIN,
            executor.traces,
            [asdict(row) for row in initial.results],
        )
        initial_by_start[bundle.start.artifact_id] = initial.accuracy
    (args.output / "common-start-train-feedback.json").write_text(
        json.dumps(feedback_by_start, ensure_ascii=False, indent=2) + "\n"
    )

    proposal_calls = {"early": 0, "late": 0}
    phase_by_improver = {
        snapshot.improver.artifact_id: snapshot.phase
        for trajectory in trajectories
        for snapshot in (trajectory.early, trajectory.late)
    }

    def factory(improver: AgentSpec):
        phase = phase_by_improver[improver.artifact_id]
        counted = CallCounter(model)

        def counted_invoke(agent, task):
            response = counted.invoke(agent, task)
            proposal_calls[phase] += 1
            return response

        class PhaseCounter:
            name = model.name
            invoke = staticmethod(counted_invoke)

        def render(current: AgentSpec, text: str) -> AgentSpec:
            feedback = feedback_by_start[current.artifact_id]
            rules = parse_patch(text, STOP_RULE_TEXT)
            validate_stop_coverage(rules, feedback)
            return render_priority_patch(current, text)

        return ModelReflectionProposer(
            PhaseCounter(),
            improver,
            max_failures=5,
            max_attempts=3,
            feedback_provider=lambda context: feedback_by_start[context.current.artifact_id],
            proposal_renderer=render,
            output_requirement=STOP_OUTPUT_REQUIREMENT,
        )

    proposer = PlasticityProposer(
        "real_dynamic_priority_patch",
        model.name,
        factory,
        config={
            "renderer": "required-record-failure-first",
            "max_failures": 5,
            "max_attempts": 3,
            "feedback": "sanitized common-start train trajectories only",
            "accept": "greedy train exact-match accuracy",
        },
    )
    manifest = make_meta_plasticity_manifest(
        "support-real-early-late-plasticity-m8",
        executor.name,
        proposer,
        candidate_budget=1,
        dev_threshold=1.0,
        trajectories=trajectories,
        bundles=bundles,
        families=families,
    )
    write_meta_plasticity_manifest(manifest, args.output)
    experiment = run_meta_plasticity(
        manifest,
        trajectories,
        bundles,
        proposer,
        executor,
        families,
    )
    experiment.summary["execution"] = {
        "model": model.model,
        "network_calls_this_run": model.network_calls,
        "cache_hits_this_run": model.cache_hits,
        "cached_response_files": len(list((args.output / "cache").glob("*.json"))),
        "model_steps": len(executor.traces),
        "proposal_calls": proposal_calls,
        "common_start_train_accuracy": initial_by_start,
        "paid_api_calls": 0,
    }
    evidence_path, report_path = write_meta_plasticity_evidence(
        experiment,
        args.output,
        limitations=[
            "Early and late snapshots come from actual local stop-patch and dynamic-patch optimization runs on qwen2.5:7b.",
            "Only two real trajectories and two paired follow-up seeds were run; trajectory-level uncertainty has little statistical power.",
            "Follow-up selection uses one training candidate and greedy exact-match accuracy; it does not include trajectory-gated acceptance.",
            "Source and transfer families are small synthetic support tasks, not production traffic or an open-ended RSI evaluation.",
            "Threshold speed uses accepted development evaluations; unreached runs remain right-censored.",
        ],
        scope_notice=(
            "早晚快照来自本机 qwen2.5:7b 的真实训练运行。后续优化从同一个 BASE Agent 出发，"
            "候选预算、训练任务和测试任务一致。样本只有 2 条轨迹 × 2 个后续种子，"
            "结果只描述这组受限提示修改，不能证明开放式 RSI 可塑性。"
        ),
    )
    (args.output / "agent_traces.json").write_text(
        json.dumps(executor.traces, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "summary": experiment.summary,
        "evidence": str(evidence_path),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
