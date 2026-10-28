import json
import tempfile
import unittest
from pathlib import Path

from merit.agent_contract import AgentResponse, AgentSpec, AgentTask, Usage
from merit.agent_demo import LocalTextNormalizer
from merit.agent_methods import ModelReflectionProposer
from merit.agent_pair_demo import make_local_agent_bundles
from merit.meta_plasticity import (
    PlasticityProposer,
    make_meta_plasticity_manifest,
    run_meta_plasticity,
)
from merit.meta_plasticity_demo import (
    make_local_meta_plasticity_manifest,
    make_local_plasticity_families,
    make_local_plasticity_proposer,
    make_local_plasticity_trajectories,
    run_local_meta_plasticity_demo,
)
from merit.meta_plasticity_report import write_meta_plasticity_evidence, write_meta_plasticity_manifest


class MockPlasticityProposerExecutor:
    name = "mock_plasticity_proposer"

    def __init__(self):
        self.requests = []

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        self.requests.append({"agent": agent, "task": task})
        return AgentResponse(
            output="Revise only from the supplied training feedback.",
            usage=Usage(30, 10, 0.01, 4.0, cost_is_estimate=True),
        )


class MetaPlasticityTests(unittest.TestCase):
    def test_current_agent_strength_is_separate_from_successor_quality(self):
        experiment = run_local_meta_plasticity_demo(trajectory_count=3, seed_count=3)
        summary = experiment.summary
        self.assertEqual(summary["paired_run_count"], 9)
        self.assertEqual(summary["mean_early_current_agent_accuracy"], 0.25)
        self.assertEqual(summary["mean_late_current_agent_accuracy"], 1.0)
        self.assertEqual(summary["mean_current_agent_delta_late_minus_early"], 0.75)
        self.assertEqual(summary["mean_early_successor_accuracy"], 1.0)
        self.assertEqual(summary["mean_late_successor_accuracy"], 0.5)
        self.assertEqual(summary["mean_successor_delta_late_minus_early"], -0.5)
        self.assertEqual(summary["trajectory_bootstrap_95_ci"], [-0.5, -0.5])
        self.assertEqual(summary["early_mean_evaluations_to_threshold_when_reached"], 2.0)
        self.assertIsNone(summary["late_mean_evaluations_to_threshold_when_reached"])
        self.assertEqual(summary["early_threshold_reached_rate"], 1.0)
        self.assertEqual(summary["late_threshold_reached_rate"], 0.0)

    def test_common_start_equal_budget_and_trajectory_statistics(self):
        experiment = run_local_meta_plasticity_demo(trajectory_count=3, seed_count=2)
        self.assertEqual(len(experiment.summary["per_trajectory"]), 3)
        for run in experiment.runs:
            self.assertEqual(run.early_followup.start, run.late_followup.start)
            self.assertEqual(run.early_followup.train_tasks, run.late_followup.train_tasks)
            self.assertEqual(run.early_followup.heldout_tasks, run.late_followup.heldout_tasks)
            self.assertEqual(run.early_followup.summary["candidate_evaluations"], 4)
            self.assertEqual(run.late_followup.summary["candidate_evaluations"], 4)
            for followup in (run.early_followup, run.late_followup):
                accepted = [entry.candidate for entry in followup.history if entry.accepted]
                self.assertEqual(accepted[0].parent, run.start_artifact_id)
                self.assertEqual(accepted[-1], followup.successor)
                for parent, child in zip(accepted, accepted[1:]):
                    self.assertEqual(child.parent, parent.artifact_id)
        self.assertEqual(experiment.summary["uncertainty_unit"].split(";")[0], "trajectory")

    def test_transfer_regression_and_family_costs_are_recorded(self):
        experiment = run_local_meta_plasticity_demo(trajectory_count=2, seed_count=2)
        summary = experiment.summary
        self.assertEqual(summary["mean_transfer_successor_delta_late_minus_early"], -0.5)
        self.assertEqual(summary["early_worst_source_improvement"], -0.75)
        self.assertEqual(summary["late_worst_source_improvement"], -0.5)
        self.assertEqual(set(summary["family_summary"]), {"plasticity_verbatim_echo", "plasticity_unicode_normalization"})
        for run in experiment.runs:
            self.assertEqual(len(run.family_comparisons), 2)
            for comparison in run.family_comparisons:
                self.assertEqual(len(comparison.common_start.results), 4)
                self.assertEqual(len(comparison.early_successor.results), 4)
                self.assertEqual(len(comparison.late_successor.results), 4)
        self.assertEqual(summary["total_observed_cost_usd"], 0.0)

    def test_manifest_rejects_snapshot_substitution(self):
        trajectories = list(make_local_plasticity_trajectories(1))
        manifest = make_local_meta_plasticity_manifest(trajectory_count=1, seed_count=1)
        original = trajectories[0]
        late = original.late
        changed_late = type(late)(
            late.trajectory_id,
            late.phase,
            late.improver.evolve(instructions="Changed after manifest creation"),
            late.current_agent,
        )
        trajectories[0] = type(original)(original.trajectory_id, original.early, changed_late)
        with self.assertRaisesRegex(ValueError, "Manifest snapshots"):
            run_meta_plasticity(
                manifest,
                trajectories,
                make_local_agent_bundles(1),
                make_local_plasticity_proposer(),
                LocalTextNormalizer(),
                make_local_plasticity_families(),
            )

    def test_model_reflection_has_separate_cost_and_no_hidden_task_access(self):
        mock = MockPlasticityProposerExecutor()
        proposer = PlasticityProposer(
            "model_reflection",
            mock.name,
            lambda improver: ModelReflectionProposer(mock, improver),
            config={"max_failures": 5},
        )
        trajectories = make_local_plasticity_trajectories(1)
        original = make_local_agent_bundles(1)[0]
        heldout = (
            AgentTask("secret-one", "UNSEEN SECRET ONE", "unseen secret one"),
            AgentTask("secret-two", "  HIDDEN TWO  ", "hidden two"),
            AgentTask("secret-three", "  PRIVATE THREE  ", "private three"),
            AgentTask("secret-four", "reserved", "reserved"),
        )
        bundles = (type(original)(original.seed, original.start, original.train_tasks, heldout),)
        manifest = make_meta_plasticity_manifest(
            "mock-plasticity",
            LocalTextNormalizer.name,
            proposer,
            2,
            1.0,
            trajectories,
            bundles,
        )
        experiment = run_meta_plasticity(
            manifest,
            trajectories,
            bundles,
            proposer,
            LocalTextNormalizer(),
        )
        self.assertEqual(len(mock.requests), 4)
        self.assertEqual(experiment.summary["early_method_cost_usd"], 0.02)
        self.assertEqual(experiment.summary["late_method_cost_usd"], 0.02)
        self.assertEqual(experiment.summary["total_observed_cost_usd"], 0.04)
        run = experiment.runs[0]
        self.assertEqual(run.early_followup.summary["proposer_cost_usd"], 0.02)
        self.assertEqual(run.late_followup.summary["proposer_cost_usd"], 0.02)
        for request in mock.requests:
            prompt = request["task"].prompt
            self.assertIn("train", prompt)
            for task in bundles[0].heldout_tasks:
                self.assertNotIn(task.task_id, prompt)
                self.assertNotIn(task.prompt, prompt)

    def test_evidence_contains_snapshots_thresholds_and_nested_statistics(self):
        experiment = run_local_meta_plasticity_demo(trajectory_count=2, seed_count=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest_path = write_meta_plasticity_manifest(experiment.manifest, output)
            evidence_path, html_path = write_meta_plasticity_evidence(experiment, output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_digest"], experiment.manifest.digest)
            self.assertEqual(evidence["manifest_digest"], experiment.manifest.digest)
            payload = evidence["experiment"]
            self.assertEqual(len(payload["summary"]["per_trajectory"]), 2)
            self.assertEqual(len(payload["summary"]["per_run"]), 4)
            self.assertEqual(len(payload["trajectories"]), 2)
            for trajectory in payload["trajectories"]:
                self.assertIn("instructions", trajectory["early"]["improver"])
                self.assertIn("config_json", trajectory["late"]["improver"])
                self.assertIn("artifact_id", trajectory["early"]["current_agent"])
                self.assertIn("artifact_id", trajectory["late"]["current_agent"])
            for run in payload["runs"]:
                self.assertIn("early_snapshot_id", run)
                self.assertIn("late_snapshot_id", run)
                self.assertIn("proposer_usage", run["early_followup"]["history"][0])
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("早期／晚期改进器可塑性", html)
            self.assertIn("轨迹层级", html)
            self.assertIn("不是 LLM", html)
            self.assertIn("+75.00%", html)
            self.assertIn("-50.00%", html)

    def test_real_run_notice_replaces_constructed_snapshot_limitations(self):
        experiment = run_local_meta_plasticity_demo(trajectory_count=1, seed_count=1)
        with tempfile.TemporaryDirectory() as directory:
            evidence_path, html_path = write_meta_plasticity_evidence(
                experiment,
                Path(directory),
                limitations=["Snapshots came from a real local model run."],
                scope_notice="真实模型快照；不是开放式 RSI 结论。",
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["limitations"], ["Snapshots came from a real local model run."])
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("真实模型快照", html)
            self.assertNotIn("不是 LLM", html)


if __name__ == "__main__":
    unittest.main()
