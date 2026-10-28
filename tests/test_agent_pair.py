import json
import tempfile
import unittest
from pathlib import Path

from merit.agent_demo import LocalTextNormalizer
from merit.agent_pair import run_agent_paired
from merit.agent_pair_demo import (
    make_local_agent_bundles,
    make_local_agent_controls,
    make_local_agent_paired_manifest,
    propose_lowercase_only,
    run_local_agent_paired_demo,
)
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.agent_contract import AgentOptimizerAdapter
from merit.agent_demo import propose_normalizer


class AgentPairTests(unittest.TestCase):
    def test_paired_agent_comparison_uses_equal_budgets_and_paired_stats(self):
        experiment = run_local_agent_paired_demo(seed_count=4, candidate_budget=4)
        summary = experiment.summary
        self.assertEqual(summary["run_count"], 4)
        self.assertEqual(summary["mean_baseline_heldout_accuracy"], 0.5)
        self.assertEqual(summary["mean_challenger_heldout_accuracy"], 1.0)
        self.assertEqual(summary["mean_heldout_accuracy_improvement"], 0.5)
        self.assertEqual(summary["paired_bootstrap_95_ci"], [0.5, 0.5])
        self.assertEqual(summary["baseline_candidate_evaluations"], 16)
        self.assertEqual(summary["challenger_candidate_evaluations"], 16)
        for run in experiment.runs:
            self.assertEqual(run.baseline.start, run.challenger.start)
            self.assertEqual(run.baseline.train_tasks, run.challenger.train_tasks)
            self.assertEqual(run.baseline.heldout_tasks, run.challenger.heldout_tasks)
            self.assertEqual(run.baseline.summary["candidate_evaluations"], 4)
            self.assertEqual(run.challenger.summary["candidate_evaluations"], 4)
            self.assertEqual(
                run.baseline.baseline_heldout.accuracy,
                run.challenger.baseline_heldout.accuracy,
            )

    def test_local_train_and_heldout_inputs_are_disjoint(self):
        for bundle in make_local_agent_bundles(20):
            train_inputs = {task.prompt for task in bundle.train_tasks}
            heldout_inputs = {task.prompt for task in bundle.heldout_tasks}
            self.assertTrue(train_inputs.isdisjoint(heldout_inputs))

    def test_paired_agent_run_is_deterministic(self):
        first = run_local_agent_paired_demo(seed_count=3, candidate_budget=4)
        second = run_local_agent_paired_demo(seed_count=3, candidate_budget=4)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_paired_manifest_rejects_bundle_substitution(self):
        bundles = list(make_local_agent_bundles(2))
        manifest = make_local_agent_paired_manifest(seed_count=2)
        bundles[0] = type(bundles[0])(
            bundles[0].seed,
            bundles[0].start,
            bundles[0].train_tasks[:2],
            bundles[0].heldout_tasks,
        )
        baseline = AgentOptimizerAdapter("lowercase_only", propose_lowercase_only)
        challenger = AgentOptimizerAdapter("toggle_normalization", propose_normalizer)
        with self.assertRaisesRegex(ValueError, "seed specifications"):
            run_agent_paired(
                manifest,
                bundles,
                baseline,
                challenger,
                LocalTextNormalizer(),
                controls=make_local_agent_controls(),
            )

    def test_paired_evidence_contains_both_arms_and_interval(self):
        experiment = run_local_agent_paired_demo(seed_count=3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest_path = write_agent_pair_manifest(experiment.manifest, output)
            evidence_path, html_path = write_agent_pair_evidence(experiment, output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_digest"], experiment.manifest.digest)
            self.assertEqual(evidence["manifest_digest"], experiment.manifest.digest)
            self.assertEqual(len(evidence["experiment"]["runs"]), 3)
            self.assertEqual(evidence["experiment"]["summary"]["paired_bootstrap_95_ci"], [0.5, 0.5])
            for run in evidence["experiment"]["runs"]:
                self.assertEqual(len(run["baseline"]["history"]), 4)
                self.assertEqual(len(run["challenger"]["history"]), 4)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("多种子 Agent 改进对照", html)
            self.assertIn("+50.00%", html)
            self.assertIn("不是 LLM", html)


if __name__ == "__main__":
    unittest.main()
