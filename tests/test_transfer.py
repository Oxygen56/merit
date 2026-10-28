import json
import tempfile
import unittest
from pathlib import Path

from merit.agent_demo import LocalTextNormalizer, run_local_agent_demo
from merit.transfer import make_transfer_manifest, run_transfer_evaluation
from merit.transfer_demo import make_local_task_families, run_local_transfer_demo
from merit.transfer_report import write_transfer_evidence, write_transfer_manifest


class TransferTests(unittest.TestCase):
    def test_transfer_matrix_detects_gain_and_old_capability_regression(self):
        experiment = run_local_transfer_demo()
        summary = experiment.summary
        self.assertEqual(summary["target_baseline_accuracy"], 0.25)
        self.assertEqual(summary["target_successor_accuracy"], 1.0)
        self.assertEqual(summary["target_improvement"], 0.75)
        self.assertEqual(summary["mean_transfer_baseline_accuracy"], 0.25)
        self.assertEqual(summary["mean_transfer_successor_accuracy"], 1.0)
        self.assertEqual(summary["mean_transfer_improvement"], 0.75)
        self.assertEqual(summary["min_source_baseline_accuracy"], 1.0)
        self.assertEqual(summary["min_source_successor_accuracy"], 0.25)
        self.assertEqual(summary["worst_source_accuracy_delta"], -0.75)
        self.assertEqual(summary["additional_agent_invocations"], 16)
        self.assertGreater(summary["additional_input_tokens"], 0)
        self.assertEqual(summary["additional_cost_usd"], 0.0)
        target = next(comparison for comparison in experiment.comparisons if comparison.role == "target")
        self.assertTrue(target.reused_base_evaluation)
        self.assertIs(target.baseline, experiment.base.baseline_heldout)
        self.assertIs(target.successor, experiment.base.successor_heldout)

    def test_transfer_manifest_rejects_task_family_substitution(self):
        base = run_local_agent_demo()
        families = list(make_local_task_families(base))
        manifest = make_transfer_manifest("local-transfer-regression", base, families)
        transfer = families[-1]
        families[-1] = type(transfer)(transfer.name, transfer.role, transfer.tasks[:2])
        with self.assertRaises(ValueError):
            run_transfer_evaluation(manifest, base, LocalTextNormalizer(), families)

    def test_transfer_evidence_contains_every_family_and_output(self):
        experiment = run_local_transfer_demo()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest_path = write_transfer_manifest(experiment.manifest, output)
            evidence_path, html_path = write_transfer_evidence(experiment, output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_digest"], experiment.manifest.digest)
            self.assertEqual(evidence["manifest_digest"], experiment.manifest.digest)
            comparisons = evidence["experiment"]["comparisons"]
            self.assertEqual({item["role"] for item in comparisons}, {"source", "target", "transfer"})
            for comparison in comparisons:
                self.assertEqual(len(comparison["successor"]["results"]), 4)
                self.assertTrue(all("output" in result for result in comparison["successor"]["results"]))
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("迁移与旧能力回归", html)
            self.assertIn("-75.00%", html)
            self.assertIn("不是 LLM", html)


if __name__ == "__main__":
    unittest.main()
