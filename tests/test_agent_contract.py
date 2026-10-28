import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from merit import AgentSpec, Budget, Usage, tasks_digest
from merit.agent_contract import AgentOptimizerAdapter, run_agent_contract
from merit.agent_demo import (
    LocalTextNormalizer,
    make_local_agent_manifest,
    make_local_agent_start,
    make_local_agent_tasks,
    propose_normalizer,
    run_local_agent_demo,
)
from merit.agent_report import write_agent_evidence, write_agent_manifest


class AgentContractTests(unittest.TestCase):
    def test_agent_spec_is_immutable_and_tracks_lineage(self):
        config = {"model": {"name": "local"}, "enabled": False}
        start = AgentSpec("instructions", config)
        config["enabled"] = True
        self.assertFalse(start.config["enabled"])
        returned_config = start.config
        returned_config["enabled"] = True
        self.assertFalse(start.config["enabled"])
        self.assertEqual(start.artifact_id, AgentSpec("instructions", {"enabled": False, "model": {"name": "local"}}).artifact_id)
        child = start.evolve(config_updates={"enabled": True})
        self.assertEqual(child.parent, start.artifact_id)
        self.assertNotEqual(child.artifact_id, start.artifact_id)

    def test_usage_aggregates_and_rejects_invalid_values(self):
        total = Usage(2, 3, 0.01, 4.0) + Usage(5, 7, 0.02, 6.0)
        self.assertEqual(total, Usage(7, 10, 0.03, 10.0))
        with self.assertRaises(ValueError):
            Usage(input_tokens=-1)
        with self.assertRaises(ValueError):
            Usage(cost_usd=float("nan"))

    def test_local_agent_contract_improves_without_hidden_task_access(self):
        seen_task_ids = []

        def propose(context):
            seen_task_ids.extend(task.task_id for task in context.train_tasks)
            return propose_normalizer(context)

        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        optimizer = AgentOptimizerAdapter("toggle_normalization", propose)
        experiment = run_agent_contract(
            make_local_agent_manifest(), start, train, heldout, optimizer, LocalTextNormalizer()
        )
        summary = experiment.summary
        self.assertEqual(summary["initial_dev_accuracy"], 0.25)
        self.assertEqual(summary["successor_dev_accuracy"], 1.0)
        self.assertEqual(summary["baseline_heldout_accuracy"], 0.25)
        self.assertEqual(summary["successor_heldout_accuracy"], 1.0)
        self.assertEqual(summary["heldout_accuracy_improvement"], 0.75)
        self.assertEqual(summary["candidate_evaluations"], 4)
        self.assertEqual(summary["accepted_candidates"], 2)
        self.assertEqual(summary["dev_agent_invocations"], 20)
        self.assertEqual(summary["heldout_agent_invocations"], 8)
        self.assertGreater(summary["total_input_tokens"], 0)
        self.assertEqual(summary["total_cost_usd"], 0.0)
        self.assertTrue(all(task_id.startswith("train-") for task_id in seen_task_ids))
        self.assertEqual(experiment.successor.config, {"lowercase": True, "strip": True})
        current_id = start.artifact_id
        for entry in experiment.history:
            self.assertEqual(entry.candidate.parent, current_id)
            if entry.accepted:
                current_id = entry.candidate.artifact_id
        self.assertEqual(experiment.successor.artifact_id, current_id)

    def test_agent_adapter_rejects_invalid_proposals(self):
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        for proposal, exception in (
            (lambda context: context.current.config, TypeError),
            (lambda context: AgentSpec("orphan"), ValueError),
        ):
            with self.subTest(exception=exception):
                optimizer = AgentOptimizerAdapter("toggle_normalization", proposal)
                with self.assertRaises(exception):
                    run_agent_contract(
                        make_local_agent_manifest(), start, train, heldout, optimizer, LocalTextNormalizer()
                    )

    def test_agent_manifest_detects_task_substitution(self):
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        optimizer = AgentOptimizerAdapter("toggle_normalization", propose_normalizer)
        with self.assertRaises(ValueError):
            run_agent_contract(
                make_local_agent_manifest(), start, train, heldout[:2], optimizer, LocalTextNormalizer()
            )

    def test_train_and_heldout_inputs_must_be_disjoint(self):
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        contaminated = (
            replace(heldout[0], prompt=train[0].prompt, expected=train[0].expected),
            *heldout[1:],
        )
        manifest = replace(
            make_local_agent_manifest(),
            heldout_task_digest=tasks_digest(contaminated),
        )
        optimizer = AgentOptimizerAdapter("toggle_normalization", propose_normalizer)
        with self.assertRaisesRegex(ValueError, "inputs must be disjoint"):
            run_agent_contract(
                manifest, start, train, contaminated, optimizer, LocalTextNormalizer()
            )

    def test_agent_evidence_contains_task_outputs_and_costs(self):
        experiment = run_local_agent_demo()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest_path = write_agent_manifest(experiment.manifest, output)
            evidence_path, html_path = write_agent_evidence(experiment, output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_digest"], experiment.manifest.digest)
            self.assertEqual(evidence["manifest_digest"], experiment.manifest.digest)
            self.assertEqual(len(evidence["experiment"]["history"]), 4)
            self.assertEqual(len(evidence["experiment"]["successor_heldout"]["results"]), 4)
            self.assertIn("usage", evidence["experiment"]["successor_heldout"]["results"][0])
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("本地 Agent 契约样例", html)
            self.assertIn("不是 LLM", html)
            self.assertIn("100.00%", html)


if __name__ == "__main__":
    unittest.main()
