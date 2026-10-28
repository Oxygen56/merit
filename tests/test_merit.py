import json
import tempfile
import unittest
from pathlib import Path

from merit import AgentArtifact, Budget, BudgetExceeded, NoChangeOptimizer, PythonOptimizerAdapter
from merit.demo import default_manifest, make_start, make_tasks, run_toy_experiment
from merit.optimizers import RandomSearchOptimizer
from merit.report import write_evidence, write_manifest
from merit.runner import PairRunner
from merit.stats import bootstrap_mean_ci, sign_flip_pvalue


class RogueOptimizer:
    name = "rogue"

    def optimize(self, start, initial_loss, budget, seed, evaluate):
        del initial_loss, seed
        candidate = start.with_values(tuple(value + 1.0 for value in start.values))
        loss = evaluate(candidate)
        evaluate(candidate)  # Deliberately exceed a one-candidate budget.
        from merit.model import OptimizeResult

        return OptimizeResult(candidate, loss, ())


class MeritTests(unittest.TestCase):
    def test_artifact_lineage_and_content_digest(self):
        start = AgentArtifact((0.0, 1.0))
        child = start.with_values((0.5, 1.0))
        self.assertEqual(child.parent, start.artifact_id)
        self.assertNotEqual(child.artifact_id, start.artifact_id)
        self.assertEqual(child.artifact_id, AgentArtifact((0.5, 1.0), parent=start.artifact_id).artifact_id)

    def test_budget_cannot_be_exceeded(self):
        budget = Budget(1)
        budget.consume()
        with self.assertRaises(BudgetExceeded):
            budget.consume()

    def test_pairing_uses_same_start_and_equal_caps(self):
        experiment = run_toy_experiment(default_manifest(seed_count=2, candidate_budget=8))
        for run in experiment.runs:
            self.assertEqual(run.start_artifact_id, run.baseline.start_artifact_id)
            self.assertEqual(run.start_artifact_id, run.challenger.start_artifact_id)
            self.assertEqual(run.baseline.initial_heldout_rmse, run.challenger.initial_heldout_rmse)
            self.assertEqual(run.baseline.candidate_evaluations, 8)
            self.assertEqual(run.challenger.candidate_evaluations, 8)

    def test_repeated_run_is_deterministic(self):
        manifest = default_manifest(seed_count=2, candidate_budget=8)
        self.assertEqual(run_toy_experiment(manifest).to_dict(), run_toy_experiment(manifest).to_dict())

    def test_identical_optimizer_has_zero_paired_delta(self):
        baseline = RandomSearchOptimizer()
        challenger = RandomSearchOptimizer()
        challenger.name = "random_search_copy"
        manifest = default_manifest(seed_count=2, candidate_budget=8)
        object.__setattr__(manifest, "challenger", "random_search_copy")
        object.__setattr__(manifest, "controls", ("no_change",))
        runner = PairRunner(baseline, challenger, make_start, make_tasks, controls=(NoChangeOptimizer(),))
        experiment = runner.run(manifest)
        self.assertEqual(experiment.summary["mean_heldout_rmse_improvement"], 0.0)
        for run in experiment.runs:
            self.assertEqual(run.baseline.heldout_rmse, run.challenger.heldout_rmse)

    def test_rogue_optimizer_is_stopped_by_budget(self):
        baseline = RandomSearchOptimizer()
        rogue = RogueOptimizer()
        manifest = default_manifest(seed_count=1, candidate_budget=1)
        object.__setattr__(manifest, "challenger", rogue.name)
        object.__setattr__(manifest, "controls", ("no_change",))
        runner = PairRunner(baseline, rogue, make_start, make_tasks, controls=(NoChangeOptimizer(),))
        with self.assertRaises(BudgetExceeded):
            runner.run(manifest)

    def test_no_change_preserves_start_without_spending_budget(self):
        experiment = run_toy_experiment(default_manifest(seed_count=2, candidate_budget=4))
        self.assertIn("no_change", experiment.manifest.controls)
        for run in experiment.runs:
            control = {arm.optimizer: arm for arm in run.controls}["no_change"]
            self.assertEqual(control.start_artifact_id, run.start_artifact_id)
            self.assertEqual(control.successor.artifact_id, run.start_artifact_id)
            self.assertEqual(control.heldout_rmse, control.initial_heldout_rmse)
            self.assertEqual(control.final_dev_rmse, control.initial_dev_rmse)
            self.assertEqual(control.candidate_evaluations, 0)
            self.assertEqual(control.history, ())
            self.assertEqual(control.accepted_candidates, 0)

    def test_random_edit_and_memory_reset_control_semantics(self):
        experiment = run_toy_experiment(default_manifest(seed_count=2, candidate_budget=8))
        self.assertEqual(experiment.manifest.controls, ("no_change", "random_edit", "memory_reset"))
        for run in experiment.runs:
            controls = {arm.optimizer: arm for arm in run.controls}
            random_edit = controls["random_edit"]
            self.assertEqual(random_edit.start_artifact_id, run.start_artifact_id)
            self.assertEqual(random_edit.initial_heldout_rmse, run.baseline.initial_heldout_rmse)
            self.assertEqual(random_edit.candidate_evaluations, 8)
            self.assertEqual(random_edit.accepted_candidates, 8)
            self.assertEqual(len(random_edit.history), 8)
            self.assertTrue(all(entry.accepted for entry in random_edit.history))
            self.assertEqual(random_edit.successor.artifact_id, random_edit.history[-1].candidate_artifact_id)

            memory_reset = controls["memory_reset"]
            self.assertEqual(memory_reset.start_artifact_id, run.start_artifact_id)
            self.assertEqual(memory_reset.initial_heldout_rmse, run.baseline.initial_heldout_rmse)
            self.assertEqual(memory_reset.candidate_evaluations, 8)
            self.assertEqual(len(memory_reset.history), 8)
            self.assertTrue(all(entry.candidate.parent == run.start_artifact_id for entry in memory_reset.history))
            best_loss = memory_reset.initial_dev_rmse
            for entry in memory_reset.history:
                self.assertEqual(entry.accepted, entry.dev_rmse < best_loss)
                if entry.accepted:
                    best_loss = entry.dev_rmse
            self.assertEqual(memory_reset.final_dev_rmse, best_loss)

    def test_python_adapter_integrates_with_runner_and_preserves_lineage(self):
        def propose(context):
            values = list(context.current.values)
            values[context.iteration % len(values)] += 0.25
            return context.current.with_values(tuple(values))

        challenger = PythonOptimizerAdapter("python_increment", propose)
        manifest = default_manifest(seed_count=1, candidate_budget=4)
        object.__setattr__(manifest, "challenger", challenger.name)
        object.__setattr__(manifest, "controls", ("no_change",))
        runner = PairRunner(
            RandomSearchOptimizer(), challenger, make_start, make_tasks, controls=(NoChangeOptimizer(),)
        )
        run = runner.run(manifest).runs[0]
        self.assertEqual(run.challenger.candidate_evaluations, 4)
        self.assertEqual(len(run.challenger.history), 4)
        current_id = run.start_artifact_id
        for entry in run.challenger.history:
            self.assertEqual(entry.candidate.parent, current_id)
            self.assertEqual(entry.candidate_artifact_id, entry.candidate.artifact_id)
            if entry.accepted:
                current_id = entry.candidate.artifact_id
        self.assertEqual(run.challenger.successor.artifact_id, current_id)

    def test_python_adapter_rejects_non_artifact_proposals(self):
        adapter = PythonOptimizerAdapter("invalid", lambda context: context.current.values)

        def evaluate(candidate):
            raise AssertionError(f"Invalid candidate must not be evaluated: {candidate}")

        with self.assertRaises(TypeError):
            adapter.optimize(AgentArtifact((0.0, 1.0)), 1.0, Budget(1), 7, evaluate)

    def test_paired_statistics(self):
        values = [1.0, 2.0, 3.0, 4.0]
        lower, upper = bootstrap_mean_ci(values, resamples=200)
        self.assertLessEqual(lower, 2.5)
        self.assertGreaterEqual(upper, 2.5)
        self.assertLess(sign_flip_pvalue(values), 1.0)
        self.assertEqual(sign_flip_pvalue([0.0, 0.0]), 1.0)

    def test_manifest_and_evidence_files(self):
        manifest = default_manifest(seed_count=2, candidate_budget=8)
        experiment = run_toy_experiment(manifest)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest_path = write_manifest(manifest, output)
            evidence_path, html_path = write_evidence(experiment, output)
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["manifest_digest"], manifest.digest)
            self.assertEqual(manifest_data["controls"], ["no_change", "random_edit", "memory_reset"])
            self.assertEqual(evidence["manifest_digest"], manifest.digest)
            self.assertEqual(len(evidence["experiment"]["runs"]), 2)
            first_history = evidence["experiment"]["runs"][0]["baseline"]["history"][0]
            self.assertIn("values", first_history["candidate"])
            self.assertIn("parent", first_history["candidate"])
            for run in evidence["experiment"]["runs"]:
                controls = {arm["optimizer"]: arm for arm in run["controls"]}
                self.assertEqual(set(controls), {"no_change", "random_edit", "memory_reset"})
                self.assertEqual(controls["no_change"]["candidate_evaluations"], 0)
                self.assertEqual(len(controls["random_edit"]["history"]), 8)
                self.assertEqual(len(controls["memory_reset"]["history"]), 8)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Merit Evidence Card", html)
            for name in ("no_change", "random_edit", "memory_reset"):
                self.assertIn(f"{name} RMSE", html)
            self.assertIn("不是真实记忆模块", html)


if __name__ == "__main__":
    unittest.main()
