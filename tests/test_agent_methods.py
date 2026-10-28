import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from merit import AgentResponse, AgentSpec, AgentTask, Usage
from merit.agent_contract import AgentOptimizerAdapter, run_agent_contract
from merit.agent_demo import LocalTextNormalizer, make_local_agent_manifest, make_local_agent_start, make_local_agent_tasks
from merit.agent_methods import FailureFeedbackProposer, ModelReflectionProposer
from merit.agent_pair_demo import run_local_agent_paired_demo
from merit.agent_pair_report import write_agent_pair_evidence


class MockProposerExecutor:
    name = "mock_proposer"

    def __init__(self):
        self.requests = []

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        self.requests.append({"agent": agent, "task": task})
        return AgentResponse(
            output="Return normalized text and preserve unrelated behavior.",
            usage=Usage(100, 20, 0.01, 5.0, cost_is_estimate=True),
        )


class SequenceProposerExecutor(MockProposerExecutor):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        self.requests.append({"agent": agent, "task": task})
        return AgentResponse(
            output=self.outputs.pop(0),
            usage=Usage(100, 20, 0.01, 5.0, cost_is_estimate=True),
        )


class AgentMethodTests(unittest.TestCase):
    def test_failure_feedback_uses_only_train_failures(self):
        manifest = make_local_agent_manifest(candidate_budget=1, optimizer_name="failure_feedback")
        optimizer = AgentOptimizerAdapter("failure_feedback", FailureFeedbackProposer())
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        experiment = run_agent_contract(
            manifest, start, train, heldout, optimizer, LocalTextNormalizer()
        )
        instructions = experiment.history[0].candidate.instructions
        self.assertIn("train-lowercase", instructions)
        self.assertIn("train-strip", instructions)
        for task in heldout:
            self.assertNotIn(task.task_id, instructions)
            self.assertNotIn(task.expected, instructions)

    def test_model_reflection_cost_is_separate_and_hidden_tasks_are_not_shared(self):
        proposer_executor = MockProposerExecutor()
        proposer = ModelReflectionProposer(
            proposer_executor,
            AgentSpec("Improve the agent instructions using only supplied train failures."),
        )
        optimizer = AgentOptimizerAdapter(
            "model_reflection", proposer, proposer_source=proposer_executor.name
        )
        manifest = replace(
            make_local_agent_manifest(candidate_budget=2, optimizer_name="model_reflection"),
            proposer_source=proposer_executor.name,
        )
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        experiment = run_agent_contract(
            manifest, start, train, heldout, optimizer, LocalTextNormalizer()
        )
        summary = experiment.summary
        self.assertEqual(summary["total_cost_usd"], 0.0)
        self.assertEqual(summary["proposer_cost_usd"], 0.02)
        self.assertEqual(summary["combined_total_cost_usd"], 0.02)
        self.assertEqual(summary["proposer_input_tokens"], 200)
        self.assertEqual(summary["proposer_output_tokens"], 40)
        self.assertEqual(summary["proposer_latency_ms"], 10.0)
        self.assertEqual(summary["proposal_calls"], 2)
        self.assertEqual(summary["proposer_source"], "mock_proposer")
        self.assertTrue(summary["proposer_cost_reported"])
        self.assertTrue(summary["proposer_cost_is_estimate"])
        self.assertEqual(len(proposer_executor.requests), 2)
        for request in proposer_executor.requests:
            prompt = request["task"].prompt
            self.assertIn("train-lowercase", prompt)
            for task in heldout:
                self.assertNotIn(task.task_id, prompt)
                self.assertNotIn(task.prompt, prompt)
        self.assertTrue(all(entry.proposer_usage.cost_usd == 0.01 for entry in experiment.history))

    def test_invalid_reflection_retries_and_aggregates_proposer_usage(self):
        valid = "Return normalized text and preserve unrelated behavior."
        proposer_executor = SequenceProposerExecutor([
            '{"action":"lookup","argument":"key"}', valid,
        ])
        proposer = ModelReflectionProposer(
            proposer_executor,
            AgentSpec("Improve the agent instructions using only supplied train failures."),
            instruction_validator=lambda text: None if text == valid else "Not a complete prompt",
            max_attempts=2,
            feedback_provider=lambda context: {"literal_placeholder_arguments": ["order_key"]},
        )
        optimizer = AgentOptimizerAdapter(
            "model_reflection", proposer, proposer_source=proposer_executor.name
        )
        manifest = replace(
            make_local_agent_manifest(candidate_budget=1, optimizer_name="model_reflection"),
            proposer_source=proposer_executor.name,
        )
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        experiment = run_agent_contract(
            manifest, start, train, heldout, optimizer, LocalTextNormalizer()
        )
        self.assertEqual(experiment.history[0].candidate.instructions, valid)
        self.assertEqual(len(proposer_executor.requests), 2)
        retry_prompt = json.loads(proposer_executor.requests[1]["task"].prompt)
        self.assertEqual(retry_prompt["validation_error"], "Not a complete prompt")
        self.assertIn("previous_invalid_instructions", retry_prompt)
        for request in proposer_executor.requests:
            prompt = json.loads(request["task"].prompt)
            self.assertEqual(
                prompt["structured_train_feedback"]["literal_placeholder_arguments"],
                ["order_key"],
            )
            for task in heldout:
                self.assertNotIn(task.prompt, request["task"].prompt)
        usage = experiment.history[0].proposer_usage
        self.assertEqual(usage.input_tokens, 200)
        self.assertEqual(usage.output_tokens, 40)
        self.assertEqual(usage.cost_usd, 0.02)
        self.assertEqual(usage.latency_ms, 10.0)

    def test_reflection_stops_after_bounded_invalid_attempts(self):
        proposer_executor = SequenceProposerExecutor(["{}", "[]"])
        proposer = ModelReflectionProposer(
            proposer_executor,
            AgentSpec("Improve the agent instructions using only supplied train failures."),
            instruction_validator=lambda text: "Not a complete prompt",
            max_attempts=2,
        )
        optimizer = AgentOptimizerAdapter(
            "model_reflection", proposer, proposer_source=proposer_executor.name
        )
        manifest = replace(
            make_local_agent_manifest(candidate_budget=1, optimizer_name="model_reflection"),
            proposer_source=proposer_executor.name,
        )
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        with self.assertRaisesRegex(ValueError, "after 2 attempts") as caught:
            run_agent_contract(
                manifest, start, train, heldout, optimizer, LocalTextNormalizer()
            )
        self.assertEqual(len(proposer_executor.requests), 2)
        self.assertEqual(caught.exception.usage.input_tokens, 200)
        self.assertEqual(caught.exception.usage.output_tokens, 40)
        self.assertEqual(caught.exception.usage.cost_usd, 0.02)

    def test_structured_proposal_renderer_retries_invalid_json_patch(self):
        def render(current, text):
            payload = json.loads(text)
            if set(payload) != {"suffix"}:
                raise ValueError("Patch must contain only suffix")
            return current.evolve(instructions=current.instructions + "\n" + payload["suffix"])

        proposer_executor = SequenceProposerExecutor([
            '{"rules":[]}', '{"suffix":"Return normalized text."}',
        ])
        proposer = ModelReflectionProposer(
            proposer_executor,
            AgentSpec("Improve the agent using only supplied train failures."),
            max_attempts=2,
            proposal_renderer=render,
            output_requirement="Return only a JSON suffix patch without held-out tasks.",
        )
        optimizer = AgentOptimizerAdapter(
            "model_reflection", proposer, proposer_source=proposer_executor.name
        )
        manifest = replace(
            make_local_agent_manifest(candidate_budget=1, optimizer_name="model_reflection"),
            proposer_source=proposer_executor.name,
        )
        start = make_local_agent_start()
        train, heldout = make_local_agent_tasks()
        experiment = run_agent_contract(
            manifest, start, train, heldout, optimizer, LocalTextNormalizer()
        )
        self.assertTrue(experiment.history[0].candidate.instructions.endswith("Return normalized text."))
        self.assertEqual(len(proposer_executor.requests), 2)
        first_prompt = json.loads(proposer_executor.requests[0]["task"].prompt)
        retry_prompt = json.loads(proposer_executor.requests[1]["task"].prompt)
        self.assertEqual(first_prompt["requirement"], "Return only a JSON suffix patch without held-out tasks.")
        self.assertEqual(retry_prompt["validation_error"], "Patch must contain only suffix")
        self.assertEqual(experiment.history[0].proposer_usage.input_tokens, 200)
        for request in proposer_executor.requests:
            for task in heldout:
                self.assertNotIn(task.prompt, request["task"].prompt)

    def test_agent_ablation_controls_and_evidence(self):
        experiment = run_local_agent_paired_demo(seed_count=2, candidate_budget=4)
        controls = experiment.summary["controls"]
        self.assertEqual(set(controls), {"no_change", "random_prompt_edit", "failure_feedback"})
        self.assertEqual(controls["no_change"]["mean_candidate_evaluations"], 0.0)
        self.assertEqual(controls["random_prompt_edit"]["mean_candidate_evaluations"], 4.0)
        self.assertEqual(controls["failure_feedback"]["mean_candidate_evaluations"], 4.0)
        self.assertEqual(controls["no_change"]["mean_heldout_accuracy"], 0.25)
        self.assertEqual(controls["random_prompt_edit"]["mean_heldout_accuracy"], 0.25)
        self.assertEqual(controls["failure_feedback"]["mean_heldout_accuracy"], 0.25)
        self.assertEqual(experiment.summary["mean_heldout_accuracy_improvement"], 0.5)
        for run in experiment.runs:
            arms = {arm.manifest.optimizer: arm for arm in run.controls}
            self.assertEqual(arms["no_change"].successor.artifact_id, run.start_artifact_id)
            self.assertEqual(arms["no_change"].summary["candidate_evaluations"], 0)
            self.assertEqual(arms["random_prompt_edit"].summary["accepted_candidates"], 4)
            self.assertEqual(arms["failure_feedback"].summary["accepted_candidates"], 0)
        with tempfile.TemporaryDirectory() as directory:
            evidence_path, html_path = write_agent_pair_evidence(experiment, Path(directory))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertIn("proposer_usage", evidence["experiment"]["runs"][0]["baseline"]["history"][0])
            self.assertEqual(len(evidence["experiment"]["runs"][0]["controls"]), 3)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("消融对照", html)
            self.assertIn("random_prompt_edit", html)
            self.assertIn("proposer 成本", html)


if __name__ == "__main__":
    unittest.main()
