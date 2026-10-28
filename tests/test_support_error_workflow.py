import importlib.util
import json
import sys
import unittest
from pathlib import Path

from merit.agent_contract import AgentResponse, AgentSpec, Usage
from merit.multi_step_executor import MultiStepToolExecutor


MODULE_PATH = Path(__file__).parents[1] / "examples" / "ollama_support_error_workflow.py"
SPEC = importlib.util.spec_from_file_location("support_error_workflow_example", MODULE_PATH)
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


class QueueModel:
    name = "queue-model"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def invoke(self, agent, task):
        self.prompts.append(task.prompt)
        return AgentResponse(next(self.outputs), Usage(10, 2, 0.0, 3.0))


class SupportErrorWorkflowTests(unittest.TestCase):
    def test_normal_and_missing_record_cases_are_consistent(self):
        for case in support.TRAIN_CASES + support.HELDOUT_CASES:
            self.assertEqual(support.LOOKUP_VALUES[case.ticket], case.order)
            if case.missing == "order":
                self.assertNotIn(case.order, support.LOOKUP_VALUES)
                self.assertEqual(support.expected_output(case), "REFUND_UNAVAILABLE")
            else:
                order = json.loads(support.LOOKUP_VALUES[case.order])
                self.assertEqual(order["policy"], case.policy)
                if case.missing == "policy":
                    self.assertNotIn(case.policy, support.LOOKUP_VALUES)
                    self.assertEqual(support.expected_output(case), "REFUND_UNAVAILABLE")
                else:
                    self.assertEqual(case.missing, "none")
                    self.assertEqual(float(support.LOOKUP_VALUES[case.policy]), case.rate)
                    self.assertEqual(support.expected_output(case), str(case.total * case.rate))

    def test_normal_and_error_paths_follow_the_required_steps(self):
        cases = (
            (support.TRAIN_CASES[0], [
                '{"action":"lookup","argument":"ticket-5001"}',
                '{"action":"lookup","argument":"order-5001"}',
                '{"action":"lookup","argument":"policy-N01"}',
                '{"action":"calculator","argument":"120 * 0.5"}',
                '{"action":"final","argument":"60.0"}',
            ], "60.0", 0, 1),
            (support.TRAIN_CASES[1], [
                '{"action":"lookup","argument":"ticket-5002"}',
                '{"action":"lookup","argument":"order-X71"}',
                '{"action":"final","argument":"REFUND_UNAVAILABLE"}',
            ], "REFUND_UNAVAILABLE", 1, 0),
            (support.TRAIN_CASES[2], [
                '{"action":"lookup","argument":"ticket-5003"}',
                '{"action":"lookup","argument":"order-5003"}',
                '{"action":"lookup","argument":"policy-X71"}',
                '{"action":"final","argument":"REFUND_UNAVAILABLE"}',
            ], "REFUND_UNAVAILABLE", 1, 0),
        )
        for case, outputs, expected, error_count, calculator_count in cases:
            with self.subTest(case=case.task_id):
                model = QueueModel(outputs)
                executor = MultiStepToolExecutor(
                    model, support.LOOKUP_VALUES, max_steps=6, recover_tool_errors=True
                )
                response = executor.invoke(
                    AgentSpec(support.ERROR_WORKFLOW_INSTRUCTIONS), support.make_task(case)
                )
                self.assertEqual(response.output, expected)
                self.assertEqual(
                    sum(trace["tool_error"] is not None for trace in executor.traces), error_count
                )
                self.assertEqual(
                    sum(trace["action"] == "calculator" for trace in executor.traces),
                    calculator_count,
                )
                if error_count:
                    self.assertIn("error: Unknown lookup key", model.prompts[-1])


if __name__ == "__main__":
    unittest.main()
