import importlib.util
import json
import sys
import unittest
from pathlib import Path

from merit.agent_contract import AgentResponse, AgentSpec, Usage
from merit.multi_step_executor import MultiStepToolExecutor, safe_calculator


MODULE_PATH = Path(__file__).parents[1] / "examples" / "ollama_support_workflow_discrimination.py"
SPEC = importlib.util.spec_from_file_location("support_workflow_example", MODULE_PATH)
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


class SupportWorkflowTests(unittest.TestCase):
    def test_case_records_and_expected_refunds_are_consistent(self):
        train_ids = {case.task_id for case in support.TRAIN_CASES}
        heldout_ids = {case.task_id for case in support.HELDOUT_CASES}
        self.assertFalse(train_ids & heldout_ids)
        for case in support.TRAIN_CASES + support.HELDOUT_CASES:
            self.assertEqual(support.LOOKUP_VALUES[case.ticket], case.order)
            order = json.loads(support.LOOKUP_VALUES[case.order])
            self.assertEqual(order, {"total": case.total, "policy": case.policy})
            self.assertEqual(float(support.LOOKUP_VALUES[case.policy]), case.rate)
            self.assertEqual(
                support.expected_refund(case),
                str(safe_calculator(f"{case.total} * {case.rate}")),
            )

    def test_dependent_ticket_order_policy_flow(self):
        case = support.TRAIN_CASES[0]
        model = QueueModel([
            '{"action":"lookup","argument":"ticket-1042"}',
            '{"action":"lookup","argument":"order-7821"}',
            '{"action":"lookup","argument":"policy-D17"}',
            '{"action":"calculator","argument":"240 * 0.5"}',
            '{"action":"final","argument":"120.0"}',
        ])
        executor = MultiStepToolExecutor(model, support.LOOKUP_VALUES, max_steps=6)
        response = executor.invoke(AgentSpec(support.WORKFLOW_INSTRUCTIONS), support.TRAIN[0])
        self.assertEqual(response.output, "120.0")
        self.assertEqual(response.usage.input_tokens, 50)
        self.assertEqual(response.usage.output_tokens, 10)
        self.assertEqual(len(executor.traces), 5)
        self.assertIn("ticket-1042 => order-7821", model.prompts[1])
        self.assertIn('"total":240', model.prompts[2])
        self.assertIn("policy-D17 => 0.5", model.prompts[3])
        self.assertIn("calculator 240 * 0.5 => 120.0", model.prompts[4])


if __name__ == "__main__":
    unittest.main()
