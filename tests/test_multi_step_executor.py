import json
import unittest

from merit.agent_contract import AgentResponse, AgentSpec, AgentTask, Usage
from merit.multi_step_executor import MultiStepToolExecutor, safe_calculator


class QueueModel:
    name = "queue-model"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def invoke(self, agent, task):
        self.prompts.append(task.prompt)
        return AgentResponse(next(self.outputs), Usage(10, 2, 0.0, 3.0))


class MultiStepExecutorTests(unittest.TestCase):
    def test_host_results_return_to_model_and_usage_is_aggregated(self):
        model = QueueModel([
            '{"action":"lookup","argument":"price-apple"}',
            '{"action":"lookup","argument":"quantity-apple"}',
            '{"action":"calculator","argument":"7*6"}',
            '{"action":"final","argument":"42"}',
        ])
        executor = MultiStepToolExecutor(model, {"price-apple": "7", "quantity-apple": "6"})
        response = executor.invoke(
            AgentSpec("Use tools"),
            AgentTask("task", "Multiply the stored values", "42"),
        )
        self.assertEqual(response.output, "42")
        self.assertEqual(response.usage.input_tokens, 40)
        self.assertEqual(response.usage.output_tokens, 8)
        self.assertEqual(response.usage.latency_ms, 12.0)
        self.assertEqual(len(executor.traces), 4)
        self.assertIn("price-apple => 7", model.prompts[-1])
        self.assertIn("calculator 7*6 => 42", model.prompts[-1])
        self.assertEqual(sum(trace["action"] == "lookup" for trace in executor.traces), 2)
        self.assertEqual(sum(trace["action"] == "calculator" for trace in executor.traces), 1)

    def test_invalid_action_stops_without_host_tool_execution(self):
        model = QueueModel(["not-json"])
        executor = MultiStepToolExecutor(model, {})
        response = executor.invoke(AgentSpec("Use tools"), AgentTask("task", "question", "answer"))
        self.assertEqual(response.output, "not-json")
        self.assertEqual(len(executor.traces), 1)
        self.assertIsNotNone(executor.traces[0]["tool_error"])
        self.assertIsNone(executor.traces[0]["tool_result"])

    def test_lookup_error_stops_by_default(self):
        model = QueueModel(['{"action":"lookup","argument":"missing-order"}'])
        executor = MultiStepToolExecutor(model, {})
        response = executor.invoke(AgentSpec("Use tools"), AgentTask("task", "question", "answer"))
        self.assertEqual(response.output, '{"action":"lookup","argument":"missing-order"}')
        self.assertEqual(len(executor.traces), 1)
        self.assertEqual(executor.traces[0]["tool_error"], "Unknown lookup key")

    def test_lookup_error_returns_to_model_when_recovery_is_enabled(self):
        model = QueueModel([
            '{"action":"lookup","argument":"missing-order"}',
            '{"action":"final","argument":"REFUND_UNAVAILABLE"}',
        ])
        executor = MultiStepToolExecutor(model, {}, recover_tool_errors=True)
        response = executor.invoke(AgentSpec("Use tools"), AgentTask("task", "question", "REFUND_UNAVAILABLE"))
        self.assertEqual(response.output, "REFUND_UNAVAILABLE")
        self.assertEqual(len(executor.traces), 2)
        self.assertIn("missing-order => error: Unknown lookup key", model.prompts[1])
        self.assertIsNone(executor.traces[0]["tool_result"])
        self.assertEqual(executor.traces[1]["tool_error"], None)

    def test_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            safe_calculator("__import__('os').system('true')")
        self.assertEqual(safe_calculator("12.5*4"), 50.0)


if __name__ == "__main__":
    unittest.main()
