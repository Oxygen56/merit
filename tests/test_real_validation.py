import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from merit import AgentResponse, AgentSpec, AgentTask, Usage


SPEC = importlib.util.spec_from_file_location(
    "real_validation",
    Path(__file__).parents[1] / "examples" / "real_validation.py",
)
real_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_validation)


class FakeExecutor:
    name = "fake_external"
    model = "fake-model"

    def __init__(self, output="TOOL calculator 6*7"):
        self.output = output
        self.calls = 0

    def invoke(self, agent, task):
        self.calls += 1
        return AgentResponse(self.output, Usage(11, 7, latency_ms=2.5, cost_reported=False))


class RealValidationOfflineTests(unittest.TestCase):
    def test_safe_calculator_allows_arithmetic_and_rejects_code(self):
        self.assertEqual(real_validation.safe_calculator("(19-4)*3"), 45)
        for expression in (
            "__import__('os').system('true')",
            "open('file')",
            "2**1000",
            "[1][0]",
            "1/0",
        ):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                real_validation.safe_calculator(expression)

    def test_budget_is_checked_before_network_and_cache_does_not_hide_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            inner = FakeExecutor()
            executor = real_validation.BoundedExecutor(inner, 1, state)
            agent = AgentSpec("Use a tool")
            task = AgentTask("one", "6*7", "42")
            self.assertEqual(executor.invoke(agent, task).output, "TOOL calculator 6*7")
            with self.assertRaisesRegex(RuntimeError, "hard call cap"):
                executor.invoke(agent, AgentTask("two", "6*8", "48"))
            self.assertEqual(inner.calls, 1)
            resumed_inner = FakeExecutor()
            resumed = real_validation.BoundedExecutor(resumed_inner, 1, state)
            replay = resumed.invoke(agent, task)
            self.assertEqual(replay.output, "TOOL calculator 6*7")
            self.assertEqual(replay.usage.latency_ms, 2.5)
            self.assertEqual(resumed_inner.calls, 0)
            self.assertEqual(resumed.audit()["recorded_network_calls"], 1)
            self.assertEqual(resumed.audit()["cache_hits_this_process"], 1)
            calls = [json.loads(line) for line in (state / "provider-calls.jsonl").read_text().splitlines()]
            self.assertNotIn("api_key", json.dumps(calls))

    def test_tool_agent_records_raw_output_and_host_tool_result(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = real_validation.ToolAgentExecutor(
                real_validation.BoundedExecutor(FakeExecutor(), 1, Path(directory))
            )
            response = executor.invoke(AgentSpec("Use a tool"), AgentTask("math", "6*7", "42"))
            self.assertEqual(response.output, "42")
            self.assertEqual(executor.traces[0]["raw_model_output"], "TOOL calculator 6*7")
            self.assertEqual(executor.traces[0]["tool_name"], "calculator")
            self.assertEqual(executor.traces[0]["tool_result"], "42")
            self.assertEqual(response.usage.input_tokens, 11)

    def test_unknown_lookup_is_not_returned_as_a_value(self):
        with tempfile.TemporaryDirectory() as directory:
            inner = FakeExecutor("TOOL lookup unknown-key")
            executor = real_validation.ToolAgentExecutor(
                real_validation.BoundedExecutor(inner, 1, Path(directory))
            )
            response = executor.invoke(AgentSpec("Use a tool"), AgentTask("lookup", "unknown", ""))
            self.assertIsNone(executor.traces[0]["tool_result"])
            self.assertIn("Unknown lookup key", executor.traces[0]["tool_error"])
            self.assertEqual(response.output, "TOOL lookup unknown-key")


if __name__ == "__main__":
    unittest.main()
