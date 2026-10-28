import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from merit import AgentSpec, AgentTask
from merit.openai_executor import OpenAICompatibleExecutor, TokenPricing


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(
            {"path": self.path, "authorization": self.headers.get("Authorization"), "body": body}
        )
        status, payload = self.server.responses.pop(0)
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


class OpenAIExecutorTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        self.server.requests = []
        self.server.responses = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.environment = patch.dict(os.environ, {"MERIT_TEST_API_KEY": "test-secret"}, clear=False)
        self.environment.start()
        self.agent = AgentSpec("Normalize text", {"temperature": 0})
        self.task = AgentTask("task-1", "HELLO", "hello")

    def tearDown(self):
        self.environment.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def executor(self, **kwargs):
        return OpenAICompatibleExecutor(
            self.base_url,
            "mock-model",
            api_key_env="MERIT_TEST_API_KEY",
            timeout_seconds=2,
            retry_backoff_seconds=0,
            **kwargs,
        )

    def test_request_response_usage_and_estimated_cost(self):
        self.server.responses = [
            (200, {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
        ]
        executor = self.executor(max_retries=0, pricing=TokenPricing(0.1, 0.2))
        response = executor.invoke(self.agent, self.task)
        self.assertEqual(response.output, "hello")
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertEqual(response.usage.output_tokens, 3)
        self.assertAlmostEqual(response.usage.cost_usd, 0.0000018)
        self.assertTrue(response.usage.tokens_reported)
        self.assertTrue(response.usage.cost_reported)
        self.assertTrue(response.usage.cost_is_estimate)
        self.assertGreater(response.usage.latency_ms, 0)
        request = self.server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["authorization"], "Bearer test-secret")
        self.assertEqual(request["body"]["model"], "mock-model")
        self.assertEqual(request["body"]["temperature"], 0)
        self.assertEqual(request["body"]["messages"][0]["content"], "Normalize text")
        self.assertEqual(request["body"]["messages"][1]["content"], "HELLO")

    def test_missing_usage_is_explicitly_incomplete_not_free(self):
        self.server.responses = [(200, {"choices": [{"message": {"content": "hello"}}]})]
        response = self.executor(max_retries=0).invoke(self.agent, self.task)
        self.assertFalse(response.usage.tokens_reported)
        self.assertFalse(response.usage.cost_reported)
        self.assertEqual(response.usage.cost_usd, 0.0)
        aggregate = response.usage + response.usage
        self.assertFalse(aggregate.tokens_reported)
        self.assertFalse(aggregate.cost_reported)

    def test_retryable_failure_is_retried(self):
        self.server.responses = [
            (503, {"error": "temporary"}),
            (200, {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}),
        ]
        response = self.executor(max_retries=1).invoke(self.agent, self.task)
        self.assertEqual(response.output, "hello")
        self.assertEqual(len(self.server.requests), 2)

    def test_non_retryable_failure_does_not_retry(self):
        self.server.responses = [(401, {"error": "unauthorized"})]
        with self.assertRaisesRegex(RuntimeError, "status 401"):
            self.executor(max_retries=3).invoke(self.agent, self.task)
        self.assertEqual(len(self.server.requests), 1)

    def test_missing_key_fails_before_network_and_model_is_fixed(self):
        with patch.dict(os.environ, {"MERIT_TEST_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Missing API key"):
                self.executor(max_retries=0).invoke(self.agent, self.task)
        self.assertEqual(self.server.requests, [])
        other_model = AgentSpec("instructions", {"model": "other-model"})
        with self.assertRaisesRegex(ValueError, "model is fixed"):
            self.executor(max_retries=0).invoke(other_model, self.task)
        self.assertEqual(self.server.requests, [])


if __name__ == "__main__":
    unittest.main()
