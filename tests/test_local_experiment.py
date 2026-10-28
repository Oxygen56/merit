import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from merit.agent_contract import AgentResponse, AgentSpec, AgentTask, Usage
from merit.evidence import read_evidence, validate_result_directory
from merit.experiment import run_config
from merit.gepa_optimizer import structured_json_feedback
from merit.langgraph_executor import LangGraphToolExecutor
from merit.ollama_executor import OllamaExecutor


def provider_payload(content, latency=12.5):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }


class StructuredFeedbackTests(unittest.TestCase):
    def test_parameter_paths_and_types_are_explained_without_changing_score(self):
        expected = json.dumps({
            "name": "search",
            "arguments": {"filters": {"level": 2, "tags": ["a"]}, "limit": 3},
        })
        output = json.dumps({
            "name": "find",
            "arguments": {"filters": {"level": "2", "tags": ["b"], "extra": True}},
        })
        feedback = structured_json_feedback(expected, output, 0.0)
        self.assertIn("Exact-match score: 0", feedback)
        self.assertIn("name: tool mismatch", feedback)
        self.assertIn("arguments.limit: missing argument", feedback)
        self.assertIn("arguments.filters.extra: extra argument", feedback)
        self.assertIn("arguments.filters.level: type mismatch", feedback)
        self.assertIn("arguments.filters.tags: value mismatch", feedback)
        self.assertIn("unseen tools and questions", feedback)
        self.assertIn("Do not copy task questions", feedback)

    def test_invalid_json_has_actionable_feedback(self):
        feedback = structured_json_feedback('{"name":"search","arguments":{}}', "not-json", 0.0)
        self.assertIn("not valid JSON", feedback)
        self.assertIn("one JSON object only", feedback)


class OllamaTests(unittest.TestCase):
    def test_rejects_non_loopback_endpoint(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            OllamaExecutor("test", base_url="https://api.example.com/v1")

    def test_caches_successful_response_and_preserves_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = OllamaExecutor("test", cache_dir=Path(directory), max_network_calls=1)
            with patch.object(executor, "_request", return_value=provider_payload("hello")) as request:
                agent = AgentSpec("Answer", {"seed": 7})
                task = AgentTask("task", "Say hello", "hello")
                first = executor.invoke(agent, task)
                second = executor.invoke(agent, task)
            self.assertEqual(first.output, "hello")
            self.assertEqual(second.output, "hello")
            self.assertEqual(first.usage.latency_ms, second.usage.latency_ms)
            self.assertEqual(executor.network_calls, 1)
            self.assertEqual(executor.cache_hits, 1)
            request.assert_called_once()


class StubModel:
    name = "stub"

    def invoke(self, agent, task):
        call = {"arguments": {"x": 3}, "name": "double"}
        return AgentResponse(json.dumps(call), Usage(latency_ms=2))


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph extra is not installed")
class LangGraphTests(unittest.TestCase):
    def test_graph_validates_and_records_tool_call(self):
        executor = LangGraphToolExecutor(StubModel())
        prompt = json.dumps({
            "question": "Double 3",
            "tools": [{"name": "double", "parameters": {"properties": {"x": {}}, "required": ["x"]}}],
        })
        response = executor.invoke(AgentSpec("Use tools"), AgentTask("task", prompt, ""))
        self.assertEqual(json.loads(response.output), {"name": "double", "arguments": {"x": 3}})
        self.assertTrue(executor.tool_traces[0]["valid_tool_call"])


class ConfigRunTests(unittest.TestCase):
    def test_run_config_writes_valid_results_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tasks.json"
            dataset.write_text(json.dumps({
                "source": {"name": "offline-test", "metric_notice": "Test only"},
                "train": [{"id": "train", "prompt": "answer", "expected": "ok"}],
                "heldout": [{"id": "test", "prompt": "answer again", "expected": "ok"}],
            }), encoding="utf-8")
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "experiment": "offline-config-test",
                "model": {"name": "stub"},
                "dataset": "tasks.json",
                "seeds": [7],
                "agent": {"instructions": "Answer"},
            }), encoding="utf-8")

            def response(payload, api_key):
                instructions = payload["messages"][0]["content"]
                content = "Return only ok." if instructions.startswith("Improve") else "ok"
                return provider_payload(content)

            with patch.object(OllamaExecutor, "_request", side_effect=response):
                result = run_config(config, root / "results", max_network_calls=20)
            output = root / "results"
            self.assertEqual(result["summary"]["mean_challenger_heldout_accuracy"], 1.0)
            self.assertEqual(result["summary"]["execution"]["paid_api_calls"], 0)
            self.assertLess(result["summary"]["execution"]["network_calls_this_run"], 11)
            self.assertGreater(result["summary"]["execution"]["cached_response_files"], 0)
            self.assertTrue((output / "report.html").exists())
            self.assertTrue((output / "run-config.json").exists())
            validate_result_directory(output)
            evidence = read_evidence(output / "evidence.json")
            self.assertEqual(evidence["experiment"]["manifest"]["executor"], "ollama:stub")

    def test_fixed_prompt_and_harmful_controls_use_the_same_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tasks.json"
            dataset.write_text(json.dumps({
                "source": {"name": "offline-method-test", "metric_notice": "Test only"},
                "train": [{"id": "train", "prompt": "training question", "expected": "ok"}],
                "heldout": [{"id": "heldout", "prompt": "private heldout question", "expected": "ok"}],
            }), encoding="utf-8")
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "experiment": "offline-method-test",
                "model": {"name": "stub"},
                "dataset": "tasks.json",
                "seeds": [7],
                "candidate_budget": 1,
                "agent": {"instructions": "Answer"},
                "methods": {
                    "baseline": "no_change",
                    "challenger": "schema_guided_prompt",
                    "controls": ["empty_json"],
                },
            }), encoding="utf-8")

            def response(payload, api_key):
                instructions = payload["messages"][0]["content"]
                self.assertNotIn("private heldout question", instructions)
                return provider_payload("{}" if instructions.startswith("Return only {}") else "ok")

            with patch.object(OllamaExecutor, "_request", side_effect=response):
                result = run_config(config, root / "results", max_network_calls=20)
            summary = result["summary"]
            self.assertEqual(summary["mean_baseline_heldout_accuracy"], 1.0)
            self.assertEqual(summary["mean_challenger_heldout_accuracy"], 1.0)
            self.assertEqual(summary["controls"]["empty_json"]["mean_heldout_accuracy"], 0.0)
            validate_result_directory(root / "results")

    @unittest.skipUnless(importlib.util.find_spec("gepa"), "GEPA extra is not installed")
    def test_gepa_config_uses_train_tasks_and_candidate_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tasks.json"
            dataset.write_text(json.dumps({
                "source": {"name": "offline-gepa-test", "metric_notice": "Test only"},
                "train": [{"id": "train", "prompt": "training question", "expected": "ok"}],
                "heldout": [{"id": "heldout", "prompt": "private heldout question", "expected": "ok"}],
            }), encoding="utf-8")
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "experiment": "offline-gepa-test",
                "model": {"name": "stub"},
                "dataset": "tasks.json",
                "seeds": [7],
                "candidate_budget": 1,
                "agent": {"instructions": "Answer"},
                "methods": {"baseline": "no_change", "challenger": "gepa", "controls": []},
                "gepa": {"reflection_minibatch_size": 1, "feedback_format": "structured_json_diff"},
            }), encoding="utf-8")
            proposer_prompts = []

            def response(payload, api_key):
                instructions = payload["messages"][0]["content"]
                if instructions.startswith("Revise"):
                    proposer_prompts.append(payload["messages"][1]["content"])
                    return provider_payload("Return only ok.")
                return provider_payload("ok" if instructions == "Return only ok." else "bad")

            with patch.object(OllamaExecutor, "_request", side_effect=response):
                result = run_config(config, root / "results", max_network_calls=20)
            summary = result["summary"]
            self.assertEqual(summary["mean_baseline_heldout_accuracy"], 0.0)
            self.assertEqual(summary["mean_challenger_heldout_accuracy"], 1.0)
            self.assertEqual(summary["challenger_candidate_evaluations"], 1)
            self.assertTrue(proposer_prompts)
            self.assertNotIn("private heldout question", "\n".join(proposer_prompts))
            self.assertIn("Structured differences", "\n".join(proposer_prompts))
            self.assertIn("Do not copy task questions", "\n".join(proposer_prompts))
            evidence = read_evidence(root / "results" / "evidence.json")
            challenger = evidence["experiment"]["runs"][0]["challenger"]
            details = challenger["summary"]["optimizer_details"]
            self.assertEqual(details["library"], "gepa")
            self.assertEqual(details["selection_data"], "train_only")
            self.assertEqual(details["feedback_format"], "structured_json_diff")
            self.assertEqual(details["reflection_calls"], 1)
            self.assertEqual(challenger["summary"]["proposer_input_tokens"], 10)
            validate_result_directory(root / "results")

    @unittest.skipUnless(importlib.util.find_spec("gepa"), "GEPA extra is not installed")
    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph extra is not installed")
    def test_gepa_validation_selects_without_exposing_validation_or_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = json.dumps(
                {"name": "double", "arguments": {"x": 3}},
                sort_keys=True,
                separators=(",", ":"),
            )

            def prompt(question):
                return json.dumps({
                    "question": question,
                    "tools": [{
                        "name": "double",
                        "parameters": {"properties": {"x": {}}, "required": ["x"]},
                    }],
                })

            dataset = root / "tasks.json"
            dataset.write_text(json.dumps({
                "source": {"name": "offline-validation-test", "metric_notice": "Test only"},
                "train": [{"id": "train", "prompt": prompt("training question"), "expected": expected}],
                "validation": [{"id": "validation", "prompt": prompt("private validation question"), "expected": expected}],
                "heldout": [{"id": "heldout", "prompt": prompt("private heldout question"), "expected": expected}],
            }), encoding="utf-8")
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "experiment": "offline-validation-test",
                "executor": "langgraph_tool",
                "model": {"name": "stub"},
                "dataset": "tasks.json",
                "seeds": [7],
                "candidate_budget": 1,
                "agent": {"instructions": "Answer"},
                "methods": {"baseline": "no_change", "challenger": "gepa", "controls": []},
                "gepa": {"reflection_minibatch_size": 1, "selection_data": "validation"},
            }), encoding="utf-8")
            proposer_prompts = []

            def response(payload, api_key):
                instructions = payload["messages"][0]["content"]
                prompt_text = payload["messages"][1]["content"]
                if instructions.startswith("Revise"):
                    proposer_prompts.append(prompt_text)
                    return provider_payload("Return only ok.")
                question = json.loads(prompt_text)["question"]
                if instructions == "Answer" and question != "training question":
                    return provider_payload('```json\n{"name":"double","arguments":{"x":3}}\n```')
                if instructions == "Return only ok." and question == "training question":
                    return provider_payload('{"name":"double","arguments":{"x":3}}')
                return provider_payload("bad")

            with patch.object(OllamaExecutor, "_request", side_effect=response):
                result = run_config(config, root / "results", max_network_calls=20)
            summary = result["summary"]
            self.assertEqual(summary["mean_baseline_heldout_accuracy"], 1.0)
            self.assertEqual(summary["mean_challenger_heldout_accuracy"], 1.0)
            self.assertEqual(summary["challenger_candidate_evaluations"], 1)
            joined_prompts = "\n".join(proposer_prompts)
            self.assertIn("training question", joined_prompts)
            self.assertNotIn("private validation question", joined_prompts)
            self.assertNotIn("private heldout question", joined_prompts)
            evidence = read_evidence(root / "results" / "evidence.json")
            run = evidence["experiment"]["runs"][0]
            challenger = run["challenger"]
            details = challenger["summary"]["optimizer_details"]
            self.assertEqual(details["selection_data"], "validation_only")
            self.assertEqual(details["validation_task_count"], 1)
            self.assertEqual(challenger["summary"]["initial_validation_accuracy"], 1.0)
            self.assertEqual(challenger["summary"]["successor_validation_accuracy"], 1.0)
            self.assertEqual(challenger["summary"]["dev_agent_invocations"], 4)
            self.assertEqual(challenger["successor"]["instructions"], "Answer")
            selection_result = challenger["history"][0]["selection_evaluation"]["results"][0]
            self.assertEqual(selection_result["task_id"], "validation")
            self.assertEqual(challenger["initial_validation"]["results"][0]["output"], expected)
            self.assertIsNotNone(run["validation_task_digest"])
            validate_result_directory(root / "results")


if __name__ == "__main__":
    unittest.main()
