import importlib.util
import json
import sys
import unittest
from pathlib import Path

from merit import AgentSpec


EXAMPLES = Path(__file__).parents[1] / "examples"
for name in (
    "ollama_support_error_reflection",
    "ollama_support_error_trajectory_reflection",
    "ollama_support_error_structured_patch",
    "ollama_support_error_recovery_patch",
    "ollama_support_error_stop_patch",
):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

support = sys.modules["ollama_support_error_reflection"]
trajectory = sys.modules["ollama_support_error_trajectory_reflection"]
stop = sys.modules["ollama_support_error_stop_patch"]


def empty_feedback():
    return {
        "detected_issues": {
            key: [] for key in stop.SELECTION_ISSUE_KEYS
        }
    }


class SupportStopPatchTests(unittest.TestCase):
    def test_feedback_flags_actions_after_required_record_failure(self):
        traces = [
            {"task_id": "train-normal-5001", "step": 0, "action": "lookup", "argument": "ticket-5001", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 1, "action": "lookup", "argument": "order-5001", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 2, "action": "lookup", "argument": "policy-N01", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 3, "action": "calculator", "argument": "120*0.5", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 4, "action": "final", "argument": "60.0", "tool_error": None},
            {"task_id": "train-missing-order-5002", "step": 0, "action": "lookup", "argument": "ticket-5002", "tool_error": None},
            {"task_id": "train-missing-order-5002", "step": 1, "action": "lookup", "argument": "order-X71", "tool_error": "Unknown lookup key"},
            {"task_id": "train-missing-order-5002", "step": 2, "action": "lookup", "argument": "policy-X71", "tool_error": "Unknown lookup key"},
            {"task_id": "train-missing-order-5002", "step": 3, "action": "final", "argument": "REFUND_UNAVAILABLE", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 0, "action": "lookup", "argument": "ticket-5003", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 1, "action": "lookup", "argument": "order-5003", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 2, "action": "lookup", "argument": "policy-X71", "tool_error": "Unknown lookup key"},
            {"task_id": "train-missing-policy-5003", "step": 3, "action": "calculator", "argument": "90*0", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 4, "action": "final", "argument": "REFUND_UNAVAILABLE", "tool_error": None},
        ]
        results = [{"task_id": task.task_id, "correct": True} for task in support.TRAIN]
        feedback = trajectory.build_structured_feedback(support.TRAIN, traces, results)
        self.assertEqual(
            feedback["detected_issues"]["tasks_continuing_after_required_record_failure"],
            ["train-missing-order-5002", "train-missing-policy-5003"],
        )
        rows = {row["task_id"]: row for row in feedback["task_traces"]}
        self.assertEqual(rows["train-missing-order-5002"]["actions_after_required_record_failure"], 1)
        self.assertEqual(rows["train-missing-policy-5003"]["actions_after_required_record_failure"], 1)
        self.assertFalse(stop.accept_candidate(1.0, 1.0, feedback, feedback))
        serialized = json.dumps(feedback)
        for key in support.LOOKUP_VALUES:
            self.assertNotIn(key, serialized)

    def test_equal_accuracy_can_replace_invalid_trajectories_only_with_valid_ones(self):
        invalid = empty_feedback()
        invalid["detected_issues"]["tasks_continuing_after_required_record_failure"] = [
            "train-missing-policy-5003"
        ]
        valid = empty_feedback()
        self.assertTrue(stop.accept_candidate(1.0, 1.0, invalid, valid))
        self.assertFalse(stop.accept_candidate(1.0, 1.0, valid, valid))
        self.assertFalse(stop.accept_candidate(1.0, 0.5, invalid, valid))
        self.assertTrue(stop.accept_candidate(0.5, 1.0, valid, valid))

    def test_stop_rule_is_required_and_rendered_without_changing_base_rules(self):
        feedback = empty_feedback()
        feedback["detected_issues"]["tasks_continuing_after_required_record_failure"] = [
            "train-missing-order-5002"
        ]
        recovery_rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in stop.RECOVERY_RULE_TEXT
        ]
        with self.assertRaisesRegex(ValueError, "required_record_failure"):
            stop.validate_stop_coverage(recovery_rules, feedback)
        all_rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in stop.STOP_RULE_TEXT
        ]
        stop.validate_stop_coverage(all_rules, feedback)
        current = AgentSpec(support.BASE_INSTRUCTIONS)
        candidate = stop.render_patch(current, json.dumps({"rules": all_rules}), stop.STOP_RULE_TEXT)
        self.assertIn("return REFUND_UNAVAILABLE immediately", candidate.instructions)
        self.assertIn("do not perform any further lookups or calculations", candidate.instructions)


if __name__ == "__main__":
    unittest.main()
