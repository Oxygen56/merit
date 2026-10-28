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
):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

support = sys.modules["ollama_support_error_reflection"]
trajectory = sys.modules["ollama_support_error_trajectory_reflection"]
patch = sys.modules["ollama_support_error_structured_patch"]
recovery = sys.modules["ollama_support_error_recovery_patch"]


class SupportRecoveryPatchTests(unittest.TestCase):
    def test_feedback_flags_premature_calculation_and_repeated_failed_lookup(self):
        traces = [
            {"task_id": "train-normal-5001", "step": 0, "action": "lookup", "argument": "ticket-5001", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 1, "action": "calculator", "argument": "120*policy", "tool_error": "Only numeric"},
            {"task_id": "train-normal-5001", "step": 2, "action": "lookup", "argument": "policy-N01", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 3, "action": "calculator", "argument": "120*0.5", "tool_error": None},
            {"task_id": "train-normal-5001", "step": 4, "action": "final", "argument": "60.0", "tool_error": None},
            {"task_id": "train-missing-order-5002", "step": 0, "action": "lookup", "argument": "ticket-5002", "tool_error": None},
            {"task_id": "train-missing-order-5002", "step": 1, "action": "lookup", "argument": "order-X71", "tool_error": "Unknown lookup key"},
            {"task_id": "train-missing-order-5002", "step": 2, "action": "lookup", "argument": "order-X71", "tool_error": "Unknown lookup key"},
            {"task_id": "train-missing-order-5002", "step": 3, "action": "final", "argument": "REFUND_UNAVAILABLE", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 0, "action": "lookup", "argument": "ticket-5003", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 1, "action": "lookup", "argument": "order-5003", "tool_error": None},
            {"task_id": "train-missing-policy-5003", "step": 2, "action": "lookup", "argument": "policy-X71", "tool_error": "Unknown lookup key"},
            {"task_id": "train-missing-policy-5003", "step": 3, "action": "final", "argument": "REFUND_UNAVAILABLE", "tool_error": None},
        ]
        results = [
            {"task_id": "train-normal-5001", "correct": False},
            {"task_id": "train-missing-order-5002", "correct": True},
            {"task_id": "train-missing-policy-5003", "correct": True},
        ]
        feedback = trajectory.build_structured_feedback(support.TRAIN, traces, results)
        issues = feedback["detected_issues"]
        self.assertEqual(issues["tasks_calculating_before_policy_lookup"], ["train-normal-5001"])
        self.assertEqual(issues["tasks_repeating_failed_lookups"], ["train-missing-order-5002"])
        rows = {row["task_id"]: row for row in feedback["task_traces"]}
        self.assertEqual(rows["train-missing-order-5002"]["repeated_failed_lookup_count"], 1)
        self.assertFalse(rows["train-missing-policy-5003"]["calculation_before_successful_policy_lookup"])
        serialized = json.dumps(feedback)
        for key in support.LOOKUP_VALUES:
            self.assertNotIn(key, serialized)
        self.assertNotIn("heldout-", serialized)

    def test_recovery_rules_extend_only_the_new_renderer(self):
        rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in recovery.RECOVERY_RULE_TEXT
        ]
        payload = json.dumps({"rules": rules})
        current = AgentSpec(support.BASE_INSTRUCTIONS)
        candidate = recovery.render_patch(current, payload, recovery.RECOVERY_RULE_TEXT)
        self.assertEqual(len(candidate.instructions.splitlines()), 15)
        self.assertIn("Do not calculate before the policy lookup has succeeded", candidate.instructions)
        self.assertIn("do not repeat the same failed lookup", candidate.instructions)
        recovery_only = json.dumps({"rules": [rules[-1]]})
        with self.assertRaisesRegex(ValueError, "Unsupported rule combination"):
            patch.parse_patch(recovery_only)

    def test_detected_issues_require_matching_recovery_rules(self):
        feedback = {
            "detected_issues": {
                "tasks_calculating_before_policy_lookup": ["train-normal-5001"],
                "tasks_repeating_failed_lookups": ["train-missing-policy-5003"],
            }
        }
        base_rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in patch.RULE_TEXT
        ]
        with self.assertRaisesRegex(ValueError, "before_policy_found, lookup_failed"):
            recovery.validate_recovery_coverage(base_rules, feedback)
        all_rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in recovery.RECOVERY_RULE_TEXT
        ]
        recovery.validate_recovery_coverage(all_rules, feedback)
        no_issues = {
            "detected_issues": {
                "tasks_calculating_before_policy_lookup": [],
                "tasks_repeating_failed_lookups": [],
            }
        }
        recovery.validate_recovery_coverage(base_rules, no_issues)


if __name__ == "__main__":
    unittest.main()
