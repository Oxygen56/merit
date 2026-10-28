import importlib.util
import json
import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).parents[1] / "examples"
SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "ollama_support_error_reflection", EXAMPLES / "ollama_support_error_reflection.py"
)
support = importlib.util.module_from_spec(SUPPORT_SPEC)
sys.modules[SUPPORT_SPEC.name] = support
SUPPORT_SPEC.loader.exec_module(support)
SPEC = importlib.util.spec_from_file_location(
    "support_trajectory_reflection_example",
    EXAMPLES / "ollama_support_error_trajectory_reflection.py",
)
trajectory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trajectory
SPEC.loader.exec_module(trajectory)


class SupportTrajectoryFeedbackTests(unittest.TestCase):
    def test_feedback_flags_placeholder_and_ticket_skip_without_record_keys(self):
        traces = []
        results = []
        for index, task in enumerate(support.TRAIN):
            traces.extend([
                {
                    "task_id": task.task_id,
                    "step": 0,
                    "action": "lookup",
                    "argument": "order_key",
                    "tool_error": "Unknown lookup key",
                },
                {
                    "task_id": task.task_id,
                    "step": 1,
                    "action": "final",
                    "argument": "REFUND_UNAVAILABLE",
                    "tool_error": None,
                },
            ])
            results.append({"task_id": task.task_id, "correct": index > 0})
        feedback = trajectory.build_structured_feedback(support.TRAIN, traces, results)
        self.assertEqual(
            feedback["detected_issues"]["literal_placeholder_arguments"], ["order_key"]
        )
        self.assertEqual(
            len(feedback["detected_issues"]["tasks_skipping_input_ticket_lookup"]), 3
        )
        serialized = json.dumps(feedback)
        for key in support.LOOKUP_VALUES:
            self.assertNotIn(key, serialized)
        self.assertNotIn("heldout-", serialized)

    def test_feedback_rejects_heldout_tasks(self):
        task = support.HELDOUT[0]
        traces = [{
            "task_id": task.task_id,
            "step": 0,
            "action": "final",
            "argument": "REFUND_UNAVAILABLE",
            "tool_error": None,
        }]
        results = [{"task_id": task.task_id, "correct": True}]
        with self.assertRaisesRegex(ValueError, "must not include held-out"):
            trajectory.build_structured_feedback((task,), traces, results)


if __name__ == "__main__":
    unittest.main()
