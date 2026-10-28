import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "examples" / "ollama_support_error_reflection.py"
SPEC = importlib.util.spec_from_file_location("support_error_reflection_example", MODULE_PATH)
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


class SupportReflectionTests(unittest.TestCase):
    def test_start_has_tool_protocol_but_no_task_records(self):
        for action in ("lookup", "calculator", "final"):
            self.assertIn(action, support.BASE_INSTRUCTIONS)
        for task in support.TRAIN + support.HELDOUT:
            ticket = task.prompt.split()[3]
            self.assertNotIn(ticket, support.BASE_INSTRUCTIONS)
            self.assertNotIn(task.expected, support.BASE_INSTRUCTIONS)
        self.assertNotIn("policy", support.BASE_INSTRUCTIONS)

    def test_complete_prompt_validator_rejects_action_only_output(self):
        action_only = '{"action":"lookup","argument":"<order_key>"}'
        self.assertIsNotNone(support.validate_complete_instructions(action_only))
        self.assertIsNone(support.validate_complete_instructions(support.BASE_INSTRUCTIONS))
        without_trailing_periods = "\n".join(
            line.rstrip(".") for line in support.BASE_INSTRUCTIONS.splitlines()
        )
        self.assertIsNone(support.validate_complete_instructions(without_trailing_periods))
        self.assertIsNotNone(support.validate_complete_instructions(
            support.BASE_INSTRUCTIONS + "\n```json"
        ))

    def test_proposer_is_restricted_to_general_train_feedback(self):
        self.assertIn("training failures", support.PROPOSER_INSTRUCTIONS)
        self.assertIn("Do not include ticket, order, or policy keys", support.PROPOSER_INSTRUCTIONS)
        self.assertIn("Do not ask for held-out tasks", support.PROPOSER_INSTRUCTIONS)
        train_ids = {task.task_id for task in support.TRAIN}
        heldout_ids = {task.task_id for task in support.HELDOUT}
        self.assertFalse(train_ids & heldout_ids)


if __name__ == "__main__":
    unittest.main()
