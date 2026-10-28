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
):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

support = sys.modules["ollama_support_error_reflection"]
patch = sys.modules["ollama_support_error_structured_patch"]


class SupportStructuredPatchTests(unittest.TestCase):
    def test_allowed_rules_render_complete_prompt_without_executable_placeholders(self):
        rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in patch.RULE_TEXT
        ]
        current = AgentSpec(support.BASE_INSTRUCTIONS)
        candidate = patch.render_patch(current, json.dumps({"rules": rules}))
        self.assertEqual(candidate.parent, current.artifact_id)
        self.assertEqual(len(patch.parse_patch(candidate.instructions and json.dumps({"rules": rules}))), 6)
        for line in support.BASE_INSTRUCTIONS.splitlines():
            self.assertIn(line, candidate.instructions)
        for value in patch.RULE_TEXT.values():
            self.assertIn(value, candidate.instructions)
        self.assertNotIn("order_key", candidate.instructions)
        self.assertNotIn("<ticket_key>", candidate.instructions)
        for key in support.LOOKUP_VALUES:
            self.assertNotIn(key, candidate.instructions)

    def test_unknown_placeholder_extra_fields_and_duplicate_conditions_are_rejected(self):
        invalid_rules = [
            [{"when": "task_start", "action": "lookup", "argument_source": "<ticket_key>"}],
            [{"when": "task_start", "action": "lookup", "argument_source": "ticket_key_from_input", "extra": "x"}],
            [
                {"when": "task_start", "action": "lookup", "argument_source": "ticket_key_from_input"},
                {"when": "task_start", "action": "lookup", "argument_source": "ticket_key_from_input"},
            ],
        ]
        for rules in invalid_rules:
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                patch.parse_patch(json.dumps({"rules": rules}))

    def test_single_outer_json_fence_is_accepted_but_prose_is_not(self):
        rule = {"when": "task_start", "action": "lookup", "argument_source": "ticket_key_from_input"}
        payload = json.dumps({"rules": [rule]})
        self.assertEqual(patch.parse_patch(f"```json\n{payload}\n```"), [rule])
        with self.assertRaises(ValueError):
            patch.parse_patch("Here is the patch\n" + payload)


if __name__ == "__main__":
    unittest.main()
