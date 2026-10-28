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
    "ollama_support_error_dynamic_patch",
):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

support = sys.modules["ollama_support_error_reflection"]
dynamic = sys.modules["ollama_support_error_dynamic_patch"]


class SupportDynamicPatchTests(unittest.TestCase):
    def test_renderer_prioritizes_failure_stop_before_workflow_rules(self):
        rules = [
            {"when": when, "action": action, "argument_source": source}
            for when, action, source in reversed(dynamic.STOP_RULE_TEXT)
        ]
        current = AgentSpec(support.BASE_INSTRUCTIONS)
        candidate = dynamic.render_priority_patch(current, json.dumps({"rules": rules}))
        rendered_rules = [
            line for line in candidate.instructions.splitlines() if line.startswith("- ")
        ]
        self.assertEqual(len(rendered_rules), 9)
        self.assertIn("return REFUND_UNAVAILABLE immediately", rendered_rules[0])
        self.assertIn("Do not calculate before the policy lookup", rendered_rules[1])
        self.assertIn("At task start", rendered_rules[3])
        self.assertEqual(candidate.parent, current.artifact_id)


if __name__ == "__main__":
    unittest.main()
