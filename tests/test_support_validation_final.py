import importlib.util
import sys
import unittest
from pathlib import Path

from merit.multi_step_executor import safe_calculator


EXAMPLES = Path(__file__).parents[1] / "examples"
for name in (
    "ollama_support_error_reflection",
    "ollama_support_error_trajectory_reflection",
    "ollama_support_error_structured_patch",
    "ollama_support_error_recovery_patch",
    "ollama_support_error_stop_patch",
    "ollama_support_error_validation_final",
):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

support = sys.modules["ollama_support_error_reflection"]
experiment = sys.modules["ollama_support_error_validation_final"]


class SupportValidationFinalTests(unittest.TestCase):
    def test_train_validation_and_final_tasks_are_disjoint(self):
        splits = (support.TRAIN, support.HELDOUT, experiment.FINAL)
        ids = [task.task_id for split in splits for task in split]
        prompts = [task.prompt for split in splits for task in split]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertTrue(all(task.task_id.startswith("final-") for task in experiment.FINAL))

    def test_final_records_cover_normal_and_missing_branches(self):
        values = {**support.LOOKUP_VALUES, **experiment.FINAL_LOOKUP_VALUES}
        order = values[values["ticket-7001"]]
        self.assertIn('"total":200', order)
        self.assertEqual(values["policy-F01"], "0.4")
        self.assertEqual(safe_calculator("200*0.4"), 80.0)
        self.assertNotIn("order-Z81", values)
        self.assertNotIn("policy-Z81", values)
        self.assertEqual(experiment.FINAL[0].expected, "80.0")


if __name__ == "__main__":
    unittest.main()
