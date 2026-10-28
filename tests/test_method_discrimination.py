import tempfile
import unittest
from pathlib import Path

from examples.method_discrimination import run_experiments


class MethodDiscriminationTests(unittest.TestCase):
    def test_effective_neutral_and_harmful_methods_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            summaries = run_experiments(Path(directory))
        effective = summaries["effective"]
        harmful = summaries["harmful"]
        self.assertGreater(effective["mean_heldout_accuracy_improvement"], 0)
        self.assertEqual(effective["mean_baseline_heldout_accuracy"], 0.25)
        self.assertEqual(effective["mean_challenger_heldout_accuracy"], 1.0)
        self.assertEqual(effective["controls"]["random_prompt_edit"]["mean_heldout_accuracy"], 0.25)
        self.assertEqual(effective["controls"]["failure_feedback"]["mean_heldout_accuracy"], 0.25)
        self.assertEqual(effective["controls"]["lowercase_only"]["mean_heldout_accuracy"], 0.5)
        self.assertLess(harmful["mean_heldout_accuracy_improvement"], 0)
        self.assertEqual(harmful["mean_baseline_heldout_accuracy"], 1.0)
        self.assertEqual(harmful["mean_challenger_heldout_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
