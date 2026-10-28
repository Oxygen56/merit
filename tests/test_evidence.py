import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from merit.agent_demo import run_local_agent_demo
from merit.agent_pair_demo import run_local_agent_paired_demo
from merit.agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from merit.agent_report import write_agent_evidence, write_agent_manifest
from merit.demo import default_manifest, run_toy_experiment
from merit.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    compress_evidence,
    evidence_schema,
    read_evidence,
    validate_evidence_envelope,
    validate_result_directory,
)
from merit.meta_plasticity_demo import run_local_meta_plasticity_demo
from merit.meta_plasticity_report import write_meta_plasticity_evidence, write_meta_plasticity_manifest
from merit.report import write_evidence, write_manifest
from merit.transfer_demo import run_local_transfer_demo
from merit.transfer_report import write_transfer_evidence, write_transfer_manifest


class EvidenceReleaseTests(unittest.TestCase):
    def test_every_demo_writes_the_stable_versioned_envelope(self):
        cases = (
            (run_toy_experiment(default_manifest(seed_count=1, candidate_budget=1)), write_manifest, write_evidence),
            (run_local_agent_demo(candidate_budget=1), write_agent_manifest, write_agent_evidence),
            (run_local_agent_paired_demo(seed_count=1, candidate_budget=1), write_agent_pair_manifest, write_agent_pair_evidence),
            (run_local_transfer_demo(candidate_budget=1), write_transfer_manifest, write_transfer_evidence),
            (run_local_meta_plasticity_demo(1, 1, 1), write_meta_plasticity_manifest, write_meta_plasticity_evidence),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (experiment, manifest_writer, evidence_writer) in enumerate(cases):
                output = root / str(index)
                manifest_writer(experiment.manifest, output)
                evidence_path, _ = evidence_writer(experiment, output)
                evidence = read_evidence(evidence_path)
                self.assertEqual(evidence["schema_version"], EVIDENCE_SCHEMA_VERSION)
                metadata = validate_result_directory(output)
                self.assertEqual(metadata["manifest_digest"], experiment.manifest.digest)
                self.assertEqual(metadata["protocol_version"], experiment.manifest.protocol_version)

    def test_packaged_schema_matches_the_envelope_contract(self):
        schema = evidence_schema()
        self.assertEqual(schema["properties"]["schema_version"]["const"], EVIDENCE_SCHEMA_VERSION)
        self.assertIn("manifest_digest", schema["required"])
        self.assertIn("experiment", schema["required"])

    def test_compression_is_deterministic_and_round_trips(self):
        experiment = run_local_agent_demo(candidate_budget=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence_path, _ = write_agent_evidence(experiment, output)
            first = compress_evidence(evidence_path, output / "first.json.gz")
            second = compress_evidence(evidence_path, output / "second.json.gz")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(read_evidence(first), read_evidence(evidence_path))
            self.assertLess(first.stat().st_size, evidence_path.stat().st_size)

    def test_validation_rejects_tampering_and_unsupported_schema(self):
        experiment = run_local_agent_demo(candidate_budget=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_agent_manifest(experiment.manifest, output)
            evidence_path, _ = write_agent_evidence(experiment, output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["manifest_digest"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_result_directory(output)
            evidence = read_evidence(evidence_path)
            unsupported = copy.deepcopy(evidence)
            unsupported["schema_version"] = "future-unknown"
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                validate_evidence_envelope(unsupported)
            missing = copy.deepcopy(evidence)
            del missing["limitations"]
            with self.assertRaisesRegex(ValueError, "limitations"):
                validate_evidence_envelope(missing)

    def test_external_report_does_not_reuse_local_source_claims(self):
        experiment = run_local_agent_demo(candidate_budget=1)
        experiment = replace(
            experiment,
            manifest=replace(experiment.manifest, executor="external_mock_v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence_path, html_path = write_agent_evidence(
                experiment,
                Path(directory),
                limitations=["External mock source; this test does not make a network request."],
            )
            evidence = read_evidence(evidence_path)
            self.assertEqual(evidence["limitations"], ["External mock source; this test does not make a network request."])
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("external_mock_v1", html)
            self.assertIn("Agent 执行实验", html)
            self.assertNotIn("不是 LLM", html)

    def test_external_pair_and_transfer_reports_use_external_source(self):
        paired = run_local_agent_paired_demo(seed_count=1, candidate_budget=1)
        paired = replace(
            paired,
            manifest=replace(paired.manifest, executor="external_mock_v1"),
        )
        transfer = run_local_transfer_demo(candidate_budget=1)
        transfer = replace(
            transfer,
            base=replace(
                transfer.base,
                manifest=replace(transfer.base.manifest, executor="external_mock_v1"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (experiment, writer) in enumerate(
                ((paired, write_agent_pair_evidence), (transfer, write_transfer_evidence))
            ):
                evidence_path, html_path = writer(
                    experiment,
                    root / str(index),
                    limitations=["External mock source; no network request was made."],
                )
                evidence = read_evidence(evidence_path)
                self.assertEqual(
                    evidence["limitations"],
                    ["External mock source; no network request was made."],
                )
                html = html_path.read_text(encoding="utf-8")
                self.assertIn("external_mock_v1", html)
                self.assertNotIn("不是 LLM", html)


if __name__ == "__main__":
    unittest.main()
