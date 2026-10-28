"""Stable evidence-envelope validation and deterministic compression utilities."""

from __future__ import annotations

import gzip
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any


EVIDENCE_SCHEMA_VERSION = "merit-evidence-v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def evidence_schema() -> dict[str, Any]:
    """Load the packaged JSON Schema without optional runtime dependencies."""

    schema_path = resources.files("merit").joinpath("schemas/evidence-envelope.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_evidence_envelope(payload: Any) -> None:
    """Validate the stable envelope; experiment-specific protocols remain versioned separately."""

    if not isinstance(payload, dict):
        raise ValueError("Evidence must be a JSON object")
    required = {
        "schema_version",
        "project",
        "version",
        "generated_at",
        "environment",
        "manifest_digest",
        "limitations",
        "experiment",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Evidence is missing required fields: {', '.join(missing)}")
    if payload["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evidence schema: {payload['schema_version']}")
    for field in ("project", "version", "generated_at"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"Evidence {field} must be a non-empty string")
    if payload["project"] != "Merit":
        raise ValueError("Evidence project must be Merit")
    environment = payload["environment"]
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(field), str) or not environment[field]
        for field in ("python", "platform")
    ):
        raise ValueError("Evidence environment requires python and platform strings")
    if not isinstance(payload["manifest_digest"], str) or not _DIGEST.fullmatch(payload["manifest_digest"]):
        raise ValueError("Evidence manifest_digest must be a lowercase SHA-256 hex digest")
    if not isinstance(payload["limitations"], list) or any(
        not isinstance(item, str) or not item for item in payload["limitations"]
    ):
        raise ValueError("Evidence limitations must be a list of non-empty strings")
    if not isinstance(payload["experiment"], dict):
        raise ValueError("Evidence experiment must be an object")
    manifest = payload["experiment"].get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("protocol_version"), str):
        raise ValueError("Evidence experiment.manifest requires a protocol_version")


def read_evidence(path: Path) -> dict[str, Any]:
    """Read plain JSON or gzip-compressed JSON and validate its envelope."""

    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    validate_evidence_envelope(payload)
    return payload


def compress_evidence(path: Path, output_path: Path | None = None) -> Path:
    """Write deterministic gzip output with no source name or timestamp in its header."""

    path = Path(path)
    payload = read_evidence(path)
    output_path = Path(output_path) if output_path is not None else path.with_suffix(path.suffix + ".gz")
    if output_path.resolve() == path.resolve():
        raise ValueError("Compressed output must differ from the source evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(serialized)
    read_evidence(output_path)
    return output_path


def validate_result_directory(path: Path) -> dict[str, str]:
    """Cross-check manifest.json and evidence.json before offline sharing or replay."""

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("protocol_version"), str):
        raise ValueError("Result manifest requires a protocol_version")
    digest = manifest.get("manifest_digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("Result manifest requires a valid manifest_digest")
    evidence_path = path / "evidence.json"
    if not evidence_path.exists():
        evidence_path = path / "evidence.json.gz"
    evidence = read_evidence(evidence_path)
    if evidence["manifest_digest"] != digest:
        raise ValueError("Manifest digest differs between manifest and evidence")
    experiment_manifest = evidence["experiment"]["manifest"]
    if experiment_manifest["protocol_version"] != manifest["protocol_version"]:
        raise ValueError("Protocol version differs between manifest and evidence")
    return {
        "schema_version": evidence["schema_version"],
        "protocol_version": manifest["protocol_version"],
        "manifest_digest": digest,
    }
