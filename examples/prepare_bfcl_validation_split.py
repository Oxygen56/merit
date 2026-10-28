"""Create fixed train/validation/test splits using source structure only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_bfcl_diagnostic_subset import convert, read_jsonl

ROOT = Path(__file__).parent / "data"
QUESTIONS = ROOT / "source" / "BFCL_v4_simple_python.json"
ANSWERS = ROOT / "source" / "BFCL_v4_simple_python_answers.json"
TRAINING = ROOT / "bfcl_tool_diagnostic_subset.json"
PREVIOUS = ROOT / "bfcl_tool_subset.json"
OUTPUT = ROOT / "bfcl_tool_validation_split.json"
SELECTION_SALT = "merit-bfcl-train-validation-test-v1"
GROUPS = ("nested", "array", "float", "optional", "scalar")


def capabilities(row: dict[str, Any]) -> tuple[str, ...]:
    prompt = json.loads(row["prompt"])
    schema = prompt["tools"][0]["parameters"]
    optional = set(schema.get("properties", {})) - set(schema.get("required", []))
    types: set[str] = set()

    def walk(node: dict[str, Any], is_root: bool = False) -> None:
        kind = node.get("type")
        if isinstance(kind, str) and (not is_root or kind not in {"dict", "object"}):
            types.add(kind)
        for child in node.get("properties", {}).values():
            walk(child)
        if isinstance(node.get("items"), dict):
            walk(node["items"])

    walk(schema, is_root=True)
    tags = set()
    if optional:
        tags.add("optional")
    if types.intersection({"array"}):
        tags.add("array")
    if types.intersection({"dict", "object"}):
        tags.add("nested")
    if types.intersection({"float", "number"}):
        tags.add("float")
    return tuple(tag for tag in ("optional", "array", "nested", "float") if tag in tags) or ("scalar",)


def primary_group(tags: tuple[str, ...]) -> str:
    return next((group for group in GROUPS if group in tags), "scalar")


def order_key(task_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{task_id}".encode()).hexdigest()


def main() -> None:
    training = json.loads(TRAINING.read_text())
    previous = json.loads(PREVIOUS.read_text())
    questions = {row["id"]: row for row in read_jsonl(QUESTIONS)}
    answers = {row["id"]: row["ground_truth"] for row in read_jsonl(ANSWERS)}
    excluded = {
        row["id"]
        for row in training["train"] + training["heldout"] + previous["train"] + previous["heldout"]
    }
    groups: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    for task_id in sorted(questions.keys() - excluded):
        try:
            row = convert(questions[task_id], answers, ())
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        tags = capabilities(row)
        row["capabilities"] = list(tags)
        groups[primary_group(tags)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: order_key(row["id"]))

    validation: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    assignments = Counter()
    while len(validation) < 30 or len(heldout) < 30:
        added = False
        for group in GROUPS:
            if len(validation) == 30 and len(heldout) == 30:
                break
            if not groups[group]:
                continue
            row = groups[group].pop(0)
            prefer_validation = assignments[group] % 2 == 0
            assignments[group] += 1
            if prefer_validation and len(validation) < 30:
                validation.append(row)
            elif len(heldout) < 30:
                heldout.append(row)
            elif len(validation) < 30:
                validation.append(row)
            else:
                raise AssertionError("A full split received another task")
            added = True
        if not added:
            break
    if len(validation) != 30 or len(heldout) != 30:
        raise ValueError(f"Need 30 tasks per split, found {len(validation)} and {len(heldout)}")
    payload = {
        "source": {
            **training["source"],
            "selection": (
                "The existing 22 training tasks are unchanged. Thirty validation and thirty final-test tasks "
                "were selected from unused source rows by deterministic structure-stratified round robin, "
                "alternating validation and test within each stratum. No model outputs were consulted. "
                f"Selection salt: {SELECTION_SALT}."
            ),
            "metric_notice": (
                training["source"]["metric_notice"]
                + " The validation and final-test splits in this file were fixed before their first model run."
            ),
        },
        "train": training["train"],
        "validation": validation,
        "heldout": heldout,
    }
    splits = (payload["train"], payload["validation"], payload["heldout"])
    assert [len(rows) for rows in splits] == [22, 30, 30]
    assert payload["train"] == training["train"]
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            assert not {row["id"] for row in left}.intersection(row["id"] for row in right)
            assert not {row["prompt"] for row in left}.intersection(row["prompt"] for row in right)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT}: train=22 validation=30 heldout=30")
    for name, rows in zip(("validation", "heldout"), (validation, heldout), strict=True):
        counts = Counter(primary_group(tuple(row["capabilities"])) for row in rows)
        print(name, dict(counts))
        print("IDs:", ",".join(row["id"] for row in rows))


if __name__ == "__main__":
    main()
