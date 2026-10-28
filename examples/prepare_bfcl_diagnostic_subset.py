"""Build a structure-stratified training set; preserve the original test set byte-for-byte.

The additions were selected from source features before running the new model
experiment. The original held-out outputs were inspected in an earlier pilot,
so this is a diagnostic dataset, not an untouched benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent / "data"
QUESTIONS = ROOT / "source" / "BFCL_v4_simple_python.json"
ANSWERS = ROOT / "source" / "BFCL_v4_simple_python_answers.json"
ORIGINAL = ROOT / "bfcl_tool_subset.json"
OUTPUT = ROOT / "bfcl_tool_diagnostic_subset.json"

# Each tuple is (source ID, structural capabilities exercised by this task).
# Selection does not use any result on the new training tasks.
ADDITIONS = (
    ("simple_python_0", ("optional_string", "exact_value")),
    ("simple_python_15", ("optional_enum", "exact_value")),
    ("simple_python_28", ("optional_float", "float_value")),
    ("simple_python_34", ("optional_string", "multiple_required")),
    ("simple_python_89", ("nested_object", "omitted_optional")),
    ("simple_python_94", ("nested_object", "optional_default")),
    ("simple_python_108", ("string_array", "optional_boolean")),
    ("simple_python_119", ("integer_arrays", "optional_float")),
    ("simple_python_120", ("integer_arrays", "optional_boolean")),
    ("simple_python_122", ("nested_arrays", "omitted_optional")),
    ("simple_python_126", ("string_array", "exact_value")),
    ("simple_python_260", ("nested_objects", "optional_object")),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_value(schema: dict[str, Any], alternatives: Any, path: str) -> Any:
    """Choose the first BFCL alternative while respecting array-valued arguments."""
    kind = schema.get("type")
    if kind in {"dict", "object"} or "properties" in schema:
        # The top-level ground truth is the argument map itself; nested objects
        # wrap each candidate object in BFCL's list of alternatives.
        if isinstance(alternatives, dict):
            value = alternatives
        elif isinstance(alternatives, list) and alternatives and isinstance(alternatives[0], dict):
            value = alternatives[0]
        else:
            raise ValueError(f"Expected object alternatives at {path}")
        result = {}
        for key, child_alternatives in value.items():
            decoded = decode_value(schema.get("properties", {}).get(key, {}), child_alternatives, f"{path}.{key}")
            if decoded != "":
                result[key] = decoded
        return result
    if kind == "array":
        if not isinstance(alternatives, list) or not alternatives or not isinstance(alternatives[0], list):
            raise ValueError(f"Expected array alternatives at {path}")
        # BFCL's outer list contains alternatives; the chosen list is the argument.
        return alternatives[0]
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError(f"Expected scalar alternatives at {path}")
    return alternatives[0]


def convert(row: dict[str, Any], answers: dict[str, Any], capabilities: tuple[str, ...]) -> dict[str, Any]:
    if len(row["function"]) != 1:
        raise ValueError(f"Expected one tool for {row['id']}")
    tool = row["function"][0]
    schema = tool["parameters"]
    ground_truth = answers[row["id"]][0][tool["name"]]
    arguments = decode_value(schema, ground_truth, row["id"])
    missing = set(schema.get("required", [])) - set(arguments)
    if missing:
        raise ValueError(f"Missing required arguments for {row['id']}: {sorted(missing)}")
    prompt = canonical({"question": row["question"][0][0]["content"], "tools": [tool]})
    return {
        "id": row["id"],
        "prompt": prompt,
        "expected": canonical({"name": tool["name"], "arguments": arguments}),
        "capabilities": list(capabilities),
    }


def main() -> None:
    original_text = ORIGINAL.read_text(encoding="utf-8")
    original = json.loads(original_text)
    questions = {row["id"]: row for row in read_jsonl(QUESTIONS)}
    answers = {row["id"]: row["ground_truth"] for row in read_jsonl(ANSWERS)}
    excluded = {row["id"] for row in original["train"] + original["heldout"]}
    duplicate = excluded & {task_id for task_id, _ in ADDITIONS}
    if duplicate:
        raise ValueError(f"New training tasks overlap the original experiment: {sorted(duplicate)}")
    added = [convert(questions[task_id], answers, capabilities) for task_id, capabilities in ADDITIONS]
    train = original["train"] + added
    if len({row["id"] for row in train}) != len(train):
        raise ValueError("Duplicate training task IDs")
    payload = {
        "source": {
            **original["source"],
            "selection": (
                "The original 10 training tasks plus 12 source-feature-stratified additions "
                "covering optional parameters, arrays, nested objects, numeric types, and exact values. "
                "The original 30 held-out tasks are unchanged. Difficulty is assessed on training tasks only."
            ),
            "metric_notice": (
                original["source"]["metric_notice"]
                + " Optional defaults use the first BFCL alternative; nested objects and arrays follow the supplied schema. "
                "This is a diagnostic follow-up after inspecting earlier test errors, not an untouched benchmark."
            ),
        },
        "train": train,
        "heldout": original["heldout"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Assert semantic identity of every held-out field, not merely task IDs.
    assert payload["heldout"] == original["heldout"]
    print(f"Wrote {OUTPUT}: {len(original['train'])} original + {len(added)} diagnostic training tasks; 30 unchanged test tasks")
    for row in added:
        print(row["id"], ", ".join(row["capabilities"]), row["expected"])


if __name__ == "__main__":
    main()
