"""Build an unambiguous tool-call subset from the public BFCL v4 JSONL files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent / "data"
QUESTIONS = ROOT / "source" / "BFCL_v4_simple_python.json"
ANSWERS = ROOT / "source" / "BFCL_v4_simple_python_answers.json"
SOURCE_URL = "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    answers = {row["id"]: row["ground_truth"] for row in read_jsonl(ANSWERS)}
    tasks = []
    for row in read_jsonl(QUESTIONS):
        if len(row["function"]) != 1:
            continue
        tool = row["function"][0]
        parameters = tool["parameters"]
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if set(properties) != set(required):
            continue
        ground_truth = answers[row["id"]][0][tool["name"]]
        if not set(required).issubset(ground_truth):
            continue
        arguments = {name: ground_truth[name][0] for name in required}
        question = row["question"][0][0]["content"]
        prompt = json.dumps({"question": question, "tools": [tool]}, ensure_ascii=False)
        expected = json.dumps({"name": tool["name"], "arguments": arguments}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        tasks.append({"id": row["id"], "prompt": prompt, "expected": expected})
        if len(tasks) == 40:
            break
    if len(tasks) < 40:
        raise RuntimeError("Not enough unambiguous BFCL tasks")
    payload = {
        "source": {
            "name": "BFCL v4 simple_python",
            "url": SOURCE_URL,
            "license": "Apache-2.0",
            "selection": "First 40 single-tool tasks with no optional parameters",
            "metric_notice": "BFCL-derived required-argument JSON exact match; not an official BFCL score. Public tasks may occur in model training data.",
        },
        "train": tasks[:10],
        "heldout": tasks[10:],
    }
    output = ROOT / "bfcl_tool_subset.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}: 10 train + 30 held-out tasks")


if __name__ == "__main__":
    main()
