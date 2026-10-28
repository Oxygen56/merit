"""Pilot the fixed schema prompt on train and validation tasks only."""
from __future__ import annotations

import json
from pathlib import Path

from merit.agent_contract import AgentSpec, AgentTask, evaluate_agent
from merit.experiment import SCHEMA_GUIDED_SUFFIX
from merit.langgraph_executor import LangGraphToolExecutor
from merit.ollama_executor import OllamaExecutor

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "examples/data/bfcl_tool_validation_split.json").read_text())
OUTPUT = ROOT / "results/schema-guided-prompt-pilot"
OUTPUT.mkdir(parents=True, exist_ok=True)
model = OllamaExecutor(
    "qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    timeout_seconds=120,
    cache_dir=OUTPUT / "cache",
    max_network_calls=200,
)
executor = LangGraphToolExecutor(model)
start = AgentSpec(
    "Read the JSON input. Select the appropriate tool and return a JSON object with name and arguments.",
    {"temperature": 0, "max_tokens": 192, "seed": 101},
)
candidate = start.evolve(instructions=f"{start.instructions}\n{SCHEMA_GUIDED_SUFFIX}")
summary = {"seed": 101, "heldout_tasks_evaluated": 0}
for split in ("train", "validation"):
    tasks = tuple(AgentTask(row["id"], row["prompt"], row["expected"]) for row in DATA[split])
    initial = evaluate_agent(executor, start, tasks)
    revised = evaluate_agent(executor, candidate, tasks)
    summary[split] = {"initial_accuracy": initial.accuracy, "candidate_accuracy": revised.accuracy}
summary["network_calls_this_run"] = model.network_calls
summary["cache_hits_this_run"] = model.cache_hits
(OUTPUT / "pilot.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUTPUT / "tool_traces.json").write_text(json.dumps(executor.tool_traces, indent=2) + "\n")
print(json.dumps(summary, indent=2))
