# Merit

**Evaluate the improver, not just the agent.**

Merit compares methods for improving agents from the same starting point, tasks, and candidate budget. It measures unseen-task performance, transfer, regressions, execution traces, and token use—not just the final score.

[![CI](https://github.com/Oxygen56/merit/actions/workflows/ci.yml/badge.svg)](https://github.com/Oxygen56/merit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**English** | [简体中文](README.zh-CN.md)

## Why Merit?

A better agent does not necessarily mean a better improvement method. Merit keeps the comparison controlled:

- **Paired controls:** compare no-change, random edits, heuristics, model reflection, and GEPA from one starting agent.
- **Comparable budgets:** cap candidate evaluations and record proposer usage separately.
- **Task isolation:** optimize on training tasks, select on validation tasks, and evaluate on separate final tasks.
- **Beyond accuracy:** inspect transfer, source-task regressions, tool trajectories, tokens, and latency.
- **Reproducible runs:** save candidate parents, per-task outputs, settings, manifests, and HTML reports.

Merit is an evaluation framework, not another agent framework. It works with a small Python executor interface and has no core runtime dependencies.

## Install

```bash
git clone https://github.com/Oxygen56/merit.git
cd merit
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Python 3.10 or newer is required.

## Quickstart

Run a paired local experiment. No API key or model download is needed:

```bash
merit paired-agent-demo --seed-count 2 --budget 2 --output results/quickstart
merit validate results/quickstart
merit report results/quickstart
```

The output directory contains:

```text
manifest.json   # settings written before execution
evidence.json   # candidates, per-task outputs, usage, and statistics
report.html     # a report to open in your browser
```

Try transfer and early/late improver comparisons next:

```bash
merit transfer-demo --budget 2 --output results/transfer
merit plasticity-demo --trajectory-count 2 --seed-count 2 --budget 2 --output results/plasticity
```

The bundled demos use fixed local responses to exercise the experiment runner. They are interface examples, not model benchmark results.

## What the current runs show

| Experiment | Result | Scope |
|---|---|---|
| Simple real-model reflection | 0% → 100% final accuracy | 2 seeds, small synthetic tasks |
| BFCL-adapted subset | No change / reflection / GEPA: 63.3%; random edit: 60.0% | No test-transfer gain; not an official BFCL score |
| Support workflow | 100% → 100%; missing-policy path: 5 → 4 steps | Better trajectory, no accuracy gain |
| Early vs. late improver | 100% successor accuracy on both sides; delta 0 | 4 pairs sharing one early source run |

The negative results are part of the project: current runs do not establish a general ranking of improvement methods or open-ended recursive self-improvement. See [all results and reproduction commands](docs/RESULTS.md).

## Connect your agent

Implement one method; Merit handles the optimization loop, evaluation, budgets, and run records:

```python
from merit import AgentResponse, Usage

class MyExecutor:
    name = "my-agent-v1"

    def invoke(self, agent, task):
        result = my_runtime.run(
            instructions=agent.instructions,
            config=agent.config,
            input=task.prompt,
        )
        return AgentResponse(
            output=result.text,
            usage=Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                cost_reported=False,
            ),
        )
```

Run the complete local example with:

```bash
python examples/custom_executor.py --output results/custom
merit validate results/custom
```

See [executor integration](docs/EXECUTORS.md) for OpenAI-compatible endpoints, LangGraph, AgentScope, and tool executors.

## Local models

Use a running local Ollama service for model-backed experiments:

```bash
python -m pip install -e '.[langgraph,yaml]'
ollama pull qwen2.5:0.5b
merit run examples/ollama_langgraph.yaml --output results/ollama-pilot
merit report results/ollama-pilot
```

`OllamaExecutor` accepts only loopback addresses and does not read an API key. Successful responses are cached inside the result directory so an interrupted run can resume. GEPA support is available through the `gepa` extra.

## Documentation

- [Results and reproduction](docs/RESULTS.md)
- [Experiment protocol and result format](docs/PROTOCOL.md)
- [Executor integration](docs/EXECUTORS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src examples tests
```

Default tests and demos make no external model calls.

## License

[MIT](LICENSE)
