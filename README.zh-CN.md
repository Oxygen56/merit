# Merit

**评测改进方法，而不只是 Agent 的最终分数。**

Merit 让不同改进方法从同一个 Agent、同一批任务和相同的候选预算开始，再比较新任务表现、迁移、旧能力退化、执行轨迹和 token 用量，而不是只看最终分数。

[![CI](https://github.com/Oxygen56/merit/actions/workflows/ci.yml/badge.svg)](https://github.com/Oxygen56/merit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | **简体中文**

## 为什么选择 Merit？

改出的 Agent 更好，不代表改进方法本身更好。Merit 把比较条件固定下来：

- **配对对照：**从同一个起点比较不修改、随机编辑、启发式方法、模型反思和 GEPA。
- **可比预算：**限制候选评估次数，单独记录 proposer 用量。
- **任务隔离：**训练任务用于优化，验证任务用于选择，最终测试独立评估。
- **不只看准确率：**同时检查迁移、旧能力退化、工具轨迹、token 和耗时。
- **可复现记录：**保存候选父子关系、逐题输出、实验设置、manifest 和 HTML 报告。

Merit 是评测框架，不是另一个 Agent 开发框架。接入只需要一个很小的 Python 执行器接口，核心包没有第三方运行依赖。

## 安装

```bash
git clone https://github.com/Oxygen56/merit.git
cd merit
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

需要 Python 3.10 或更新版本。

## 快速开始

运行一个本地配对实验，不需要 API key，也不用下载模型：

```bash
merit paired-agent-demo --seed-count 2 --budget 2 --output results/quickstart
merit validate results/quickstart
merit report results/quickstart
```

输出目录包含：

```text
manifest.json   # 执行前写入的实验设置
evidence.json   # 候选、逐题输出、用量和统计结果
report.html     # 可直接在浏览器打开的报告
```

继续尝试迁移和早晚改进器比较：

```bash
merit transfer-demo --budget 2 --output results/transfer
merit plasticity-demo --trajectory-count 2 --seed-count 2 --budget 2 --output results/plasticity
```

内置 demo 使用固定的本地响应来跑通实验流程，属于接口示例，不是模型基准成绩。

## 当前运行结果

| 实验 | 结果 | 范围 |
|---|---|---|
| 简单任务真实模型反思 | 最终准确率 0% → 100% | 2 个种子，小规模合成任务 |
| BFCL 改编子集 | 不修改／反思／GEPA 均为 63.3%，随机修改为 60.0% | 没有测试迁移提升；不是官方 BFCL 分数 |
| 售后工作流 | 准确率 100% → 100%，缺规则路径 5 → 4 步 | 轨迹改善，没有准确率提升 |
| 早晚改进器 | 两边后继准确率均为 100%，差值为 0 | 4 组配对，共享同一个早期来源 |

负结果同样保留：目前不能据此给通用改进方法排名，也不能说明开放式递归自我改进有效。查看[完整结果与复现命令](docs/RESULTS.md)。

## 接入你的 Agent

实现一个方法即可，Merit 负责优化流程、评估、预算和运行记录：

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

运行完整的本地接入示例：

```bash
python examples/custom_executor.py --output results/custom
merit validate results/custom
```

OpenAI-compatible 接口、LangGraph、AgentScope 和工具执行器的接入方式见[执行器接入指南](docs/EXECUTORS.md)。

## 本地模型

启动本机 Ollama 后，可以运行模型实验：

```bash
python -m pip install -e '.[langgraph,yaml]'
ollama pull qwen2.5:0.5b
merit run examples/ollama_langgraph.yaml --output results/ollama-pilot
merit report results/ollama-pilot
```

`OllamaExecutor` 只接受本机回环地址，不读取 API key。成功响应会缓存在结果目录中，中断后可以继续运行。安装 `gepa` 可选依赖后可使用 GEPA 对照。

## 文档

- [实验结果与复现](docs/RESULTS.md)
- [实验协议与结果格式](docs/PROTOCOL.md)
- [执行器接入指南](docs/EXECUTORS.md)
- [故障排查](docs/TROUBLESHOOTING.md)
- [路线图](ROADMAP.md) · [更新记录](CHANGELOG.md)
- [贡献指南](CONTRIBUTING.md) · [安全说明](SECURITY.md)

## 开发

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src examples tests
```

默认测试和 demo 不调用外部模型。

## 许可证

[MIT](LICENSE)
