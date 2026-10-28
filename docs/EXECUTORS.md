# 执行器与 Agent 框架接入指南

## 1. 稳定接入边界

Merit 不接管 Agent 框架的内部状态。任何模型、工作流或 Agent 运行时都只需实现：

```python
class Executor:
    name = "my_executor_version"

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        ...
```

- `agent.instructions` 是系统指令；`agent.config` 是 JSON 配置；
- `task.prompt` 与 `task.expected` 属于当前任务，但执行器只需要 prompt，不能利用 expected 生成答案；
- 返回 `AgentResponse(output, usage)`，exact-match 由 runner 统一计算；
- executor 的 `name` 应能区分模型或运行时版本，manifest 会固定它；
- API key、cookie、token 等秘密不得写入 `AgentSpec`、任务或实验记录。

可直接运行的离线接入示例见 `examples/custom_executor.py`。

## 2. OpenAI-compatible 执行器

```python
import os
from merit import OpenAICompatibleExecutor, TokenPricing

executor = OpenAICompatibleExecutor(
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ["OPENAI_MODEL"],
    api_key_env="OPENAI_API_KEY",
    timeout_seconds=60,
    max_retries=2,
    # 可选：只有明确核对供应商单价后才配置。
    pricing=TokenPricing(input_per_million=1.0, output_per_million=2.0),
)
```

`base_url` 可以填服务根地址或完整 `/chat/completions` 地址。模型固定在 executor 上，artifact 不能用 config 偷换模型。支持的 artifact 参数为 `temperature`、`top_p`、`max_tokens`、`stop`、`seed`、`response_format`；供应商是否支持某个参数需要单独确认。

供应商未返回 usage 时，`tokens_reported=False`。未配置价格时，`cost_reported=False`。这两种情况都不能解释为 0 token 或免费。配置价格产生的成本会标记 `cost_is_estimate=True`，不冒充供应商账单。

`examples/openai_smoke.py` 提供小规模调用脚本。它需要显式 `--accept-paid-run`，默认不会运行；当前仓库的测试和 demo 均不调用它。

`examples/real_validation.py` 包含已经运行过的 `llm` 与 `agent` 两组实验，固定模型为 `kimi-for-coding`，支持网络调用次数上限、断点缓存、原始调用日志、逐任务工具轨迹和结果目录校验。缓存重放保留首次供应商 latency，不将本地缓存耗时当成模型延迟。完整结果见 `results/real-llm/` 与 `results/real-agent/`；两种实验均为小规模探索，不是通用模型排名或开放式 RSI 证明。

## 3. 模型反思 proposer

被评 Agent 与 proposer 可以使用不同 executor：

```python
from merit import AgentOptimizerAdapter, AgentSpec, ModelReflectionProposer

reflection = ModelReflectionProposer(
    proposer_executor,
    AgentSpec("Revise instructions using only supplied training failures."),
    max_failures=5,
)
optimizer = AgentOptimizerAdapter(
    "model_reflection",
    reflection,
    proposer_source=proposer_executor.name,
)
```

manifest 的 `proposer_source` 必须相同。每次提案返回 `AgentProposal`，独立记录 proposer usage。训练失败样例最多取 `max_failures` 条；隐藏任务不会进入 proposer prompt。反思输出必须是完整的新 instructions，而不是 diff 或解释文字。

## 4. LangGraph 接入模式

以下是与具体 LangGraph 版本解耦的适配模板，不是对第三方 API 的实时验证。`compiled_graph` 由使用者在当前环境中构建；消息映射、状态字段与 usage 位置需要按实际 graph 调整。

```python
from merit import AgentResponse, AgentSpec, AgentTask, Usage

class LangGraphExecutor:
    def __init__(self, compiled_graph, name="langgraph:my_graph_v1"):
        self.graph = compiled_graph
        self.name = name

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        state = self.graph.invoke(
            {
                "instructions": agent.instructions,
                "agent_config": agent.config,
                "messages": [{"role": "user", "content": task.prompt}],
            },
            config={"configurable": {"thread_id": f"merit-{task.task_id}"}},
        )
        output = state["messages"][-1].content
        # 若运行时没有 usage，必须显式标记未知。
        return AgentResponse(output, Usage(tokens_reported=False, cost_reported=False))
```

注意：

- 每个任务使用独立 thread，避免上一题记忆污染下一题；若研究记忆，必须把它作为显式实验变量；
- 不要把图内检索到的 held-out 语料意外暴露给 proposer；
- 若输出是结构化消息，先定义唯一、确定的文本提取规则；
- token 和成本应从模型响应或计费记录映射，不能用字符数假扮供应商 usage。

## 5. AgentScope 接入模式

同样推荐写薄适配器，不让 Merit 依赖 AgentScope 内部对象。下面的 `invoke_agent` 是由使用者提供的单轮调用函数，因此模板不假设某个 AgentScope 版本的消息类或返回类型：

```python
from merit import AgentResponse, AgentSpec, AgentTask, Usage

class AgentScopeExecutor:
    def __init__(self, invoke_agent, name="agentscope:my_agent_v1"):
        self._invoke_agent = invoke_agent
        self.name = name

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        result = self._invoke_agent(
            instructions=agent.instructions,
            config=agent.config,
            prompt=task.prompt,
            session_id=f"merit-{task.task_id}",
        )
        return AgentResponse(
            output=result["text"],
            usage=result.get("usage", Usage(tokens_reported=False, cost_reported=False)),
        )
```

`invoke_agent` 应创建或选择干净的会话，将 AgentScope 的文本输出和 usage 映射到上述字典。多 Agent 工作流也按同一方式封装，但 artifact 必须包含影响行为的配置，否则两个后继可能实际上不可复现。

## 6. 模型实验检查清单

1. 先用本地 mock 检查任务隔离、预算和报告，再运行 1～4 次外部模型调用；
2. 明确模型、base URL、参数、超时、重试、任务数、候选预算和费用上限；
3. 逐条保存供应商响应，失败响应不能静默当作正确或免费；
4. 优化集、held-out、transfer 和 source 任务分别命名并检查 digest；
5. 多轮 Agent 会话默认隔离，跨任务记忆必须单独声明；
6. 报告的实验限制必须与实际调用一致，不能沿用本地“不是 LLM”的模板；
7. 小样本只做 smoke 或探索性分析，不把单次成功率当成稳定模型排名。
