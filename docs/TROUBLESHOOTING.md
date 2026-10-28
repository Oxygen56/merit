# 故障排查

## 安装与离线运行

**找不到 `merit` 命令**

确认虚拟环境已激活，或直接使用 `.venv/bin/merit`。源码目录也可以运行 `PYTHONPATH=src python -m merit --help`。

**没有 API key 是否能测试**

可以。全部单元测试、mock HTTP 测试和五个本地 demo 都不需要外部账号。外部模型调用示例需要单独显式运行。

**Python 版本**

要求 Python 3.10 或更高版本。核心运行无第三方依赖；安装构建工具需要的环境与运行时依赖是两回事。

## Manifest 与任务错误

**`Manifest ... does not match`**

manifest 写入后又替换了起点、任务、对照臂、executor、proposer 或预算。应重新生成 manifest 并开始新实验，不要绕过 digest 检查。

**`Train and held-out task IDs must be disjoint`**

同一 split 内 task ID 必须唯一，训练与隐藏任务 ID 也不能重复。还应额外检查输入内容重复；ID 不同并不自动代表语义上不泄漏。

**`Agent proposal must be a direct child`**

候选必须通过 `context.current.evolve(...)` 或等价方式从当前 artifact 派生。返回旧 artifact、另一个分支或直接新建无 parent 的 `AgentSpec` 都会被拒绝。

**预算超限**

每次候选开发评价都必须调用 runner 提供的 `evaluate(candidate)`。不要在 optimizer 内自行调用 executor 来规避候选预算；额外的 proposer 调用应通过 `AgentProposal.usage` 独立记账。

## 模型调用与成本

**缺少密钥**

错误会指出缺失的环境变量名。不要把密钥写进 `AgentSpec.config`、代码、manifest 或 issue。

**HTTP 429／5xx／超时**

内置执行器只对有限状态和网络错误做有限重试，并使用指数退避。持续失败应减少任务或稍后再运行，不要在 tight loop 中反复执行付费 smoke。

**成本显示未完整记录**

检查供应商是否返回 usage，以及是否显式配置了 `TokenPricing`。Merit 不猜测价格。`cost_reported=False` 与 `cost_usd=0` 的含义不同。

**模型结果与本地结果不同**

这是预期现象。本地文本规范化器只验证协议，不预测 LLM 行为。检查模型、参数、prompt、输出提取规则和任务是否一致。

## 统计解释

**为什么区间在小样本上很宽或退化**

少量种子或轨迹无法支持稳定排名。确定性样例的所有差值相同时，区间会退化为一个点，这不代表模型实验有同样确定性。

**Meta-Plasticity 为什么不能把所有 seed 当作独立样本**

同一轨迹内的种子共享改进器历史。Merit 先求轨迹内配对均值，再按轨迹做 Bootstrap 和符号翻转，避免伪重复。

**未达到阈值为什么显示 `null`／未达到**

这是预算内右删失，不是 0 次，也不能直接填候选预算。报告同时给出达阈率与仅在达阈运行中的平均评估数。

## 实验记录、压缩与迁移

**校验结果目录**

```bash
merit validate results/agent-paired
```

失败信息会区分缺少 envelope 字段、未知 schema、digest 不一致和 protocol 不一致。不要用删除校验字段的方式修复。

**实验记录文件太大**

```bash
merit compress-evidence results/agent-paired/evidence.json
```

生成 `evidence.json.gz` 后可保留压缩文件进行校验；JSON 内容不变。分享前仍需检查任务、输出和供应商数据是否可公开。

**老实验记录没有 `schema_version`**

它属于开发期旧格式。按 `docs/PROTOCOL.md` 重新运行并核对结果，不要只补版本号后宣称迁移完成。
