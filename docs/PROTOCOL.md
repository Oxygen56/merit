# Merit 协议与迁移说明

## 版本层次

Merit 将三类版本分开管理：

- 包版本：当前为 `0.1.0`，表示 Python API 与命令行的发布版本；
- 结果文件外层 schema：当前为 `merit-evidence-v1`，约束所有结果文件的公共结构；
- 实验 protocol：约束 `experiment.manifest` 及其对应的实验数据结构。

| 实验 | protocol_version |
|---|---|
| 玩具配对实验 | `merit-paired-v0.1` |
| Agent 单次运行 | `merit-agent-contract-v0.1` |
| 多种子 Agent 配对与消融 | `merit-agent-paired-v0.1` |
| 迁移与旧能力回归 | `merit-transfer-v0.1` |
| Meta-Plasticity | `merit-meta-plasticity-v0.1` |

包版本升级不自动改变实验含义。任何已有字段的语义变更都必须伴随相应 protocol 新版本和迁移说明，不能只改字段名称后继续沿用旧 digest 口径。

## 结果文件外层（evidence envelope）

每份 `evidence.json` 必须包含：

| 字段 | 含义 |
|---|---|
| `schema_version` | 公共外层 schema 版本 |
| `project` / `version` | 项目名和生成它的 Merit 包版本 |
| `generated_at` | UTC 生成时间；不要求两次运行相同 |
| `environment` | Python 与平台信息 |
| `manifest_digest` | 与预运行 manifest 对齐的小写 SHA-256 |
| `limitations` | 当前实验不能外推的边界 |
| `experiment` | 具体 protocol 的完整实验数据 |

打包的 JSON Schema 位于 `src/merit/schemas/evidence-envelope.schema.json`，也可通过 `merit.evidence.evidence_schema()` 读取。envelope 只约束公共结构；实验字段由各自的 `protocol_version` 解释。

```bash
merit validate results/agent-paired
merit compress-evidence results/agent-paired/evidence.json
```

`validate` 交叉检查 manifest digest 与 protocol 版本，支持目录中只有 `evidence.json.gz` 的情况。`compress-evidence` 先校验再生成确定性 gzip，并重新读取压缩结果验证；压缩不会改变 JSON 内容。

## 核心不变量

- `AgentSpec` 不可变，`artifact_id` 由 instructions、config、parent 和协议版本共同决定；候选必须是当前节点的直接子节点。
- manifest 在执行前固定起点、任务 digest、预算、方法和 proposer 配置；runner 拒绝替换。
- 优化器只能收到训练任务与训练反馈。训练与隐藏任务不仅 ID 必须不同，输入文本也不得重复；隐藏任务、迁移任务和 source 回归任务由 runner 独立复评。
- 主动优化臂的候选评估预算相同。no-change 是零候选锚点，不参与“等预算智能方法”的解释。
- `total_*` 只覆盖被评 Agent；`proposer_*` 独立；`combined_total_*` 才是二者之和。缺失 usage 不能当作免费。
- Meta-Plasticity 的统计单位是轨迹，种子只在轨迹内配对；开发阈值未达运行是右删失，不能填成 0 或预算值。

## 迁移策略

### 迁移没有 `schema_version` 的旧结果文件

早期开发版本的 evidence 没有公共 `schema_version`。这些文件不是 `merit-evidence-v1` 的合格输入，应该用对应开发版本重新运行实验，而不是只手工补一个版本号就声称完成迁移。重新运行后必须核对：

1. manifest digest 与原实验配置和任务定义一致；
2. 数值结果及 candidate lineage 一致；
3. 新增字段没有覆盖原字段或改变其含义；
4. limitations 与执行器来源仍然一致。

### 后续兼容规则

- envelope 的 patch 兼容变更只能增加可选元数据，不改变已有字段语义；
- 删除字段、修改类型或改变统计口径需要新 schema 版本；
- 实验协议变更需要新的 `protocol_version`，并提供从旧版本重跑或转换的具体步骤；
- 读取器遇到未知版本必须明确报错，不得猜测结构并生成看似有效的结果；
- 迁移时保留随机种子、预算、任务 digest 和 artifact parent，不要重写。

当前仍处于 0.x 阶段。发布 1.0 前如需破坏性格式调整，会在本文记录迁移工具和受影响协议，不承诺无版本标识的永久兼容。
