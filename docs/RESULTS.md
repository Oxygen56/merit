# 实验结果与复现

本文记录仓库中已完成的运行。不同任务族不混合计算平均分；训练试点不当作最终提升；模拟响应只用于离线流程检查。

## 结果总览

| 实验 | 任务与规模 | 主要结果 | 结论边界 |
|---|---|---|---|
| 真实模型配对反思 | 2 个种子，每个 3 条训练、4 条测试 | 不修改 0%，随机修改 0%，模型反思 100% | 简单合成任务上的正例 |
| BFCL 改编子集 | 22 条训练、30 条验证、30 条最终测试，候选预算 3 | 不修改、模型反思、GEPA 均为 63.3%，随机修改为 60.0% | 没有最终测试提升，不是官方 BFCL 分数 |
| 多步售后工作流 | 2 个基础种子，各重复 2 次 | 最终准确率均为 100%，缺规则路径由 5 步降为 4 步 | 改善执行过程，没有答案分数提升 |
| 早晚改进器 | 2 条轨迹 × 2 个后续种子，共 4 组 | 早晚当前 Agent 和后继均为 100%，差值为 0 | 两条轨迹共享一个早期来源，不能据此判断 RSI 可塑性 |

## 真实模型基础实验

使用 `kimi-for-coding` 的小规模运行保留了模型原始回答和逐题结果。

| 实验 | 设置 | 结果 |
|---|---|---|
| LLM 配对反思 | 2 个种子；每个 3 条训练、4 条测试 | 不修改 0%，一次随机编辑 0%，一次模型反思候选 100% |
| LLM 迁移与回归 | 改名字段迁移和逐字保留源任务 | 平均迁移 0% → 16.7%，平均源任务 66.7% → 83.3% |
| 工具 Agent | 3 条训练、4 条测试；2 个候选；1 条轨迹 | 训练 66.7% → 100%，测试 100% → 75%；29 次任务执行中调用宿主工具 16 次 |
| Agent 迁移与回归 | 未见工具参数和直接算术 | 未见工具任务保持 100%，直接算术 66.7% → 100% |

结果文件：[LLM 配对报告](../results/real-llm/report.html)、[LLM 结果数据](../results/real-llm/evidence.json)、[工具 Agent 报告](../results/real-agent/report.html)、[工具调用记录](../results/real-agent/agent_traces.json)。

这些结果只覆盖对应任务，不能用于模型通用排名或开放式 RSI 结论。供应商 token 和耗时已记录；未配置 token 单价，因此美元成本未报告。

## BFCL 改编子集

基础配置使用 BFCL v4 单轮工具调用改编子集，比较函数名和必需参数。初始 0.5B 运行中，不修改、随机修改、模型反思和 GEPA 的平均测试准确率分别为 66.7%、68.3%、66.7% 和 66.7%。[基础报告](../results/ollama-langgraph-bfcl/report.html)

换用本机 `qwen2.5:7b` 后，四种方法分别为 81.7%、83.3%、81.7% 和 81.7%。两个种子的初始训练准确率均为 100%，GEPA 按 `skip_perfect_score` 跳过反思，因此这轮不能比较两个模型的候选质量。[7B 报告](../results/ollama-langgraph-gepa-qwen25-7b/report.html)

后续只把训练任务从 10 条增至 22 条，使初始训练准确率降为 81.8%。GEPA 开始生成候选，但平均测试准确率仍为 81.7%。一个接受的候选把训练从 18/22 提高到 21/22，同时完整复制一道训练题；测试修复与回归各 1 题，净变化为 0。[诊断报告](../results/ollama-langgraph-gepa-qwen25-7b-harder/report.html)

把反馈改成参数级 JSON 差异并加入防复制要求后，GEPA 平均训练准确率为 86.4%，测试仍为 81.7%。一个候选减少了复制，但没有形成测试净提升。[结构化反馈报告](../results/ollama-langgraph-gepa-qwen25-7b-structured-feedback/report.html)

固定 22 条训练、30 条验证和 30 条最终测试后，两个候选都把训练从 18/22 提高到 20/22，但验证从 20/30 降为 18/30，因而被拒绝。候选预算为 1 时，不修改、GEPA、随机修改和模型反思的最终准确率分别为 63.3%、63.3%、65.0% 和 63.3%。[独立验证报告](../results/ollama-langgraph-gepa-qwen25-7b-validation/report.html)

保持任务划分不变，把候选预算从 1 增至 3 后，没有 GEPA 候选超过初始验证成绩。最终结果为不修改 63.3%、模型反思 63.3%、GEPA 63.3%、随机修改 60.0%。更多搜索没有解决提案质量问题。[三候选配置](../examples/ollama_langgraph_gepa_qwen25_7b_validation_budget3.yaml)、[三候选报告](../results/ollama-langgraph-gepa-qwen25-7b-validation-budget3/report.html)

复现命令：

```bash
python -m pip install -e '.[langgraph,yaml,gepa]'
merit run examples/ollama_langgraph_gepa_qwen25_7b_validation_budget3.yaml \
  --output results/ollama-langgraph-gepa-qwen25-7b-validation-budget3
```

## 方法辨别与多步工具 Agent

8 个受控种子中，有效方法把准确率从 25% 提高到 100%，随机修改和失败文本反馈保持 25%，有害方法把另一组已有能力从 100% 降到 25%。[受控正例](../results/method-discrimination/effective/report.html)、[受控负例](../results/method-discrimination/harmful/report.html)

本机 7B 对照中，固定 schema 提示没有训练提升并被拒绝，最终保持 63.3%；强制空 JSON 的对照降为 0%。这些运行说明框架能区分正、零、负变化，不表示 GEPA 已在 BFCL 子集上取得提升。[本机方法辨别报告](../results/ollama-langgraph-method-discrimination-qwen25-7b/report.html)

本机 `qwen2.5:7b` 多步实验支持 `lookup`、`calculator` 和 `final`。2 个基础种子各重复 2 次，固定多步提示把 4 条合成任务的最终准确率从 0% 提高到 100%，空 JSON 方法反向运行则从 100% 降到 0%。固定提示在最终测试中平均使用 1297 个输入 token、98 个输出 token、9.60 秒，执行 8 个模型步骤、4 次查询和 2 次计算。提示是预设规则，不是 GEPA 学到的策略。[复现脚本](../examples/ollama_multi_step_discrimination.py)、[正例报告](../results/multi-step-method-discrimination/effective/report.html)

## 售后工作流与模型反思

带依赖调用的流程为工单 → 订单 → 退款规则 → 计算 → 回答。4 次重复运行中，预设工作流提示把最终准确率从 0% 提高到 100%，空 JSON 对照从 100% 降到 0%。每次最终测试平均使用 2766 个输入 token、147 个输出 token、15.31 秒。[复现脚本](../examples/ollama_support_workflow_discrimination.py)、[正例报告](../results/support-workflow-discrimination/effective/report.html)

异常分支混合正常退款、订单缺失和规则缺失。工具错误回传后，模型会在查询失败时返回 `REFUND_UNAVAILABLE`，不再猜测或计算。4 次运行从 0% 提高到 100%，空 JSON 对照从 100% 降到 0%。每次最终测试记录 12 个模型步骤、8 次查询、2 次预期错误和 1 次计算，平均使用 3810 个输入 token、170 个输出 token、20.32 秒。[异常分支脚本](../examples/ollama_support_error_workflow.py)、[异常分支正例](../results/support-error-workflow-discrimination/effective/report.html)

随后不再提供预设工作流，改由模型只根据训练失败提出修改：

1. **自由反思：**4 个提案都只生成 action JSON 示例，训练从 2/3 降为 0/3，全部被拒绝；不修改和模型反思最终均为 66.7%，随机修改平均 50.0%。[报告](../results/support-error-model-reflection/report.html)、[复算](../results/analysis/merit-support-reflection-analysis.json)
2. **结构重试：**第二个提案通过结构检查但训练仍为 2/3；轨迹显示它查询字面占位符 `order_key`，没有运行最终测试。[复算](../results/analysis/merit-support-reflection-retry-analysis.json)
3. **轨迹反馈：**三个提案仍缺少完整提示并保留占位符，候选评估和最终测试均为 0。[复算](../results/analysis/merit-support-trajectory-feedback-analysis.json)
4. **受限 JSON 修改：**候选修复正常退款，却在缺规则后反复查询直到步骤上限，训练保持 2/3，被拒绝。[复算](../results/analysis/merit-support-structured-patch-analysis.json)
5. **恢复规则：**第二个提案加入两条恢复规则，训练从 2/3 提高到 3/3，但必要记录缺失后仍有多余操作，没有运行最终测试。[复算](../results/analysis/merit-support-recovery-patch-analysis.json)
6. **停止规则：**候选修复缺订单分支，训练答案保持 3/3，但缺规则后仍执行计算，因此轨迹联合筛选拒绝候选。[复算](../results/analysis/merit-support-stop-patch-analysis.json)

## M6：动态反馈与重复训练

动态反馈使用最新被拒绝轨迹，并把必要记录失败后的停止规则放在提示首位。2 个基础种子各重复 2 次，4 个候选训练答案均为 3/3，正常、缺订单、缺规则路径分别为 5、3、4 步，全部通过轨迹联合筛选。训练阶段最终测试评估为 0。

复现脚本：[动态反馈](../examples/ollama_support_error_dynamic_patch.py)、[多种子重复](../examples/ollama_support_error_dynamic_repeated.py)。复算见 [M6／M7 复算结果](../results/analysis/merit-support-dynamic-m6-m7-analysis.json)。

## M7：独立验证、最终测试与迁移

训练、验证和最终测试各 3 条且互不重叠。4 次验证均接受候选，基线和后继最终准确率均为 100%，没有分数提升。后继在缺规则路径不再失败后计算，步骤从 5 降为 4。BFCL 改编子集单独报告，不与售后结果平均。

复现脚本：[最终评估](../examples/ollama_support_error_validation_final.py)。复算见 [M6／M7 复算结果](../results/analysis/merit-support-dynamic-m6-m7-analysis.json)。

## M8：真实早晚轨迹配对

M8 从停止规则试点取早期快照，从两个动态修改运行取晚期快照，在两个共同 BASE Agent 上各进行 1 次候选后续优化。4 组配对中，早晚当前 Agent 和后继准确率均为 100%，后继提示完全相同，差值为 0。两边迁移任务均从 50% 提高到 100%，源任务族均保持 50%。

两条轨迹共享同一个早期来源，不是两个独立早期优化重复。结果只用于小规模配对诊断，不能证明开放式 RSI 可塑性。

复现脚本：[M8 配对实验](../examples/ollama_support_meta_plasticity.py)。结果文件：[报告](../results/support-meta-plasticity-real/report.html)、[结果数据](../results/support-meta-plasticity-real/evidence.json)、[独立复算](../results/analysis/merit-support-meta-plasticity-analysis.json)。

```bash
python examples/ollama_support_meta_plasticity.py \
  --output results/support-meta-plasticity-real
```

## 复现与校验

默认离线验收不需要模型服务：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src examples tests
merit validate results/real-llm
merit validate results/real-agent
merit validate results/support-error-model-reflection
merit validate results/support-meta-plasticity-real
```

本机模型复现需要自行启动 Ollama，并准备脚本或 YAML 中指定的模型。结果目录中的成功响应缓存可以避免中断后重复请求；公开分享前应检查任务文本、模型输出和运行记录是否适合公开。
