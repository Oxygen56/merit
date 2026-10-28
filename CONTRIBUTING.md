# 贡献指南

## 开发原则

Merit 优先保证实验可复核，而不是堆叠 Agent 框架适配器。提交改动时请保持：

- 同起点、同任务、可比较预算；
- 训练／隐藏任务隔离；
- 方法与随机对照分开；
- 被评 Agent、proposer、迁移复评成本分开；
- 负结果与限制原样保留；
- 默认离线，不自动调用付费服务。

## 本地开发

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests examples
```

新增实验命令至少验证一次小规模运行，并检查 `manifest.json`、`evidence.json`、`report.html` 三者一致。所有公开示例必须与当前 API 同步。

## 修改协议时

1. 先添加会暴露问题的测试，再改协议；
2. 不静默改变已有指标、digest 或统计单位；
3. 需要破坏性变更时增加 protocol／schema 版本；
4. 更新 `docs/PROTOCOL.md` 的迁移步骤；
5. 保留 artifact lineage、任务 digest、逐任务输出和 usage；
6. 不用模拟数据宣称模型效果、延迟或成本结论。

## 测试要求

- 新功能需要正常路径和至少一个失败／篡改路径；
- 外部服务调用不得进入默认测试；使用本地 mock；
- 统计测试固定种子，避免随机失败；
- 测试不得读取用户的密钥或依赖本机私有路径；
- 文档中的离线示例应实际运行验证。

## 报告问题

问题描述请包含最小复现、Merit 版本、Python 版本、协议版本和错误文本。不要提交 API key、私有任务内容、个人数据或未脱敏实验记录文件。
