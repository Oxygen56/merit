"""Evidence files for the local agent-contract experiment."""

from __future__ import annotations

import html
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent_contract import AgentContractExperiment, AgentContractManifest


def write_agent_manifest(manifest: AgentContractManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    payload = asdict(manifest)
    payload["manifest_digest"] = manifest.digest
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def render_agent_html(experiment: AgentContractExperiment) -> str:
    summary = experiment.summary
    history_rows = "".join(
        "<tr>"
        f"<td>{entry.iteration}</td>"
        f"<td><code>{html.escape(entry.candidate.artifact_id[:12])}</code></td>"
        f"<td>{entry.evaluation.accuracy:.2%}</td>"
        f"<td>{'是' if entry.accepted else '否'}</td>"
        f"<td>{entry.evaluation.usage.input_tokens + entry.evaluation.usage.output_tokens}</td>"
        f"<td>{entry.remaining_budget}</td>"
        "</tr>"
        for entry in experiment.history
    )
    heldout_by_id = {task.task_id: task for task in experiment.heldout_tasks}
    baseline_outputs = {result.task_id: result for result in experiment.baseline_heldout.results}
    heldout_rows = "".join(
        "<tr>"
        f"<td>{html.escape(result.task_id)}</td>"
        f"<td>{html.escape(heldout_by_id[result.task_id].prompt)}</td>"
        f"<td>{html.escape(baseline_outputs[result.task_id].output)}</td>"
        f"<td>{html.escape(result.output)}</td>"
        f"<td>{html.escape(heldout_by_id[result.task_id].expected)}</td>"
        f"<td>{'✓' if result.correct else '✗'}</td>"
        "</tr>"
        for result in experiment.successor_heldout.results
    )
    traces = summary.get("tool_traces", [])
    trace_rows = "".join(
        "<tr>"
        f"<td>{html.escape(trace['task_id'])}</td>"
        f"<td><code>{html.escape(trace['artifact_id'][:12])}</code></td>"
        f"<td>{html.escape(trace['raw_model_output'])}</td>"
        f"<td>{html.escape(trace['tool_name'] or '未调用')}</td>"
        f"<td>{html.escape(trace['tool_argument'] or '')}</td>"
        f"<td>{html.escape(trace['tool_result'] or trace['tool_error'] or '')}</td>"
        f"<td>{html.escape(trace['final_output'])}</td>"
        "</tr>"
        for trace in traces
    )
    tool_cards = ""
    tool_section = ""
    if traces:
        tool_cards = (
            f"<div class=\"card\"><div class=\"label\">实际工具调用</div><div class=\"value\">{summary['tool_invocations']}</div></div>"
            f"<div class=\"card\"><div class=\"label\">Agent + proposer 累计延迟</div><div class=\"value\">{summary['combined_total_latency_ms'] / 1000:.1f}s</div></div>"
        )
        tool_section = (
            "<h2>真实工具调用轨迹</h2><p class=\"muted\">包含训练、隐藏复评与迁移阶段；原始模型输出未经清洗，空输出原样保留。</p>"
            '<div class="table-wrap"><table><thead><tr><th>任务</th><th>Artifact</th><th>模型原始输出</th><th>工具</th><th>参数</th><th>工具结果／错误</th><th>最终输出</th></tr></thead>'
            f"<tbody>{trace_rows}</tbody></table></div>"
        )
    is_local = experiment.manifest.executor == "local_text_normalizer"
    heading = "本地 Agent 契约样例" if is_local else "Agent 执行实验"
    if is_local:
        boundary = "<strong>边界：</strong>执行器是确定性的本地文本规范化器，不是 LLM，也没有调用外部模型。token 是空白分词的本地代理计数，成本为 0；这里验证的是接口与证据链，不是模型能力。"
    else:
        boundary = (
            f"<strong>来源：</strong>执行器 <code>{html.escape(experiment.manifest.executor)}</code>。"
            "结果仅适用于本次任务、artifact 与执行环境；usage 完整性和实验限制见 evidence.json。"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merit · Agent Contract</title>
<style>
:root{{--ink:#15263b;--muted:#64748b;--line:#dce4ec;--paper:#f5f7fa;--accent:#0f766e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 22px}}h1{{margin:6px 0;font-size:34px}}h2{{margin-top:30px}}.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.1em}}
.subtitle,.muted{{color:var(--muted)}}.notice{{background:#fff7df;border:1px solid #ecd28a;border-radius:12px;padding:14px 18px;margin:20px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}}
.label{{color:var(--muted);font-size:13px}}.value{{font-size:26px;font-weight:750}}.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}
table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#eaf0f5;color:#475569;font-size:12px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}footer{{margin-top:26px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<div class="eyebrow">MERIT AGENT CONTRACT</div><h1>{heading}</h1>
<p class="subtitle">同一份 artifact 先只在训练任务上改进，再对起点与后继 artifact 分别执行隐藏任务复评，并汇总调用量、token 与成本。</p>
<div class="notice">{boundary}</div>
<section class="cards">
<div class="card"><div class="label">起点隐藏集准确率</div><div class="value">{summary['baseline_heldout_accuracy']:.2%}</div></div>
<div class="card"><div class="label">后继隐藏集准确率</div><div class="value">{summary['successor_heldout_accuracy']:.2%}</div></div>
<div class="card"><div class="label">候选评估 / 采纳</div><div class="value">{summary['candidate_evaluations']} / {summary['accepted_candidates']}</div></div>
<div class="card"><div class="label">总 token</div><div class="value">{summary['total_input_tokens'] + summary['total_output_tokens']}</div></div>
<div class="card"><div class="label">总成本</div><div class="value">{f"${summary['total_cost_usd']:.2f}" if summary['total_cost_reported'] else '未完整记录'}</div></div>
{tool_cards}
</section>
<h2>Artifact 与执行边界</h2>
<div class="card">
<p><strong>执行器：</strong><code>{html.escape(experiment.manifest.executor)}</code></p>
<p><strong>起点：</strong><code>{experiment.start.artifact_id}</code></p>
<p><strong>后继：</strong><code>{experiment.successor.artifact_id}</code></p>
<p><strong>后继配置：</strong><code>{html.escape(json.dumps(experiment.successor.config, ensure_ascii=False, sort_keys=True))}</code></p>
<p><strong>开发 / 隐藏执行次数：</strong>{summary['dev_agent_invocations']} / {summary['heldout_agent_invocations']}</p>
<p><strong>Manifest digest：</strong><code>{experiment.manifest.digest}</code></p>
</div>
<h2>训练集候选历史</h2>
<div class="table-wrap"><table><thead><tr><th>迭代</th><th>候选 artifact</th><th>训练准确率</th><th>采纳</th><th>token</th><th>剩余预算</th></tr></thead>
<tbody>{history_rows}</tbody></table></div>
<h2>隐藏集逐任务复评</h2>
<div class="table-wrap"><table><thead><tr><th>任务</th><th>输入</th><th>起点输出</th><th>后继输出</th><th>期望输出</th><th>后继正确</th></tr></thead>
<tbody>{heldout_rows}</tbody></table></div>
{tool_section}
<footer>协议：{experiment.manifest.protocol_version} · 完整任务定义、逐任务输出、usage、候选历史与 lineage 见 evidence.json。</footer>
</main></body></html>"""


def write_agent_evidence(
    experiment: AgentContractExperiment,
    output_dir: Path,
    limitations: list[str] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    html_path = output_dir / "report.html"
    if limitations is None:
        limitations = [
            "The executor is a deterministic local text normalizer, not an LLM or external agent.",
            "Token counts are whitespace-based local proxies, not provider billing records.",
            "Latency is zero-filled by this local executor; no wall-clock benchmark is claimed.",
            "This run validates task, artifact, budget, execution and cost contracts only.",
        ]
    payload = {
        "schema_version": "merit-evidence-v1",
        "project": "Merit",
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "manifest_digest": experiment.manifest.digest,
        "limitations": limitations,
        "experiment": experiment.to_dict(),
    }
    evidence_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_agent_html(experiment), encoding="utf-8")
    return evidence_path, html_path
