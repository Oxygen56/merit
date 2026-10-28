"""Evidence output for multi-seed agent improvement comparisons."""

from __future__ import annotations

import html
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent_pair import AgentPairedExperiment, AgentPairedManifest


def write_agent_pair_manifest(manifest: AgentPairedManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    payload = asdict(manifest)
    payload["manifest_digest"] = manifest.digest
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def render_agent_pair_html(experiment: AgentPairedExperiment) -> str:
    summary = experiment.summary
    control_headers = "".join(f"<th>{html.escape(name)}</th>" for name in experiment.manifest.controls)
    rows = []
    for run in experiment.runs:
        controls = {arm.manifest.optimizer: arm for arm in run.controls}
        control_cells = "".join(
            f"<td>{controls[name].successor_heldout.accuracy:.2%}</td>"
            for name in experiment.manifest.controls
        )
        rows.append(
            "<tr>"
            f"<td>{run.seed}</td>"
            f"<td><code>{run.start_artifact_id[:12]}</code></td>"
            f"<td>{run.baseline.successor_heldout.accuracy:.2%}</td>"
            f"<td>{run.challenger.successor_heldout.accuracy:.2%}</td>"
            f"{control_cells}"
            f"<td class=\"{'positive' if run.heldout_accuracy_improvement >= 0 else 'negative'}\">{run.heldout_accuracy_improvement:+.2%}</td>"
            f"<td>{run.baseline.summary['candidate_evaluations']}</td>"
            f"<td>{run.challenger.summary['candidate_evaluations']}</td>"
            "</tr>"
        )
    control_summary = "".join(
        f"<p><strong>{html.escape(name)}</strong>：平均准确率 {data['mean_heldout_accuracy']:.2%}；"
        f"平均候选评估 {data['mean_candidate_evaluations']:g}；proposer 成本 ${data['total_proposer_cost_usd']:.2f}</p>"
        for name, data in summary["controls"].items()
    )
    ci = summary["paired_bootstrap_95_ci"]
    is_local = experiment.manifest.executor == "local_text_normalizer"
    if is_local:
        boundary = "<strong>边界：</strong>这是确定性本地文本 Agent 的协议样例，不是 LLM 实验，也不代表外部 Agent 的效果。失败反馈是启发式规则，不是模型反思；本地执行器的行为由 config 决定，因此仅修改 instructions 的对照不变是预期现象，不能据此判断 prompt 方法是否有效。"
    else:
        boundary = (
            f"<strong>来源：</strong>执行器 <code>{html.escape(experiment.manifest.executor)}</code>。"
            "结果仅适用于本次配对任务与预算；usage 完整性和实验限制见 evidence.json。"
        )
    execution = summary.get("execution", {})
    execution_html = ""
    if execution:
        source = execution.get("dataset_source", {})
        execution_html = (
            f"<div class=\"notice\"><strong>运行环境：</strong>模型 <code>{html.escape(execution['model'])}</code>；"
            f"地址 <code>{html.escape(execution['endpoint'])}</code>；付费 API 调用 {execution['paid_api_calls']} 次。<br>"
            f"本次命令网络请求 {execution['network_calls_this_run']} 次，缓存命中 {execution['cache_hits_this_run']} 次；"
            f"结果目录共有 {execution['cached_response_files']} 份缓存响应。续跑次数分别记录，不把单次进程的请求数当成实验总数。<br>"
            f"<strong>数据与评分：</strong>{html.escape(source.get('name', 'custom dataset'))}；"
            f"{html.escape(source.get('metric_notice', '见结果文件中的任务定义。'))}</div>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merit · Paired Agent Improvement</title>
<style>
:root{{--ink:#15263b;--muted:#64748b;--line:#dce4ec;--paper:#f5f7fa;--accent:#0f766e}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 22px}}
h1{{margin:6px 0;font-size:34px}}h2{{margin-top:28px}}.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.1em}}.subtitle{{color:var(--muted)}}.notice{{background:#fff7df;border:1px solid #ecd28a;border-radius:12px;padding:14px 18px;margin:20px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px}}.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}}.label{{color:var(--muted);font-size:13px}}.value{{font-size:25px;font-weight:750}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;margin-top:26px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#eaf0f5;color:#475569;font-size:12px}}th:nth-child(1),th:nth-child(2),td:nth-child(1),td:nth-child(2){{text-align:left}}
.positive{{color:#047857;font-weight:700}}.negative{{color:#b91c1c;font-weight:700}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}footer{{margin-top:26px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<div class="eyebrow">MERIT PAIRED AGENT EVALUATION</div><h1>多种子 Agent 改进对照</h1>
<p class="subtitle">每个种子内，所有方法使用同一起点、相同训练／隐藏任务；主动优化臂使用相同候选预算。改善值 = 挑战者后继准确率 − 基线后继准确率。</p>
<div class="notice">{boundary}</div>
{execution_html}
<section class="cards">
<div class="card"><div class="label">基线平均隐藏集准确率</div><div class="value">{summary['mean_baseline_heldout_accuracy']:.2%}</div></div>
<div class="card"><div class="label">挑战者平均隐藏集准确率</div><div class="value">{summary['mean_challenger_heldout_accuracy']:.2%}</div></div>
<div class="card"><div class="label">平均改善值</div><div class="value positive">{summary['mean_heldout_accuracy_improvement']:+.2%}</div></div>
<div class="card"><div class="label">配对 Bootstrap 95% CI</div><div class="value">[{ci[0]:+.2%}, {ci[1]:+.2%}]</div></div>
<div class="card"><div class="label">主动优化臂候选预算</div><div class="value">{summary['candidate_budget_per_arm']}</div></div>
</section>
<h2>消融对照</h2><div class="card">{control_summary or '<p>未注册对照臂</p>'}</div>
<div class="table-wrap"><table><thead><tr><th>Seed</th><th>共同起点</th><th>基线后继</th><th>挑战者后继</th>{control_headers}<th>改善值</th><th>基线候选</th><th>挑战者候选</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<footer>协议：{experiment.manifest.protocol_version} · no-change 不消耗候选预算 · 基线／挑战者 proposer 成本：${summary['baseline_proposer_cost_usd']:.2f} / ${summary['challenger_proposer_cost_usd']:.2f} · 被评 Agent 成本与 proposer 成本分账保存在 evidence.json。</footer>
</main></body></html>"""


def write_agent_pair_evidence(
    experiment: AgentPairedExperiment,
    output_dir: Path,
    limitations: list[str] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    html_path = output_dir / "report.html"
    if limitations is None:
        limitations = [
            "Deterministic local text executor only; no LLM or real-agent performance claim.",
            "The percentile bootstrap and sign-flip test operate on paired per-seed accuracy differences.",
            "The small deterministic example validates protocol behavior, not statistical power.",
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
    html_path.write_text(render_agent_pair_html(experiment), encoding="utf-8")
    return evidence_path, html_path
