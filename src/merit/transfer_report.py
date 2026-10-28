"""Static evidence report for transfer and regression matrices."""

from __future__ import annotations

import html
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .transfer import TransferExperiment, TransferManifest


def write_transfer_manifest(manifest: TransferManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    payload = asdict(manifest)
    payload["manifest_digest"] = manifest.digest
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def render_transfer_html(experiment: TransferExperiment) -> str:
    matrix_rows = "".join(
        "<tr>"
        f"<td>{html.escape(comparison.family)}</td>"
        f"<td>{html.escape(comparison.role)}</td>"
        f"<td>{comparison.baseline.accuracy:.2%}</td>"
        f"<td>{comparison.successor.accuracy:.2%}</td>"
        f"<td class=\"{'positive' if comparison.improvement >= 0 else 'negative'}\">{comparison.improvement:+.2%}</td>"
        f"<td>{comparison.baseline.usage.input_tokens + comparison.baseline.usage.output_tokens + comparison.successor.usage.input_tokens + comparison.successor.usage.output_tokens}</td>"
        "</tr>"
        for comparison in experiment.comparisons
    )
    detail_sections = []
    for comparison in experiment.comparisons:
        expected = {task.task_id: task.expected for task in comparison.tasks}
        baseline = {result.task_id: result.output for result in comparison.baseline.results}
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(result.task_id)}</td>"
            f"<td>{html.escape(baseline[result.task_id])}</td>"
            f"<td>{html.escape(result.output)}</td>"
            f"<td>{html.escape(expected[result.task_id])}</td>"
            f"<td>{'✓' if result.correct else '✗'}</td>"
            "</tr>"
            for result in comparison.successor.results
        )
        detail_sections.append(
            f"<h3>{html.escape(comparison.family)} · {html.escape(comparison.role)}</h3>"
            '<div class="table-wrap"><table><thead><tr><th>任务</th><th>起点输出</th><th>后继输出</th><th>期望输出</th><th>后继正确</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    summary = experiment.summary
    is_local = experiment.base.manifest.executor == "local_text_normalizer"
    if is_local:
        boundary = "<strong>边界：</strong>这是本地文本规范化器的契约样例，不是 LLM 或真实 Agent 的迁移结论。source 的逐字保留要求与 target 的规范化要求有意设置冲突，用于验证回归检测链路。"
    else:
        boundary = (
            f"<strong>来源：</strong>执行器 <code>{html.escape(experiment.base.manifest.executor)}</code>。"
            "迁移与回归结果仅适用于本次任务族；usage 完整性和实验限制见 evidence.json。"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merit · Transfer and Regression</title>
<style>
:root{{--ink:#15263b;--muted:#64748b;--line:#dce4ec;--paper:#f5f7fa;--accent:#0f766e}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 22px}}
h1{{margin:6px 0;font-size:34px}}h2{{margin-top:30px}}h3{{margin-top:26px}}.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.1em}}.subtitle{{color:var(--muted)}}
.notice{{background:#fff7df;border:1px solid #ecd28a;border-radius:12px;padding:14px 18px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}}.label{{color:var(--muted);font-size:13px}}.value{{font-size:26px;font-weight:750}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:#eaf0f5;color:#475569;font-size:12px}}.positive{{color:#047857;font-weight:700}}.negative{{color:#b91c1c;font-weight:700}}footer{{margin-top:26px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<div class="eyebrow">MERIT TRANSFER MATRIX</div><h1>迁移与旧能力回归</h1>
<p class="subtitle">对同一个起点 artifact 和后继 artifact 逐任务族复评。改善值 = 后继准确率 − 起点准确率；source 为旧能力，target 为优化目标，transfer 为未参与优化的新任务族。</p>
<div class="notice">{boundary}</div>
<section class="cards">
<div class="card"><div class="label">目标域提升</div><div class="value positive">{summary['target_improvement']:+.2%}</div></div>
<div class="card"><div class="label">迁移任务族提升</div><div class="value positive">{summary['mean_transfer_improvement']:+.2%}</div></div>
<div class="card"><div class="label">旧能力最大回归</div><div class="value negative">{summary['worst_source_accuracy_delta']:+.2%}</div></div>
<div class="card"><div class="label">新增 Agent 调用</div><div class="value">{summary['additional_agent_invocations']}</div></div>
</section>
<h2>任务族矩阵</h2>
<div class="table-wrap"><table><thead><tr><th>任务族</th><th>角色</th><th>起点准确率</th><th>后继准确率</th><th>改善值</th><th>token 代理</th></tr></thead>
<tbody>{matrix_rows}</tbody></table></div>
<h2>逐任务证据</h2>{''.join(detail_sections)}
<footer>协议：{experiment.manifest.protocol_version} · 起点 artifact <code>{experiment.base.start.artifact_id[:12]}</code> · 后继 artifact <code>{experiment.base.successor.artifact_id[:12]}</code> · 完整数据见 evidence.json。</footer>
</main></body></html>"""


def write_transfer_evidence(
    experiment: TransferExperiment,
    output_dir: Path,
    limitations: list[str] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    html_path = output_dir / "report.html"
    if limitations is None:
        limitations = [
            "This is a deterministic local executor example, not an LLM transfer result.",
            "The source family intentionally conflicts with the target normalization behavior to test regression detection.",
            "Token counts are local proxies; cost and latency are not real provider measurements.",
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
    html_path.write_text(render_transfer_html(experiment), encoding="utf-8")
    return evidence_path, html_path
