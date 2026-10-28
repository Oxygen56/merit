"""Human-readable HTML and machine-readable JSON evidence output."""

from __future__ import annotations

import html
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .model import Experiment, ExperimentManifest


def write_manifest(manifest: ExperimentManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    payload = asdict(manifest)
    payload["manifest_digest"] = manifest.digest
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def render_html(experiment: Experiment) -> str:
    summary = experiment.summary
    manifest = experiment.manifest
    rows = []
    for run in experiment.runs:
        delta_class = "positive" if run.heldout_rmse_improvement > 0 else "negative"
        controls = {arm.optimizer: arm for arm in run.controls}
        control_cells = "".join(
            f"<td>{_fmt(controls[name].heldout_rmse)}</td>" for name in manifest.controls
        )
        rows.append(
            "<tr>"
            f"<td>{run.seed}</td>"
            f"<td><code>{run.start_artifact_id[:12]}</code></td>"
            f"<td>{_fmt(run.baseline.heldout_rmse)}</td>"
            f"<td>{_fmt(run.challenger.heldout_rmse)}</td>"
            f"{control_cells}"
            f'<td class="{delta_class}">{_fmt(run.heldout_rmse_improvement)}</td>'
            f"<td>{run.baseline.accepted_candidates}/{run.baseline.candidate_evaluations}</td>"
            f"<td>{run.challenger.accepted_candidates}/{run.challenger.candidate_evaluations}</td>"
            "</tr>"
        )
    control_headers = "".join(f"<th>{html.escape(name)} RMSE</th>" for name in manifest.controls)
    control_descriptions = {
        "no_change": "保留起点",
        "random_edit": "随机编辑并无条件采纳",
        "memory_reset": "每轮从共同起点重新提案，按开发集贪心保留最佳候选；这是搜索状态重置，不是真实记忆模块",
    }
    control_summary = "".join(
        f"<dt>对照：{html.escape(name)}</dt>"
        f"<dd>{html.escape(control_descriptions.get(name, '独立对照'))}；"
        f"平均 RMSE {_fmt(summary['controls'][name]['mean_heldout_rmse'])}；"
        f"平均候选评估 {summary['controls'][name]['mean_candidate_evaluations']:g} 次</dd>"
        for name in manifest.controls
    )
    ci = summary["paired_bootstrap_95_ci"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merit · {html.escape(manifest.experiment)}</title>
<style>
:root{{--ink:#15263b;--muted:#64748b;--line:#dce4ec;--paper:#f5f7fa;--card:#fff;--accent:#0f766e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:48px 24px}}header{{margin-bottom:28px}}.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase}}
h1{{font-size:36px;line-height:1.2;margin:8px 0}}h2{{margin-top:32px}}.subtitle{{color:var(--muted);max-width:760px;font-size:17px}}
.notice{{background:#fff7df;border:1px solid #ecd28a;border-radius:12px;padding:14px 18px;margin:22px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.label{{color:var(--muted);font-size:13px}}.value{{font-size:25px;font-weight:750;margin-top:5px}}table{{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#eaf0f5;color:#475569;font-size:12px}}th:nth-child(1),th:nth-child(2),td:nth-child(1),td:nth-child(2){{text-align:left}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}.positive{{color:#047857;font-weight:700}}.negative{{color:#b91c1c;font-weight:700}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}footer{{margin-top:30px;color:var(--muted);font-size:13px}}dt{{color:var(--muted);float:left;clear:left;width:210px}}dd{{margin-left:230px;margin-bottom:6px}}
@media(max-width:640px){{main{{padding:28px 14px}}h1{{font-size:28px}}dt{{float:none;width:auto}}dd{{margin-left:0;margin-bottom:12px}}}}
</style>
</head>
<body><main>
<header><div class="eyebrow">Merit Evidence Card</div><h1>{html.escape(manifest.experiment)}</h1>
<p class="subtitle">同一起点、相同优化候选预算、按种子配对的改进方法对照。评价指标为 held-out RMSE，改善值 = 基线 RMSE − 挑战者 RMSE，正值表示挑战者更好。</p></header>
<div class="notice"><strong>边界：</strong>这是确定性线性玩具任务，只用于验证 Merit 的实验协议和报告链路，不是 LLM、真实 Agent 或 RSI 的效果结论。</div>
<section class="cards">
<div class="card"><div class="label">平均 held-out RMSE 改善</div><div class="value">{_fmt(summary['mean_heldout_rmse_improvement'])}</div></div>
<div class="card"><div class="label">配对 Bootstrap 95% CI</div><div class="value">[{_fmt(ci[0])}, {_fmt(ci[1])}]</div></div>
<div class="card"><div class="label">优化臂候选预算</div><div class="value">{manifest.candidate_budget}</div></div>
<div class="card"><div class="label">配对种子数</div><div class="value">{len(experiment.runs)}</div></div>
</section>
<h2>结果摘要</h2>
<div class="card"><dl>
<dt>基线</dt><dd>{html.escape(manifest.baseline)}，平均 RMSE {_fmt(summary['mean_baseline_heldout_rmse'])}</dd>
<dt>挑战者</dt><dd>{html.escape(manifest.challenger)}，平均 RMSE {_fmt(summary['mean_challenger_heldout_rmse'])}</dd>
{control_summary}
<dt>单侧符号翻转检验</dt><dd>p = {_fmt(summary['one_sided_sign_flip_p_value'])}；探索性参考，不替代正式功效分析</dd>
<dt>Manifest digest</dt><dd><code>{manifest.digest}</code></dd>
</dl></div>
<h2>逐种子结果</h2>
<div class="table-wrap"><table><thead><tr><th>Seed</th><th>共同起点</th><th>基线 RMSE</th><th>挑战者 RMSE</th>{control_headers}<th>改善值</th><th>基线采纳</th><th>挑战者采纳</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<footer>协议：{manifest.protocol_version} · no-change 对照不消耗候选评估；所有候选历史、产物 lineage、任务摘要和环境信息见 evidence.json。</footer>
</main></body></html>"""


def write_evidence(experiment: Experiment, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    html_path = output_dir / "report.html"
    payload = {
        "schema_version": "merit-evidence-v1",
        "project": "Merit",
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "manifest_digest": experiment.manifest.digest,
        "limitations": [
            "Toy linear tasks only; no LLM or real-agent claim.",
            "The percentile bootstrap and sign-flip test operate on paired per-seed differences.",
            "A small seed count is for protocol validation, not a powered research conclusion.",
        ],
        "experiment": experiment.to_dict(),
    }
    evidence_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(experiment), encoding="utf-8")
    return evidence_path, html_path
