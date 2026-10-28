"""Static evidence output for early/late improver plasticity comparisons."""

from __future__ import annotations

import html
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .meta_plasticity import MetaPlasticityExperiment, MetaPlasticityManifest


def write_meta_plasticity_manifest(manifest: MetaPlasticityManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    payload = asdict(manifest)
    payload["manifest_digest"] = manifest.digest
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _percent(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "未配置"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _threshold(value: float | None) -> str:
    return "未达到" if value is None else f"{value:g}"


def render_meta_plasticity_html(
    experiment: MetaPlasticityExperiment,
    *,
    scope_notice: str | None = None,
) -> str:
    summary = experiment.summary
    notice = scope_notice or (
        "这是直接构造快照的确定性本地文本样例，不是 LLM、真实 Agent 或真实 RSI 可塑性结论。"
        "样例刻意让晚期当前 Agent 更强、晚期改进器后继更弱，仅用于验证两类指标能够分离。"
        f"区间以轨迹为重采样单位；{summary['trajectory_count']} 条轨迹的小样本只作协议演示。"
    )
    ci = summary["trajectory_bootstrap_95_ci"]
    trajectory_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['trajectory_id'])}</td><td>{item['seed_count']}</td>"
        f"<td>{_percent(item['mean_current_agent_delta'], True)}</td>"
        f"<td class=\"{'positive' if item['mean_successor_delta'] >= 0 else 'negative'}\">{_percent(item['mean_successor_delta'], True)}</td>"
        f"<td>{_threshold(item['early_mean_evaluations_to_threshold_when_reached'])}</td>"
        f"<td>{_threshold(item['late_mean_evaluations_to_threshold_when_reached'])}</td>"
        "</tr>"
        for item in summary["per_trajectory"]
    )
    family_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td><td>{html.escape(item['role'])}</td>"
        f"<td>{_percent(item['early_mean_accuracy'])}</td><td>{_percent(item['late_mean_accuracy'])}</td>"
        f"<td>{_percent(item['early_mean_improvement_from_common_start'], True)}</td>"
        f"<td>{_percent(item['late_mean_improvement_from_common_start'], True)}</td>"
        f"<td>{_percent(item['mean_successor_delta_late_minus_early'], True)}</td>"
        "</tr>"
        for name, item in summary["family_summary"].items()
    ) or '<tr><td colspan="7">未配置额外任务族</td></tr>'
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merit · Meta-Plasticity</title>
<style>
:root{{--ink:#15263b;--muted:#64748b;--line:#dce4ec;--paper:#f5f7fa;--accent:#0f766e}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1200px;margin:auto;padding:42px 22px}}
h1{{margin:6px 0;font-size:34px}}h2{{margin-top:30px}}.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.1em}}.subtitle{{color:var(--muted)}}
.notice{{background:#fff7df;border:1px solid #ecd28a;border-radius:12px;padding:14px 18px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}}.label{{color:var(--muted);font-size:13px}}.value{{font-size:26px;font-weight:750}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{background:#eaf0f5;color:#475569;font-size:12px}}th:first-child,td:first-child{{text-align:left}}.positive{{color:#047857;font-weight:700}}.negative{{color:#b91c1c;font-weight:700}}footer{{margin-top:26px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<div class="eyebrow">MERIT META-PLASTICITY</div><h1>早期／晚期改进器可塑性</h1>
<p class="subtitle">当前 Agent 与改进器快照分开评测。早晚改进器从共同后继起点出发，使用相同后续任务、候选预算与 proposer 配置；差值均为晚期 − 早期。</p>
<div class="notice"><strong>边界：</strong>{html.escape(notice)}</div>
<section class="cards">
<div class="card"><div class="label">当前 Agent 差值（晚 − 早）</div><div class="value">{_percent(summary['mean_current_agent_delta_late_minus_early'], True)}</div></div>
<div class="card"><div class="label">后继质量差值（晚 − 早）</div><div class="value negative">{_percent(summary['mean_successor_delta_late_minus_early'], True)}</div></div>
<div class="card"><div class="label">轨迹层级 Bootstrap 95% CI</div><div class="value">[{ci[0]:+.2%}, {ci[1]:+.2%}]</div></div>
<div class="card"><div class="label">早期／晚期达开发阈值的平均评估数</div><div class="value">{_threshold(summary['early_mean_evaluations_to_threshold_when_reached'])} / {_threshold(summary['late_mean_evaluations_to_threshold_when_reached'])}</div></div>
</section>
<p>开发阈值：{summary['dev_threshold']:.0%}；早期／晚期达阈率：{_percent(summary['early_threshold_reached_rate'])} / {_percent(summary['late_threshold_reached_rate'])}。未达到预算内阈值的运行按右删失保留，不当作 0 次或任意填入预算值。</p>
<h2>逐轨迹结果</h2>
<div class="table-wrap"><table><thead><tr><th>轨迹</th><th>配对种子</th><th>当前 Agent 差值</th><th>后继质量差值</th><th>早期达阈评估</th><th>晚期达阈评估</th></tr></thead><tbody>{trajectory_rows}</tbody></table></div>
<h2>迁移与旧能力</h2>
<div class="table-wrap"><table><thead><tr><th>任务族</th><th>角色</th><th>早期后继</th><th>晚期后继</th><th>早期相对共同起点</th><th>晚期相对共同起点</th><th>后继差值（晚 − 早）</th></tr></thead><tbody>{family_rows}</tbody></table></div>
<h2>成本口径</h2>
<div class="card"><p>早期／晚期方法成本：${summary['early_method_cost_usd']:.4f} / ${summary['late_method_cost_usd']:.4f}。方法成本包含后续被评 Agent、proposer 和迁移后继复评。</p><p>共同起点复评、早晚当前 Agent 快照复评单独记录；全部观测成本 ${summary['total_observed_cost_usd']:.4f}。逐阶段 token、usage 和调用记录见 evidence.json。</p></div>
<footer>协议：{experiment.manifest.protocol_version} · 轨迹 {summary['trajectory_count']} 条 × 每条 {summary['seed_count_per_trajectory']} 个种子 · 每阶段候选预算 {summary['candidate_budget_per_phase']} · 统计单位是轨迹而不是全部种子。</footer>
</main></body></html>"""


def write_meta_plasticity_evidence(
    experiment: MetaPlasticityExperiment,
    output_dir: Path,
    *,
    limitations: list[str] | None = None,
    scope_notice: str | None = None,
) -> tuple[Path, Path]:
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
        "limitations": limitations or [
            "This local example constructs snapshots directly; it does not train a real improver or evaluate an LLM.",
            "Bootstrap intervals and sign-flip tests use trajectory-level means, not pooled seeds as independent replicates.",
            "Small deterministic trajectory samples demonstrate the protocol, not statistical power or real RSI plasticity.",
            "Threshold speed uses accepted development evaluations; unreached runs are right-censored.",
        ],
        "experiment": experiment.to_dict(),
    }
    evidence_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(
        render_meta_plasticity_html(experiment, scope_notice=scope_notice),
        encoding="utf-8",
    )
    return evidence_path, html_path
