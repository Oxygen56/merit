"""Merit command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .agent_demo import LocalTextNormalizer, make_local_agent_manifest, run_local_agent_demo
from .agent_pair_demo import make_local_agent_paired_manifest, run_local_agent_paired_demo
from .agent_pair_report import write_agent_pair_evidence, write_agent_pair_manifest
from .agent_report import write_agent_evidence, write_agent_manifest
from .demo import default_manifest, run_toy_experiment
from .evidence import compress_evidence, read_evidence, validate_result_directory
from .experiment import run_config
from .meta_plasticity_demo import make_local_meta_plasticity_manifest, run_local_meta_plasticity_demo
from .meta_plasticity_report import write_meta_plasticity_evidence, write_meta_plasticity_manifest
from .report import write_evidence, write_manifest
from .transfer import run_transfer_evaluation
from .transfer_demo import make_local_task_families, make_local_transfer_manifest
from .transfer_report import write_transfer_evidence, write_transfer_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merit", description="Controlled comparisons for agent improvement methods")
    parser.add_argument("--version", action="version", version=f"Merit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the deterministic toy paired experiment")
    demo.add_argument("--output", type=Path, default=Path("results/toy-comparison"))
    demo.add_argument("--seed-count", type=int, default=12)
    demo.add_argument("--budget", type=int, default=64)
    agent_demo = subparsers.add_parser("agent-demo", help="Validate agent contracts with a local executor")
    agent_demo.add_argument("--output", type=Path, default=Path("results/local-agent-contract"))
    agent_demo.add_argument("--budget", type=int, default=4)
    transfer_demo = subparsers.add_parser("transfer-demo", help="Run local transfer and regression checks")
    transfer_demo.add_argument("--output", type=Path, default=Path("results/transfer-regression"))
    transfer_demo.add_argument("--budget", type=int, default=4)
    pair_demo = subparsers.add_parser("paired-agent-demo", help="Run a multi-seed local agent comparison")
    pair_demo.add_argument("--output", type=Path, default=Path("results/agent-paired"))
    pair_demo.add_argument("--seed-count", type=int, default=8)
    pair_demo.add_argument("--budget", type=int, default=4)
    plasticity_demo = subparsers.add_parser("plasticity-demo", help="Compare early and late improver snapshots")
    plasticity_demo.add_argument("--output", type=Path, default=Path("results/meta-plasticity"))
    plasticity_demo.add_argument("--trajectory-count", type=int, default=4)
    plasticity_demo.add_argument("--seed-count", type=int, default=4)
    plasticity_demo.add_argument("--budget", type=int, default=4)
    plasticity_demo.add_argument("--dev-threshold", type=float, default=1.0)
    validate = subparsers.add_parser("validate", help="Validate a result manifest and evidence envelope")
    validate.add_argument("path", type=Path)
    compress = subparsers.add_parser("compress-evidence", help="Validate and gzip an evidence JSON file")
    compress.add_argument("path", type=Path)
    compress.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run", help="Run a configured local Ollama experiment")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--max-network-calls", type=int)
    report = subparsers.add_parser("report", help="Validate and summarize a saved experiment report")
    report.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        output = args.output or Path("results") / args.config.stem
        result = run_config(args.config, output, args.max_network_calls)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    if args.command == "report":
        output = args.path.resolve()
        validate_result_directory(output)
        evidence_path = output / "evidence.json"
        if not evidence_path.exists():
            evidence_path = output / "evidence.json.gz"
        evidence = read_evidence(evidence_path)
        print(f"Report: {output / 'report.html'}")
        print(json.dumps(evidence["experiment"].get("summary", {}), indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        metadata = validate_result_directory(args.path)
        print(f"Valid Merit result: {args.path.resolve()}")
        print(f"Schema:   {metadata['schema_version']}")
        print(f"Protocol: {metadata['protocol_version']}")
        print(f"Manifest: {metadata['manifest_digest']}")
        return 0
    if args.command == "compress-evidence":
        output = compress_evidence(args.path, args.output)
        print(f"Compressed evidence: {output.resolve()}")
        return 0
    if args.command == "demo":
        manifest = default_manifest(seed_count=args.seed_count, candidate_budget=args.budget)
        output_dir = args.output.resolve()
        manifest_path = write_manifest(manifest, output_dir)
        experiment = run_toy_experiment(manifest)
        evidence_path, html_path = write_evidence(experiment, output_dir)
        summary = experiment.summary
        print(f"Manifest:  {manifest_path}")
        print(f"Evidence:  {evidence_path}")
        print(f"Report:    {html_path}")
        print(f"Mean held-out RMSE improvement: {summary['mean_heldout_rmse_improvement']:.6f}")
        print(f"Paired 95% CI: {summary['paired_bootstrap_95_ci']}")
        return 0
    if args.command == "agent-demo":
        output_dir = args.output.resolve()
        manifest = make_local_agent_manifest(candidate_budget=args.budget)
        manifest_path = write_agent_manifest(manifest, output_dir)
        experiment = run_local_agent_demo(candidate_budget=args.budget)
        if experiment.manifest != manifest:
            raise RuntimeError("Agent manifest changed before execution")
        evidence_path, html_path = write_agent_evidence(experiment, output_dir)
        summary = experiment.summary
        print(f"Manifest:  {manifest_path}")
        print(f"Evidence:  {evidence_path}")
        print(f"Report:    {html_path}")
        print(f"Baseline held-out accuracy: {summary['baseline_heldout_accuracy']:.2%}")
        print(f"Successor held-out accuracy: {summary['successor_heldout_accuracy']:.2%}")
        print(f"Candidate evaluations: {summary['candidate_evaluations']}")
        print(f"Total cost USD: {summary['total_cost_usd']:.2f}")
        return 0
    if args.command == "transfer-demo":
        output_dir = args.output.resolve()
        base = run_local_agent_demo(candidate_budget=args.budget)
        families = make_local_task_families(base)
        manifest = make_local_transfer_manifest(base)
        manifest_path = write_transfer_manifest(manifest, output_dir)
        experiment = run_transfer_evaluation(manifest, base, LocalTextNormalizer(), families)
        evidence_path, html_path = write_transfer_evidence(experiment, output_dir)
        summary = experiment.summary
        print(f"Manifest:  {manifest_path}")
        print(f"Evidence:  {evidence_path}")
        print(f"Report:    {html_path}")
        print(f"Target improvement: {summary['target_improvement']:+.2%}")
        print(f"Transfer improvement: {summary['mean_transfer_improvement']:+.2%}")
        print(f"Worst source accuracy delta: {summary['worst_source_accuracy_delta']:+.2%}")
        return 0
    if args.command == "paired-agent-demo":
        output_dir = args.output.resolve()
        manifest = make_local_agent_paired_manifest(args.seed_count, args.budget)
        manifest_path = write_agent_pair_manifest(manifest, output_dir)
        experiment = run_local_agent_paired_demo(args.seed_count, args.budget)
        if experiment.manifest != manifest:
            raise RuntimeError("Agent paired manifest changed before execution")
        evidence_path, html_path = write_agent_pair_evidence(experiment, output_dir)
        summary = experiment.summary
        print(f"Manifest:  {manifest_path}")
        print(f"Evidence:  {evidence_path}")
        print(f"Report:    {html_path}")
        print(f"Mean held-out accuracy improvement: {summary['mean_heldout_accuracy_improvement']:+.2%}")
        print(f"Paired 95% CI: {summary['paired_bootstrap_95_ci']}")
        return 0
    if args.command == "plasticity-demo":
        output_dir = args.output.resolve()
        manifest = make_local_meta_plasticity_manifest(
            args.trajectory_count,
            args.seed_count,
            args.budget,
            args.dev_threshold,
        )
        manifest_path = write_meta_plasticity_manifest(manifest, output_dir)
        experiment = run_local_meta_plasticity_demo(
            args.trajectory_count,
            args.seed_count,
            args.budget,
            args.dev_threshold,
        )
        if experiment.manifest != manifest:
            raise RuntimeError("Meta-Plasticity manifest changed before execution")
        evidence_path, html_path = write_meta_plasticity_evidence(experiment, output_dir)
        summary = experiment.summary
        print(f"Manifest:  {manifest_path}")
        print(f"Evidence:  {evidence_path}")
        print(f"Report:    {html_path}")
        print(f"Current-agent delta (late - early): {summary['mean_current_agent_delta_late_minus_early']:+.2%}")
        print(f"Successor delta (late - early): {summary['mean_successor_delta_late_minus_early']:+.2%}")
        print(f"Trajectory 95% CI: {summary['trajectory_bootstrap_95_ci']}")
        return 0
    return 2
