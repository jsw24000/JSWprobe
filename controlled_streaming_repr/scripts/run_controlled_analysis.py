#!/usr/bin/env python3
"""Run pluggable controlled analyses over saved token bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from controlled_common import (
    PROJECT_DIR,
    add_common_filters,
    filtered_manifest_records,
    load_config,
    load_json,
    output_root,
    project_name,
    resolve_index_path,
    resolve_manifest_path,
    save_json,
    save_resolved_config,
    setup_logging,
)

from adapters.token_schema import load_token_bundle
from analysis_methods.base_analysis import AnalysisRun
from analysis_methods.registry import create_analysis_method, list_analysis_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled analysis methods.")
    parser.add_argument("--experiment-config", type=Path, default=PROJECT_DIR / "configs" / "controlled_scannet_v1.yaml")
    parser.add_argument("--analysis-config", type=Path, default=PROJECT_DIR / "configs" / "analysis_controlled_v1.yaml")
    parser.add_argument("--manifest-index", type=Path, default=PROJECT_DIR / "outputs" / "manifests" / "controlled_scannet_v1" / "index.json")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    add_common_filters(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_config = load_config(args.experiment_config)
    analysis_config = load_config(args.analysis_config)
    logger = setup_logging(experiment_config, "run_controlled_analysis")
    project = project_name(experiment_config)
    index_path = resolve_index_path(args.manifest_index, experiment_config)
    index = load_json(index_path)
    records = filtered_manifest_records(index, args)

    analysis_root = analysis_config.get("analysis", {})
    methods = args.methods or list(analysis_root.get("methods", []))
    models = args.models or list(experiment_config.get("extraction", {}).get("models", ["lingbot-map"]))
    logger.info("Registered analysis methods: %s", ", ".join(list_analysis_methods()))
    logger.info("Selected %d manifests, models=%s, methods=%s", len(records), models, methods)

    runs = collect_runs(records, index_path=index_path, config=experiment_config, project=project, models=models, logger=logger)
    summary: dict[str, Any] = {"runs": [], "missing_token_bundles": []}
    summary["missing_token_bundles"] = [run for run in runs.get("missing", [])]
    analysis_runs: list[AnalysisRun] = runs.get("runs", [])

    for method_name in methods:
        method_config = dict(analysis_root)
        method_config.update(analysis_root.get(method_name, {}))
        method_config["random_seed"] = experiment_config.get("project", {}).get("random_seed", 2026)
        method = create_analysis_method(method_name, config=method_config)
        logger.info("Running analysis method: %s", method_name)
        if method.supports_group_run:
            results = method.run_group(analysis_runs)
        else:
            results = [method.run(run) for run in analysis_runs]
        summary["runs"].append(
            {
                "method": method_name,
                "result_count": len(results),
                "ok": sum(1 for result in results if result.get("status") == "ok"),
                "skipped": sum(1 for result in results if result.get("status") == "skipped"),
            }
        )

    analysis_out = output_root(experiment_config) / "metrics" / project
    save_resolved_config(experiment_config, analysis_out, filename="resolved_experiment_config.json")
    save_resolved_config(analysis_config, analysis_out, filename="resolved_analysis_config.json")
    save_json(analysis_out / "last_analysis_summary.json", summary)
    logger.info("Analysis complete. Summary: %s", analysis_out / "last_analysis_summary.json")
    return 0


def collect_runs(
    records: list[dict[str, Any]],
    *,
    index_path: Path,
    config: dict[str, Any],
    project: str,
    models: list[str],
    logger: Any,
) -> dict[str, Any]:
    output = {"runs": [], "missing": []}
    for record in records:
        manifest_path = resolve_manifest_path(record["path"], index_path)
        manifest = load_json(manifest_path)
        for model_name in models:
            token_dir = (
                output_root(config)
                / "tokens"
                / project
                / model_name
                / manifest["scene_id"]
                / manifest["setting_name"]
                / manifest["condition_id"]
            )
            bundle_path = token_dir / "token_bundle.json"
            if not bundle_path.is_file():
                missing = {
                    "model_name": model_name,
                    "scene_id": manifest["scene_id"],
                    "setting_name": manifest["setting_name"],
                    "condition_id": manifest["condition_id"],
                    "token_bundle": str(bundle_path),
                }
                logger.warning("Missing token bundle: %s", bundle_path)
                output["missing"].append(missing)
                continue
            bundle = load_token_bundle(bundle_path)
            metrics_dir = (
                output_root(config)
                / "metrics"
                / project
                / model_name
                / manifest["scene_id"]
                / manifest["setting_name"]
                / manifest["condition_id"]
            )
            figures_dir = (
                output_root(config)
                / "figures"
                / project
                / model_name
                / manifest["scene_id"]
                / manifest["setting_name"]
                / manifest["condition_id"]
            )
            metrics_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)
            output["runs"].append(
                AnalysisRun(
                    bundle=bundle,
                    manifest=manifest,
                    bundle_path=bundle_path,
                    metrics_dir=metrics_dir,
                    figures_dir=figures_dir,
                    config=config,
                )
            )
    return output


if __name__ == "__main__":
    raise SystemExit(main())
