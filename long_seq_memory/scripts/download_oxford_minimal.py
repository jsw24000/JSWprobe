#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, write_json  # noqa: E402


DEFAULT_CONFIG = "/home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/dataset_download_one_sequence.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "complete_local_file": path.exists() and path.stat().st_size > 0,
    }


def scan_local_dir(local_dir: Path, patterns: list[str]) -> dict[str, Any]:
    cache_dir = local_dir / ".cache" / "huggingface"
    incomplete = sorted(cache_dir.glob("**/*.incomplete")) if cache_dir.exists() else []
    metadata = sorted(cache_dir.glob("**/*.metadata")) if cache_dir.exists() else []
    return {
        "local_dir": str(local_dir),
        "files": {pattern: file_status(local_dir / pattern) for pattern in patterns},
        "cache_dir": str(cache_dir),
        "cache_exists": cache_dir.exists(),
        "cache_size_bytes": sum(p.stat().st_size for p in cache_dir.glob("**/*") if p.is_file()) if cache_dir.exists() else 0,
        "metadata_files": len(metadata),
        "incomplete_files": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in incomplete
        ],
    }


def configure_proxy(args: argparse.Namespace) -> dict[str, Any]:
    before = {name: os.environ.get(name) for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]}
    if args.unset_proxy:
        for name in before:
            os.environ.pop(name, None)
    if args.proxy:
        for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ[name] = args.proxy
    after = {name: os.environ.get(name) for name in before}
    return {"before": before, "after": after}


def download_files(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    repo_id = str(cfg["repo_id"])
    repo_type = str(cfg.get("repo_type", "dataset"))
    revision = str(cfg.get("branch", cfg.get("revision", "main")))
    local_dir = Path(cfg["local_dir"]).expanduser()
    patterns = [str(item) for item in cfg.get("patterns", [])]
    results = []
    for pattern in patterns:
        destination = local_dir / pattern
        if destination.exists() and destination.stat().st_size > 0 and not args.force_download:
            print(f"[skip] {pattern} already exists ({destination.stat().st_size} bytes)")
            results.append({"pattern": pattern, "status": "exists", "path": str(destination), "size_bytes": destination.stat().st_size})
            continue
        print(f"[download] {pattern}")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=pattern,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(local_dir),
                force_download=bool(args.force_download),
                local_files_only=bool(args.local_files_only),
            )
            p = Path(path)
            results.append({"pattern": pattern, "status": "downloaded", "path": str(p), "size_bytes": p.stat().st_size if p.exists() else 0})
        except Exception as exc:
            results.append({"pattern": pattern, "status": "failed", "error": repr(exc)})
            print(f"[failed] {pattern}: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", help="Only materialize files already present in the HuggingFace cache.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--unset-proxy", action="store_true", help="Clear HTTP(S)/ALL proxy environment variables for this process.")
    parser.add_argument("--proxy", default=None, help="Set HTTP_PROXY/HTTPS_PROXY/ALL_PROXY for this process, e.g. http://127.0.0.1:7890 or socks5h://127.0.0.1:7897.")
    parser.add_argument("--manifest", default=str(EXPERIMENT_ROOT / "outputs" / "causal_branch_4414" / "path_check" / "oxford_minimal_download_status.json"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    local_dir = Path(cfg["local_dir"]).expanduser()
    patterns = [str(item) for item in cfg.get("patterns", [])]
    proxy_status = configure_proxy(args)
    before = scan_local_dir(local_dir, patterns)
    payload: dict[str, Any] = {
        "config": str(Path(args.config).expanduser()),
        "repo_id": cfg.get("repo_id"),
        "repo_type": cfg.get("repo_type"),
        "revision": cfg.get("branch", cfg.get("revision", "main")),
        "local_dir": str(local_dir),
        "patterns": patterns,
        "proxy": proxy_status,
        "before": before,
        "download_results": [],
    }

    if args.dry_run:
        payload["status"] = "dry_run"
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        ensure_dir(local_dir)
        payload["download_results"] = download_files(cfg, args)
        payload["after"] = scan_local_dir(local_dir, patterns)
        complete = all(item["complete_local_file"] for item in payload["after"]["files"].values())
        payload["status"] = "complete" if complete else "incomplete"
        print(f"status={payload['status']}")

    write_json(args.manifest, payload)
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
