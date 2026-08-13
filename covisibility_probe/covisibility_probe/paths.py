from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parent


def resolve_project_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return ((base or project_root()) / raw).resolve()


def candidate_scannet_roots() -> list[Path]:
    root = workspace_root()
    return [
        root / "data1" / "3dsm" / "ScanNet" / "scans_extracted",
        root / "data1" / "3dsm",
        Path("/home/data1/3dsm/ScanNet/scans_extracted"),
        Path("/home/data1/ScanNet/scans_extracted"),
        Path("/home/data1/ScanNet/scans"),
        Path("/data1/3dsm/ScanNet/scans_extracted"),
        Path("/data1/ScanNet/scans_extracted"),
        Path("/disk1/3dsm/ScanNet/scans_extracted"),
        Path("/disk1/ScanNet/scans_extracted"),
    ]


def auto_scannet_root() -> Path | None:
    for candidate in candidate_scannet_roots():
        if candidate.is_dir() and any(p.is_dir() and p.name.startswith("scene") for p in candidate.iterdir()):
            return candidate.resolve()
    return None


def default_paths() -> dict[str, Any]:
    root = workspace_root()
    return {
        "project_root": project_root(),
        "workspace_root": root,
        "lingbot_repo": root / "lingbot-map",
        "lingbot_checkpoint": root / "lingbot-map" / "checkpoints" / "lingbot-map.pt",
        "dinov2_repo": root / "dinov2",
        "dinov2_checkpoint": root / "weights" / "dinov2" / "dinov2_vitl14_reg4_pretrain.pth",
        "vggt_repo": root / "vggt",
        "vggt_checkpoint": root / "vggt" / "checkpoints" / "model.pt",
        "scannet_root": auto_scannet_root(),
    }


def make_run_id(tag: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tag = (tag or "run").replace(" ", "_")
    return f"{stamp}_{clean_tag}"


def resolve_run_dir(cfg: dict, run_id: str | None = None) -> Path:
    run_cfg = cfg.setdefault("run", {})
    rid = run_id or run_cfg.get("run_id") or make_run_id(run_cfg.get("tag"))
    run_cfg["run_id"] = rid
    output_root = Path(run_cfg.get("output_root", project_root() / "outputs")).expanduser().resolve()
    return output_root / rid
