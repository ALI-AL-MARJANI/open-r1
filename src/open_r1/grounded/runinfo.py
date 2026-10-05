"""Provenance recorded with every result file: code version, libraries and hardware."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm", "openai")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["accelerator"] = "cuda"
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                }
                for i in range(torch.cuda.device_count())
            ]
        elif torch.backends.mps.is_available():
            info["accelerator"] = "mps"
        else:
            info["accelerator"] = "cpu"
    except ImportError:
        info["accelerator"] = None
    return info


def collect_run_info() -> dict[str, Any]:
    """Snapshot of the environment at the time of the call."""
    # Result files are written while runs are in progress, so they do not count as code changes.
    status = _git("status", "--porcelain", "--", ".", ":!results")
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "hardware": _hardware(),
        "command": " ".join(sys.argv),
    }


def write_json(path: str | Path, payload: Any) -> Path:
    """Write `payload` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path
