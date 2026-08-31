"""Reproducibility metadata captured at the start of each research run."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    return {
        "repository_root": str(root),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--porcelain"),
        "source_sha256": _source_hashes(root),
        "python": sys.version,
        "platform": platform.platform(),
        "command": list(sys.argv),
        "packages": _packages(),
    }


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.glob("**/*.py")):
        if any(part in {".venv", ".git", "runs", "runs_phase1_smoke", "runs_phase2_smoke"} for part in path.parts):
            continue
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _packages() -> dict[str, str]:
    names = ("numpy", "torch", "pytest")
    return {name: importlib.metadata.version(name) for name in names if _installed(name)}


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
