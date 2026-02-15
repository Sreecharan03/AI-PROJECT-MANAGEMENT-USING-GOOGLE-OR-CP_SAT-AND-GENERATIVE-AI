# ai_pm/core/storage.py
# Phase 1 — Run-folder creation, input snapshotting, and hashing.
# Stack: Python 3.11
#
# Features:
#   - new_run_dir(project, runs_root) -> Path              # creates runs/<project>/<timestamp>/
#   - snapshot_inputs(run_dir, inputs) -> manifest dict    # copies files/URLs into inputs/ and computes SHA256
#   - writes logs/hashes.json                              # reproducible record of inputs
#   - CLI main()                                           # quick manual test
#
# Notes:
#   - Reads config/settings.yaml to honor `data_dir` (default: "runs").
#   - Accepts local file paths and http(s) URLs (basic download).
#   - No schema validation here — this is JUST storage. Validation lives in core/schemas.py
#
# Usage:
#   cd ai_pm
#   python -m core.storage --project "Demo" --inputs samples/team.csv samples/skills.csv
#   # (adds a run under runs/Demo/<timestamp>/ with inputs/ + logs/hashes.json)
#
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import yaml
import requests
from zoneinfo import ZoneInfo


# ----------------------------
# Small dataclasses for clarity
# ----------------------------

@dataclass
class InputRecord:
    """One input artifact copied or downloaded into the run."""
    dest_name: str           # e.g., "team.csv"
    sha256: str              # hex digest
    bytes: int               # size in bytes
    mime: str                # best-guess from mimetypes
    source: str              # original local path or URL

@dataclass
class SnapshotResult:
    project: str
    timestamp: str
    run_dir: str
    inputs_dir: str
    logs_dir: str
    records: List[InputRecord]


# --------------------------------
# Helpers: repo/config/time/stores
# --------------------------------

def _repo_root() -> Path:
    # ai_pm/core/storage.py -> repo root is parent of ai_pm
    return Path(__file__).resolve().parents[1]

def _load_settings() -> dict:
    """Load config/settings.yaml if present; fall back to defaults."""
    cfg = _repo_root() / "config" / "settings.yaml"
    defaults = {
        "timezone": "Asia/Kolkata",
        "workweek": ["Mon","Tue","Wed","Thu","Fri"],
        "workday_hours": 8,
        "data_dir": "runs",
    }
    if cfg.exists():
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                defaults.update(data)
        except Exception:
            pass
    return defaults

def _now_stamp(tz_name: str) -> str:
    """Return a filesystem-friendly timestamp in the target timezone."""
    tz = ZoneInfo(tz_name)
    dt = datetime.now(tz)
    return dt.strftime("%Y%m%d_%H%M%S")  # e.g., 20251021_224105

_SLUG_RX = re.compile(r"[^a-zA-Z0-9._-]+")

def safe_project_slug(name: str) -> str:
    """Make a safe folder slug from an arbitrary project name."""
    name = name.strip()
    name = name.replace(" ", "_")
    name = _SLUG_RX.sub("-", name)
    return name or "default"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> Tuple[str, int]:
    """Compute SHA256 and return (hex_digest, num_bytes)."""
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            total += len(b)
            h.update(b)
    return h.hexdigest(), total

def _guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(path.as_posix())
    return mt or "application/octet-stream"

def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False


# ----------------------------
# Run directory creation
# ----------------------------

def new_run_dir(project: str, runs_root: Optional[Path] = None) -> Path:
    """
    Create runs/<project>/<timestamp>/ with inputs/ and logs/ subfolders.
    The base `runs_root` is taken from settings.data_dir if not passed.
    """
    settings = _load_settings()
    base = runs_root or (_repo_root() / settings.get("data_dir", "runs"))
    slug = safe_project_slug(project)
    stamp = _now_stamp(settings.get("timezone", "Asia/Kolkata"))

    run_dir = base / slug / stamp
    _ensure_dir(run_dir / "inputs")
    _ensure_dir(run_dir / "logs")

    return run_dir


# ----------------------------
# Copy & download inputs
# ----------------------------

def _download_to(path_or_url: str, dest_dir: Path, suggested_name: Optional[str] = None) -> Path:
    """
    If path_or_url is a URL, download it; otherwise treat it as a local path and copy.
    Returns destination file path inside dest_dir with chosen name.
    """
    dest_dir = dest_dir.resolve()
    _ensure_dir(dest_dir)

    if not is_url(path_or_url):
        src = Path(path_or_url).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input not found: {src}")
        dest = dest_dir / (suggested_name or src.name)
        # Copy while preserving timestamps
        data = src.read_bytes()
        dest.write_bytes(data)
        return dest

    # URL case — basic download
    url = path_or_url
    name = suggested_name
    if not name:
        # try to get a name from the URL path
        parsed = urlparse(url)
        base = Path(parsed.path).name or "download.bin"
        name = base

    dest = dest_dir / name
    with requests.get(url, timeout=30) as r:
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def snapshot_inputs(run_dir: Path, inputs: Iterable[str]) -> SnapshotResult:
    """
    Copy/download provided inputs into run_dir/inputs and write logs/hashes.json.
    Returns a SnapshotResult describing what was stored.
    """
    run_dir = run_dir.resolve()
    inputs_dir = run_dir / "inputs"
    logs_dir = run_dir / "logs"
    _ensure_dir(inputs_dir)
    _ensure_dir(logs_dir)

    settings = _load_settings()
    project = run_dir.parts[-2] if len(run_dir.parts) >= 2 else "unknown"
    stamp = run_dir.name

    records: List[InputRecord] = []
    used_names: set[str] = set()

    for idx, item in enumerate(inputs, start=1):
        # Deduplicate destination names if needed
        suggested = None
        if is_url(item):
            # Prefer a stable name per URL index
            parsed = urlparse(item)
            base = Path(parsed.path).name
            suggested = base or f"input_{idx}.csv"
        else:
            suggested = Path(item).name

        # Avoid collisions: add numeric suffix if name already used
        base_name = suggested
        suffix = 1
        while base_name in used_names:
            stem = Path(suggested).stem
            ext = Path(suggested).suffix
            base_name = f"{stem}_{suffix}{ext}"
            suffix += 1

        dest = _download_to(item, inputs_dir, suggested_name=base_name)
        used_names.add(dest.name)

        sha, nbytes = compute_sha256(dest)
        rec = InputRecord(
            dest_name=dest.name,
            sha256=sha,
            bytes=nbytes,
            mime=_guess_mime(dest),
            source=item,
        )
        records.append(rec)

    # Write hashes.json
    manifest = {
        "project": project,
        "timestamp": stamp,
        "settings": settings,
        "inputs": [asdict(r) for r in records],
    }
    out = logs_dir / "hashes.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SnapshotResult(
        project=project,
        timestamp=stamp,
        run_dir=str(run_dir),
        inputs_dir=str(inputs_dir),
        logs_dir=str(logs_dir),
        records=records,
    )


# ----------------------------
# CLI for quick manual test
# ----------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Example:
      python -m core.storage --project "Demo" --inputs samples/team.csv samples/skills.csv
      python -m core.storage --project "Demo" --inputs https://example.com/file.csv
    """
    parser = argparse.ArgumentParser(description="Create a run and snapshot inputs with SHA256 logs.")
    parser.add_argument("--project", required=True, help="Project name (will be slugged for the folder).")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="List of input file paths or http(s) URLs to snapshot."
    )
    args = parser.parse_args(argv)

    run_dir = new_run_dir(args.project)
    print(f"[OK] Created run: {run_dir}")

    try:
        result = snapshot_inputs(run_dir, args.inputs)
    except Exception as e:
        print(f"[ERROR] snapshot failed: {e}", file=sys.stderr)
        return 2

    print(f"[OK] Inputs -> {result.inputs_dir}")
    print(f"[OK] Log    -> {result.logs_dir}/hashes.json")
    for r in result.records:
        print(f"  - {r.dest_name}  {r.bytes} bytes  sha256={r.sha256[:10]}…  src={r.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
