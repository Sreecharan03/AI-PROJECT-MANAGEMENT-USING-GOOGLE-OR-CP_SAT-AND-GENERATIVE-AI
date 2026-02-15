#!/usr/bin/env python3
# ai_pm/scripts/ingest_validate.py
# Phase 1 — End-to-end ingestion CLI: validate CSVs/Sheet URLs, then snapshot.
#
# What this script does:
#   1) Accept paths or URLs for team.csv, skills.csv, and optional holidays.csv, history.csv.
#   2) If a Google Sheets URL is provided, convert it to a CSV export URL.
#   3) Download/Read inputs, parse as CSV, validate rows with core.schemas.
#   4) On success, create a new run folder and snapshot inputs via core.storage (copy/download + SHA256).
#
# Usage examples:
#   cd ai_pm
#   python -m scripts.ingest_validate \
#     --project "Demo" \
#     --team samples/team.csv \
#     --skills samples/skills.csv \
#     --holidays samples/holidays.csv \
#     --history samples/history.csv
#
#   # from Google Sheets:
#   python -m scripts.ingest_validate \
#     --project "SheetDemo" \
#     --team "https://docs.google.com/spreadsheets/d/<<SHEET_ID>>/edit#gid=0" \
#     --skills "https://docs.google.com/spreadsheets/d/<<SHEET_ID_SKILLS>>/edit#gid=0" \
#     --dry-run
#
# Flags:
#   --dry-run   Validate only; do not snapshot.
#
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, parse_qs

import requests

from core import schemas
from core.storage import (
    new_run_dir,
    snapshot_inputs,
    is_url,  # reuse helper
)

# ----------------------------
# Google Sheets helpers
# ----------------------------

def _is_google_sheet(url: str) -> bool:
    try:
        u = urlparse(url)
        return "docs.google.com" in u.netloc and "/spreadsheets" in u.path
    except Exception:
        return False

def _to_sheets_csv_export(url: str) -> str:
    """
    Convert typical Google Sheets URL to a CSV export URL.
    Examples we accept:
      https://docs.google.com/spreadsheets/d/<ID>/edit#gid=<GID>
      https://docs.google.com/spreadsheets/d/<ID>/view#gid=<GID>
    We output:
      https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID or 0>
    """
    u = urlparse(url)
    parts = u.path.strip("/").split("/")
    sheet_id = None
    for i, part in enumerate(parts):
        if part == "d" and i + 1 < len(parts):
            sheet_id = parts[i + 1]
            break
    # fallback gid=0 if not present
    gid = parse_qs(u.fragment).get("gid", ["0"])[0]
    if not sheet_id:
        # Not in the expected form — just return the original URL
        return url
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def _normalize_source(path_or_url: str) -> str:
    """Return either a local path or a normalized URL (Sheets → CSV export)."""
    if is_url(path_or_url) and _is_google_sheet(path_or_url):
        return _to_sheets_csv_export(path_or_url)
    return path_or_url


# ----------------------------
# CSV loading (local or URL)
# ----------------------------

def _read_csv_from_local(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    return schemas._read_csv_dicts(path)  # reuse core helper

def _read_csv_from_url(url: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    sio = io.StringIO(text)
    reader = csv.DictReader(sio)
    header = reader.fieldnames or []
    rows = [dict(r) for r in reader]
    return header, rows

def _load_csv_any(src: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    src = _normalize_source(src)
    if is_url(src):
        return _read_csv_from_url(src)
    return _read_csv_from_local(Path(src).expanduser().resolve())


# ----------------------------
# Header checks per file type
# ----------------------------

def _require_columns(header: Sequence[str], required: Sequence[str]) -> List[str]:
    return [c for c in required if c not in header]

def _check_header_or_raise(kind: str, header: List[str]) -> None:
    if kind == "team":
        req = schemas.TEAM_REQUIRED_COLUMNS
    elif kind == "skills":
        req = schemas.SKILLS_REQUIRED_COLUMNS
    elif kind == "holidays":
        req = schemas.HOLIDAYS_REQUIRED_COLUMNS
    elif kind == "history":
        req = schemas.HISTORY_REQUIRED_COLUMNS
    else:
        raise RuntimeError(f"unknown kind: {kind}")
    missing = _require_columns(header, req)
    if missing:
        raise RuntimeError(f"{kind}.csv missing columns: {missing}")


# ----------------------------
# Validation orchestration
# ----------------------------

def _validate_team(src: str) -> Tuple[int, int]:
    header, rows = _load_csv_any(src)
    _check_header_or_raise("team", header)
    valid, errs = schemas.validate_team_rows(rows, "team.csv")
    for e in errs[:10]:
        print(f"[team] row {e.row_index}: {e.message}")
    return len(valid), len(errs)

def _validate_skills(src: str) -> Tuple[int, int]:
    header, rows = _load_csv_any(src)
    _check_header_or_raise("skills", header)
    valid, errs = schemas.validate_skills_rows(rows, "skills.csv")
    for e in errs[:10]:
        print(f"[skills] row {e.row_index}: {e.message}")
    return len(valid), len(errs)

def _validate_holidays(src: Optional[str]) -> Tuple[int, int]:
    if not src:
        return 0, 0
    header, rows = _load_csv_any(src)
    _check_header_or_raise("holidays", header)
    valid, errs = schemas.validate_holidays_rows(rows, "holidays.csv")
    for e in errs[:10]:
        print(f"[holidays] row {e.row_index}: {e.message}")
    return len(valid), len(errs)

def _validate_history(src: Optional[str]) -> Tuple[int, int]:
    if not src:
        return 0, 0
    header, rows = _load_csv_any(src)
    _check_header_or_raise("history", header)
    valid, errs = schemas.validate_history_rows(rows, "history.csv")
    for e in errs[:10]:
        print(f"[history] row {e.row_index}: {e.message}")
    return len(valid), len(errs)


# ----------------------------
# CLI main()
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate team/skills (+ optional holidays/history) from paths or URLs, then snapshot."
    )
    parser.add_argument("--project", required=True, help="Project name; used for runs/<project>/<timestamp>/")
    parser.add_argument("--team", required=True, help="Path or URL to team.csv")
    parser.add_argument("--skills", required=True, help="Path or URL to skills.csv")
    parser.add_argument("--holidays", help="Path or URL to holidays.csv (optional)")
    parser.add_argument("--history", help="Path or URL to history.csv (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not snapshot or create a run.")
    args = parser.parse_args(argv)

    # Validate each input
    ok = True

    tv, te = _validate_team(args.team)
    print(f"[team] valid={tv} errors={te}")
    if te > 0: ok = False

    sv, se = _validate_skills(args.skills)
    print(f"[skills] valid={sv} errors={se}")
    if se > 0: ok = False

    hv, he = _validate_holidays(args.holidays)
    if args.holidays:
        print(f"[holidays] valid={hv} errors={he}")
        if he > 0: ok = False

    yv, ye = _validate_history(args.history)
    if args.history:
        print(f"[history] valid={yv} errors={ye}")
        if ye > 0: ok = False

    if not ok:
        print("[FAIL] Validation errors present. No snapshot created.")
        return 2

    if args.dry_run:
        print("[OK] Dry run complete — validation passed.")
        return 0

    # Snapshot on success
    inputs_list = [args.team, args.skills]
    if args.holidays:
        inputs_list.append(args.holidays)
    if args.history:
        inputs_list.append(args.history)

    run_dir = new_run_dir(args.project)
    print(f"[OK] Created run: {run_dir}")
    result = snapshot_inputs(run_dir, inputs_list)
    print(f"[OK] Inputs -> {result.inputs_dir}")
    print(f"[OK] Log    -> {result.logs_dir}/hashes.json")
    for r in result.records:
        print(f"  - {r.dest_name}  {r.bytes} bytes  sha256={r.sha256[:10]}…  src={r.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
