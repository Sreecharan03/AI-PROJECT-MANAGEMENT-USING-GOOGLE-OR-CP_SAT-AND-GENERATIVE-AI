# ai_pm/core/schemas.py
# Phase 1 — Authoritative data schemas + validators for ingestion.
# Stack: Python 3.11, Pydantic v2.
#
# What this file provides:
# - Pydantic models for TeamMember, SkillMap, Holiday, HistoryEntry.
# - Helpers to parse CSV rows and validate required columns/types/ranges.
# - CLI main() for quick checks (no snapshots here; storage.py & Streamlit page come next).
#
# Usage (examples):
#   cd ai_pm
#   python -m core.schemas --team ./samples/team.csv
#   python -m core.schemas --skills ./samples/skills.csv
#   python -m core.schemas --holidays ./samples/holidays.csv
#   python -m core.schemas --history ./samples/history.csv
#
# Notes:
# - We avoid pandas to keep Phase 1 core light; Streamlit page may use DataFrames for preview.
# - Team.skills expects a JSON list like: [{"name":"python","level":4}, ...]
# - Skills.synonyms accepts either 'a|b|c' or JSON list '["a","b","c"]'.
# - Timezone validated against IANA database via zoneinfo.ZoneInfo.

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator
from zoneinfo import ZoneInfo


# ---------------------------
# Error container for batch validation
# ---------------------------

@dataclass
class RowError:
    source: str         # logical source name e.g., "team.csv"
    row_index: int      # 1-based CSV row index (excluding header) for user clarity
    message: str        # human-friendly error message
    row: Dict[str, Any] # original row content for context (best-effort)


# ---------------------------
# Models — Team
# ---------------------------

class SkillItem(BaseModel):
    """One skill entry in a team member's skills list."""
    name: str = Field(..., min_length=1)
    level: int = Field(..., ge=0, le=5, description="0–5 inclusive")

class TeamMember(BaseModel):
    """A single team member with capacity and timezone."""
    member_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    seniority_level: int = Field(..., ge=0, le=5, description="0–5 inclusive")
    weekly_capacity_hours: float = Field(..., ge=0.0)
    current_load_hours: float = Field(..., ge=0.0)
    timezone: str = Field(..., min_length=1, description="IANA TZ like Asia/Kolkata")
    skills: List[SkillItem] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        # ensure it's a valid IANA zone
        try:
            _ = ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Invalid IANA timezone: {v}") from e
        return v

# ---------------------------
# Models — Skills ontology
# ---------------------------

class SkillMap(BaseModel):
    """Canonical skill row with synonyms."""
    canonical_skill: str = Field(..., min_length=1)
    synonyms: List[str] = Field(default_factory=list)

# ---------------------------
# Models — Optional Holidays
# ---------------------------

class Holiday(BaseModel):
    region: str = Field(..., min_length=1)
    date: date

# ---------------------------
# Models — Optional History
# ---------------------------

class HistoryEntry(BaseModel):
    task_id: str = Field(..., min_length=1)
    member_id: str = Field(..., min_length=1)
    outcome: str = Field(..., min_length=1)           # free text or enum later
    review_score: Optional[float] = Field(None, ge=0, le=5)
    cycle_time_hrs: Optional[float] = Field(None, ge=0.0)


# ======================================================
# Parser/validator helpers
# ======================================================

TEAM_REQUIRED_COLUMNS = [
    "member_id", "name", "role", "seniority_level",
    "weekly_capacity_hours", "current_load_hours",
    "timezone", "skills",
]

SKILLS_REQUIRED_COLUMNS = ["canonical_skill", "synonyms"]
HOLIDAYS_REQUIRED_COLUMNS = ["region", "date"]
HISTORY_REQUIRED_COLUMNS = ["task_id", "member_id", "outcome", "review_score", "cycle_time_hrs"]


def _require_columns(header: Sequence[str], required: Sequence[str]) -> List[str]:
    missing = [c for c in required if c not in header]
    return missing


def _parse_member_skills(val: Any) -> List[SkillItem]:
    """
    'skills' column in team.csv is expected to be a JSON list of {name, level}.
    Example: [{"name":"python","level":4},{"name":"streamlit","level":3}]
    We'll accept empty/blank as [].
    """
    if val is None:
        return []
    if isinstance(val, list):
        payload = val
    else:
        s = str(val).strip()
        if not s:
            return []
        try:
            payload = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError("skills must be JSON list of {name,level}") from e
    if not isinstance(payload, list):
        raise ValueError("skills must be a list")
    return [SkillItem.model_validate(item) for item in payload]


def _parse_synonyms(val: Any) -> List[str]:
    """
    Accept either a JSON list string or a pipe-delimited string.
    Examples:
      ["ml ops","mlops","machine learning operations"]
      ml ops|mlops|machine learning operations
    """
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    # Try JSON first
    try:
        maybe = json.loads(s)
        if isinstance(maybe, list):
            return [str(x).strip() for x in maybe if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback to pipe-delimited
    parts = [p.strip() for p in s.split("|")]
    return [p for p in parts if p]


def validate_team_rows(rows: Iterable[Mapping[str, Any]], source_name: str = "team.csv") -> Tuple[List[TeamMember], List[RowError]]:
    valid: List[TeamMember] = []
    errors: List[RowError] = []
    for i, row in enumerate(rows, start=1):
        # Build a dict with typed fields
        materialized = dict(row)
        try:
            materialized["seniority_level"] = int(materialized.get("seniority_level", ""))
            materialized["weekly_capacity_hours"] = float(materialized.get("weekly_capacity_hours", ""))
            materialized["current_load_hours"] = float(materialized.get("current_load_hours", ""))
            materialized["skills"] = _parse_member_skills(materialized.get("skills"))
            member = TeamMember.model_validate(materialized)
            valid.append(member)
        except Exception as e:
            errors.append(RowError(source=source_name, row_index=i, message=str(e), row=materialized))
    return valid, errors


def validate_skills_rows(rows: Iterable[Mapping[str, Any]], source_name: str = "skills.csv") -> Tuple[List[SkillMap], List[RowError]]:
    valid: List[SkillMap] = []
    errors: List[RowError] = []
    for i, row in enumerate(rows, start=1):
        materialized = dict(row)
        try:
            materialized["synonyms"] = _parse_synonyms(materialized.get("synonyms"))
            item = SkillMap.model_validate(materialized)
            valid.append(item)
        except Exception as e:
            errors.append(RowError(source=source_name, row_index=i, message=str(e), row=materialized))
    return valid, errors


def validate_holidays_rows(rows: Iterable[Mapping[str, Any]], source_name: str = "holidays.csv") -> Tuple[List[Holiday], List[RowError]]:
    valid: List[Holiday] = []
    errors: List[RowError] = []
    for i, row in enumerate(rows, start=1):
        materialized = dict(row)
        try:
            # date is ISO YYYY-MM-DD; date.fromisoformat will validate.
            if isinstance(materialized.get("date"), str):
                materialized["date"] = date.fromisoformat(materialized["date"])
            item = Holiday.model_validate(materialized)
            valid.append(item)
        except Exception as e:
            errors.append(RowError(source=source_name, row_index=i, message=str(e), row=materialized))
    return valid, errors


def validate_history_rows(rows: Iterable[Mapping[str, Any]], source_name: str = "history.csv") -> Tuple[List[HistoryEntry], List[RowError]]:
    valid: List[HistoryEntry] = []
    errors: List[RowError] = []
    for i, row in enumerate(rows, start=1):
        materialized = dict(row)
        try:
            # cast numerics if present
            if materialized.get("review_score", "") != "":
                materialized["review_score"] = float(materialized["review_score"])
            else:
                materialized["review_score"] = None
            if materialized.get("cycle_time_hrs", "") != "":
                materialized["cycle_time_hrs"] = float(materialized["cycle_time_hrs"])
            else:
                materialized["cycle_time_hrs"] = None
            item = HistoryEntry.model_validate(materialized)
            valid.append(item)
        except Exception as e:
            errors.append(RowError(source=source_name, row_index=i, message=str(e), row=materialized))
    return valid, errors


# ======================================================
# CSV utilities (no pandas dependency)
# ======================================================

def _read_csv_dicts(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        rows = [dict(r) for r in rd]
    return header, rows


def _check_header_or_die(header: List[str], required: List[str], source: str) -> None:
    missing = _require_columns(header, required)
    if missing:
        raise RuntimeError(f"{source}: missing required columns: {missing}")


# ======================================================
# CLI main() — quick local validation helper
# ======================================================

def main(argv: Optional[List[str]] = None) -> int:
    """
    Minimal command-line validator for raw CSVs.
    This is a developer convenience; the official CLI and Streamlit UI come next.
    """
    parser = argparse.ArgumentParser(description="Phase 1 schemas/validator (quick checks)")
    parser.add_argument("--team", type=Path, help="Path to team.csv")
    parser.add_argument("--skills", type=Path, help="Path to skills.csv")
    parser.add_argument("--holidays", type=Path, help="Path to holidays.csv (optional)")
    parser.add_argument("--history", type=Path, help="Path to history.csv (optional)")
    args = parser.parse_args(argv)

    exit_code = 0

    if args.team:
        header, rows = _read_csv_dicts(args.team)
        _check_header_or_die(header, TEAM_REQUIRED_COLUMNS, "team.csv")
        valid, errs = validate_team_rows(rows, "team.csv")
        print(f"[team] valid={len(valid)} errors={len(errs)}")
        for e in errs[:10]:
            print(f"  - row {e.row_index}: {e.message}")
        if errs:
            exit_code = 2

    if args.skills:
        header, rows = _read_csv_dicts(args.skills)
        _check_header_or_die(header, SKILLS_REQUIRED_COLUMNS, "skills.csv")
        valid, errs = validate_skills_rows(rows, "skills.csv")
        print(f"[skills] valid={len(valid)} errors={len(errs)}")
        for e in errs[:10]:
            print(f"  - row {e.row_index}: {e.message}")
        if errs:
            exit_code = 2

    if args.holidays:
        header, rows = _read_csv_dicts(args.holidays)
        _check_header_or_die(header, HOLIDAYS_REQUIRED_COLUMNS, "holidays.csv")
        valid, errs = validate_holidays_rows(rows, "holidays.csv")
        print(f"[holidays] valid={len(valid)} errors={len(errs)}")
        for e in errs[:10]:
            print(f"  - row {e.row_index}: {e.message}")
        if errs:
            exit_code = 2

    if args.history:
        header, rows = _read_csv_dicts(args.history)
        _check_header_or_die(header, HISTORY_REQUIRED_COLUMNS, "history.csv")
        valid, errs = validate_history_rows(rows, "history.csv")
        print(f"[history] valid={len(valid)} errors={len(errs)}")
        for e in errs[:10]:
            print(f"  - row {e.row_index}: {e.message}")
        if errs:
            exit_code = 2

    if not any([args.team, args.skills, args.holidays, args.history]):
        parser.print_help()
        return 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
