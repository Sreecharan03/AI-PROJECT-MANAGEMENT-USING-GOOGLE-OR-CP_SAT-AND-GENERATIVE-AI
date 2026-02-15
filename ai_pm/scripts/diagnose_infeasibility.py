#!/usr/bin/env python3
"""
scripts/diagnose_infeasibility.py
Purpose: Explain why CP-SAT found no feasible assignment.

What it does:
  - Loads team.csv and normalized task_graph.json
  - (Optional) Loads skills.csv ontology and maps team skill names to canonical
  - For each task: lists eligible members and flags tasks with ZERO eligible members
  - Summarizes total hours vs capacity and suggests a minimum horizon in weeks

Usage:
  python -m scripts.diagnose_infeasibility \
    --team samples/team.csv \
    --tasks /tmp/task_graph.normalized.json \
    --settings config/settings.yaml \
    --skills runs/Demo/20251022_002650/normalized/skills.csv  # optional but recommended

Exit codes:
  0 = diagnostics printed successfully
  2 = bad/missing inputs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# --- minimal local helpers to avoid cross-import churn ---
def _yaml_load(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def _load_settings(path: Path) -> Dict[str, Any]:
    cfg = _yaml_load(path)
    return {
        "timezone": cfg.get("timezone", "Asia/Kolkata"),
        "workweek": cfg.get("workweek", ["Mon","Tue","Wed","Thu","Fri"]),
        "workday_hours": int(cfg.get("workday_hours", 8)),
        "data_dir": cfg.get("data_dir", "runs"),
    }

def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _load_team_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    team: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            mid = (r.get("member_id") or "").strip()
            if not mid:
                continue
            # skills as dict name->level
            skills: Dict[str, int] = {}
            raw = (r.get("skills") or "").strip()
            try:
                arr = json.loads(raw) if raw else []
                if isinstance(arr, list):
                    for it in arr:
                        nm = str(it.get("name", "")).strip()
                        lv = it.get("level", 0)
                        try: lv = int(lv)
                        except Exception: lv = 0
                        if nm:
                            skills[nm] = max(0, min(5, lv))
            except Exception:
                pass
            team[mid] = {
                "member_id": mid,
                "name": (r.get("name") or "").strip(),
                "role": (r.get("role") or "").strip(),
                "seniority": int(float(r.get("seniority_level") or 0)),
                "weekly_capacity": float(r.get("weekly_capacity_hours") or 0.0),
                "current_load": float(r.get("current_load_hours") or 0.0),
                "skills": skills,
            }
    return team

def _canonicalize_team_skills(team: Dict[str, Dict[str, Any]], skills_csv: Optional[Path]) -> None:
    """If an ontology is provided, map EVERY team skill name -> canonical."""
    if not skills_csv or not skills_csv.exists():
        return
    # inline ontology parser (pipe/JSON)
    import csv as _csv, json as _json
    def _parse_syns(s: str) -> List[str]:
        if not s: return []
        s = s.strip()
        if not s: return []
        try:
            maybe = _json.loads(s)
            if isinstance(maybe, list):
                return [str(x).strip() for x in maybe if str(x).strip()]
        except Exception:
            pass
        return [p.strip() for p in s.split("|") if p.strip()]
    # build token->canonical
    from collections import defaultdict
    token2canon: Dict[str, str] = {}
    def _norm(x: str) -> str:
        x = (x or "").strip().lower()
        for ch in "-_/.,": x = x.replace(ch, " ")
        return " ".join(x.split())
    with skills_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rd = _csv.DictReader(f)
        for row in rd:
            canon = (row.get("canonical_skill") or "").strip()
            if not canon: continue
            token2canon[_norm(canon)] = canon
            for s in _parse_syns(row.get("synonyms","") or ""):
                token2canon[_norm(s)] = canon
    # apply
    for m in team.values():
        new_sk = {}
        for nm, lv in m["skills"].items():
            c = token2canon.get(_norm(nm))
            new_sk[c if c else nm] = lv
        m["skills"] = new_sk

def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = _json_load(path)
    return list(data.get("tasks", []))

def _member_meets(task: Dict[str, Any], m: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons = []
    tskills = task.get("required_skills") or []
    if not tskills:
        return True, reasons
    for req in tskills:
        nm = str(req.get("name","")).strip()
        minlvl = int(float(req.get("level", 0)))
        lvl = m["skills"].get(nm, -1)
        if lvl < minlvl:
            reasons.append(f"needs {nm}≥{minlvl}, has {lvl if lvl>=0 else 'none'}")
    return (len(reasons)==0), reasons

def _sum_hours(tasks: List[Dict[str, Any]]) -> float:
    return sum(float(t.get("estimate_h") or 0.0) for t in tasks)

def _total_available(team: Dict[str, Dict[str, Any]]) -> float:
    return sum(max(0.0, m["weekly_capacity"] - m["current_load"]) for m in team.values())

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Explain infeasibility: zero-eligible tasks and capacity shortfall.")
    ap.add_argument("--team", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--settings", type=Path, default=Path("config/settings.yaml"))
    ap.add_argument("--skills", type=Path, help="Optional skills.csv to canonicalize TEAM skills")
    ap.add_argument("--horizon-weeks", type=int, help="Planned horizon (weeks) used for capacity comparison")
    args = ap.parse_args(argv)

    if not args.team.exists() or not args.tasks.exists():
        print("[ERROR] team or tasks file not found")
        return 2

    settings = _load_settings(args.settings)
    team = _load_team_csv(args.team)
    tasks = _load_tasks(args.tasks)
    _canonicalize_team_skills(team, args.skills)

    total_hours = _sum_hours(tasks)
    weekly_avail = _total_available(team)
    if weekly_avail <= 0: weekly_avail = 1.0
    min_weeks = int((total_hours + weekly_avail - 1) // weekly_avail) if total_hours > 0 else 1
    if args.horizon_weeks:
        horizon = args.horizon_weeks
    else:
        horizon = max(1, min_weeks)

    print(f"[INFO] tasks={len(tasks)}  team_members={len(team)}")
    print(f"[INFO] total task hours={total_hours:.1f}")
    print(f"[INFO] total weekly available={weekly_avail:.1f}")
    print(f"[INFO] suggested minimum horizon (weeks)≈{min_weeks}  (you used {args.horizon_weeks or '(auto) ' + str(horizon)})")

    # Eligibility per task
    zero_elig: List[Tuple[str, List[str]]] = []
    for t in tasks:
        tid = t.get("task_id")
        title = t.get("title","")
        reqs = t.get("required_skills") or []
        eligible = []
        why_by_member: Dict[str, List[str]] = {}
        for mid, m in team.items():
            ok, reasons = _member_meets(t, m)
            if ok:
                eligible.append(mid)
            else:
                why_by_member[mid] = reasons
        if not eligible:
            zero_elig.append((tid, [f"{mid}: {', '.join(reasons)}" for mid, reasons in why_by_member.items()]))

    if zero_elig:
        print("\n[ISSUE] Tasks with ZERO eligible members (skill/level mismatch):")
        for tid, details in zero_elig:
            print(f"  - {tid}:")
            for line in details[:10]:
                print(f"     • {line}")
            if len(details) > 10:
                print("     • ...")
        print("\n[ACTION] Fix by either:")
        print("  1) Updating skills ontology so task skill names map to what team has (e.g., 'frontend' -> 'react'), then re-validate tasks.")
        print("  2) Adding the missing skills/levels to team.csv (or upskilling levels).")
        print("  3) Passing --skills <normalized skills.csv> to this tool so team skills are canonicalized to match tasks.")
    else:
        print("\n[OK] Every task has at least one eligible member.")

    # Capacity check against chosen horizon
    cap_hours = weekly_avail * horizon
    if total_hours > cap_hours:
        short = total_hours - cap_hours
        print(f"\n[ISSUE] Capacity shortfall over {horizon} week(s): need {total_hours:.1f}h, have {cap_hours:.1f}h (short {short:.1f}h).")
        print(f"[ACTION] Increase --horizon-weeks to at least ≈{min_weeks}, or reduce scope/estimates.")
    else:
        print(f"\n[OK] Capacity fits within {horizon} week(s).")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
