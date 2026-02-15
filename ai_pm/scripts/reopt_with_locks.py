# ai_pm/scripts/reopt_with_locks.py
# Phase 5 — Re-optimize with locks (assignment + date windows)
#
# WHAT THIS SCRIPT DOES
# --------------------
# • Reads team.csv, normalized task_graph.json, settings.yaml, and a prior plan.json.
# • Reads locks.json:
#       {
#         "assignment_locks": {"T2-1":"u2","T3-2":"u1"},
#         "date_locks": {
#           "T2-1": {"start_after":"2025-10-23T09:00:00+05:30"},
#           "T4":   {"end_before":"2025-10-31T18:00:00+05:30"}
#         },
#         "theta": {"skill_fit":0.5,"fairness":0.2,"continuity":0.2,"deadline_risk":0.1},
#         "horizon_weeks": 4
#       }
# • Reduces member capacity for locked hours (spread across horizon), solves remaining tasks,
#   rebuilds schedule for ALL tasks, then verifies date windows.
# • If any date lock is violated, prints a clear report and exits non-zero (never breaks locks silently).
#
# USAGE
# -----
# python -m scripts.reopt_with_locks \
#   --team samples/team.csv \
#   --tasks runs/Demo/<ts>/normalized/task_graph.json \
#   --settings config/settings.yaml \
#   --plan runs/Demo/<ts>/plan/plan.json \
#   --locks /path/to/locks.json \
#   --out /tmp/plan.locked.json \
#   [--kpis-out /tmp/kpis.locked.json] \
#   [--history samples/history.csv]
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Core
from core.optimizer import (
    load_settings,
    load_team_csv,
    load_task_graph,
    solve_assignments,
    build_schedule,
)
from core.kpis import compute_kpis


# ---------------------------
# Helpers
# ---------------------------

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)  # handles offsets like +05:30
    except Exception:
        return None

def _sum_locked_hours(plan_json: Path, assignment_locks: Dict[str, str]) -> Dict[str, float]:
    """Return hours per member consumed by locked tasks (looked up from existing plan)."""
    data = json.loads(plan_json.read_text(encoding="utf-8"))
    hours_by_member: Dict[str, float] = {}
    for t in data.get("tasks", []):
        tid = (t.get("task_id") or "").strip()
        if tid in assignment_locks:
            mid = assignment_locks[tid]
            est = float(t.get("estimate_h") or 0.0)
            hours_by_member[mid] = hours_by_member.get(mid, 0.0) + est
    return hours_by_member

def _apply_capacity_adjustment(members, locked_hours_by_member: Dict[str, float], horizon_weeks: int) -> None:
    """Decrease each member's available weekly capacity by locked_hours/horizon."""
    H = max(1, int(horizon_weeks))
    for mid, locked in locked_hours_by_member.items():
        if mid in members:
            m = members[mid]
            m.current_load = float(m.current_load) + (float(locked) / H)
            m.available_weekly = max(0.0, float(m.weekly_capacity) - float(m.current_load))

def _date_lock_violations(plan_obj: Dict[str, Any], date_locks: Dict[str, Dict[str, str]]) -> List[str]:
    """Return a list of human-readable violations against date locks."""
    idx = {t["task_id"]: t for t in plan_obj.get("tasks", [])}
    msgs: List[str] = []
    for tid, win in (date_locks or {}).items():
        pt = idx.get(tid)
        if not pt:
            msgs.append(f"[{tid}] not present in plan; cannot enforce date window.")
            continue
        start = _parse_iso(pt.get("start"))
        end   = _parse_iso(pt.get("end"))
        if "start" in win:
            want = _parse_iso(win["start"])
            if want and start and start != want:
                msgs.append(f"[{tid}] start exact lock mismatch: got {start}, want {want}.")
        if "end" in win:
            want = _parse_iso(win["end"])
            if want and end and end != want:
                msgs.append(f"[{tid}] end exact lock mismatch: got {end}, want {want}.")
        if "start_after" in win:
            cut = _parse_iso(win["start_after"])
            if cut and start and start < cut:
                msgs.append(f"[{tid}] start {start} is before locked window start_after {cut}.")
        if "end_before" in win:
            cut = _parse_iso(win["end_before"])
            if cut and end and end > cut:
                msgs.append(f"[{tid}] end {end} exceeds locked window end_before {cut}.")
    return msgs


# ---------------------------
# Core routine
# ---------------------------

def reopt_with_locks(
    team_csv: Path,
    tasks_json: Path,
    settings_yaml: Path,
    prior_plan_json: Path,
    locks_json: Path,
    out_plan: Path,
    out_kpis: Optional[Path] = None,
    history_csv: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Re-solve around assignment/date locks and write artifacts."""
    # Load inputs
    settings = load_settings(settings_yaml)
    members  = load_team_csv(team_csv)
    tasks    = load_task_graph(tasks_json, settings["workday_hours"])  # same loader signature as used elsewhere

    locks = json.loads(locks_json.read_text(encoding="utf-8"))
    assignment_locks: Dict[str, str] = locks.get("assignment_locks", {}) or {}
    date_locks: Dict[str, Dict[str, str]] = locks.get("date_locks", {}) or {}
    theta: Dict[str, float] = locks.get("theta", {"skill_fit":0.5,"fairness":0.2,"continuity":0.2,"deadline_risk":0.1})
    horizon_weeks: int = int(locks.get("horizon_weeks", 4))

    # Capacity adjustment for locked hours (from prior plan’s estimates)
    locked_hours_by_member = _sum_locked_hours(prior_plan_json, assignment_locks)
    _apply_capacity_adjustment(members, locked_hours_by_member, horizon_weeks)

    # Split tasks: locked vs unlocked
    locked_set = set(assignment_locks.keys())
    tasks_unlocked = [t for t in tasks if t.task_id not in locked_set]

    # Solve only for unlocked
    assign_unlocked, loads, obj = solve_assignments(members, tasks_unlocked, theta=theta, horizon_weeks=horizon_weeks)

    # Merge (locked + new)
    assignment_all = dict(assign_unlocked)
    assignment_all.update(assignment_locks)

    # Build full schedule for ALL tasks
    plan_obj = build_schedule(assignment_all, tasks, members, settings)
    plan_obj["objective"] = obj

    # Validate date locks (hard)
    violations = _date_lock_violations(plan_obj, date_locks)
    if violations:
        # Do NOT write outputs if locks are violated.
        msg = "\n".join(f"  - {m}" for m in violations)
        raise RuntimeError(f"Date lock violations detected:\n{msg}")

    # Save outputs
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(json.dumps(plan_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    kpis_obj: Optional[Dict[str, Any]] = None
    if out_kpis is not None:
        kpis_obj = compute_kpis(team_csv, tasks_json, out_plan, horizon_weeks=horizon_weeks)
        out_kpis.parent.mkdir(parents=True, exist_ok=True)
        out_kpis.write_text(json.dumps(kpis_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return plan_obj, kpis_obj


# ---------------------------
# CLI
# ---------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Re-optimize around assignment/date locks.")
    ap.add_argument("--team", required=True, help="team.csv")
    ap.add_argument("--tasks", required=True, help="normalized task_graph.json")
    ap.add_argument("--settings", required=True, help="config/settings.yaml")
    ap.add_argument("--plan", required=True, help="prior plan.json (used to measure locked hours)")
    ap.add_argument("--locks", required=True, help="locks.json (assignment_locks/date_locks/theta/horizon_weeks)")
    ap.add_argument("--out", required=True, help="output plan.json")
    ap.add_argument("--kpis-out", default=None, help="optional kpis.json output")
    ap.add_argument("--history", default=None, help="optional history.csv (not strictly needed here)")
    return ap.parse_args()

def main() -> int:
    args = _parse_args()
    try:
        plan_obj, kpis_obj = reopt_with_locks(
            team_csv=Path(args.team),
            tasks_json=Path(args.tasks),
            settings_yaml=Path(args.settings),
            prior_plan_json=Path(args.plan),
            locks_json=Path(args.locks),
            out_plan=Path(args.out),
            out_kpis=Path(args.kpis_out) if args.kpis_out else None,
            history_csv=Path(args.history) if args.history else None,
        )
        print(f"[OK] Re-optimized plan written -> {args.out}")
        if args.kpis_out:
            print(f"[OK] KPIs written -> {args.kpis_out}")
        return 0
    except Exception as e:
        print(f"[ERROR] Re-optimization failed: {e}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
