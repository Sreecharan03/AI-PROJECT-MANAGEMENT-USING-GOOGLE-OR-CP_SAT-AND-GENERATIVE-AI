# ai_pm/core/kpis.py
# Phase 7 — KPIs & Analytics helpers
#
# WHAT THIS FILE DOES
# -------------------
# • compute_kpis(team_csv, tasks_json, plan_json, horizon_weeks)
#     - coverage, capacity_violations, avg_skill_fit, utilization_stddev,
#       critical_path_hours, due_by_violations, counts
# • compute_slack_by_task(plan_json)
#     - per-task slack to due_by (in hours); None if no due_by
# • critical_path(tasks_json)
#     - returns (order: List[str], total_hours: float) for longest path in DAG
#
# USAGE (CLI)
# -----------
#   python -m core.kpis \
#       --team samples/team.csv \
#       --tasks runs/Demo/.../normalized/task_graph.json \
#       --plan  runs/Demo/.../plan/plan.json \
#       --horizon-weeks 4
#
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


# ----------------------------
# Utilities
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _end_of_day(date_yyyy_mm_dd: str, tz: ZoneInfo) -> Optional[datetime]:
    try:
        y, m, d = [int(x) for x in date_yyyy_mm_dd.split("-")]
        return datetime(y, m, d, 23, 59, 59, tzinfo=tz)
    except Exception:
        return None

def _parse_iso_opt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _tz_from_settings(plan_or_settings: Dict[str, Any]) -> ZoneInfo:
    # plan.json carries settings; else fallback
    settings = plan_or_settings.get("settings", plan_or_settings)
    tz_str = settings.get("timezone") if isinstance(settings, dict) else None
    try:
        return ZoneInfo(tz_str or "Asia/Kolkata")
    except Exception:
        return ZoneInfo("Asia/Kolkata")


# ----------------------------
# Data readers
# ----------------------------

@dataclass
class MemberSkills:
    member_id: str
    name: str
    skills: Dict[str, float]

def _load_team_skills(team_csv: Path) -> Dict[str, MemberSkills]:
    out: Dict[str, MemberSkills] = {}
    with team_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            mid = (r.get("member_id") or "").strip()
            if not mid:
                continue
            nm = (r.get("name") or mid).strip()
            try:
                sk = r.get("skills") or "[]"
                skl = json.loads(sk) if isinstance(sk, str) else sk
            except Exception:
                skl = []
            m: Dict[str, float] = {}
            for it in skl or []:
                sname = (it.get("name") or "").strip().lower()
                lvl = float(it.get("level") or 0.0)
                if sname:
                    m[sname] = max(0.0, lvl)
            out[mid] = MemberSkills(mid, nm, m)
    return out


# ----------------------------
# Core KPI calculations
# ----------------------------

def _skill_fit_for_task(assigned_mid: str, reqs: List[Dict[str, Any]], team: Dict[str, MemberSkills]) -> float:
    if not assigned_mid or assigned_mid not in team or not reqs:
        return 0.0
    mem = team[assigned_mid]
    per_req: List[float] = []
    for r in reqs:
        sname = (r.get("name") or "").strip().lower()
        req_lvl = float(r.get("level") or 0.0)
        have = float(mem.skills.get(sname, 0.0))
        # If no level specified, treat as ≥1
        need = req_lvl if req_lvl > 0 else 1.0
        per_req.append(min(1.0, (have / need) if need > 0 else 0.0))
    return sum(per_req) / len(per_req) if per_req else 0.0

def _aggregate_skill_fit(plan: Dict[str, Any], tasks: Dict[str, Any], team: Dict[str, MemberSkills]) -> float:
    # Build lookup of required skills per task_id from the tasks graph
    reqs_by_tid: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks.get("tasks", []):
        reqs_by_tid[t["task_id"]] = t.get("required_skills") or []

    vals: List[float] = []
    for t in plan.get("tasks", []):
        tid = t.get("task_id")
        mid = t.get("member_id")
        est = float(t.get("estimate_h") or 0.0)
        sf = _skill_fit_for_task(mid, reqs_by_tid.get(tid, []), team)
        # weight by hours (more significant tasks matter more)
        vals.append(sf * max(1.0, est))
    if not vals:
        return 0.0
    return sum(vals) / sum(max(1.0, float(t.get("estimate_h") or 0.0)) for t in plan.get("tasks", []))

def _utilization_stddev(plan: Dict[str, Any], horizon_weeks: int) -> float:
    members = plan.get("summary", {}).get("members", [])
    if not members:
        return 0.0
    util: List[float] = []
    H = max(1, int(horizon_weeks))
    for m in members:
        # Optimizer stored weekly_available; we scale by horizon weeks
        weekly_avail = float(m.get("weekly_available") or 0.0)
        cap = weekly_avail * H
        load = float(m.get("load_hours") or 0.0)
        u = 0.0 if cap <= 0 else min(1.0, max(0.0, load / cap))
        util.append(u)
    try:
        return float(pstdev(util)) if len(util) >= 2 else 0.0
    except Exception:
        return 0.0

def _capacity_violations(plan: Dict[str, Any], horizon_weeks: int) -> int:
    members = plan.get("summary", {}).get("members", [])
    H = max(1, int(horizon_weeks))
    viol = 0
    for m in members:
        weekly_avail = float(m.get("weekly_available") or 0.0)
        cap = weekly_avail * H
        load = float(m.get("load_hours") or 0.0)
        if cap < load - 1e-6:
            viol += 1
    return viol

def _critical_path_hours(tasks_json: Path) -> float:
    _, total = critical_path(tasks_json)
    return total

def _coverage(tasks_json: Path, plan: Dict[str, Any]) -> float:
    total = len(_read_json(tasks_json).get("tasks", []))
    scheduled = len(plan.get("tasks", []))
    return 0.0 if total <= 0 else scheduled / total

def _due_by_violations(plan: Dict[str, Any]) -> int:
    tz = _tz_from_settings(plan.get("settings", {}))
    cnt = 0
    for t in plan.get("tasks", []):
        end = _parse_iso_opt(t.get("end"))
        due = t.get("due_by")
        if not (end and due):
            continue
        due_eod = _end_of_day(due, tz)
        if due_eod and end > due_eod:
            cnt += 1
    return cnt

def compute_kpis(team_csv: Path, tasks_json: Path, plan_json: Path, horizon_weeks: int = 4) -> Dict[str, Any]:
    """Primary KPI bundle used by pages 03/04/05."""
    team = _load_team_skills(team_csv)
    tasks = _read_json(tasks_json)
    plan = _read_json(plan_json)

    kpis: Dict[str, Any] = {
        "coverage": _coverage(tasks_json, plan),
        "capacity_violations": _capacity_violations(plan, horizon_weeks),
        "avg_skill_fit": round(_aggregate_skill_fit(plan, tasks, team), 4),
        "utilization_stddev": round(_utilization_stddev(plan, horizon_weeks), 4),
        "critical_path_hours": round(_critical_path_hours(tasks_json), 4),
        "due_by_violations": _due_by_violations(plan),
        "counts": {
            "tasks_total": len(tasks.get("tasks", [])),
            "tasks_scheduled": len(plan.get("tasks", [])),
            "members": len(plan.get("summary", {}).get("members", [])),
        },
    }
    return kpis


# ----------------------------
# New analytics helpers (Phase 7)
# ----------------------------

def compute_slack_by_task(plan_json: Path) -> Dict[str, Optional[float]]:
    """
    Returns {task_id: slack_hours | None} where slack = due_by_eod - end.
    None means the task has no due_by or no end timestamp.
    """
    plan = _read_json(plan_json)
    tz = _tz_from_settings(plan.get("settings", {}))
    out: Dict[str, Optional[float]] = {}
    for t in plan.get("tasks", []):
        tid = t.get("task_id")
        end_dt = _parse_iso_opt(t.get("end"))
        due = t.get("due_by")
        if not (tid and end_dt and due):
            out[tid] = None
            continue
        due_eod = _end_of_day(due, tz)
        if not due_eod:
            out[tid] = None
            continue
        out[tid] = round((due_eod - end_dt).total_seconds() / 3600.0, 2)
    return out

def critical_path(tasks_json: Path) -> Tuple[List[str], float]:
    """
    Longest path in DAG by estimated hours.
    Returns (order_of_task_ids_on_longest_path, total_hours).
    """
    data = _read_json(tasks_json)
    tasks = {t["task_id"]: t for t in data.get("tasks", [])}
    # Build DAG
    indeg: Dict[str, int] = {tid: 0 for tid in tasks}
    succ: Dict[str, List[str]] = {tid: [] for tid in tasks}
    for t in tasks.values():
        for d in (t.get("dependencies") or []):
            if d in indeg:
                indeg[t["task_id"]] += 1
                succ[d].append(t["task_id"])

    # Topo + DP for longest path
    from collections import deque
    Q = deque([tid for tid, k in indeg.items() if k == 0])
    best_sum: Dict[str, float] = {tid: float(tasks[tid].get("estimate_h") or 0.0) for tid in tasks}
    prev: Dict[str, Optional[str]] = {tid: None for tid in tasks}

    order: List[str] = []
    while Q:
        u = Q.popleft()
        order.append(u)
        for v in succ[u]:
            cand = best_sum[u] + float(tasks[v].get("estimate_h") or 0.0)
            if cand > best_sum[v] + 1e-12:
                best_sum[v] = cand
                prev[v] = u
            indeg[v] -= 1
            if indeg[v] == 0:
                Q.append(v)

    if len(order) != len(tasks):
        # Cycle -> not a valid DAG; return empty but safe values
        return ([], 0.0)

    # Find endpoint with maximum sum
    end = max(best_sum, key=lambda k: best_sum[k]) if best_sum else None
    total = float(best_sum.get(end, 0.0)) if end else 0.0

    # Reconstruct path
    path: List[str] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return (path, total)


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser("Compute KPIs & analytics helpers")
    ap.add_argument("--team", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--horizon-weeks", type=int, default=4)
    ap.add_argument("--print-slack", action="store_true", help="Also print slack per task.")
    ap.add_argument("--print-critpath", action="store_true", help="Also print critical path task ids.")
    args = ap.parse_args()

    k = compute_kpis(args.team, args.tasks, args.plan, horizon_weeks=args.horizon_weeks)
    print(json.dumps(k, indent=2))

    if args.print_slack:
        s = compute_slack_by_task(args.plan)
        print(json.dumps({"slack_hours": s}, indent=2))

    if args.print_critpath:
        path, total = critical_path(args.tasks)
        print(json.dumps({"critical_path": path, "total_hours": total}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
