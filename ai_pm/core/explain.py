# ai_pm/core/explain.py
# Phase 5 — Assignment rationales for Review/Locks
#
# WHAT / WHY
# ----------
# Given team.csv, task_graph.json, plan.json (+ optional history.csv),
# compute per-task metrics that explain an assignment and surface the top 2–3
# textual driver bullets (e.g., "Strong skill match", "Improves fairness").
#
# USAGE (CLI)
# -----------
# python -m core.explain \
#   --team samples/team.csv \
#   --tasks runs/<...>/normalized/task_graph.json \
#   --plan  runs/<...>/plan/plan.json \
#   [--history samples/history.csv] \
#   [--horizon-weeks 4] \
#   [--out /tmp/rationales.json]
#
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Data models (lightweight)
# -----------------------------

@dataclass
class Member:
    member_id: str
    name: str = ""
    weekly_capacity: float = 40.0
    current_load: float = 0.0
    timezone: str = "Asia/Kolkata"
    skills: Dict[str, float] = field(default_factory=dict)  # canonical -> level


@dataclass
class TaskReq:
    task_id: str
    title: str = ""
    estimate_h: float = 0.0
    required_skills: List[Dict[str, Any]] = field(default_factory=list)
    due_by: Optional[str] = None  # YYYY-MM-DD


@dataclass
class PlanTask:
    task_id: str
    member_id: str
    member_name: str = ""
    estimate_h: float = 0.0
    start: Optional[str] = None  # ISO string
    end: Optional[str] = None    # ISO string
    due_by: Optional[str] = None
    due_violation: Optional[bool] = None


# -----------------------------
# Parsing helpers
# -----------------------------

def load_team_csv(path: Path) -> Dict[str, Member]:
    """Read team.csv into Member objects (skills = JSON list in the CSV column)."""
    members: Dict[str, Member] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            mid = (r.get("member_id") or "").strip()
            if not mid:
                continue
            m = Member(
                member_id=mid,
                name=(r.get("name") or "").strip(),
                weekly_capacity=float(r.get("weekly_capacity_hours") or 40.0),
                current_load=float(r.get("current_load_hours") or 0.0),
                timezone=(r.get("timezone") or "Asia/Kolkata").strip() or "Asia/Kolkata",
            )
            # skills: JSON list of {name, level}
            try:
                skills_json = r.get("skills") or "[]"
                arr = json.loads(skills_json) if isinstance(skills_json, str) else skills_json
                for it in (arr or []):
                    nm = (it.get("name") or "").strip().lower()
                    if nm:
                        m.skills[nm] = float(it.get("level") or 0.0)
            except Exception:
                pass  # tolerate malformed skills column
            members[mid] = m
    return members


def load_task_graph(path: Path) -> Dict[str, TaskReq]:
    data = json.loads(path.read_text(encoding="utf-8"))
    res: Dict[str, TaskReq] = {}
    for t in (data.get("tasks") or []):
        res[(t.get("task_id") or "").strip()] = TaskReq(
            task_id=(t.get("task_id") or "").strip(),
            title=(t.get("title") or "").strip(),
            estimate_h=float(t.get("estimate_h") or 0.0),
            required_skills=t.get("required_skills") or [],
            due_by=t.get("due_by"),
        )
    return res


def load_plan(path: Path) -> List[PlanTask]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[PlanTask] = []
    for t in (data.get("tasks") or []):
        out.append(
            PlanTask(
                task_id=(t.get("task_id") or "").strip(),
                member_id=(t.get("member_id") or "").strip(),
                member_name=(t.get("member_name") or "").strip(),
                estimate_h=float(t.get("estimate_h") or 0.0),
                start=t.get("start"),
                end=t.get("end"),
                due_by=t.get("due_by"),
                due_violation=t.get("due_violation"),
            )
        )
    return out


def load_history(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    return rows


# -----------------------------
# Metric computations
# -----------------------------

def _skill_match(member: Member, reqs: List[Dict[str, Any]]) -> float:
    """
    Average over required skills:
      if level_req>0  -> min(1, level_member / level_req)
      else            -> presence 1/0
    """
    if not reqs:
        return 1.0
    vals: List[float] = []
    for r in reqs:
        nm = (r.get("name") or "").strip().lower()
        lvl_req = float(r.get("level") or 0.0)
        lvl_mem = float(member.skills.get(nm, 0.0))
        if lvl_req > 0:
            vals.append(min(1.0, lvl_mem / max(0.0001, lvl_req)))
        else:
            vals.append(1.0 if lvl_mem > 0 else 0.0)
    return sum(vals) / len(vals) if vals else 1.0


def _utilization_by_member(plan: List[PlanTask], members: Dict[str, Member], horizon_weeks: float = 4.0) -> Dict[str, float]:
    """Assigned_hours / available_hours_over_horizon, clipped to [0, 10] for stability."""
    assigned: Dict[str, float] = {}
    for t in plan:
        assigned[t.member_id] = assigned.get(t.member_id, 0.0) + float(t.estimate_h or 0.0)
    util: Dict[str, float] = {}
    for mid, m in members.items():
        avail_weekly = max(0.0, float(m.weekly_capacity) - float(m.current_load))
        avail_total = max(1e-6, avail_weekly * float(horizon_weeks))
        util[mid] = min(10.0, assigned.get(mid, 0.0) / avail_total)
    return util


def _fairness_effect_for_task(t: PlanTask, util: Dict[str, float]) -> float:
    """Member utilization minus team average (≈ lower/near-zero is better)."""
    if not util:
        return 0.0
    avg = sum(util.values()) / len(util)
    return float(util.get(t.member_id, 0.0) - avg)


def _continuity(member: Member, task: TaskReq, history_rows: List[Dict[str, Any]], tasks_by_id: Dict[str, TaskReq]) -> float:
    """
    Share of task's required skills that appear in this member's historical tasks' skills (0..1).
    Fallback (no history): reuse skill_match on current skills.
    """
    if not history_rows:
        return _skill_match(member, task.required_skills)

    seen: Dict[str, float] = {}
    for h in history_rows:
        if (h.get("member_id") or "").strip() != member.member_id:
            continue
        tid = (h.get("task_id") or "").strip()
        if not tid:
            continue
        t_old = tasks_by_id.get(tid)
        if not t_old:
            continue
        for rs in (t_old.required_skills or []):
            nm = (rs.get("name") or "").strip().lower()
            if nm:
                seen[nm] = max(seen.get(nm, 0.0), float(rs.get("level") or 0.0))

    if not seen:
        return _skill_match(member, task.required_skills)

    matches, total = 0, 0
    for r in (task.required_skills or []):
        nm = (r.get("name") or "").strip().lower()
        if not nm:
            continue
        total += 1
        if nm in seen:
            matches += 1
    return (matches / total) if total else 1.0


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)  # handles "+05:30"
    except Exception:
        return None


def _end_of_day(due_by: str, tzinfo) -> Optional[datetime]:
    """Return YYYY-MM-DD at 23:59:59 (same tz as plan timestamps)."""
    try:
        y, m, d = [int(x) for x in due_by.split("-")]
        dt = datetime(y, m, d, 23, 59, 59)
        return dt.replace(tzinfo=tzinfo)
    except Exception:
        return None


def _slack_hours(pt: PlanTask) -> Optional[float]:
    """Due-by end of day minus scheduled end (hours). Positive=has slack; negative=late."""
    if not pt.due_by:
        return None
    end = _parse_iso(pt.end)
    if not end:
        return None
    due = _end_of_day(pt.due_by, end.tzinfo)
    if not due:
        return None
    delta = (due - end).total_seconds() / 3600.0
    return round(delta, 2)


# -----------------------------
# Rationale & drivers
# -----------------------------

def _drivers(skill: float, fair_eff: float, cont: float, slack: Optional[float]) -> List[str]:
    bullets: List[str] = []
    # Skill
    if skill >= 0.9:
        bullets.append(f"Strong skill match ({skill:.2f}).")
    elif skill >= 0.7:
        bullets.append(f"Good skill match ({skill:.2f}).")
    # Continuity
    if cont >= 0.8:
        bullets.append(f"High continuity with past work ({cont:.2f}).")
    elif cont >= 0.5:
        bullets.append(f"Moderate continuity ({cont:.2f}).")
    # Fairness
    if fair_eff < -0.05:
        bullets.append(f"Improves fairness (below team avg load by {abs(fair_eff):.2f}).")
    elif fair_eff > 0.10:
        bullets.append(f"Risk of overload (above avg by {fair_eff:.2f}).")
    # Deadline slack
    if slack is not None:
        if slack >= 8:
            bullets.append(f"Healthy deadline slack (+{slack:.0f} h).")
        elif slack < 0:
            bullets.append(f"Deadline risk (−{abs(slack):.0f} h).")

    # Order by importance using an explicit rank (no tricky boolean arithmetic).
    def _rank(s: str) -> int:
        if "Deadline risk" in s:
            return 0
        if "Strong skill match" in s:
            return 1
        if "Good skill match" in s:
            return 2
        if "High continuity" in s:
            return 3
        if "Moderate continuity" in s:
            return 4
        if "Improves fairness" in s:
            return 5
        if "Risk of overload" in s:
            return 6
        if "Healthy deadline slack" in s:
            return 7
        return 8

    bullets.sort(key=_rank)
    return (bullets[:3] or ["Neutral trade-offs."])


def explain_plan(
    team_csv: Path,
    tasks_json: Path,
    plan_json: Path,
    history_csv: Optional[Path] = None,
    horizon_weeks: float = 4.0,
) -> Dict[str, Any]:
    """Main API: compute rationale per task for a given plan."""
    members = load_team_csv(team_csv)
    tasks = load_task_graph(tasks_json)
    plan = load_plan(plan_json)
    history = load_history(history_csv)

    util = _utilization_by_member(plan, members, horizon_weeks=horizon_weeks)

    out: Dict[str, Any] = {"rationales": {}}
    for p in plan:
        tinfo = tasks.get(p.task_id, TaskReq(task_id=p.task_id, title=p.task_id))
        m = members.get(p.member_id)
        slack = _slack_hours(p)

        if not m:
            out["rationales"][p.task_id] = {
                "task_id": p.task_id,
                "member_id": p.member_id,
                "member_name": p.member_name,
                "skill_match": 0.0,
                "fairness_effect": None,
                "continuity": 0.0,
                "slack_hours": slack,
                "drivers": ["Unknown member; cannot compute rationale."],
            }
            continue

        skill = _skill_match(m, tinfo.required_skills)
        fair_eff = _fairness_effect_for_task(p, util)
        cont = _continuity(m, tinfo, history, tasks)
        drivers = _drivers(skill, fair_eff, cont, slack)

        out["rationales"][p.task_id] = {
            "task_id": p.task_id,
            "title": tinfo.title or p.task_id,
            "member_id": p.member_id,
            "member_name": p.member_name or m.name or p.member_id,
            "estimate_h": p.estimate_h,
            "skill_match": round(float(skill), 4),
            "fairness_effect": round(float(fair_eff), 4) if fair_eff is not None else None,
            "continuity": round(float(cont), 4),
            "slack_hours": slack,
            "drivers": drivers,
        }
    return out


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute per-assignment rationales for a plan.")
    ap.add_argument("--team", required=True, help="team.csv path")
    ap.add_argument("--tasks", required=True, help="task_graph.json path (normalized or split)")
    ap.add_argument("--plan", required=True, help="plan.json path")
    ap.add_argument("--history", default=None, help="history.csv path (optional)")
    ap.add_argument("--horizon-weeks", type=float, default=4.0, help="horizon weeks for utilization math")
    ap.add_argument("--out", default=None, help="write JSON to this path; print to stdout if omitted")
    return ap.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()
    team = Path(args.team)
    tasks = Path(args.tasks)
    plan = Path(args.plan)
    history = Path(args.history) if args.history else None

    result = explain_plan(team, tasks, plan, history_csv=history, horizon_weeks=float(args.horizon_weeks))
    txt = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
        print(f"[OK] Rationales written -> {args.out}")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
