# ai_pm/core/optimizer.py
# Phase 4 — CP-SAT assignment + earliest-feasible scheduling (post-process)
#
# What this module provides:
#   1) load_team_csv(path) -> members dict
#   2) load_task_graph(path) -> tasks list
#   3) solve_assignments(team, tasks, theta) -> (assign, loads, obj)
#      - CP-SAT chooses exactly one owner per task with feasibility & capacity
#   4) build_schedule(assign, tasks, settings) -> plan with calendar dates, flags due_by violations
#   5) solve_plan(team_csv, task_graph_json, settings_yaml, theta) -> plan dict (ready for export)
#   6) main() for CLI testing
#
# Hard constraints enforced in model:
#   - One owner per task (coverage=100%)
#   - Member must satisfy all required_skills (name + min level if provided)
#   - Total assigned hours per member <= available_weekly_hours * horizon_weeks
#
# Post-processing (deterministic):
#   - Earliest-start schedule by topological order, obeying dependencies,
#     honoring Mon–Fri, `workday_hours`, starting from "today" in settings timezone.
#   - Due-by dates are checked (flagged if violated).
#
# Objective (weighted by theta):
#   + skill_fit (higher is better)
#   + continuity bonus (from history if present)
#   - fairness penalty (L1 deviation of per-member loads from mean)
#
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from zoneinfo import ZoneInfo
from ortools.sat.python import cp_model


# ----------------------------
# Data models (lightweight)
# ----------------------------

@dataclass
class Member:
    member_id: str
    name: str
    role: str
    seniority: int
    weekly_capacity: float
    current_load: float
    tz: str
    skills: Dict[str, int]  # canonical skill -> level (0..5)
    # derived:
    available_weekly: float = 0.0
    history_bonus: float = 0.0  # small positive bonus from history (avg review_score/5)

@dataclass
class Task:
    task_id: str
    title: str
    est_h: float
    req_skills: List[Dict[str, Any]]  # [{name, level?}]
    deps: List[str]
    due_by: Optional[date] = None
    # derived:
    dur_days: int = 0  # ceil(est_h / workday_hours)


# ----------------------------
# I/O helpers
# ----------------------------

def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _yaml_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def load_team_csv(path: Path) -> Dict[str, Member]:
    """
    team.csv columns:
      member_id, name, role, seniority_level(0–5), weekly_capacity_hours, current_load_hours, timezone, skills(JSON list of {name,level})
    """
    members: Dict[str, Member] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            mid = (row.get("member_id") or "").strip()
            if not mid:
                continue
            skills_cell = (row.get("skills") or "").strip()
            skill_map: Dict[str, int] = {}
            try:
                data = json.loads(skills_cell) if skills_cell else []
                if isinstance(data, list):
                    for it in data:
                        nm = str(it.get("name", "")).strip()
                        lv = it.get("level", 0)
                        try:
                            lv = int(lv)
                        except Exception:
                            lv = 0
                        if nm:
                            skill_map[nm] = max(0, min(5, lv))
            except Exception:
                # tolerate malformed rows
                pass
            weekly = float(row.get("weekly_capacity_hours") or 0.0)
            current = float(row.get("current_load_hours") or 0.0)
            available = max(0.0, weekly - current)
            m = Member(
                member_id=mid,
                name=(row.get("name") or "").strip(),
                role=(row.get("role") or "").strip(),
                seniority=int(float(row.get("seniority_level") or 0)),
                weekly_capacity=weekly,
                current_load=current,
                tz=(row.get("timezone") or "Asia/Kolkata").strip(),
                skills=skill_map,
                available_weekly=available,
                history_bonus=0.0,
            )
            members[mid] = m
    return members

def load_history_csv(path: Optional[Path]) -> Dict[str, float]:
    """
    Optional: history.csv with columns: task_id,member_id,outcome,review_score,cycle_time_hrs
    Returns: avg review_score per member_id (for a small continuity bonus).
    """
    rating_by_member: Dict[str, List[float]] = defaultdict(list)
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            mid = (row.get("member_id") or "").strip()
            try:
                score = float(row.get("review_score") or 0.0)
            except Exception:
                score = 0.0
            if mid:
                rating_by_member[mid].append(score)
    avg_by_member = {m: (sum(v)/len(v) if v else 0.0) for m, v in rating_by_member.items()}
    return avg_by_member

def load_task_graph(path: Path, workday_hours: int) -> List[Task]:
    data = _json_load(path)
    tasks: List[Task] = []
    for t in data.get("tasks", []):
        tid = (t.get("task_id") or "").strip()
        if not tid:
            continue
        due = t.get("due_by")
        due_date = None
        if due:
            try:
                due_date = date.fromisoformat(str(due))
            except Exception:
                due_date = None
        est = float(t.get("estimate_h") or 0.0)
        dur_days = max(1, math.ceil(est / max(1, workday_hours)))
        tasks.append(Task(
            task_id=tid,
            title=t.get("title", ""),
            est_h=est,
            req_skills=list(t.get("required_skills", [])),
            deps=[str(d) for d in (t.get("dependencies", []) or [])],
            due_by=due_date,
            dur_days=dur_days,
        ))
    return tasks

def load_settings(path: Path) -> Dict[str, Any]:
    cfg = _yaml_load(path)
    return {
        "timezone": cfg.get("timezone", "Asia/Kolkata"),
        "workweek": cfg.get("workweek", ["Mon","Tue","Wed","Thu","Fri"]),
        "workday_hours": int(cfg.get("workday_hours", 8)),
        "data_dir": cfg.get("data_dir", "runs"),
    }


# ----------------------------
# Scoring helpers
# ----------------------------

def _member_meets_requirements(member: Member, task: Task) -> bool:
    for req in task.req_skills:
        nm = str(req.get("name", "")).strip()
        if not nm:
            return False
        min_level = int(float(req.get("level", 0)))
        if member.skills.get(nm, -1) < min_level:
            return False
    return True

def _skill_fit_score(member: Member, task: Task) -> float:
    """
    0..1 score. Average over required skills of (member_level / max(1, required_level or 5)).
    If task doesn't specify level, assume target level=3 for normalization.
    """
    if not task.req_skills:
        return 0.0
    scores: List[float] = []
    for req in task.req_skills:
        nm = str(req.get("name", "")).strip()
        tgt = int(float(req.get("level", 3)))  # default target level 3
        lvl = member.skills.get(nm, 0)
        scores.append(min(1.0, (lvl / max(1, tgt))))
    return sum(scores) / len(scores)

def _theta_defaults() -> Dict[str, float]:
    # defaults from prompt: skill_fit 0.5, fairness 0.2, continuity 0.2, deadline_risk 0.1 (deadline used in post KPIs)
    return {"skill_fit": 0.5, "fairness": 0.2, "continuity": 0.2, "deadline_risk": 0.1}


# ----------------------------
# CP-SAT assignment model
# ----------------------------

def solve_assignments(team: Dict[str, Member], tasks: List[Task], theta: Optional[Dict[str, float]] = None,
                      horizon_weeks: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, float], float]:
    """
    Build a CP-SAT model to assign each task to exactly one feasible member.
    Capacity is enforced as: sum(est_h of tasks assigned to m) <= available_weekly[m] * H
    Returns:
        assign: dict task_id -> member_id
        load_by_member: dict member_id -> total assigned hours
        objective_value
    """
    theta = theta or _theta_defaults()
    model = cp_model.CpModel()

    member_ids = list(team.keys())
    task_ids = [t.task_id for t in tasks]

    # Horizon weeks heuristic if not provided
    total_hours = sum(t.est_h for t in tasks)
    total_weekly_available = sum(max(0.0, m.available_weekly) for m in team.values())
    if total_weekly_available <= 0:
        total_weekly_available = 1.0
    min_weeks = math.ceil(total_hours / total_weekly_available) if total_hours > 0 else 1
    H = horizon_weeks or max(1, min_weeks + 1)  # add a little slack

    # Decision vars x[t,m] in {0,1}
    x: Dict[Tuple[str,str], cp_model.IntVar] = {}
    for t in tasks:
        for mid in member_ids:
            x[(t.task_id, mid)] = model.NewBoolVar(f"x_{t.task_id}_{mid}")

    # One owner per task
    for t in tasks:
        model.Add(sum(x[(t.task_id, mid)] for mid in member_ids) == 1)

    # Feasibility (skills): forbid assignments where member doesn't meet requirements
    for t in tasks:
        for mid in member_ids:
            if not _member_meets_requirements(team[mid], t):
                model.Add(x[(t.task_id, mid)] == 0)

    # Capacity per member across the horizon (hours)
    load_vars: Dict[str, cp_model.LinearExpr] = {}
    for mid in member_ids:
        load_expr = sum(int(round(t.est_h)) * x[(t.task_id, mid)] for t in tasks)  # int for CP-SAT
        load_vars[mid] = load_expr
        cap = int(round(team[mid].available_weekly * H))
        model.Add(load_expr <= cap)

    # Fairness: minimize L1 deviation from average
    avg_load = int(round(total_hours / len(member_ids))) if member_ids else 0
    abs_devs = []
    for mid in member_ids:
        dev = model.NewIntVar(-10**7, 10**7, f"dev_{mid}")
        model.Add(dev == load_vars[mid] - avg_load)
        abs_dev = model.NewIntVar(0, 10**7, f"abs_{mid}")
        model.AddAbsEquality(abs_dev, dev)
        abs_devs.append(abs_dev)

    # Skill fit reward & continuity bonus
    SCALE = 1000
    skill_reward_terms = []
    cont_bonus_terms = []
    for t in tasks:
        for mid in member_ids:
            sf = _skill_fit_score(team[mid], t)  # 0..1
            skill_reward_terms.append(int(round(SCALE * sf)) * x[(t.task_id, mid)])
            cont = team[mid].history_bonus  # 0..1
            cont_bonus_terms.append(int(round(SCALE * cont)) * x[(t.task_id, mid)])

    # Objective: maximize skill_fit + continuity - fairness
    model.Maximize(
        int(round(theta.get("skill_fit", 0.5) * 1_000)) * sum(skill_reward_terms)
        + int(round(theta.get("continuity", 0.2) * 1_000)) * sum(cont_bonus_terms)
        - int(round(theta.get("fairness", 0.2) * 1_000)) * sum(abs_devs)
    )

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0  # keep it snappy
    solver.parameters.num_search_workers = 8
    result = solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Optimizer could not find a feasible assignment.")

    # Extract solution
    assign: Dict[str, str] = {}
    loads: Dict[str, float] = {mid: 0.0 for mid in member_ids}
    for t in tasks:
        for mid in member_ids:
            if solver.Value(x[(t.task_id, mid)]) == 1:
                assign[t.task_id] = mid
                loads[mid] += t.est_h
                break

    return assign, loads, solver.ObjectiveValue()


# ----------------------------
# Scheduling (post-process)
# ----------------------------

def _is_workday(d: date, workweek: List[str]) -> bool:
    week = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    return week[d.weekday()] in workweek

def _next_workday(d: date, workweek: List[str]) -> date:
    cur = d
    while not _is_workday(cur, workweek):
        cur += timedelta(days=1)
    return cur

def _add_work_hours(start: datetime, hours: float, tz: ZoneInfo, workweek: List[str], workday_hours: int) -> datetime:
    """
    Advance a datetime by N work hours, skipping non-workdays and outside daily window.
    Daily window: 09:00–(09:00+workday_hours) in the provided tz.
    """
    current = start
    remaining = float(hours)

    # Normalize to a workday at 09:00
    day_start = datetime.combine(current.date(), datetime.min.time()).replace(tzinfo=tz).replace(hour=9, minute=0)
    if current < day_start:
        current = day_start
    if not _is_workday(current.date(), workweek):
        nd = _next_workday(current.date(), workweek)
        current = datetime.combine(nd, datetime.min.time()).replace(tzinfo=tz).replace(hour=9, minute=0)

    while remaining > 1e-6:
        end_today = datetime.combine(current.date(), datetime.min.time()).replace(tzinfo=tz).replace(hour=9) + timedelta(hours=workday_hours)
        span = (end_today - current).total_seconds() / 3600.0
        if remaining <= span + 1e-9:
            current = current + timedelta(hours=remaining)
            remaining = 0.0
        else:
            remaining -= span
            # move to next workday 09:00
            nd = _next_workday(current.date() + timedelta(days=1), workweek)
            current = datetime.combine(nd, datetime.min.time()).replace(tzinfo=tz).replace(hour=9, minute=0)
    return current

def build_schedule(assign: Dict[str, str], tasks: List[Task], team: Dict[str, Member], settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic, greedy forward scheduler:
      - topological order,
      - each task starts at the max of (all parent finishes, member's next free),
      - member works 8h/day, Mon–Fri as per settings.
    Returns a plan dict with task schedules and summary loads.
    """
    tz_proj = ZoneInfo(settings.get("timezone", "Asia/Kolkata"))
    workweek = settings.get("workweek", ["Mon","Tue","Wed","Thu","Fri"])
    workday_hours = int(settings.get("workday_hours", 8))

    # Build adjacency & in-degree
    task_map = {t.task_id: t for t in tasks}
    indeg = {t.task_id: 0 for t in tasks}
    children: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
    for t in tasks:
        for p in t.deps:
            if p in task_map:
                children[p].append(t.task_id)
                indeg[t.task_id] += 1

    # Topological order
    q = deque([tid for tid, d in indeg.items() if d == 0])
    topo: List[str] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != len(tasks):
        raise RuntimeError("Task graph is not a DAG; cannot schedule.")

    # Member availability pointer (datetime)
    now = datetime.now(tz_proj)
    # Align to next workday 09:00 in project tz
    start_day = _next_workday(now.date(), workweek)
    start_dt = datetime.combine(start_day, datetime.min.time()).replace(tzinfo=tz_proj).replace(hour=9)
    member_free_at: Dict[str, datetime] = {mid: start_dt for mid in team.keys()}

    # Track when each task finishes (for precedence)
    finish_at: Dict[str, datetime] = {}

    plan_tasks: List[Dict[str, Any]] = []
    due_violations = 0

    for tid in topo:
        t = task_map[tid]
        mid = assign[tid]
        m_tz = ZoneInfo(team[mid].tz or settings.get("timezone", "Asia/Kolkata"))

        # Earliest allowed by deps (in project tz)
        earliest = start_dt
        if t.deps:
            parents_finish = [finish_at[p] for p in t.deps if p in finish_at]
            if parents_finish:
                earliest = max(earliest, max(parents_finish))

        # Member available time (convert to member tz and start 09:00 local)
        avail = member_free_at[mid].astimezone(m_tz)
        earliest_local = earliest.astimezone(m_tz)
        start_time = max(avail, earliest_local)
        start_time = datetime.combine(start_time.date(), datetime.min.time()).replace(tzinfo=m_tz).replace(hour=9)
        if not _is_workday(start_time.date(), workweek):
            nd = _next_workday(start_time.date(), workweek)
            start_time = datetime.combine(nd, datetime.min.time()).replace(tzinfo=m_tz).replace(hour=9)

        end_time = _add_work_hours(start_time, t.est_h, m_tz, workweek, workday_hours)

        # Record finish in project tz and update member availability in project tz (keeps one clock)
        finish_at[tid] = end_time.astimezone(tz_proj)
        member_free_at[mid] = end_time.astimezone(tz_proj)

        due_flag = False
        if t.due_by:
            due_dt = datetime.combine(t.due_by, datetime.min.time()).replace(tzinfo=m_tz).replace(hour=18)  # COB local
            if end_time > due_dt:
                due_flag = True
                due_violations += 1

        plan_tasks.append({
            "task_id": tid,
            "title": t.title,
            "member_id": mid,
            "member_name": team[mid].name,
            "estimate_h": t.est_h,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "due_by": t.due_by.isoformat() if t.due_by else None,
            "due_violation": due_flag,
        })

    # Summary loads (hours)
    loads = defaultdict(float)
    for pt in plan_tasks:
        loads[pt["member_id"]] += float(pt["estimate_h"])

    plan = {
        "settings": settings,
        "summary": {
            "members": [
                {"member_id": mid, "name": team[mid].name, "load_hours": loads[mid], "weekly_available": team[mid].available_weekly}
                for mid in team
            ],
            "due_by_violations": due_violations,
        },
        "tasks": plan_tasks,
    }
    return plan


# ----------------------------
# Orchestration
# ----------------------------

def solve_plan(team_csv: Path, task_graph_json: Path, settings_yaml: Path,
               history_csv: Optional[Path] = None,
               theta: Optional[Dict[str, float]] = None,
               horizon_weeks: Optional[int] = None) -> Dict[str, Any]:
    """
    High-level helper: load inputs → CP-SAT assignments → greedy schedule → return plan dict.
    """
    settings = load_settings(settings_yaml)
    members = load_team_csv(team_csv)
    # history bonus (simple)
    hist = load_history_csv(history_csv)
    for mid, m in members.items():
        if mid in hist and hist[mid] > 0:
            # normalize 0..1 (5-star scale) with a small weight
            m.history_bonus = min(1.0, max(0.0, hist[mid] / 5.0))

    tasks = load_task_graph(task_graph_json, settings["workday_hours"])

    assign, loads, obj = solve_assignments(members, tasks, theta=theta, horizon_weeks=horizon_weeks)
    plan = build_schedule(assign, tasks, members, settings)
    plan["objective"] = obj
    return plan


# ----------------------------
# CLI (quick test)
# ----------------------------

def main() -> int:
    """
    Example:
      python -m core.optimizer \
        --team samples/team.csv \
        --tasks /tmp/task_graph.splitted.json \
        --settings config/settings.yaml \
        --history samples/history.csv \
        --theta 0.5 0.2 0.2 0.1 \
        --horizon-weeks 4 \
        --out /tmp/plan.json
    """
    ap = argparse.ArgumentParser(description="CP-SAT assignment + greedy scheduler → plan.json")
    ap.add_argument("--team", type=Path, required=True, help="Path to team.csv")
    ap.add_argument("--tasks", type=Path, required=True, help="Path to normalized or split task_graph.json")
    ap.add_argument("--settings", type=Path, default=Path("config/settings.yaml"))
    ap.add_argument("--history", type=Path, help="Optional path to history.csv")
    ap.add_argument("--theta", type=float, nargs=4, metavar=("SKILL","FAIR","CONT","DEADL"),
                    help="Weights (skill_fit, fairness, continuity, deadline_risk). Deadline used in KPIs later.")
    ap.add_argument("--horizon-weeks", type=int, help="Capacity horizon in weeks (default auto)")
    ap.add_argument("--out", type=Path, help="Where to write the plan.json (stdout if omitted)")
    args = ap.parse_args()

    theta = None
    if args.theta:
        theta = {"skill_fit": args.theta[0], "fairness": args.theta[1], "continuity": args.theta[2], "deadline_risk": args.theta[3]}

    try:
        plan = solve_plan(args.team, args.tasks, args.settings, history_csv=args.history, theta=theta, horizon_weeks=args.horizon_weeks)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] plan written -> {args.out}")
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
