# ai_pm/scripts/check_locks_feasibility.py
# Diagnose infeasibility from assignment/date locks and due_by, timezone-safe.

from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple, Set
from zoneinfo import ZoneInfo

# ---------- IO ----------
def _read_settings(p: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def _load_team_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Returns {member_id: {skill_name: level}}"""
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            mid = (r.get("member_id") or "").strip()
            if not mid: 
                continue
            skills_json = r.get("skills") or "[]"
            try:
                arr = json.loads(skills_json) if isinstance(skills_json, str) else skills_json
            except Exception:
                arr = []
            out[mid] = {}
            for it in arr or []:
                nm = (it.get("name") or "").strip().lower()
                lv = float(it.get("level") or 0.0)
                if nm:
                    out[mid][nm] = lv
    return out

def _load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    res: Dict[str, Dict[str, Any]] = {}
    for t in data.get("tasks", []):
        tid = (t.get("task_id") or "").strip()
        if tid:
            res[tid] = t
    return res

def _load_locks(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# ---------- Time helpers (timezone-safe) ----------
def _tz(settings: Dict[str, Any]) -> ZoneInfo:
    tz_str = settings.get("timezone") or "Asia/Kolkata"
    try:
        return ZoneInfo(tz_str)
    except Exception:
        return ZoneInfo("Asia/Kolkata")

def _parse_iso(s: Optional[str], tz: ZoneInfo) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    # make timezone-aware if needed
    return dt if dt.tzinfo else dt.replace(tzinfo=tz)

def _end_of_day(date_yyyy_mm_dd: str, tz: ZoneInfo) -> Optional[datetime]:
    try:
        y, m, d = [int(x) for x in date_yyyy_mm_dd.split("-")]
        return datetime(y, m, d, 23, 59, 59, tzinfo=tz)
    except Exception:
        return None

def _business_window(d: datetime, workday_hours: int) -> tuple[datetime, datetime]:
    # Work window 09:00–(09+workday_hours) in the same tz as d
    ws = d.replace(hour=9, minute=0, second=0, microsecond=0)
    we = d.replace(hour=9 + workday_hours, minute=0, second=0, microsecond=0)
    return ws, we

def _add_business_hours(start: datetime, hours: float, workday_hours: int, workweek: List[str]) -> datetime:
    """Advance forward by <hours> honoring workweek/workday; timezone-aware."""
    hours_left = float(hours)
    cur = start
    idx2label = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    allowed = set(workweek)
    while hours_left > 1e-9:
        label = idx2label[cur.weekday()]
        if label in allowed:
            ws, we = _business_window(cur, workday_hours)
            if cur < ws:
                cur = ws
            if cur < we:
                can = (we - cur).total_seconds()/3600.0
                take = min(can, hours_left)
                cur = cur + timedelta(hours=take)
                hours_left -= take
                if hours_left <= 1e-9:
                    break
        # next day at 09:00 (same tz)
        nxt = (cur + timedelta(days=1))
        cur = nxt.replace(hour=9, minute=0, second=0, microsecond=0)
    return cur

# ---------- Graph ----------
def _topo_order(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    indeg: Dict[str, int] = {tid: 0 for tid in tasks}
    for t in tasks.values():
        for d in (t.get("dependencies") or []):
            if d in indeg:
                indeg[t["task_id"]] += 1
    from collections import defaultdict, deque
    pred = defaultdict(list)
    for t in tasks.values():
        for d in (t.get("dependencies") or []):
            pred[d].append(t["task_id"])
    Q = deque([tid for tid, k in indeg.items() if k == 0])
    order: List[str] = []
    while Q:
        u = Q.popleft()
        order.append(u)
        for v in pred[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                Q.append(v)
    if len(order) != len(tasks):
        raise RuntimeError("Task graph contains a cycle.")
    return order

# ---------- Core checks ----------
def _eligible(member_skills: Dict[str, float], reqs: List[Dict[str, Any]]) -> bool:
    for r in (reqs or []):
        name = (r.get("name") or "").strip().lower()
        lvl_req = float(r.get("level") or 0.0)
        lvl_mem = float(member_skills.get(name, 0.0))
        if lvl_req > 0:
            if lvl_mem + 1e-9 < lvl_req:
                return False
        else:
            if lvl_mem <= 0:
                return False
    return True

def diagnose(team_csv: Path, tasks_json: Path, locks_json: Path, settings_yaml: Path, baseline_start: Optional[datetime] = None) -> Tuple[bool, List[str]]:
    team = _load_team_csv(team_csv)
    tasks = _load_tasks(tasks_json)
    locks = _load_locks(locks_json)
    settings = _read_settings(settings_yaml)
    tz = _tz(settings)

    workweek = settings.get("workweek", ["Mon","Tue","Wed","Thu","Fri"])
    workday_hours = int(settings.get("workday_hours", 8))

    baseline = baseline_start or datetime.now(tz).replace(hour=9, minute=0, second=0, microsecond=0)
    if baseline.tzinfo is None:
        baseline = baseline.replace(tzinfo=tz)

    msgs: List[str] = []
    ok = True

    # 1) assignment locks
    assign_locks: Dict[str, str] = locks.get("assignment_locks", {}) or {}
    for tid, mid in assign_locks.items():
        if tid not in tasks:
            ok = False; msgs.append(f"[ASSIGN] {tid} not in task graph.")
            continue
        if mid not in team:
            ok = False; msgs.append(f"[ASSIGN] {tid} locked to unknown member_id={mid}.")
            continue
        if not _eligible(team[mid], tasks[tid].get("required_skills") or []):
            ok = False; msgs.append(f"[ASSIGN] {tid}→{mid} infeasible: member lacks required skills/levels.")

    # 2) earliest schedule ignoring capacity
    order = _topo_order(tasks)
    earliest_end: Dict[str, datetime] = {}
    earliest_start: Dict[str, datetime] = {}
    date_locks: Dict[str, Dict[str, str]] = (locks.get("date_locks") or {})
    for tid in order:
        t = tasks[tid]
        deps = t.get("dependencies") or []
        est_h = float(t.get("estimate_h") or 0.0)

        es = baseline
        for d in deps:
            if d in earliest_end:
                es = max(es, earliest_end[d])

        dl = date_locks.get(tid, {})
        # window push
        sa = _parse_iso(dl.get("start_after"), tz) if dl else None
        if sa:
            es = max(es, sa)
        # exact start must not be before es
        s_exact = _parse_iso(dl.get("start"), tz) if dl else None
        if s_exact and s_exact < es:
            ok = False; msgs.append(f"[DATE] {tid} exact start {s_exact} < earliest allowed {es} (deps/start_after).")
        if s_exact:
            es = s_exact

        ee = _add_business_hours(es, est_h, workday_hours, workweek)

        # exact end must not be before earliest end
        e_exact = _parse_iso(dl.get("end"), tz) if dl else None
        if e_exact and e_exact < ee:
            ok = False; msgs.append(f"[DATE] {tid} exact end {e_exact} < earliest feasible end {ee}.")
        if e_exact:
            ee = e_exact

        eb = _parse_iso(dl.get("end_before"), tz) if dl else None
        if eb and eb < ee:
            ok = False; msgs.append(f"[DATE] {tid} end_before {eb} < earliest feasible end {ee}.")

        earliest_start[tid] = es
        earliest_end[tid] = ee

        # due_by sanity (EOD)
        if t.get("due_by"):
            due_eod = _end_of_day(t["due_by"], tz)
            if due_eod and due_eod < ee:
                ok = False; msgs.append(f"[DUE] {tid} due_by {due_eod} < earliest feasible end {ee} (deps/duration).")

    return ok, msgs

# ---------- CLI ----------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose infeasibility from locks and due_by constraints (timezone-safe).")
    ap.add_argument("--team", required=True, help="team.csv")
    ap.add_argument("--tasks", required=True, help="split task graph JSON")
    ap.add_argument("--locks", required=True, help="locks.json")
    ap.add_argument("--settings", required=True, help="config/settings.yaml")
    ap.add_argument("--baseline", default=None, help="optional baseline ISO start; default today 09:00 in project tz")
    return ap.parse_args()

def main() -> int:
    args = _parse_args()
    baseline = None
    if args.baseline:
        # baseline parsed in tz from settings inside diagnose; pass as aware if possible
        try:
            baseline = datetime.fromisoformat(args.baseline)
        except Exception:
            baseline = None
    ok, msgs = diagnose(Path(args.team), Path(args.tasks), Path(args.locks), Path(args.settings), baseline_start=baseline)
    if ok:
        print("[OK] Locks/date windows and due_by are feasible against deps/durations.")
        return 0
    print("[ISSUES] Found infeasibilities:")
    for m in msgs:
        print(" -", m)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
