# ai_pm/app_streamlit/04_review.py
# Phase 5 — Review, Locks & Re-solve (Scenarios A/B) + Quick Fix for due_by
# Phase 6 — Coactive Learning: "Prefer B over A" updates θ in runs/<Project>/preferences.json
#
# WHAT THIS PAGE DOES
# -------------------
# • Load Scenario A (plan.json) + team.csv + task_graph.json (+ optional history.csv and locks.json).
# • Detect due_by violations in A and offer “Quick Fix” (end_before = due_by EOD).
# • Pre-check feasibility (eligibility, deps, dates, due_by) and re-solve around locks → Scenario B.
# • Save B + KPIs in runs/<project>/<ts>/plan/, log to runs/<project>/<ts>/logs/ui.log.
# • Show KPIs A vs B, diffs, and per-task rationales.
# • (Phase 6) Button: “Prefer B over A — Update θ” stores new θ at runs/<project>/preferences.json
#
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# --- Robust import shim so 'core.*' modules resolve under Streamlit ---
BASE_DIR = Path(__file__).resolve().parents[1]  # .../ai_pm
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# ----------------------------------------------------------------------

from _state import init_app_state, sidebar_defaults
from core.optimizer import (
    load_settings,
    load_team_csv,
    load_task_graph,
    solve_assignments,
    build_schedule,
)
from core.kpis import compute_kpis
from core.storage import new_run_dir, snapshot_inputs
from core.explain import explain_plan  # rationale engine (Phase 5)

RUNS_DIR = BASE_DIR / "runs"

# ----------------------------
# Phase-6 preference helpers
# ----------------------------

def _theta_defaults() -> Dict[str, float]:
    return {"skill_fit": 0.5, "fairness": 0.2, "continuity": 0.2, "deadline_risk": 0.1}

def _prefs_path(project: str) -> Path:
    pdir = RUNS_DIR / project
    pdir.mkdir(parents=True, exist_ok=True)
    return pdir / "preferences.json"

def _load_prefs(project: str) -> Dict[str, float]:
    fp = _prefs_path(project)
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            # ensure all keys exist
            th = _theta_defaults()
            th.update({k: float(v) for k, v in (data or {}).items() if k in th})
            return th
        except Exception:
            return _theta_defaults()
    return _theta_defaults()

def _renorm(theta: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in theta.values())
    if s <= 0:
        return _theta_defaults()
    return {k: max(0.0, float(v)) / s for k, v in theta.items()}

def _nudge_theta(old: Dict[str, float], signals: Dict[str, float], eta: float) -> Dict[str, float]:
    """
    Increase weight by eta where B beats A (signal>0). Then clip [0,1] and renorm.
    signals keys: skill_fit (higher better), fairness (higher is better meaning LOWER stddev),
                  continuity (higher better), deadline_risk (higher is better meaning FEWER violations).
    """
    th = dict(old)
    for k in ["skill_fit", "fairness", "continuity", "deadline_risk"]:
        sig = float(signals.get(k, 0.0) or 0.0)
        if sig > 0:
            th[k] = float(th.get(k, 0.0)) + float(eta)
        # if sig<=0 we keep the same (conservative).
    # clip+renorm
    th = {k: min(1.0, max(0.0, float(v))) for k, v in th.items()}
    th = _renorm(th)
    return th

# ----------------------------
# Small helpers (existing)
# ----------------------------

def _load_json_bytes(file) -> Dict[str, Any]:
    return json.loads(file.getvalue().decode("utf-8"))

def _plan_to_df(plan: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for t in plan.get("tasks", []):
        rows.append({
            "task_id": t.get("task_id"),
            "title": t.get("title"),
            "member_id": t.get("member_id"),
            "member_name": t.get("member_name"),
            "estimate_h": float(t.get("estimate_h") or 0.0),
            "start": t.get("start"),
            "end": t.get("end"),
            "due_by": t.get("due_by"),
            "due_violation": bool(t.get("due_violation") or False),
        })
    return pd.DataFrame(rows)

def _parse_iso_opt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _tz_from_settings(settings: Dict[str, Any]) -> ZoneInfo:
    tz_str = settings.get("timezone") or "Asia/Kolkata"
    try:
        return ZoneInfo(tz_str)
    except Exception:
        return ZoneInfo("Asia/Kolkata")

def _end_of_day(date_yyyy_mm_dd: str, tz: ZoneInfo) -> Optional[datetime]:
    try:
        y, m, d = [int(x) for x in date_yyyy_mm_dd.split("-")]
        return datetime(y, m, d, 23, 59, 59, tzinfo=tz)
    except Exception:
        return None

def _date_lock_violations(plan_obj: Dict[str, Any], date_locks: Dict[str, Dict[str, str]]) -> List[str]:
    """Return human-readable violations for date locks vs a built plan."""
    idx = {t["task_id"]: t for t in plan_obj.get("tasks", [])}
    msgs: List[str] = []
    for tid, win in (date_locks or {}).items():
        pt = idx.get(tid)
        if not pt:
            msgs.append(f"[{tid}] not present in plan; cannot enforce date window.")
            continue
        start = _parse_iso_opt(pt.get("start"))
        end   = _parse_iso_opt(pt.get("end"))
        if "start" in win:
            want = _parse_iso_opt(win["start"])
            if want and start and start != want:
                msgs.append(f"[{tid}] start exact lock mismatch: got {start}, want {want}.")
        if "end" in win:
            want = _parse_iso_opt(win["end"])
            if want and end and end != want:
                msgs.append(f"[{tid}] end exact lock mismatch: got {end}, want {want}.")
        if "start_after" in win:
            cut = _parse_iso_opt(win["start_after"])
            if cut and start and start < cut:
                msgs.append(f"[{tid}] start {start} is before window start_after {cut}.")
        if "end_before" in win:
            cut = _parse_iso_opt(win["end_before"])
            if cut and end and end > cut:
                msgs.append(f"[{tid}] end {end} exceeds window end_before {cut}.")
    return msgs

def _sum_locked_hours(plan_a: Dict[str, Any], assignment_locks: Dict[str, str]) -> Dict[str, float]:
    hrs: Dict[str, float] = {}
    for t in plan_a.get("tasks", []):
        tid = (t.get("task_id") or "").strip()
        if tid in assignment_locks:
            mid = assignment_locks[tid]
            est = float(t.get("estimate_h") or 0.0)
            hrs[mid] = hrs.get(mid, 0.0) + est
    return hrs

def _apply_capacity_adjustment(members, locked_by_member: Dict[str, float], horizon_weeks: int) -> None:
    H = max(1, int(horizon_weeks))
    for mid, locked in locked_by_member.items():
        if mid in members:
            m = members[mid]
            m.current_load = float(m.current_load) + (float(locked) / H)
            m.available_weekly = max(0.0, float(m.weekly_capacity) - float(m.current_load))

def _build_diff(plan_a: Dict[str, Any], plan_b: Dict[str, Any]) -> pd.DataFrame:
    A = {t["task_id"]: t for t in plan_a.get("tasks", [])}
    B = {t["task_id"]: t for t in plan_b.get("tasks", [])}
    task_ids = sorted(set(A.keys()) | set(B.keys()))
    rows = []
    for tid in task_ids:
        ta, tb = A.get(tid, {}), B.get(tid, {})
        rows.append({
            "task_id": tid,
            "title": tb.get("title") or ta.get("title") or tid,
            "owner_A": f'{ta.get("member_name","") or ta.get("member_id","")}',
            "owner_B": f'{tb.get("member_name","") or tb.get("member_id","")}',
            "start_A": ta.get("start"),
            "start_B": tb.get("start"),
            "end_A": ta.get("end"),
            "end_B": tb.get("end"),
            "due_violation_A": bool(ta.get("due_violation") or False),
            "due_violation_B": bool(tb.get("due_violation") or False),
            "changed_owner": (ta.get("member_id") != tb.get("member_id")),
            "changed_time": (ta.get("start") != tb.get("start")) or (ta.get("end") != tb.get("end")),
        })
    return pd.DataFrame(rows)

def _business_window(d: datetime, workday_hours: int) -> Tuple[datetime, datetime]:
    ws = d.replace(hour=9, minute=0, second=0, microsecond=0)
    we = d.replace(hour=9 + workday_hours, minute=0, second=0, microsecond=0)
    return ws, we

def _add_business_hours(start: datetime, hours: float, workday_hours: int, workweek: List[str]) -> datetime:
    """Advance forward by <hours> respecting workweek/workday; timezone-aware."""
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
        cur = (cur + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    return cur

def _topo_order(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    indeg: Dict[str, int] = {tid: 0 for tid in tasks}
    succ: Dict[str, List[str]] = {tid: [] for tid in tasks}
    for t in tasks.values():
        for d in (t.get("dependencies") or []):
            if d in indeg:
                indeg[t["task_id"]] += 1
                succ[d].append(t["task_id"])
    from collections import deque
    Q = deque([tid for tid, k in indeg.items() if k == 0])
    order: List[str] = []
    while Q:
        u = Q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                Q.append(v)
    if len(order) != len(tasks):
        raise RuntimeError("Task graph contains a cycle.")
    return order

def _load_team_skillmap_csv(team_csv_path: Path) -> Dict[str, Dict[str, float]]:
    """member_id -> {skill: level}"""
    out: Dict[str, Dict[str, float]] = {}
    with team_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
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

def _locks_feasible(team_csv: Path, tasks_json: Path, assignment_locks: Dict[str, str],
                    date_locks: Dict[str, Dict[str, str]], settings: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Eligibility + precedence + date window + due_by feasibility (no capacity), tz-safe."""
    tz = _tz_from_settings(settings)
    workweek = settings.get("workweek", ["Mon","Tue","Wed","Thu","Fri"])
    workday_hours = int(settings.get("workday_hours", 8))

    data = json.loads(tasks_json.read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in (data.get("tasks") or [])}
    team = _load_team_skillmap_csv(team_csv)

    ok = True; msgs: List[str] = []

    # 1) assignment locks eligibility
    for tid, mid in (assignment_locks or {}).items():
        if tid not in tasks:
            ok = False; msgs.append(f"[ASSIGN] {tid} not in task graph.")
            continue
        if mid not in team:
            ok = False; msgs.append(f"[ASSIGN] {tid} locked to unknown member_id={mid}.")
            continue
        if not _eligible(team[mid], tasks[tid].get("required_skills") or []):
            ok = False; msgs.append(f"[ASSIGN] {tid}→{mid} infeasible: member lacks required skills/levels.")

    # 2) earliest schedule ignoring capacity for date & due_by sanity
    order = _topo_order(tasks)
    earliest_end: Dict[str, datetime] = {}
    earliest_start: Dict[str, datetime] = {}
    baseline = datetime.now(tz).replace(hour=9, minute=0, second=0, microsecond=0)

    for tid in order:
        t = tasks[tid]
        deps = t.get("dependencies") or []
        est_h = float(t.get("estimate_h") or 0.0)

        es = baseline
        for d in deps:
            if d in earliest_end:
                es = max(es, earliest_end[d])

        dl = (date_locks or {}).get(tid, {})
        sa = _parse_iso_opt(dl.get("start_after")) if dl else None
        if sa:
            es = max(es, sa)
        s_exact = _parse_iso_opt(dl.get("start")) if dl else None
        if s_exact and s_exact < es:
            ok = False; msgs.append(f"[DATE] {tid} exact start {s_exact} < earliest allowed {es} (deps/start_after).")
        if s_exact:
            es = s_exact

        ee = _add_business_hours(es, est_h, workday_hours, workweek)

        e_exact = _parse_iso_opt(dl.get("end")) if dl else None
        if e_exact and e_exact < ee:
            ok = False; msgs.append(f"[DATE] {tid} exact end {e_exact} < earliest feasible end {ee}.")
        if e_exact:
            ee = e_exact

        eb = _parse_iso_opt(dl.get("end_before")) if dl else None
        if eb and eb < ee:
            ok = False; msgs.append(f"[DATE] {tid} end_before {eb} < earliest feasible end {ee}.")

        earliest_start[tid] = es
        earliest_end[tid] = ee

        if t.get("due_by"):
            due_eod = _end_of_day(t["due_by"], tz)
            if due_eod and due_eod < ee:
                ok = False; msgs.append(f"[DUE] {tid} due_by {due_eod} < earliest feasible end {ee} (deps/duration).")

    return ok, msgs

def _eligibility_report(team_csv: Path, tasks_json: Path, settings: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return a list of tasks with zero eligible members and per-task reason bullets."""
    data = json.loads(tasks_json.read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in (data.get("tasks") or [])}
    team = _load_team_skillmap_csv(team_csv)

    zero: List[str] = []
    reasons: Dict[str, List[str]] = {}

    for tid, t in tasks.items():
        reqs = t.get("required_skills") or []
        eligible_any = False
        bullets: List[str] = []
        for mid, mskills in team.items():
            if _eligible(mskills, reqs):
                eligible_any = True
            else:
                miss = []
                for r in reqs:
                    nm = (r.get("name") or "").strip().lower()
                    lv = float(r.get("level") or 0.0)
                    mv = float(mskills.get(nm, 0.0))
                    if (lv > 0 and mv + 1e-9 < lv) or (lv == 0 and mv <= 0):
                        miss.append(f'needs {nm}≥{int(lv) if lv>0 else 1} (has {int(mv)})')
                bullets.append(f'• {tid}: {mid} ' + (", ".join(miss) if miss else ""))
        if not eligible_any:
            zero.append(tid)
            reasons[tid] = bullets or ["no eligible members"]
    return zero, reasons

# --- Quick-fix helpers ---

def _list_due_by_violations(plan_obj: Dict[str, Any], tz: ZoneInfo) -> List[Dict[str, str]]:
    """Return [{task_id, title, end, due_by}] for tasks ending after due_by EOD."""
    out: List[Dict[str, str]] = []
    for t in plan_obj.get("tasks", []):
        due = t.get("due_by")
        end = t.get("end")
        if not due or not end:
            continue
        due_eod = _end_of_day(due, tz)
        end_dt = _parse_iso_opt(end)
        if due_eod and end_dt and end_dt > due_eod:
            out.append({
                "task_id": t.get("task_id"),
                "title": t.get("title"),
                "end": end,
                "due_by": due,
            })
    return out

def _build_quickfix_date_locks(violations: List[Dict[str, str]], tz: ZoneInfo) -> Dict[str, Dict[str, str]]:
    """Create date_locks with end_before = due_by EOD for each violating task."""
    locks: Dict[str, Dict[str, str]] = {}
    for v in violations:
        tid = v["task_id"]
        due_eod = _end_of_day(v["due_by"], tz)
        if due_eod:
            locks[tid] = {"end_before": due_eod.isoformat()}
    return locks

# ----------------------------
# Page UI
# ----------------------------

st.set_page_config(page_title="AI-PM — Review (A/B)", page_icon="📝", layout="wide")
init_app_state()
sidebar_defaults()

st.title("04 — Review & Re-Optimize (Scenarios A/B)")
st.caption("Lock owners and/or dates, re-solve around them to produce Scenario B. Quick Fix converts due_by misses into end_before locks.")

# Inputs
c0, c1, c2 = st.columns(3)
with c0:
    project_name = st.text_input("Project name", value="Demo")
with c1:
    team_file = st.file_uploader("team.csv", type=["csv"])
with c2:
    tasks_file = st.file_uploader("task_graph.json (normalized or split)", type=["json"])

c3, c4, c5 = st.columns(3)
with c3:
    plan_file = st.file_uploader("Scenario A: plan.json", type=["json"])
with c4:
    history_file = st.file_uploader("history.csv (optional)", type=["csv"])
with c5:
    locks_file = st.file_uploader("locks.json (optional; assignment/date locks)", type=["json"])

if not (team_file and tasks_file and plan_file):
    st.info("Upload team.csv, task_graph.json, and an existing plan.json to begin.")
    st.stop()

# Settings & tz
settings_path = BASE_DIR / "config" / "settings.yaml"
settings = load_settings(settings_path)
tz = _tz_from_settings(settings)

# Create new run folder up front (so all paths/logs live here)
run_dir = Path(new_run_dir(project_name))
logs_dir = run_dir / "logs"
plan_dir = run_dir / "plan"
inputs_dir = run_dir / "inputs"
for d in [logs_dir, plan_dir, inputs_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Logger to file
log_path = logs_dir / "ui.log"
logger = logging.getLogger("ai_pm.ui.review")
logger.handlers.clear()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
logger.info("==== Review page start ====")

# Persist uploads under run_dir/inputs
team_path = inputs_dir / "team.csv"; team_path.write_bytes(team_file.getvalue())
tasks_path = inputs_dir / "task_graph.json"; tasks_path.write_bytes(tasks_file.getvalue())
planA_path = inputs_dir / "plan_A.json"; planA_path.write_bytes(plan_file.getvalue())
hist_path: Optional[Path] = None
if history_file:
    hist_path = inputs_dir / "history.csv"; hist_path.write_bytes(history_file.getvalue())
locks_json: Dict[str, Any] = {}
if locks_file:
    locks_json = _load_json_bytes(locks_file)
    (inputs_dir / "locks.json").write_text(json.dumps(locks_json, ensure_ascii=False, indent=2), encoding="utf-8")

# Load A
try:
    plan_A = json.loads(planA_path.read_text(encoding="utf-8"))
    df_A = _plan_to_df(plan_A)
    st.success(f"Scenario A loaded. Artifacts will be saved under: `{run_dir}`")
    logger.info("Loaded Scenario A, team.csv, task_graph.json")
except Exception as e:
    st.error(f"Failed to read plan.json: {e}")
    logger.exception("Failed to read A plan")
    st.stop()

# Team & member options for locks
members = load_team_csv(team_path)
member_opts = [(m.member_id, m.name or m.member_id) for m in members.values()]
member_id_list = [mid for mid, _ in member_opts]
member_label_map = {mid: label for mid, label in member_opts}

# Locks UI
st.subheader("Locks")
st.caption("Provide locks via upload OR use the controls below.")
assignment_locks: Dict[str, str] = {}
date_locks: Dict[str, Dict[str, str]] = {}
theta_default = _load_prefs(project_name)  # Phase-6: preload current θ
horizon_default = 4

if locks_json:
    assignment_locks = dict(locks_json.get("assignment_locks") or {})
    date_locks = dict(locks_json.get("date_locks") or {})
    # locks.json can also carry theta/horizon hints; non-destructive merge:
    hinted = locks_json.get("theta") or {}
    theta_default = _renorm({**theta_default, **{k: float(v) for k, v in hinted.items() if k in theta_default}})
    horizon_default = int(locks_json.get("horizon_weeks") or horizon_default)
    st.info("Locks loaded from file. You can still add/override below.")

for _, row in df_A.iterrows():
    tid = row["task_id"]; title = row["title"]; owner = row["member_id"]; start = row["start"]; end = row["end"]
    with st.container():
        st.markdown(f"**{tid}** — {title}")
        c1, c2, c3 = st.columns([1.2, 2, 2.8])
        with c1:
            do_lock = st.checkbox("Lock owner", value=(tid in assignment_locks), key=f"lock_{tid}")
        with c2:
            idx = member_id_list.index(owner) if owner in member_id_list else 0
            sel = st.selectbox("Owner",
                options=member_id_list,
                format_func=lambda mid: f"{member_label_map.get(mid, mid)} ({mid})",
                index=member_id_list.index(assignment_locks.get(tid, owner)) if assignment_locks.get(tid, owner) in member_id_list else idx,
                disabled=not do_lock,
                key=f"owner_{tid}"
            )
        with c3:
            with st.expander("Date lock (optional)"):
                st.caption("ISO like 2025-10-23T09:00:00+05:30. Fill any subset.")
                curr = date_locks.get(tid, {})
                s_exact = st.text_input("start (exact)", value=curr.get("start",""), key=f"s_exact_{tid}")
                e_exact = st.text_input("end (exact)", value=curr.get("end",""), key=f"e_exact_{tid}")
                s_after = st.text_input("start_after (window)", value=curr.get("start_after",""), key=f"s_after_{tid}")
                e_before = st.text_input("end_before (window)", value=curr.get("end_before",""), key=f"e_before_{tid}")
        if do_lock:
            assignment_locks[tid] = sel
        win: Dict[str, str] = {}
        if s_exact.strip(): win["start"] = s_exact.strip()
        if e_exact.strip(): win["end"] = e_exact.strip()
        if s_after.strip(): win["start_after"] = s_after.strip()
        if e_before.strip(): win["end_before"] = e_before.strip()
        if win:
            date_locks[tid] = win
    st.divider()

# Quick Fix — detect due_by violations in A and offer end_before locks
violations = _list_due_by_violations(plan_A, tz)
apply_quick_fix = False
if violations:
    st.subheader("Quick Fix — due_by violations in Scenario A")
    st.warning("The tasks below finish after their due_by. Apply one-click locks (end_before = due_by end-of-day) and re-solve.")
    st.table(pd.DataFrame(violations))
    apply_quick_fix = st.checkbox("Apply Quick Fix date locks for the listed tasks", value=True, help="Creates end_before = due_by 23:59:59 for each violating task.")

# Weights + horizon
st.subheader("Weights (θ) & horizon")
c5, c6, c7, c8, c9 = st.columns(5)
with c5: th_skill = st.slider("skill_fit ↑", 0.0, 1.0, float(theta_default["skill_fit"]), 0.05)
with c6: th_fair  = st.slider("fairness ↓", 0.0, 1.0, float(theta_default["fairness"]), 0.05)
with c7: th_cont  = st.slider("continuity ↑", 0.0, 1.0, float(theta_default["continuity"]), 0.05)
with c8: th_dead  = st.slider("deadline_risk ↓", 0.0, 1.0, float(theta_default["deadline_risk"]), 0.05)
with c9: horizon_weeks = st.number_input("Horizon (weeks)", min_value=1, value=int(horizon_default), step=1)
theta = {"skill_fit": th_skill, "fairness": th_fair, "continuity": th_cont, "deadline_risk": th_dead}

# PRE-CHECK BUTTON
if st.button("✅ Pre-check locks (feasibility)", type="secondary"):
    eff_date_locks = dict(date_locks)
    if apply_quick_fix:
        eff_date_locks.update(_build_quickfix_date_locks(violations, tz))
    ok, msgs = _locks_feasible(team_path, tasks_path, assignment_locks, eff_date_locks, settings)
    if ok:
        st.success("Locks/windows/due_by are feasible against deps/durations.")
        logger.info("Pre-check OK")
    else:
        st.error("Infeasible locks/windows:\n" + "\n".join(" - " + m for m in msgs))
        logger.warning("Pre-check failed: %s", msgs)

# RE-SOLVE BUTTON
run_btn = st.button("♻️ Re-solve around locks (build Scenario B)", type="primary")

if run_btn:
    logger.info("Re-solve clicked. theta=%s horizon=%s", theta, horizon_weeks)

    # Merge Quick Fix into date_locks for this run
    eff_date_locks = dict(date_locks)
    if apply_quick_fix:
        ql = _build_quickfix_date_locks(violations, tz)
        eff_date_locks.update(ql)
        logger.info("Quick Fix date_locks applied: %s", ql)

    # Hard pre-check
    ok, msgs = _locks_feasible(team_path, tasks_path, assignment_locks, eff_date_locks, settings)
    if not ok:
        st.error("Infeasible locks/windows:\n" + "\n".join(" - " + m for m in msgs))
        logger.error("Hard pre-check failed: %s", msgs)
        st.stop()

    # Load tasks
    try:
        tasks_all = load_task_graph(tasks_path, settings["workday_hours"])
    except Exception as e:
        st.error(f"Failed to load task graph: {e}")
        logger.exception("Failed to load task graph")
        st.stop()

    # Adjust capacity for locked hours measured from A
    locked_by_member = _sum_locked_hours(plan_A, assignment_locks)
    _apply_capacity_adjustment(members, locked_by_member, int(horizon_weeks))

    # Solve only for unlocked tasks
    locked_set = set(assignment_locks.keys())
    tasks_unlocked = [t for t in tasks_all if t.task_id not in locked_set]

    try:
        assign_unlocked, loads, obj = solve_assignments(members, tasks_unlocked, theta=theta, horizon_weeks=int(horizon_weeks))
        logger.info("Solve unlocked OK — obj=%s", obj)
    except Exception as e:
        zero, reasons = _eligibility_report(team_path, tasks_path, settings)
        if zero:
            st.error("Re-optimization failed: solver infeasible.\n"
                     "Tasks with ZERO eligible members:\n" +
                     "\n".join([f" - {tid}\n   " + "\n   ".join(reasons.get(tid, [])) for tid in zero]))
            logger.error("Infeasible — zero eligible: %s", zero)
        else:
            st.error(f"Re-optimization failed: {e}")
            logger.exception("Solve failed")
        st.stop()

    # Merge and schedule all
    assignments_all = dict(assign_unlocked); assignments_all.update(assignment_locks)
    try:
        plan_B = build_schedule(assignments_all, tasks_all, members, settings)
        plan_B["objective"] = obj
        logger.info("Schedule built OK")
    except Exception as e:
        st.error(f"Scheduling failed: {e}")
        logger.exception("Schedule failed")
        st.stop()

    # Validate date locks hard against B
    viol = _date_lock_violations(plan_B, eff_date_locks)
    if viol:
        st.error("Date lock violations detected:\n" + "\n".join(" - " + m for m in viol))
        logger.error("Date lock violations: %s", viol)
        st.stop()

    # Save B plan & KPIs under this run
    planB_path = plan_dir / "plan_B.json"
    planB_path.write_text(json.dumps(plan_B, ensure_ascii=False, indent=2), encoding="utf-8")

    kpis_B = {}
    try:
        kpis_B = compute_kpis(team_path, tasks_path, planB_path, horizon_weeks=int(horizon_weeks)) or {}
        (plan_dir / "kpis_B.json").write_text(json.dumps(kpis_B, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("KPIs B written")
    except Exception as e:
        st.warning(f"Failed to compute KPIs for B: {e}")
        logger.exception("KPIs for B failed")

    # KPIs for A (also persist for side-by-side)
    kpis_A = {}
    try:
        kpis_A = compute_kpis(team_path, tasks_path, planA_path, horizon_weeks=int(horizon_weeks)) or {}
        (plan_dir / "kpis_A.json").write_text(json.dumps(kpis_A, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Snapshot inputs
    try:
        snapshot_inputs(run_dir, [team_path.as_posix(), tasks_path.as_posix(), settings_path.as_posix(), planA_path.as_posix()])
    except Exception as e:
        st.warning(f"Snapshot warning: {e}")
        logger.warning("Snapshot warning: %s", e)

    st.success("Scenario B created and saved.")
    st.write(f"**Run folder:** `{run_dir}`")
    st.write(f"**Plan B:** `{planB_path}`  |  **KPIs:** `{plan_dir / 'kpis_B.json'}`")
    st.caption(f"Logs: `{log_path}`")

    # KPIs compare
    st.subheader("KPIs — A vs B")
    cols = st.columns(6)
    def _metric(cols, i: int, key: str, va, vb):
        val = "—" if vb is None else str(vb)
        if va is None or vb is None:
            cols[i].metric(key, value=val, delta=None)
        else:
            try:
                cols[i].metric(key, value=val, delta=f"{(vb - va):+0.4g}")
            except Exception:
                cols[i].metric(key, value=val, delta=None)

    for i, key in enumerate(["coverage","capacity_violations","avg_skill_fit","utilization_stddev","critical_path_hours","due_by_violations"]):
        _metric(cols, i, key, kpis_A.get(key), kpis_B.get(key))

    # Diff table
    st.subheader("Diffs — owner & timing")
    diff_df = _build_diff(plan_A, plan_B)
    st.dataframe(diff_df, use_container_width=True, hide_index=True)

    # Rationale drawer for selected task
    st.subheader("Rationale — A vs B (select a task)")
    sel_tid = st.selectbox("Task", options=list(diff_df["task_id"]))
    # Persist A/B plans to files under plan_dir for rationale engine
    tmpA = plan_dir / "__A.json"; tmpB = plan_dir / "__B.json"
    tmpA.write_text(json.dumps(plan_A, ensure_ascii=False), encoding="utf-8")
    tmpB.write_text(json.dumps(plan_B, ensure_ascii=False), encoding="utf-8")
    try:
        ratA = explain_plan(team_path, tasks_path, tmpA, history_csv=hist_path)
        ratB = explain_plan(team_path, tasks_path, tmpB, history_csv=hist_path)
    except Exception as e:
        st.warning(f"Rationale compute failed: {e}")
        ratA, ratB = {"rationales": {}}, {"rationales": {}}

    rA = (ratA.get("rationales") or {}).get(sel_tid)
    rB = (ratB.get("rationales") or {}).get(sel_tid)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Scenario A rationale**")
        if rA:
            st.json({k: rA[k] for k in ["member_name","skill_match","continuity","fairness_effect","slack_hours","drivers"] if k in rA})
        else:
            st.info("No rationale for this task in A.")
    with cB:
        st.markdown("**Scenario B rationale**")
        if rB:
            st.json({k: rB[k] for k in ["member_name","skill_match","continuity","fairness_effect","slack_hours","drivers"] if k in rB})
        else:
            st.info("No rationale for this task in B.")

    # ----------------------------
    # Phase 6 — Coactive Learning UI
    # ----------------------------
    st.subheader("Coactive learning — Prefer B over A (update θ)")
    eta = st.slider("Learning rate (η)", 0.01, 0.5, 0.10, 0.01, help="How strongly to nudge weights toward features where B outperforms A.")
    prefs_before = _load_prefs(project_name)
    st.caption(f"Current θ file: `{_prefs_path(project_name)}`")
    st.json({"old_theta": prefs_before})

    # Compute signals from KPIs; continuity via rationales if history is available
    s_skill = float((kpis_B.get("avg_skill_fit") or 0.0) - (kpis_A.get("avg_skill_fit") or 0.0))
    s_fair  = float((kpis_A.get("utilization_stddev") or 0.0) - (kpis_B.get("utilization_stddev") or 0.0))  # lower is better
    s_dead  = float((kpis_A.get("due_by_violations") or 0.0) - (kpis_B.get("due_by_violations") or 0.0))    # fewer is better

    def _avg_cont(rats: Dict[str, Any]) -> Optional[float]:
        try:
            vals = [float(v.get("continuity")) for v in (rats.get("rationales") or {}).values() if v.get("continuity") is not None]
            return sum(vals)/len(vals) if vals else None
        except Exception:
            return None

    contA = _avg_cont(ratA)
    contB = _avg_cont(ratB)
    s_cont = 0.0 if (contA is None or contB is None) else float(contB - contA)

    st.json({"signals": {"skill_fit": s_skill, "fairness": s_fair, "continuity": s_cont, "deadline_risk": s_dead}})

    if st.button("👍 Prefer B over A — Update θ", type="primary"):
        new_theta = _nudge_theta(prefs_before, {"skill_fit": s_skill, "fairness": s_fair, "continuity": s_cont, "deadline_risk": s_dead}, eta)
        _prefs_path(project_name).write_text(json.dumps(new_theta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Preferences updated. old=%s new=%s eta=%s signals=%s", prefs_before, new_theta, eta,
                    {"skill_fit": s_skill, "fairness": s_fair, "continuity": s_cont, "deadline_risk": s_dead})
        st.success("Preferences updated and saved. Optimizer (03) will use these as defaults.")
        st.json({"old_theta": prefs_before, "new_theta": new_theta})

# Per your rule: main()
def main() -> None:
    print("This is a Streamlit page. Launch with:\n  streamlit run app_streamlit/04_review.py")

if __name__ == "__main__":
    main()
