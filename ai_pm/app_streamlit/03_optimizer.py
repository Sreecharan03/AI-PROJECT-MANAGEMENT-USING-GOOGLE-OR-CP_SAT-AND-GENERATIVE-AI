# ai_pm/app_streamlit/03_optimizer.py
# Phase 4 (+ Phase 6 awareness) — Optimizer UI (θ sliders → plan.json + KPIs)
#
# WHAT'S NEW (vs your previous file)
# ----------------------------------
# 1) Auto-load θ from runs/<Project>/preferences.json (Phase 6 coactive prefs).
# 2) Banner showing which θ was applied (and when it was updated).
# 3) Writes *all* artifacts to runs/<Project>/<ts> (plan/, logs/, inputs/). Previews still read from uploads directly.
# 4) File logger at runs/<Project>/<ts>/logs/ui.log (info/errors with paths & parameters).
# 5) Infeasible hints: a clear eligibility report listing tasks with ZERO eligible members and the missing skills/levels.
#
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Local imports
from _state import init_app_state, sidebar_defaults
from core.optimizer import solve_plan
from core.kpis import compute_kpis
from core.storage import new_run_dir, snapshot_inputs

# Optional: Gantt with Altair
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False


# ----------------------------
# Helpers
# ----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # .../ai_pm

def _settings_path() -> Path:
    return BASE_DIR / "config" / "settings.yaml"

def _theta_defaults() -> Dict[str, float]:
    return {"skill_fit": 0.5, "fairness": 0.2, "continuity": 0.2, "deadline_risk": 0.1}

def _prefs_path(project: str) -> Path:
    return BASE_DIR / "runs" / project / "preferences.json"

def _load_prefs_theta(project: str) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Load theta from runs/<Project>/preferences.json if present; else defaults.
    Returns (theta, updated_at_str or None).
    """
    th = _theta_defaults()
    p = _prefs_path(project)
    if not p.exists():
        return th, None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        th_in = obj.get("theta") or {}
        for k, v in th.items():
            if isinstance(th_in.get(k), (int, float)):
                th[k] = float(th_in[k])
        return th, obj.get("updated_at")
    except Exception:
        # If preferences file is malformed, just fall back to defaults silently.
        return th, None

def _read_csv_preview_bytes(file, n: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read a few rows from an UploadedFile (no writes)."""
    try:
        file.seek(0)
        text = file.getvalue().decode("utf-8-sig")
        rows = []
        rdr = csv.DictReader(text.splitlines())
        cols = rdr.fieldnames or []
        for i, r in enumerate(rdr):
            if i >= n:
                break
            rows.append(dict(r))
        return cols, rows
    except Exception:
        return [], []

def _save_uploaded(file, dest_path: Path) -> Path:
    """Persist an UploadedFile to a destination path under the run folder."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    file.seek(0)
    data = file.getvalue()
    if not data:
        raise RuntimeError(f"Uploaded file was empty: {dest_path.name}")
    dest_path.write_bytes(data)
    return dest_path

def _plan_dataframe(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for t in plan.get("tasks", []):
        rows.append({
            "task_id": t.get("task_id"),
            "title": t.get("title"),
            "member_id": t.get("member_id"),
            "member_name": t.get("member_name"),
            "estimate_h": t.get("estimate_h"),
            "start": t.get("start"),
            "end": t.get("end"),
            "due_by": t.get("due_by"),
            "due_violation": t.get("due_violation"),
        })
    return rows

def _workload_dataframe(plan: Dict[str, Any]) -> pd.DataFrame:
    agg: Dict[str, float] = {}
    names: Dict[str, str] = {m["member_id"]: (m.get("name") or m["member_id"])
                             for m in plan.get("summary", {}).get("members", [])}
    for t in plan.get("tasks", []):
        mid = t.get("member_id")
        agg[mid] = agg.get(mid, 0.0) + float(t.get("estimate_h") or 0.0)
    rows = [{"member": names.get(mid, mid), "assigned_hours": hours} for mid, hours in agg.items()]
    return pd.DataFrame(rows)

def _gantt_dataframe(plan: Dict[str, Any]) -> pd.DataFrame:
    """Prepare data for a simple Altair Gantt as a pandas DataFrame."""
    rows = []
    for t in plan.get("tasks", []):
        try:
            rows.append({
                "task": f'{t.get("task_id")} — {t.get("title")}',
                "member": f'{t.get("member_name")} ({t.get("member_id")})',
                "start": t.get("start"),
                "end": t.get("end"),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

# ---- Infeasibility helpers (inline micro-checker, no extra deps) ----

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

def _eligibility_report(team_csv: Path, tasks_json: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return tasks with zero eligible members and per-task reasons (missing skills with levels)."""
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
                        need = int(lv) if lv > 0 else 1
                        miss.append(f'needs {nm}≥{need} (has {int(mv)})')
                bullets.append(f'• {mid}: ' + (", ".join(miss) if miss else ""))
        if not eligible_any:
            zero.append(tid)
            reasons[tid] = bullets or ["no eligible members"]
    return zero, reasons


# ----------------------------
# Page
# ----------------------------

st.set_page_config(page_title="AI-PM — Optimizer", page_icon="🧮", layout="wide")
init_app_state()
sidebar_defaults()

st.title("03 — Optimizer (CP-SAT → plan.json)")
st.caption("Upload team & tasks, choose weights, then Generate Plan. Outputs go to runs/<project>/<timestamp>/plan/.")

settings_path = _settings_path()
st.caption(f"Using `{settings_path}` for timezone/workweek/workday hours.")

# Project name (used for runs/<project>/<ts>/...)
project_name = st.text_input("Project name", value="Demo", help="Used for runs/<project>/<timestamp>/")

# Auto-load θ from preferences (Phase 6)
theta_loaded, updated_ts = _load_prefs_theta(project_name)
if updated_ts:
    st.info(f"Applied θ from `runs/{project_name}/preferences.json` (updated {updated_ts}).")
else:
    st.caption("Using default θ (no preferences.json found for this project).")

# Inputs
st.subheader("Inputs")
colA, colB = st.columns(2)
with colA:
    team_file = st.file_uploader("team.csv", type=["csv"], help="Required")
    history_file = st.file_uploader("history.csv (optional)", type=["csv"])
with colB:
    tasks_file = st.file_uploader("task_graph.json (normalized or split)", type=["json"], help="Required")

# Previews (non-blocking, no writes)
with st.expander("Preview inputs (first 5 rows)"):
    if team_file:
        cols, rows = _read_csv_preview_bytes(team_file, n=5)
        st.markdown("**team.csv**")
        st.write(rows if rows else "(empty or unreadable)")
    if history_file:
        cols, rows = _read_csv_preview_bytes(history_file, n=5)
        st.markdown("**history.csv**")
        st.write(rows if rows else "(empty or unreadable)")
    if tasks_file:
        try:
            tasks_file.seek(0)
            obj = json.loads(tasks_file.getvalue().decode("utf-8"))
            st.markdown("**task_graph.json**")
            st.json({"tasks_count": len(obj.get("tasks", []))})
        except Exception as e:
            st.warning(f"Tasks preview failed: {e}")

# Weights (θ) — default from preferences if available
st.subheader("Weights (θ)")
theta = dict(theta_loaded)  # start from loaded values
c1, c2, c3, c4 = st.columns(4)
with c1:
    theta["skill_fit"] = st.slider("skill_fit ↑", 0.0, 1.0, float(theta["skill_fit"]), 0.05)
with c2:
    theta["fairness"] = st.slider("fairness ↓", 0.0, 1.0, float(theta["fairness"]), 0.05)
with c3:
    theta["continuity"] = st.slider("continuity ↑", 0.0, 1.0, float(theta["continuity"]), 0.05)
with c4:
    theta["deadline_risk"] = st.slider("deadline_risk ↓", 0.0, 1.0, float(theta["deadline_risk"]), 0.05)

horizon_weeks = st.number_input("Horizon (weeks) for capacity", min_value=1, value=4, step=1)

# Run
run_btn = st.button("✅ Generate Plan", type="primary")

if run_btn:
    if not team_file or not tasks_file:
        st.error("Please upload both team.csv and task_graph.json.")
        st.stop()

    # Create run folder first, so all artifacts (including inputs & logs) go there
    run_dir = Path(new_run_dir(project_name))
    plan_dir = run_dir / "plan"
    inputs_dir = run_dir / "inputs"
    logs_dir = run_dir / "logs"
    for d in [plan_dir, inputs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # File logger -> runs/<project>/<ts>/logs/ui.log
    log_path = logs_dir / "ui.log"
    logger = logging.getLogger("ai_pm.ui.optimizer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=== Optimize clicked ===")
    logger.info("project=%s horizon_weeks=%s theta=%s", project_name, horizon_weeks, theta)

    # Persist uploads under run_dir/inputs
    try:
        team_path = _save_uploaded(team_file, inputs_dir / "team.csv")
        tasks_path = _save_uploaded(tasks_file, inputs_dir / "task_graph.json")
        history_path = None
        if history_file:
            history_path = _save_uploaded(history_file, inputs_dir / "history.csv")
        logger.info("inputs saved: team=%s tasks=%s history=%s", team_path, tasks_path, history_path)
    except Exception as e:
        st.error(f"Failed to save uploads: {e}")
        logger.exception("Failed to save uploads")
        st.stop()

    # Solve plan
    try:
        plan = solve_plan(team_path, tasks_path, _settings_path(),
                          history_csv=history_path, theta=theta, horizon_weeks=int(horizon_weeks))
        logger.info("solve_plan OK")
    except Exception as e:
        # Provide detailed infeasible hints
        try:
            zero, reasons = _eligibility_report(team_path, tasks_path)
            if zero:
                msg = ("Optimization failed: solver infeasible.\n"
                       "Tasks with ZERO eligible members:\n" +
                       "\n".join([f" - {tid}\n   " + "\n   ".join(reasons.get(tid, [])) for tid in zero]))
                st.error(msg)
                logger.error(msg)
            else:
                st.error(f"Optimization failed: {e}")
                logger.exception("solve_plan raised")
        except Exception:
            st.error(f"Optimization failed: {e}")
            logger.exception("solve_plan raised (no report)")
        st.stop()

    # Persist plan and KPIs
    plan_path = plan_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("plan written: %s", plan_path)

    # KPIs
    try:
        kpis = compute_kpis(team_path, tasks_path, plan_path, horizon_weeks=int(horizon_weeks))
        (plan_dir / "kpis.json").write_text(json.dumps(kpis, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("kpis written: %s", plan_dir / "kpis.json")
    except Exception as e:
        st.warning(f"KPI computation failed: {e}")
        logger.exception("kpis failed")
        kpis = {}

    # Snapshot inputs (absolute settings path)
    try:
        snap = [team_path.as_posix(), tasks_path.as_posix(), _settings_path().as_posix()]
        if history_path:
            snap.append(history_path.as_posix())
        snapshot_inputs(run_dir, snap)
        logger.info("snapshot complete")
    except Exception as e:
        st.warning(f"Snapshot warning: {e}")
        logger.warning("snapshot warning: %s", e)

    # UI — success + pointers
    st.success("Plan generated and saved.")
    st.write(f"**Run folder:** `{run_dir}`")
    st.write(f"**Plan:** `{plan_path}`  |  **KPIs:** `{plan_dir / 'kpis.json'}`")
    st.caption(f"Logs: `{log_path}`")

    # KPI strip
    st.subheader("KPIs")
    st.json(kpis if kpis else {"info": "KPIs unavailable"})

    # Workload chart
    st.subheader("Member workload (assigned hours)")
    wl_df = _workload_dataframe(plan)
    if not wl_df.empty:
        st.bar_chart(wl_df, x="member", y="assigned_hours", use_container_width=True)
    else:
        st.info("No tasks to show.")

    # Plan table
    st.subheader("Plan tasks")
    st.dataframe(_plan_dataframe(plan), use_container_width=True, hide_index=True)

    # Simple Gantt (Altair + DataFrame)
    st.subheader("Gantt")
    if ALT_AVAILABLE:
        gantt_df = _gantt_dataframe(plan)
        if not gantt_df.empty:
            chart = (
                alt.Chart(gantt_df)
                .mark_bar()
                .encode(
                    x=alt.X("start:T", title="Start"),
                    x2=alt.X2("end:T", title="End"),
                    y=alt.Y("task:N", sort=None, title=None),
                    color=alt.Color("member:N", legend=None),
                    tooltip=[alt.Tooltip("task:N"), alt.Tooltip("member:N"),
                             alt.Tooltip("start:T"), alt.Tooltip("end:T")],
                )
                .properties(height=min(500, 24 * max(1, len(gantt_df))), width="container")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No tasks to render.")
    else:
        st.info("Altair not available; showing table view above.")

# Per your rule: a small main()
def main() -> None:
    print("This is a Streamlit page. Launch with:\n  streamlit run app_streamlit/03_optimizer.py")

if __name__ == "__main__":
    main()
