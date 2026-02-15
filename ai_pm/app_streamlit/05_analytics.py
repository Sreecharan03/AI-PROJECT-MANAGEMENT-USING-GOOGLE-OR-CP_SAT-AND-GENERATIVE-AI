# ai_pm/app_streamlit/05_analytics.py
# Phase 7 — Analytics: utilization histogram, deadline risk heatmap (slack), critical path list.
#
# WHY THIS FILE
# -------------
# Shows three analytics views from real run artifacts:
#   1) Utilization histogram per member (based on scheduled window).
#   2) Deadline-risk heatmap by slack hours (negative = violation).
#   3) Critical path list (order + total hours).
# Falls back to simple tables if Altair isn't available.
#
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from _state import init_app_state, sidebar_defaults

# Optional: Altair for charts
try:
    import altair as alt
    ALT = True
except Exception:
    ALT = False

# Ensure local imports resolve under Streamlit
BASE_DIR = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa
if str(BASE_DIR) not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR))

# Helpers from KPIs (Phase-7)
from core.kpis import per_task_slack, critical_path  # noqa: E402

# ----------------------------
# Filesystem helpers
# ----------------------------

RUNS_DIR = BASE_DIR / "runs"

def _list_projects() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])

def _list_runs(project: str) -> List[Path]:
    root = RUNS_DIR / project
    if not root.exists():
        return []
    runs = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if (d / "plan" / "plan.json").exists() and (d / "plan" / "kpis.json").exists():
            runs.append(d)
    return runs

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _load_run_artifacts(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan = _load_json(run_dir / "plan" / "plan.json")
    kpis = _load_json(run_dir / "plan" / "kpis.json")
    return plan, kpis

def _kpi_rows(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    counts = kpis.get("counts", {})
    return [
        {"metric": "coverage", "value": kpis.get("coverage")},
        {"metric": "capacity_violations", "value": kpis.get("capacity_violations")},
        {"metric": "avg_skill_fit", "value": kpis.get("avg_skill_fit")},
        {"metric": "utilization_stddev", "value": kpis.get("utilization_stddev")},
        {"metric": "critical_path_hours", "value": kpis.get("critical_path_hours")},
        {"metric": "due_by_violations", "value": kpis.get("due_by_violations")},
        {"metric": "tasks_total", "value": counts.get("tasks_total")},
        {"metric": "tasks_scheduled", "value": counts.get("tasks_scheduled")},
        {"metric": "members", "value": counts.get("members")},
    ]

def _workload_df(plan: Dict[str, Any]) -> pd.DataFrame:
    name_by_id = {m["member_id"]: (m.get("name") or m["member_id"])
                  for m in plan.get("summary", {}).get("members", [])}
    agg: Dict[str, float] = {}
    for t in plan.get("tasks", []):
        mid = t.get("member_id")
        agg[mid] = agg.get(mid, 0.0) + float(t.get("estimate_h") or 0.0)
    rows = [{"member_id": mid, "member": name_by_id.get(mid, mid), "assigned_hours": hrs}
            for mid, hrs in agg.items()]
    return pd.DataFrame(rows)

def _plan_span_weeks(plan: Dict[str, Any]) -> int:
    """Approximate scheduled span in *weeks* from min start to max end."""
    def _parse(dt: Optional[str]) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(dt) if dt else None
        except Exception:
            return None
    starts = [s for s in (_parse(t.get("start")) for t in plan.get("tasks", [])) if s]
    ends = [e for e in (_parse(t.get("end")) for t in plan.get("tasks", [])) if e]
    if not starts or not ends:
        return 1
    dur_days = max(1, (max(ends) - min(starts)).days or 1)
    weeks = (dur_days + 6) // 7
    return max(1, int(weeks))

def _utilization_histogram_df(plan: Dict[str, Any]) -> pd.DataFrame:
    """Per-member utilization ≈ assigned_hours / (weekly_available * span_weeks)."""
    span_w = _plan_span_weeks(plan)
    rows = []
    summary_members = plan.get("summary", {}).get("members", [])
    wl = _workload_df(plan)
    idx = wl.set_index("member_id") if not wl.empty else pd.DataFrame()
    for m in summary_members:
        mid = m.get("member_id")
        name = m.get("name") or mid
        weekly_avail = float(m.get("weekly_available") or 0.0)
        assigned = float(idx.loc[mid]["assigned_hours"]) if not idx.empty and mid in idx.index else 0.0
        denom = (weekly_avail * span_w) if weekly_avail > 0 else None
        util = (assigned / denom) if denom else None
        rows.append({
            "member": name,
            "member_id": mid,
            "assigned_hours_total": assigned,
            "weekly_available": weekly_avail,
            "span_weeks": span_w,
            "utilization": util
        })
    return pd.DataFrame(rows)

def _slack_heatmap_df(plan: Dict[str, Any]) -> pd.DataFrame:
    """One row per task with slack (hours)."""
    rows = per_task_slack(plan)  # returns list of dicts
    df = pd.DataFrame(rows)
    # Friendly label for coloring
    def _bucket(x: Optional[float]) -> str:
        if x is None:
            return "no_deadline"
        if x < 0:
            return "late"
        if x < 8:
            return "<8h"
        if x < 24:
            return "8–24h"
        if x < 40:
            return "24–40h"
        return "≥40h"
    if not df.empty and "slack_h" in df.columns:
        df["slack_bucket"] = df["slack_h"].apply(_bucket)
    return df

def _critical_path_table(tasks_json: Path) -> Tuple[pd.DataFrame, Optional[float]]:
    """Return (table_df, total_hours) for critical path; handles missing graphs."""
    if not tasks_json.exists():
        return pd.DataFrame(), None
    order, total = critical_path(tasks_json)
    # Build a small table with order and per-task estimates
    g = _load_json(tasks_json)
    by_id = {t["task_id"]: t for t in g.get("tasks", [])}
    rows = []
    cum = 0.0
    for i, tid in enumerate(order, start=1):
        est = float(by_id.get(tid, {}).get("estimate_h") or 0.0)
        cum += est
        rows.append({"#": i, "task_id": tid, "title": by_id.get(tid, {}).get("title", tid), "estimate_h": est, "cum_h": cum})
    return pd.DataFrame(rows), total

# ----------------------------
# Page UI
# ----------------------------

st.set_page_config(page_title="AI-PM — Analytics", page_icon="📊", layout="wide")
init_app_state()
sidebar_defaults()

st.title("05 — Analytics (utilization • deadline risk • critical path)")

# Project selector
projects = _list_projects()
if not projects:
    st.warning(f"No projects found under `{RUNS_DIR}` yet. Generate a plan on 03 — Optimizer first.")
    st.stop()

project = st.selectbox("Project", projects, index=0)

# Runs selector
runs = _list_runs(project)
if not runs:
    st.warning(f"No completed runs for project '{project}'. Generate a plan first.")
    st.stop()

labels = [r.name for r in runs]
sel = st.selectbox("Select a run", options=labels, index=0)
run_dir = runs[labels.index(sel)]

# Load artifacts
plan, kpis = _load_run_artifacts(run_dir)
inputs_task_graph = run_dir / "inputs" / "task_graph.json"  # may not exist if snapshot was skipped

# KPIs overview
st.subheader("KPIs")
st.dataframe(pd.DataFrame(_kpi_rows(kpis)), hide_index=True, use_container_width=True)

# 1) Utilization histogram
st.subheader("Utilization per member (histogram)")
u_df = _utilization_histogram_df(plan)
no_utilization_data = (
    u_df.empty or
    "utilization" not in u_df.columns or
    u_df["utilization"].isna().all()
)
if no_utilization_data:
    st.info("Not enough info to compute utilization (missing weekly_available or schedule span). Showing assigned hours instead.")
    wl_df = _workload_df(plan)
    if not wl_df.empty:
        st.bar_chart(wl_df, x="member", y="assigned_hours", use_container_width=True)
    else:
        st.info("No tasks in plan.")
else:
    st.caption(f"Span weeks (approx): {int(u_df['span_weeks'].max())}")
    if ALT:
        chart = (
            alt.Chart(u_df)
            .mark_bar()
            .encode(
                x=alt.X("utilization:Q", title="Utilization (0–1)"),
                y=alt.Y("member:N", sort="-x", title=None),
                tooltip=["member:N","utilization:Q","assigned_hours_total:Q","weekly_available:Q","span_weeks:Q"],
            )
            .properties(height=min(500, 28 * max(1, len(u_df))), width="container")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(u_df[["member","utilization","assigned_hours_total","weekly_available","span_weeks"]],
                     hide_index=True, use_container_width=True)

# 2) Deadline risk heatmap (slack)
st.subheader("Deadline risk (slack hours)")
s_df = _slack_heatmap_df(plan)
if s_df.empty:
    st.info("No slack data (no due_by dates or unscheduled tasks).")
else:
    st.caption("Negative slack = past due; small positive slack = higher risk.")
    if ALT:
        hm_df = s_df.copy()
        hm_df["task_label"] = hm_df["task_id"] + " — " + hm_df["title"].fillna("")
        hm_df["member_label"] = hm_df["member_name"].fillna(hm_df["member_id"].fillna(""))
        chart = (
            alt.Chart(hm_df)
            .mark_rect()
            .encode(
                y=alt.Y("task_label:N", sort=None, title=None),
                x=alt.X("member_label:N", title=None),
                color=alt.Color("slack_h:Q", title="Slack (h)"),
                tooltip=["task_id:N","title:N","member_label:N","due_by:N","end:N","slack_h:Q","slack_bucket:N"],
            )
            .properties(height=min(500, 22 * max(1, len(hm_df))), width="container")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(s_df[["task_id","title","member_name","due_by","end","slack_h","slack_bucket"]],
                     hide_index=True, use_container_width=True)

# 3) Critical path list
st.subheader("Critical path (order + total hours)")
cp_df, cp_total = _critical_path_table(inputs_task_graph)
if cp_df.empty:
    st.info("Task graph not found in this run’s inputs. Generate a new run that snapshots inputs/task_graph.json.")
else:
    st.caption(f"Total CP length: {cp_total if cp_total is not None else '—'} hours")
    st.dataframe(cp_df, hide_index=True, use_container_width=True)

# Downloads
st.subheader("Downloads")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download slack.csv",
        data=(s_df.to_csv(index=False).encode("utf-8") if not s_df.empty else "".encode("utf-8")),
        file_name="slack.csv",
        mime="text/csv",
        disabled=s_df.empty
    )
with col2:
    st.download_button(
        "Download critical_path.csv",
        data=(cp_df.to_csv(index=False).encode("utf-8") if not cp_df.empty else "".encode("utf-8")),
        file_name="critical_path.csv",
        mime="text/csv",
        disabled=cp_df.empty
    )

# Per your rule: main()
def main() -> None:
    print("This is a Streamlit page. Launch with:\n  streamlit run app_streamlit/05_analytics.py")

if __name__ == "__main__":
    main()
