# ai_pm/scripts/patch_team_for_locks.py
# Purpose: Patch team.csv so that every assignment lock in locks.json is eligible.
# Usage:
#   python -m scripts.patch_team_for_locks \
#     --project "Demo" \
#     --team /teamspace/studios/this_studio/ai_pm/samples/team.csv \
#     --tasks /teamspace/studios/this_studio/ai_pm/scenarios/task_graph.reopt.json \
#     --locks /teamspace/studios/this_studio/ai_pm/scenarios/locks_scenario.json
#
# Output:
#   runs/<Project>/<timestamp>/inputs/team.patched.csv
#   runs/<Project>/<timestamp>/logs/patch.log
#
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# -------- helpers --------

def _repo_root() -> Path:
    # this script lives under ai_pm/scripts/
    return Path(__file__).resolve().parents[1]

def _new_run_dir(project: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = _repo_root() / "runs" / project / ts
    run.mkdir(parents=True, exist_ok=True)
    (run / "inputs").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    return run

def _load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in (data.get("tasks") or [])}
    return tasks

def _read_team_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    return rows, rd.fieldnames  # type: ignore

def _write_team_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

def _parse_skills_cell(val: str) -> List[Dict[str, Any]]:
    try:
        return json.loads(val) if isinstance(val, str) else (val or [])
    except Exception:
        return []

def _skills_index(skills_arr: List[Dict[str, Any]]) -> Dict[str, float]:
    idx: Dict[str, float] = {}
    for s in skills_arr or []:
        nm = (s.get("name") or "").strip().lower()
        lv = float(s.get("level") or 0.0)
        if nm:
            idx[nm] = lv
    return idx

def _skills_to_json(idx: Dict[str, float]) -> str:
    arr = [{"name": k, "level": v} for k, v in sorted(idx.items())]
    return json.dumps(arr, ensure_ascii=False)

# -------- patch logic --------

def patch_team_for_locks(project: str, team_csv: Path, tasks_json: Path, locks_json: Path) -> Path:
    run_dir = _new_run_dir(project)
    log_path = run_dir / "logs" / "patch.log"

    # copy inputs for traceability
    inputs_dir = run_dir / "inputs"
    team_copy = inputs_dir / "team.original.csv"
    tasks_copy = inputs_dir / "task_graph.json"
    locks_copy = inputs_dir / "locks.json"
    team_copy.write_bytes(team_csv.read_bytes())
    tasks_copy.write_bytes(tasks_json.read_bytes())
    locks_copy.write_bytes(locks_json.read_bytes())

    tasks = _load_tasks(tasks_json)
    locks = json.loads(locks_json.read_text(encoding="utf-8"))
    assign_locks: Dict[str, str] = dict(locks.get("assignment_locks") or {})

    rows, fieldnames = _read_team_csv(team_csv)
    if "skills" not in (fieldnames or []):
        raise SystemExit("team.csv is missing the 'skills' column.")

    # Create index by member_id
    members: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        mid = (r.get("member_id") or "").strip()
        if mid:
            members[mid] = r

    log_lines: List[str] = []
    changed_members: Dict[str, Dict[str, float]] = {}

    for tid, mid in assign_locks.items():
        if tid not in tasks:
            log_lines.append(f"[WARN] Task {tid} not found in task_graph; skipping.")
            continue
        if mid not in members:
            log_lines.append(f"[WARN] Member {mid} not found in team.csv; skipping {tid}.")
            continue

        reqs = tasks[tid].get("required_skills") or []
        row = members[mid]
        cur_skills_arr = _parse_skills_cell(row.get("skills") or "[]")
        idx = _skills_index(cur_skills_arr)
        before = dict(idx)

        for r in reqs:
            nm = (r.get("name") or "").strip().lower()
            lvl_req = float(r.get("level") or 0.0)
            # Our eligibility check treats level==0 as "must exist". Patch to level 1.
            needed = 1.0 if lvl_req <= 0 else lvl_req
            cur = float(idx.get(nm, 0.0))
            if cur + 1e-9 < needed:
                idx[nm] = needed

        if idx != before:
            row["skills"] = _skills_to_json(idx)
            changed_members[mid] = idx
            log_lines.append(f"[PATCH] {tid} -> {mid}: skills updated from {before} to {idx}")
        else:
            log_lines.append(f"[OK] {tid} -> {mid}: already eligible.")

    # Write patched CSV
    patched_path = inputs_dir / "team.patched.csv"
    _write_team_csv(patched_path, fieldnames, rows)

    # Log summary
    summary = [
        f"[INFO] Project={project}",
        f"[INFO] Run dir = {run_dir}",
        f"[INFO] Patched team -> {patched_path}",
        f"[INFO] Changed members = {list(changed_members.keys()) or 'None'}",
    ]
    log_path.write_text("\n".join(summary + log_lines), encoding="utf-8")

    # Also print to stdout for CLI visibility
    print("\n".join(summary))
    for line in log_lines:
        print(line)

    return patched_path

# -------- CLI --------

def main() -> None:
    ap = argparse.ArgumentParser(description="Patch team.csv so every locked assignment is eligible.")
    ap.add_argument("--project", required=True, help="Project name for runs/<Project>/<ts>/")
    ap.add_argument("--team", required=True, type=Path, help="Path to team.csv")
    ap.add_argument("--tasks", required=True, type=Path, help="Path to task_graph.json (normalized or split)")
    ap.add_argument("--locks", required=True, type=Path, help="Path to locks.json with assignment_locks")
    args = ap.parse_args()

    out = patch_team_for_locks(args.project, args.team, args.tasks, args.locks)
    print(f"[OK] Patched team written to: {out}")

if __name__ == "__main__":
    main()
