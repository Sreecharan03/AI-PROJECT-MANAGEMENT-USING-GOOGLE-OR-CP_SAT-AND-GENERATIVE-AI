#!/usr/bin/env python3
"""
split_tasks_by_skill.py
Purpose: Make the plan feasible by splitting multi-skill tasks that have ZERO eligible members
         into one-skill subtasks (preserving DAG semantics).

Input:
  --tasks : normalized task_graph.json (Phase 3 output; skills already canonical)
  --team  : team.csv
  --out   : output path for the patched task_graph.json
  --dry   : (optional) only print what would change

Behavior:
  1) A task is "multi-skill" if required_skills has length >= 2.
  2) A member is eligible for a task iff they have EVERY required skill (level >= requested; if level omitted, presence is still required).
  3) If a multi-skill task has ZERO eligible members:
       - Split into N subtasks (one per required skill).
       - Split estimate_h across subtasks (sum equals original).
       - Copy original dependencies into each subtask.
       - Replace every downstream dependency on the original with ALL new subtask ids.
       - Remove the original task from the graph.
  4) Warn if any required skill has NO holders in the team (these subtasks will still be unassignable until you fix team skills/ontology).

Example:
  python -m scripts.split_tasks_by_skill \
    --tasks /tmp/task_graph.normalized.json \
    --team samples/team.csv \
    --out /tmp/task_graph.splitted.json
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _save_json(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_team(p: Path) -> Dict[str, Dict[str, int]]:
    """
    Returns: member_id -> { skill_name -> level }
    """
    team: Dict[str, Dict[str, int]] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            mid = (row.get("member_id") or "").strip()
            if not mid:
                continue
            skills_raw = (row.get("skills") or "").strip()
            sk: Dict[str, int] = {}
            try:
                arr = json.loads(skills_raw) if skills_raw else []
                if isinstance(arr, list):
                    for it in arr:
                        nm = str(it.get("name", "")).strip()
                        lv = int(float(it.get("level", 0)))
                        if nm:
                            sk[nm] = max(0, min(5, lv))
            except Exception:
                pass
            team[mid] = sk
    return team

def _has(member_sk: Dict[str,int], name: str, min_level: int) -> bool:
    """Presence is required; missing skill counts as -1 < min_level (even if min_level=0)."""
    lvl = member_sk.get(name, -1)
    return lvl >= min_level and lvl >= 0

def _eligible_members(task: Dict[str, Any], team: Dict[str, Dict[str,int]]) -> List[str]:
    reqs = task.get("required_skills") or []
    if not isinstance(reqs, list):
        return []
    elig = []
    for mid, sk in team.items():
        ok = True
        for r in reqs:
            nm = str(r.get("name","")).strip()
            lvl = int(float(r.get("level", 0)))
            if not nm or not _has(sk, nm, lvl):
                ok = False
                break
        if ok:
            elig.append(mid)
    return elig

def _skills_covered_by_someone(reqs: List[Dict[str,Any]], team: Dict[str, Dict[str,int]]) -> Tuple[List[str], List[str]]:
    """
    Returns (covered_names, uncovered_names) based on presence (level >= requested).
    """
    covered, uncovered = [], []
    for r in reqs:
        nm = str(r.get("name","")).strip()
        lvl = int(float(r.get("level", 0)))
        if not nm:
            continue
        ok_any = any(_has(sk, nm, lvl) for sk in team.values())
        (covered if ok_any else uncovered).append(nm)
    return covered, uncovered

def _distribute_hours(total: float, n: int) -> List[float]:
    """Return n non-negative numbers summing to total (int-friendly if possible)."""
    if n <= 0:
        return []
    # Prefer integer hours; distribute remainder to the first few subtasks
    base = int(total // n)
    rem = int(round(total - base * n))
    parts = [float(base) for _ in range(n)]
    i = 0
    while sum(parts) < total - 1e-6 and i < n:
        parts[i] += 1.0
        i += 1
        if i == n and sum(parts) < total - 1e-6:
            # If total had fractional part, adjust the last one
            parts[-1] += (total - sum(parts))
    # If total < n, fallback to equal floats
    if total < n:
        parts = [total / n] * n
    return parts

def split_tasks_by_skill(tasks_obj: Dict[str,Any], team_csv: Path, dry: bool=False) -> Tuple[Dict[str,Any], List[str], List[str]]:
    """
    Returns (patched_task_graph, split_summaries, warnings)
    """
    team = _load_team(team_csv)
    tasks = deepcopy(tasks_obj.get("tasks", []))

    # Build reverse deps (tasks depending on X)
    depends_on: Dict[str, List[str]] = {}
    for t in tasks:
        for p in t.get("dependencies", []) or []:
            depends_on.setdefault(p, []).append(t["task_id"])

    new_tasks: List[Dict[str, Any]] = []
    removed_ids: List[str] = []
    split_logs: List[str] = []
    warnings: List[str] = []

    # Process each task
    for t in tasks:
        tid = t.get("task_id")
        reqs = t.get("required_skills") or []
        if not isinstance(reqs, list):
            new_tasks.append(t)
            continue

        if len(reqs) <= 1:
            # Single-skill or none → keep as-is
            new_tasks.append(t)
            continue

        elig = _eligible_members(t, team)
        if elig:
            # Some member can do all required skills → keep as-is
            new_tasks.append(t)
            continue

        # Zero-eligible & multi-skill → candidate for split
        covered, uncovered = _skills_covered_by_someone(reqs, team)
        if uncovered:
            warnings.append(f"[{tid}] uncovered skills (nobody on team has): {', '.join(sorted(set(uncovered)))}")

        # Create one subtask per required skill (keeps things simple & feasible)
        est = float(t.get("estimate_h") or 0.0)
        n = len(reqs)
        parts = _distribute_hours(est, n)

        sub_ids = []
        for idx, r in enumerate(reqs, start=1):
            sub_id = f"{tid}-{idx}"
            sub_ids.append(sub_id)
            sub_task = {
                "task_id": sub_id,
                "title": f"{t.get('title','')} — {str(r.get('name','')).strip()}",
                "estimate_h": parts[idx-1],
                "required_skills": [r],
                "dependencies": list(t.get("dependencies", []) or []),
            }
            if "due_by" in t and t.get("due_by"):
                sub_task["due_by"] = t["due_by"]
            new_tasks.append(sub_task)

        # Rewrite downstream dependencies: replace tid with all sub_ids (mutate original list)
        for child_id in depends_on.get(tid, []) or []:
            for ct in tasks:
                if ct.get("task_id") == child_id:
                    deps = [d for d in (ct.get("dependencies") or []) if d != tid]
                    ct["dependencies"] = list(dict.fromkeys(deps + sub_ids))

        removed_ids.append(tid)
        split_logs.append(f"Split {tid} into {', '.join(sub_ids)} (est_h {est} → {parts})")

    # ---------- FIXED FINAL ASSEMBLY ----------
    # Build the final list as:
    #   • all 'new_tasks' (includes subtasks + any original unsplit tasks we copied),
    #   • plus any tasks from the original list that were updated as children AND not removed
    #     AND not already present in new_tasks.
    present = {t["task_id"] for t in new_tasks}
    for t in tasks:
        tid = t["task_id"]
        if tid in present:
            continue
        if tid in removed_ids:
            # DO NOT re-add originals that were split
            continue
        new_tasks.append(t)
        present.add(tid)
    # -----------------------------------------

    return {"tasks": new_tasks}, split_logs, warnings
def main() -> int:
    ap = argparse.ArgumentParser(description="Split multi-skill tasks with zero eligible members into one-skill subtasks.")
    ap.add_argument("--tasks", type=Path, required=True, help="Path to normalized task_graph.json")
    ap.add_argument("--team", type=Path, required=True, help="Path to team.csv")
    ap.add_argument("--out", type=Path, required=True, help="Destination for patched task_graph.json")
    ap.add_argument("--dry", action="store_true", help="Dry run only (print planned splits)")
    args = ap.parse_args()

    obj = _load_json(args.tasks)
    patched, logs, warns = split_tasks_by_skill(obj, args.team, dry=args.dry)

    if args.dry:
        print("# Planned splits:")
        for line in logs:
            print(" -", line)
        if warns:
            print("\n# Warnings:")
            for w in warns:
                print(" -", w)
        return 0

    _save_json(patched, args.out)
    print(f"[OK] Patched task graph -> {args.out}")
    if logs:
        print("# Splits:")
        for l in logs:
            print(" -", l)
    if warns:
        print("# Warnings:")
        for w in warns:
            print(" -", w)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
