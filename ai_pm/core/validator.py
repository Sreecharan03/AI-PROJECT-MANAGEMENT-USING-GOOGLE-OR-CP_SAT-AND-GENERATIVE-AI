# ai_pm/core/validator.py
# Phase 3 — Validate & normalize LLM task graphs (no assignees).
#
# What this module does:
#   • validate_and_normalize(task_graph, ontology) -> (normalized_dict, topo_order)
#       - field/type checks
#       - estimate_h > 0
#       - due_by ISO date if present (YYYY-MM-DD)
#       - dependencies reference existing tasks
#       - DAG check (no cycles), returns a topological order
#       - skill names are canonicalized via SkillsOntology (strict: unknown -> error)
#   • CLI main(): validate a task_graph.json with a skills.csv and write a normalized JSON
#
# Inputs:
#   task_graph (dict) with key "tasks": list of task objects:
#     {
#       "task_id": "T1",
#       "title": "Do the thing",
#       "estimate_h": 4,         # > 0 (int or float)
#       "required_skills": [     # list of {name, level?}
#           {"name": "python", "level": 3},
#           {"name": "react"}
#       ],
#       "dependencies": ["T0"],  # task_ids
#       "due_by": "2025-10-31"   # optional
#     }
#
# Strictness:
#   • Any unknown skill (not in ontology) is an ERROR (so downstream optimizer only sees canonical names).
#
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.skills import SkillsOntology


# ---------------------------
# Basic helpers
# ---------------------------

def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None

def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None

def _is_iso_date(s: Any) -> bool:
    if not isinstance(s, str) or not s:
        return False
    try:
        date.fromisoformat(s)
        return True
    except Exception:
        return False


# ---------------------------
# DAG check (Kahn’s algorithm)
# ---------------------------

def _toposort(task_ids: List[str], deps_map: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """
    Return (is_dag, topological_order). deps_map[T] = list of parents for T.
    Edges go parent -> child.
    """
    indeg: Dict[str, int] = {tid: 0 for tid in task_ids}
    children: Dict[str, List[str]] = {tid: [] for tid in task_ids}

    for t in task_ids:
        parents = deps_map.get(t, [])
        for p in parents:
            children[p].append(t)
            indeg[t] += 1

    order: List[str] = []
    zero = [tid for tid in task_ids if indeg[tid] == 0]
    i = 0
    while i < len(zero):
        u = zero[i]
        i += 1
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                zero.append(v)

    return (len(order) == len(task_ids), order)


# ---------------------------
# Core validation
# ---------------------------

def validate_and_normalize(task_graph: Dict[str, Any], ontology: SkillsOntology) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate a raw task graph and return (normalized_graph, topo_order).
    Raises ValueError with a joined message if any errors are found.
    """
    errors: List[str] = []

    if not isinstance(task_graph, dict) or "tasks" not in task_graph:
        raise ValueError('task_graph must be a dict with key "tasks".')

    raw_tasks = task_graph.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError('"tasks" must be a non-empty list.')

    # Collect IDs & check uniqueness
    ids: List[str] = []
    id_set = set()
    for i, t in enumerate(raw_tasks, start=1):
        tid = t.get("task_id")
        if not isinstance(tid, str) or not tid.strip():
            errors.append(f"[task #{i}] task_id missing/empty.")
            continue
        tid = tid.strip()
        ids.append(tid)
        if tid in id_set:
            errors.append(f'Duplicate task_id "{tid}".')
        id_set.add(tid)

    # Pre-check dependencies reference existing ids
    deps_map: Dict[str, List[str]] = {}
    for i, t in enumerate(raw_tasks, start=1):
        tid = t.get("task_id", f"#{i}")
        deps = t.get("dependencies", [])
        if not isinstance(deps, list):
            errors.append(f'[{tid}] "dependencies" must be a list.')
            deps = []
        # ensure all deps exist
        for d in deps:
            if d not in id_set:
                errors.append(f'[{tid}] dependency "{d}" not found among task_ids.')
        deps_map[tid] = [str(d) for d in deps if isinstance(d, str)]

    # Field validation + normalization pass
    normalized_tasks: List[Dict[str, Any]] = []
    for i, t in enumerate(raw_tasks, start=1):
        tid = t.get("task_id", f"#{i}")

        # title
        title = t.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f'[{tid}] "title" missing/empty.')
            title = ""  # keep shape

        # estimate_h
        est = _as_float(t.get("estimate_h"))
        if est is None or est <= 0:
            errors.append(f'[{tid}] "estimate_h" must be a number > 0.')

        # required_skills
        rs = t.get("required_skills")
        if not isinstance(rs, list) or not rs:
            errors.append(f'[{tid}] "required_skills" must be a non-empty list.')
            rs = []

        canon_skills: List[Dict[str, Any]] = []
        for j, s in enumerate(rs, start=1):
            if not isinstance(s, dict):
                errors.append(f'[{tid}] required_skills[{j}] must be an object.')
                continue
            raw_name = s.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                errors.append(f'[{tid}] required_skills[{j}].name missing/empty.')
                continue
            canonical, _norm = ontology.map_to_canonical(raw_name)
            if not canonical:
                errors.append(f'[{tid}] unknown skill "{raw_name}" — not in ontology.')
                continue
            out_item: Dict[str, Any] = {"name": canonical}
            # optional level
            lvl = s.get("level", None)
            if lvl is not None:
                lvl_num = _as_int(lvl) or _as_float(lvl)
                if lvl_num is None or (isinstance(lvl_num, (int, float)) and lvl_num < 0):
                    errors.append(f'[{tid}] required_skills[{j}].level must be non-negative number if present.')
                else:
                    # Prefer int when exact
                    if isinstance(lvl_num, float) and lvl_num.is_integer():
                        lvl_num = int(lvl_num)
                    out_item["level"] = lvl_num
            canon_skills.append(out_item)

        # dependencies (already checked type and existence above)
        deps = deps_map.get(tid, [])

        # due_by (optional)
        due_by = t.get("due_by", None)
        if due_by is not None and due_by != "":
            if not _is_iso_date(due_by):
                errors.append(f'[{tid}] "due_by" must be YYYY-MM-DD if provided.')

        normalized_tasks.append({
            "task_id": tid,
            "title": title.strip(),
            "estimate_h": est if est is not None else 0.0,
            "required_skills": canon_skills,
            "dependencies": deps,
            **({"due_by": due_by} if due_by else {}),
        })

    # DAG check
    is_dag, topo = _toposort(ids, deps_map)
    if not is_dag:
        errors.append("Dependencies contain a cycle (graph is not a DAG).")

    if errors:
        raise ValueError("\n".join(errors))

    normalized = {"tasks": normalized_tasks}
    return normalized, topo


# ---------------------------
# CLI
# ---------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def main(argv: Optional[List[str]] = None) -> int:
    """
    Validate and normalize a task graph:
      python -m core.validator --tasks /path/to/task_graph.json --skills samples/skills.csv --out /tmp/task_graph.normalized.json
    """
    ap = argparse.ArgumentParser(description="Validate & normalize task graph JSON (DAG + fields + canonical skills).")
    ap.add_argument("--tasks", type=Path, required=True, help="Path to task_graph.json (LLM output).")
    ap.add_argument("--skills", type=Path, required=True, help="Path to skills.csv (ontology).")
    ap.add_argument("--out", type=Path, help="Destination for normalized JSON (stdout if omitted).")
    args = ap.parse_args(argv)

    try:
        raw = _load_json(args.tasks)
        onto = SkillsOntology.load_from_csv(args.skills)
        normalized, topo = validate_and_normalize(raw, onto)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    if args.out:
        _save_json(normalized, args.out)
        print(f"[OK] Normalized task graph written -> {args.out}")
        print(f"[OK] Topological order: {', '.join(topo)}")
    else:
        print(json.dumps(normalized, ensure_ascii=False, indent=2))
        print("\n# Topological order:", ", ".join(topo))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
