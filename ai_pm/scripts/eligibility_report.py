#!/usr/bin/env python3
"""
Eligibility report for tasks vs team skills.

What it does:
  • Loads normalized (or split) task_graph.json and team.csv.
  • (Optional) Canonicalizes TEAM skills using a skills.csv ontology (so 'frontend dev' → 'react', etc.).
  • For each task, lists members who are ELIGIBLE (have all required skills at required levels).
  • If no one is eligible, prints per-member reasons (missing/low skills).
Exit code:
  0  if all tasks have ≥1 eligible member
  3  if any task has zero eligible member
  2  for bad inputs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- small helpers (self-contained) ----------

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _load_team_csv(p: Path) -> Dict[str, Dict[str, Any]]:
    team: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            mid = (r.get("member_id") or "").strip()
            if not mid:
                continue
            skills: Dict[str, int] = {}
            raw = (r.get("skills") or "").strip()
            try:
                arr = json.loads(raw) if raw else []
                if isinstance(arr, list):
                    for it in arr:
                        nm = str(it.get("name", "")).strip()
                        try:
                            lvl = int(float(it.get("level", 0)))
                        except Exception:
                            lvl = 0
                        if nm:
                            skills[nm] = max(0, min(5, lvl))
            except Exception:
                pass
            team[mid] = {
                "member_id": mid,
                "name": (r.get("name") or "").strip(),
                "skills": skills,
            }
    return team

def _parse_syns(cell: str) -> List[str]:
    if not cell:
        return []
    s = cell.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in s.split("|") if p.strip()]

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    for ch in "-_/.,":
        s = s.replace(ch, " ")
    return " ".join(s.split())

def _canonize_team_skills(team: Dict[str, Dict[str, Any]], skills_csv: Optional[Path]) -> None:
    """Map team skill tokens to canonical names using skills.csv."""
    if not skills_csv or not skills_csv.exists():
        return
    token2canon: Dict[str, str] = {}
    with skills_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            canon = (row.get("canonical_skill") or "").strip()
            if not canon:
                continue
            token2canon[_norm(canon)] = canon
            for s in _parse_syns(row.get("synonyms", "") or ""):
                token2canon[_norm(s)] = canon
    # apply to each member
    for m in team.values():
        new_sk: Dict[str, int] = {}
        for nm, lv in m["skills"].items():
            c = token2canon.get(_norm(nm))
            new_sk[c if c else nm] = lv
        m["skills"] = new_sk

def _eligible(member_sk: Dict[str, int], reqs: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Return (is_eligible, reasons_if_not)."""
    reasons: List[str] = []
    for req in (reqs or []):
        nm = str(req.get("name", "")).strip()
        if not nm:
            reasons.append("blank skill name")
            continue
        try:
            minlvl = int(float(req.get("level", 0)))
        except Exception:
            minlvl = 0
        lvl = member_sk.get(nm, -1)
        if lvl < minlvl or lvl < 0:
            reasons.append(f"needs {nm}≥{minlvl}, has {('none' if lvl<0 else lvl)}")
    return (len(reasons) == 0, reasons)

def _format_req(req: Dict[str, Any]) -> str:
    """Turn a requirement into a compact 'name≥level' or 'name' string."""
    nm = str(req.get("name", "")).strip()
    lv_raw = req.get("level", None)
    if lv_raw is None or str(lv_raw) == "":
        return nm
    try:
        lv = int(float(lv_raw))
    except Exception:
        lv = None
    return f"{nm}≥{lv}" if lv is not None else nm

# ---------- main reporting ----------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Print eligible members per task and reasons for ineligibility.")
    ap.add_argument("--team", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--skills", type=Path, help="Optional skills.csv to canonicalize TEAM skills")
    ap.add_argument("--show-all", action="store_true", help="Also print reasons for non-eligible members")
    args = ap.parse_args(argv)

    if not args.team.exists() or not args.tasks.exists():
        print("[ERROR] team or tasks path not found")
        return 2

    team = _load_team_csv(args.team)
    tasks_obj = _load_json(args.tasks)
    _canonize_team_skills(team, args.skills)

    tasks = list(tasks_obj.get("tasks", []))
    zero_elig_count = 0

    print(f"[INFO] tasks={len(tasks)} members={len(team)}")
    for t in tasks:
        tid = t.get("task_id")
        title = t.get("title","")
        reqs = t.get("required_skills") or []
        if not isinstance(reqs, list):
            # if malformed, try to coerce into a 1-item list
            reqs = [{"name": str(reqs)}] if reqs else []

        elig: List[str] = []
        not_ok: List[Tuple[str, List[str]]] = []
        for mid, m in team.items():
            ok, why = _eligible(m["skills"], reqs)
            if ok:
                elig.append(f"{mid} ({m['name']})")
            else:
                not_ok.append((mid, why))

        req_str = ", ".join(_format_req(r) for r in reqs) if reqs else "—"
        print(f"\nTask {tid}: {title}")
        print("  required_skills:", req_str)
        if elig:
            print(f"  ✅ eligible: {', '.join(elig)}")
        else:
            zero_elig_count += 1
            print("  ❌ eligible: none")
            if args.show_all:
                for mid, reasons in not_ok:
                    name = team[mid]["name"]
                    print(f"     - {mid} ({name}): " + "; ".join(reasons))

    if zero_elig_count:
        print(f"\n[SUMMARY] {zero_elig_count} task(s) have ZERO eligible members.")
        return 3

    print("\n[SUMMARY] All tasks have at least one eligible member.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
