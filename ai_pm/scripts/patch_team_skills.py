# ai_pm/scripts/patch_team_skills.py
"""
Patch team.csv skills for one or more members.

Usage examples:
  # Add git=1 to member u1
  python -m scripts.patch_team_skills --team samples/team.csv --member u1 --add git=1

  # Add multiple skills to u1
  python -m scripts.patch_team_skills --team samples/team.csv --member u1 --add git=1 --add frontend=3

  # Apply to multiple members
  python -m scripts.patch_team_skills --team samples/team.csv --member u1 --member u2 --add git=1

  # Dry run (see what would change, without writing)
  python -m scripts.patch_team_skills --team samples/team.csv --member u1 --add git=1 --dry
"""
from __future__ import annotations
import argparse, csv, io, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def _parse_add(arg: str) -> Tuple[str, int]:
    # "skill=level" -> ("skill", int(level))
    if "=" not in arg:
        raise ValueError(f"Malformed --add '{arg}', expected skill=level")
    k, v = arg.split("=", 1)
    k = k.strip()
    try:
        lvl = int(float(v.strip()))
    except Exception:
        raise ValueError(f"Level must be a number in '{arg}'")
    return k, max(0, min(5, lvl))

def _load_rows(team_csv: Path) -> List[Dict[str, str]]:
    with team_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def _dump_rows(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(out.getvalue(), encoding="utf-8", newline="")

def _parse_skills_cell(cell: str) -> List[Dict[str, int]]:
    if not cell:
        return []
    try:
        data = json.loads(cell)
        if isinstance(data, list):
            # Normalize to [{"name":..., "level":...}]
            out = []
            for it in data:
                if isinstance(it, dict) and "name" in it:
                    lvl = it.get("level", 0)
                    try: lvl = int(float(lvl))
                    except Exception: lvl = 0
                    out.append({"name": str(it["name"]).strip(), "level": max(0, min(5, lvl))})
                elif isinstance(it, str):
                    out.append({"name": it.strip(), "level": 0})
            return out
    except Exception:
        pass
    return []

def _skills_to_cell(arr: List[Dict[str, int]]) -> str:
    return json.dumps(arr, ensure_ascii=False)

def patch_team(team_csv: Path, member_ids: List[str], adds: List[Tuple[str, int]], dry: bool=False) -> List[str]:
    """
    Apply the skill additions/updates to the given member ids.
    Returns a list of human-readable change logs.
    """
    rows = _load_rows(team_csv)
    changes: List[str] = []
    idx_by_id = { (r.get("member_id") or "").strip(): i for i, r in enumerate(rows) }
    for mid in member_ids:
        if mid not in idx_by_id:
            changes.append(f"[WARN] member_id '{mid}' not found — skipping")
            continue
        i = idx_by_id[mid]
        row = rows[i]
        skills = _parse_skills_cell(row.get("skills") or "")
        by_name = { s["name"]: s for s in skills if isinstance(s, dict) and "name" in s }

        for (name, lvl) in adds:
            if name in by_name:
                old = by_name[name]["level"]
                by_name[name]["level"] = lvl
                changes.append(f"[{mid}] {name}: {old} → {lvl}")
            else:
                skills.append({"name": name, "level": lvl})
                by_name[name] = {"name": name, "level": lvl}
                changes.append(f"[{mid}] {name}: (new) {lvl}")

        row["skills"] = _skills_to_cell(list(by_name.values()))

    if not dry:
        _dump_rows(rows, team_csv)
        changes.append(f"[OK] Updated {team_csv}")
    else:
        changes.append("[DRY] No changes written")
    return changes

def main() -> int:
    ap = argparse.ArgumentParser(description="Add/update skills for members in team.csv")
    ap.add_argument("--team", type=Path, required=True, help="Path to team.csv")
    ap.add_argument("--member", action="append", required=True, help="member_id to patch (repeatable)")
    ap.add_argument("--add", action="append", default=[], help="skill=level (repeatable)")
    ap.add_argument("--dry", action="store_true", help="Dry run (print only)")
    args = ap.parse_args()

    if not args.add:
        print("[ERROR] Provide at least one --add skill=level")
        return 2

    try:
        adds = [_parse_add(x) for x in args.add]
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    logs = patch_team(args.team, args.member, adds, dry=args.dry)
    for line in logs:
        print(line)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
