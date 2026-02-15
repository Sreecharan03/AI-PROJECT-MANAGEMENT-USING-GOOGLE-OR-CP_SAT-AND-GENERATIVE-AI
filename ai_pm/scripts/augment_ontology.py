#!/usr/bin/env python3
"""
augment_ontology.py
Purpose: Patch skills.csv with synonym mappings or new canonical skills.

Inputs:
  - --in  : path to existing skills.csv (columns: canonical_skill, synonyms)
  - --out : output path (can be same as --in to edit in-place)
  - --map : repeated "TERM=CANONICAL" entries to add synonyms (e.g., "frontend=react")
  - --add-canonical : repeated canonical skills to ensure exist (e.g., "git")

Behavior:
  - Reads synonyms as JSON list or pipe-separated.
  - De-duplicates, normalizes spacing, writes JSON arrays in 'synonyms' column.

Example:
  python -m scripts.augment_ontology \
    --in samples/skills.csv \
    --out samples/skills.csv \
    --map "frontend=react" \
    --map "frontend dev=react" \
    --map "backend=python" \
    --map "backend dev=python" \
    --add-canonical git
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

def _norm(s: str) -> str:
    s = (s or "").strip()
    return " ".join(s.split())

def _parse_synonyms(cell: str) -> List[str]:
    if not cell:
        return []
    t = cell.strip()
    if not t:
        return []
    # Try JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [_norm(str(x)) for x in obj if _norm(str(x))]
    except Exception:
        pass
    # Fallback: pipe-separated
    return [_norm(x) for x in t.split("|") if _norm(x)]

def _to_json_syns(syns: List[str]) -> str:
    return json.dumps(sorted(list({ _norm(x) for x in syns if _norm(x) })), ensure_ascii=False)

def _load_skills_csv(path: Path) -> Dict[str, List[str]]:
    """
    Returns: dict canonical -> synonyms(list)
    """
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if "canonical_skill" not in (rd.fieldnames or []) or "synonyms" not in (rd.fieldnames or []):
            raise ValueError("skills.csv must contain columns: canonical_skill, synonyms")
        for row in rd:
            canon = _norm(row.get("canonical_skill", ""))
            syns = _parse_synonyms(row.get("synonyms", "") or "")
            if canon:
                out[canon] = syns
    return out

def _save_skills_csv(sk: Dict[str, List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["canonical_skill", "synonyms"])
        wr.writeheader()
        for canon in sorted(sk.keys()):
            wr.writerow({"canonical_skill": canon, "synonyms": _to_json_syns(sk[canon])})

def _apply_mappings(sk: Dict[str, List[str]], pairs: List[Tuple[str, str]]) -> None:
    for term, canon in pairs:
        term_n = _norm(term)
        canon_n = _norm(canon)
        if not canon_n:
            continue
        if canon_n not in sk:
            sk[canon_n] = []
        if term_n and term_n not in sk[canon_n] and term_n != canon_n:
            sk[canon_n].append(term_n)

def main() -> int:
    ap = argparse.ArgumentParser(description="Augment skills.csv with synonyms and canonical skills.")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="Path to existing skills.csv")
    ap.add_argument("--out", dest="out", type=Path, required=True, help="Path to write updated skills.csv")
    ap.add_argument("--map", dest="maps", action="append", default=[], help='Mapping "TERM=CANONICAL", repeatable')
    ap.add_argument("--add-canonical", dest="adds", action="append", default=[], help="Canonical skills to ensure exist")
    args = ap.parse_args()

    skills = _load_skills_csv(args.inp)

    # Ensure canonicals
    for c in args.adds or []:
        c = _norm(c)
        if c and c not in skills:
            skills[c] = []

    # Parse and apply mappings
    pairs: List[Tuple[str, str]] = []
    for m in args.maps or []:
        if "=" not in m:
            print(f"[WARN] ignoring malformed --map: {m}")
            continue
        term, canon = m.split("=", 1)
        pairs.append((_norm(term), _norm(canon)))
    _apply_mappings(skills, pairs)

    _save_skills_csv(skills, args.out)
    print(f"[OK] Updated ontology -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
