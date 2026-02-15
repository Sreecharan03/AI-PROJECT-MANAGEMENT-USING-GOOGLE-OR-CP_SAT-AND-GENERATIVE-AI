# ai_pm/core/skills.py
# Phase 2/3 — Skills Ontology & Mapping Helpers (no external deps).
#
# What this module provides:
#   - SkillsOntology: load from CSV, normalize strings, map_to_canonical(term)
#   - add_synonym(canonical, synonym) and persist_csv(dest_path)
#   - find_unknown_terms(terms) utility for UI highlights
#
# CSV contract (same as Phase 1 skills.csv):
#   columns: canonical_skill, synonyms
#   synonyms may be a pipe string "a|b|c" or a JSON list '["a","b","c"]'
#
# CLI usage examples:
#   cd ai_pm
#   python -m core.skills --skills samples/skills.csv --term python py "React.js" reactjs
#   python -m core.skills --skills samples/skills.csv --add "react:react native|rn" --persist /tmp/skills_updated.csv
#
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ----------------------------
# Normalization helpers
# ----------------------------

def _normalize(s: str) -> str:
    """
    Normalize a skill token for matching:
      - strip leading/trailing spaces
      - lowercase
      - unify common separators to a single space: [-_/.,]
      - collapse repeated spaces
      - drop surrounding parentheses
    This keeps things predictable while preserving intent (e.g., "c++" stays "c++").
    """
    if s is None:
        return ""
    t = s.strip().lower()
    # replace common separators with spaces
    for ch in ["-", "_", "/", ".", ","]:
        t = t.replace(ch, " ")
    # remove simple wrapping parentheses
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    # collapse multiple spaces
    t = " ".join(t.split())
    return t


def _parse_synonyms(val: str) -> List[str]:
    """
    Accept either a JSON list string or a pipe-delimited string.
    Returns a list of trimmed strings (may be empty).
    """
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    # try JSON first
    try:
        maybe = json.loads(s)
        if isinstance(maybe, list):
            return [str(x).strip() for x in maybe if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # fallback to pipe-delimited
    return [p.strip() for p in s.split("|") if p.strip()]


# ----------------------------
# Ontology model
# ----------------------------

@dataclass
class SkillsOntology:
    # canonical name (original case preserved) -> set of synonym strings (original case preserved)
    canonical_to_synonyms: Dict[str, Set[str]]
    # normalized token -> canonical name (original case)
    token_to_canonical: Dict[str, str]

    @classmethod
    def load_from_csv(cls, path: Path) -> "SkillsOntology":
        """
        Build ontology from a skills.csv file.
        """
        c2s: Dict[str, Set[str]] = {}
        t2c: Dict[str, str] = {}

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            header = rd.fieldnames or []
            if "canonical_skill" not in header or "synonyms" not in header:
                raise RuntimeError("skills.csv must have columns: canonical_skill, synonyms")

            for row in rd:
                canonical = (row.get("canonical_skill") or "").strip()
                if not canonical:
                    # skip empty canonical rows
                    continue
                syns = _parse_synonyms(row.get("synonyms", "") or "")
                # ensure set
                syn_set: Set[str] = set(syns)

                # store originals
                if canonical not in c2s:
                    c2s[canonical] = set()
                c2s[canonical].update(syn_set)

                # normalized mapping: map canonical itself and its synonyms
                t2c[_normalize(canonical)] = canonical
                for s in syn_set:
                    t2c[_normalize(s)] = canonical

        return cls(canonical_to_synonyms=c2s, token_to_canonical=t2c)

    # ------------- core API -------------

    def canonical_terms(self) -> List[str]:
        return sorted(self.canonical_to_synonyms.keys(), key=lambda s: s.lower())

    def map_to_canonical(self, term: str) -> Tuple[Optional[str], str]:
        """
        Map an arbitrary term to a canonical skill.
        Returns (canonical_or_None, normalized_input).
        """
        norm = _normalize(term)
        canon = self.token_to_canonical.get(norm)
        return canon, norm

    def add_synonym(self, canonical: str, synonym: str) -> None:
        """
        Append a synonym mapping in-memory (both forward and reverse maps).
        If canonical didn't exist, it will be created as a new canonical bucket.
        """
        canonical = canonical.strip()
        synonym = synonym.strip()
        if not canonical or not synonym:
            return
        # forward
        if canonical not in self.canonical_to_synonyms:
            self.canonical_to_synonyms[canonical] = set()
        self.canonical_to_synonyms[canonical].add(synonym)
        # reverse
        self.token_to_canonical[_normalize(canonical)] = canonical
        self.token_to_canonical[_normalize(synonym)] = canonical

    def persist_csv(self, dest: Path, json_synonyms: bool = True) -> None:
        """
        Write the ontology back to CSV with columns: canonical_skill, synonyms.
        By default, synonyms are stored as a JSON list string for clarity.
        Set json_synonyms=False to write a pipe-delimited string instead.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, str]] = []
        for canonical in self.canonical_terms():
            syns = sorted(self.canonical_to_synonyms.get(canonical, []), key=lambda s: s.lower())
            if json_synonyms:
                syn_cell = json.dumps(syns, ensure_ascii=False)
            else:
                syn_cell = "|".join(syns)
            rows.append({"canonical_skill": canonical, "synonyms": syn_cell})

        with dest.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["canonical_skill", "synonyms"])
            wr.writeheader()
            wr.writerows(rows)

    # ------------- helpers for UI -------------

    def find_unknown_terms(self, terms: Iterable[str]) -> List[str]:
        """
        Return a sorted de-duplicated list of terms that do not map to any canonical.
        """
        unknown: Set[str] = set()
        for t in terms:
            canon, _ = self.map_to_canonical(t)
            if not canon:
                unknown.add(t)
        return sorted(unknown, key=lambda s: _normalize(s))


# -------------------------------------
# Utilities to pull skills from team CSV
# -------------------------------------

def extract_skill_terms_from_team_csv(team_csv: Path) -> List[str]:
    """
    Read team.csv and extract all 'skills' item names (JSON list per member row),
    returning a flat list of skill name strings.
    Schema reminder:
      team.csv columns include: skills (JSON list of {name, level})
    """
    terms: List[str] = []
    with team_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        if "skills" not in header:
            return terms
        for row in rd:
            cell = (row.get("skills") or "").strip()
            if not cell:
                continue
            try:
                payload = json.loads(cell)
                if isinstance(payload, list):
                    for item in payload:
                        name = str(item.get("name", "")).strip()
                        if name:
                            terms.append(name)
            except json.JSONDecodeError:
                # ignore malformed rows here; the schema validator will flag them elsewhere
                continue
    return terms


# ----------------------------
# CLI for quick manual checks
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Skills ontology loader and mapper")
    parser.add_argument("--skills", type=Path, required=True, help="Path to skills.csv")
    parser.add_argument("--term", nargs="*", help="Ad-hoc terms to map (e.g., python React.js rn)")
    parser.add_argument("--team", type=Path, help="Optional team.csv; will extract skill names and report unknowns")
    parser.add_argument("--add", action="append",
                        help='Add synonym(s): format "canonical:syn1|syn2|syn3" (repeatable)')
    parser.add_argument("--persist", type=Path, help="Write updated ontology to this CSV path")
    args = parser.parse_args(argv)

    onto = SkillsOntology.load_from_csv(args.skills)
    print(f"[OK] Loaded skills: {len(onto.canonical_terms())} canonical entries")

    # Map ad-hoc terms
    if args.term:
        for t in args.term:
            canon, norm = onto.map_to_canonical(t)
            print(f'  - "{t}" (norm="{norm}") -> {canon or "UNKNOWN"}')

    # Show unknowns from team.csv, if provided
    if args.team:
        toks = extract_skill_terms_from_team_csv(args.team)
        unknown = onto.find_unknown_terms(toks)
        print(f"[team] extracted={len(toks)} unknown={len(unknown)}")
        for u in unknown[:20]:
            print(f"  ? {u}")

    # Apply additions
    if args.add:
        for spec in args.add:
            if ":" not in spec:
                print(f"[WARN] ignoring malformed --add spec: {spec}")
                continue
            canonical, rhs = spec.split(":", 1)
            syns = [s for s in rhs.split("|") if s.strip()]
            for s in syns:
                onto.add_synonym(canonical.strip(), s.strip())
        print("[OK] additions applied in memory")

    # Persist if asked
    if args.persist:
        onto.persist_csv(args.persist, json_synonyms=True)
        print(f"[OK] wrote updated ontology -> {args.persist}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
