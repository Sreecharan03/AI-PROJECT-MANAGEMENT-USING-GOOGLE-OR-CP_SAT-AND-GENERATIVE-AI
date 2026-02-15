# ai_pm/core/skills_groq.py
# Dynamic skill mapping suggestions via Groq, layered on top of core.skills.
#
# What this module provides:
#   - rank_candidates(ontology, term, k)  -> List[str] (cheap local similarity)
#   - suggest_with_groq(ontology, term)   -> Suggestion(canonical|None, reason, norm, candidates)
#   - apply_and_persist(...)              -> add synonym + write updated CSV where you want
#   - CLI main() for quick testing
#
# Requirements:
#   - ai_pm/core/llm_extractor.py (already added) with get_groq_client()
#   - ai_pm/core/skills.py (already added)
#
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import json
import math

from core.llm_extractor import get_groq_client
from core.skills import SkillsOntology, _normalize

# ----------------------------
# Cheap local similarity
# ----------------------------

def _tokenize_norm(s: str) -> List[str]:
    return _normalize(s).split()

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _substr_boost(a: str, b: str) -> float:
    """Heuristic: if one string contains the other, give a small boost."""
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.15
    return 0.0

def _score(term: str, candidate: str) -> float:
    ta = _tokenize_norm(term)
    tb = _tokenize_norm(candidate)
    base = _jaccard(ta, tb)
    base += _substr_boost(_normalize(term), _normalize(candidate))
    # light length penalty to avoid mapping to very long names if tokens differ
    pen = 0.02 * max(0, abs(len(" ".join(ta)) - len(" ".join(tb))) // 5)
    return max(0.0, base - pen)

def rank_candidates(ontology: SkillsOntology, term: str, k: int = 20) -> List[str]:
    """
    Rank canonical skills by a simple similarity score, return top-k names.
    This trims the search space so the LLM sees a short, relevant candidate set.
    """
    cands = []
    for canonical in ontology.canonical_terms():
        cands.append((canonical, _score(term, canonical)))
    cands.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in cands[:k]]


# ----------------------------
# LLM suggestion
# ----------------------------

@dataclass
class Suggestion:
    term: str
    norm: str
    candidates: List[str]
    canonical: Optional[str]  # LLM's chosen canonical, or None/"UNKNOWN"
    reason: str

def _build_prompt(term: str, candidates: List[str]) -> List[Dict[str, str]]:
    """
    Build a minimal chat-style prompt for Groq that forces a JSON result.
    The model must pick EXACTLY ONE canonical from the provided candidates or return null.
    """
    system = (
        "You map arbitrary skill strings to a controlled vocabulary of CANONICAL skills.\n"
        "Always answer with strict JSON: {\"canonical\": <string or null>, \"reason\": <string>}.\n"
        "Pick ONLY from the provided candidates. If nothing matches, set canonical to null.\n"
        "Be concise but specific in reason; do not add extra keys."
    )
    user = (
        f"TERM: {term}\n"
        f"CANDIDATES ({len(candidates)}): {candidates}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def suggest_with_groq(ontology: SkillsOntology, term: str, model: str = "llama-3.1-8b-instant") -> Suggestion:
    """
    Ask Groq to choose the best canonical skill for `term` from a narrowed candidate list.
    If none matches, returns canonical=None.
    """
    # If ontology already maps it, short-circuit (no call needed)
    canon, norm = ontology.map_to_canonical(term)
    if canon:
        return Suggestion(term=term, norm=norm, candidates=[canon], canonical=canon, reason="Already in ontology.")

    candidates = rank_candidates(ontology, term, k=20)
    msgs = _build_prompt(term, candidates)

    client = get_groq_client()
    # Low temperature for stable outputs
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.1,
        max_tokens=200,
        top_p=1.0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        obj = json.loads(content)
        chosen = obj.get("canonical", None)
        reason = str(obj.get("reason", "")).strip()
        # Enforce constraint: must be one of the candidates (otherwise treat as unknown)
        if chosen not in candidates:
            chosen = None
            if not reason:
                reason = "Model did not select a valid candidate."
    except Exception:
        chosen, reason = None, "Could not parse model JSON."

    return Suggestion(term=term, norm=norm, candidates=candidates, canonical=chosen, reason=reason)


# ----------------------------
# Apply & persist convenience
# ----------------------------

def apply_and_persist(ontology: SkillsOntology, term: str, chosen_canonical: str, dest_csv: Path) -> None:
    """
    Add `term` as a synonym of `chosen_canonical` and write an updated skills.csv to `dest_csv`.
    """
    ontology.add_synonym(chosen_canonical, term)
    ontology.persist_csv(dest_csv, json_synonyms=True)


# ----------------------------
# CLI demo
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Groq-assisted canonical mapping for skills.")
    parser.add_argument("--skills", type=Path, required=True, help="Path to skills.csv (ontology)")
    parser.add_argument("--term", action="append", required=True, help="Unknown term (repeatable)")
    parser.add_argument("--persist", type=Path, help="If provided with --accept, writes updated CSV to this path")
    parser.add_argument("--accept", help="If set, auto-accept this canonical for ALL terms (careful).")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model id")
    args = parser.parse_args(argv)

    onto = SkillsOntology.load_from_csv(args.skills)
    for t in args.term:
        sug = suggest_with_groq(onto, t, model=args.model)
        print(json.dumps(asdict(sug), ensure_ascii=False, indent=2))
        if args.accept and sug.canonical:
            onto.add_synonym(sug.canonical, t)

    if args.accept and args.persist:
        onto.persist_csv(args.persist, json_synonyms=True)
        print(f"[OK] persisted updated ontology -> {args.persist}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
