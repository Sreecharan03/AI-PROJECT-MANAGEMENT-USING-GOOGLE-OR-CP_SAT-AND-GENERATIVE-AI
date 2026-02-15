# ai_pm/core/llm_extractor.py
# Phase 3 — LLM Task Builder (Groq) → task_graph.json
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
try:
    from groq import Groq
except Exception as e:
    raise RuntimeError("Install deps: pip install -r requirements.txt") from e

HARDCODED_GROQ_API_KEY: str = ""

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _load_models_cfg() -> Dict[str, Any]:
    cfg = _repo_root() / "config" / "models.yaml"
    data = {"task_extractor": "llama-3.1-8b-instant", "safety": ""}
    if cfg.exists():
        try:
            y = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            if isinstance(y, dict): data.update(y)
        except Exception:
            pass
    return data

def resolve_groq_api_key() -> str:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return key or HARDCODED_GROQ_API_KEY

def mask_key(k: Optional[str]) -> str:
    return f"{k[:4]}...{k[-4:]}" if k and len(k)>8 else ("" if not k else "*"*len(k))

def get_groq_client() -> Groq:
    key = resolve_groq_api_key()
    if not key: raise RuntimeError("No GROQ API key available.")
    return Groq(api_key=key)

EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RX = re.compile(r"(?:\+?\d[\s\-()]*){7,}")

def redact_brief(text: str, known_names: Optional[List[str]] = None) -> str:
    if not text: return ""
    red = EMAIL_RX.sub("[EMAIL]", text)
    red = PHONE_RX.sub("[PHONE]", red)
    if known_names:
        for n in sorted(known_names, key=len, reverse=True):
            if n.strip():
                red = re.compile(re.escape(n), re.IGNORECASE).sub("[NAME]", red)
    return red

SYSTEM_PROMPT = (
  "You are a project task planner. Return STRICT JSON with a 'tasks' array.\n"
  "Each task must have: task_id, title, estimate_h (>0), required_skills([{name,level?}]), "
  "dependencies([task_id]), optional due_by(YYYY-MM-DD). No people/dates except due_by."
)

def _build_messages(redacted_brief: str) -> List[Dict[str,str]]:
    user = (
      "BRIEF (PII-redacted):\n"+redacted_brief+"\n\n"
      "Return JSON like:\n"
      "{\n"
      '  "tasks":[{"task_id":"T1","title":"Set up repo","estimate_h":4,'
      '"required_skills":[{"name":"git"},{"name":"python","level":3}],"dependencies":[],'
      '"due_by":"2025-10-25"}]\n}\n'
      "Ensure dependency ids exist; DAG only."
    )
    return [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}]

def extract_tasks_from_brief(brief: str, *, model_id: Optional[str]=None, known_names: Optional[List[str]]=None) -> Dict[str,Any]:
    if not brief or not brief.strip(): raise ValueError("Brief text is empty.")
    model = model_id or _load_models_cfg().get("task_extractor","llama-3.1-8b-instant")
    msgs = _build_messages(redact_brief(brief, known_names))
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=0.1, top_p=1.0, max_tokens=2048,
        response_format={"type":"json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
            raise ValueError('Model must return {"tasks":[...]}')
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse task JSON: {e}; raw={content[:1000]}") from e

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Groq client & task extractor")
    ap.add_argument("--check", action="store_true", help="Instantiate client and print masked key.")
    ap.add_argument("--extract", action="store_true", help="Extract task graph from a brief.")
    ap.add_argument("--brief-file", type=Path, help="Path to a brief file (.txt/.md).")
    ap.add_argument("--brief", type=str, help="Inline brief text.")
    ap.add_argument("--model", type=str, help="Override model id.")
    ap.add_argument("--names", type=str, nargs="*", help="Known names to redact.")
    args = ap.parse_args(argv)

    if args.check and not args.extract:
        print("[OK] Groq client is ready.")
        print(f"[INFO] Using key: {mask_key(resolve_groq_api_key())}")
        print(f"[INFO] Model (config): {_load_models_cfg().get('task_extractor')}")
        return 0

    if args.extract:
        brief = args.brief or (_read_text(args.brief_file) if args.brief_file else "")
        if not brief:
            print("[ERROR] Provide --brief or --brief-file", file=sys.stderr); return 2
        data = extract_tasks_from_brief(brief, model_id=args.model, known_names=args.names)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    ap.print_help(); return 1

if __name__ == "__main__":
    sys.exit(main())
