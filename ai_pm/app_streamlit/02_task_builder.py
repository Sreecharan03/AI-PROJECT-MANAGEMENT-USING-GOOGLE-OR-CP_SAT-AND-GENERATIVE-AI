# ai_pm/app_streamlit/02_task_builder.py
# Phase 3 — Task Builder UI (brief → Groq tasks → edit → validate & save)
#
# CHANGELOG (fix):
#   • Add the required `--extract` flag when invoking core/llm_extractor.py.
#   • Keep absolute paths + PYTHONPATH so `core` imports resolve from Streamlit.

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import requests

from _state import init_app_state, sidebar_defaults
from core.storage import new_run_dir, snapshot_inputs

# ----------------------------
# Helpers
# ----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # .../ai_pm
LLM_EXTRACTOR_PY = BASE_DIR / "core" / "llm_extractor.py"
VALIDATOR_PY = BASE_DIR / "core" / "validator.py"

def _write_temp_text(text: str, suffix: str = ".txt") -> Path:
    """Save inline text to a temp file and return its path."""
    p = tempfile.NamedTemporaryFile("w", delete=False, suffix=suffix, encoding="utf-8")
    p.write(text)
    p.flush()
    p.close()
    return Path(p.name)

def _write_uploaded(upload, suffix: str) -> Path:
    """Persist an UploadedFile/bytes to a temp path; rewind first so multiple reads work."""
    if hasattr(upload, "seek"):
        try:
            upload.seek(0)
        except Exception:
            pass
    buf = bytes(upload.getbuffer()) if hasattr(upload, "getbuffer") else (upload.read() if hasattr(upload, "read") else upload)
    if not buf:
        raise RuntimeError("Uploaded file was empty.")
    p = tempfile.NamedTemporaryFile("wb", delete=False, suffix=suffix)
    p.write(buf)
    p.flush()
    p.close()
    return Path(p.name)

def _download_to_tmp(url: str, suffix: str) -> Path:
    """Download a CSV (e.g., Google Sheet export) to a temp file."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    p = tempfile.NamedTemporaryFile("wb", delete=False, suffix=suffix)
    p.write(r.content)
    p.flush()
    p.close()
    return Path(p.name)

def _env_with_repo() -> Dict[str, str]:
    """Return env with PYTHONPATH including ai_pm root so 'core' imports resolve."""
    env = os.environ.copy()
    root = str(BASE_DIR)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env

def _run_llm_extractor(brief_txt_path: Path) -> Dict[str, Any]:
    """
    Invoke extractor script with the required --extract flag.
    Returns parsed JSON: {"tasks":[...]}
    """
    cmd = [sys.executable, str(LLM_EXTRACTOR_PY), "--extract", "--brief-file", str(brief_txt_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_repo())
    if proc.returncode != 0:
        raise RuntimeError(f"Extractor failed: {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        return json.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"Extractor returned non-JSON output: {e}")

def _run_validator(raw_json_path: Path, skills_csv_path: Path, out_json_path: Path) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Invoke validator script to canonicalize skills and DAG-check.
    Returns (parsed_normalized, error_message|None)
    """
    cmd = [
        sys.executable, str(VALIDATOR_PY),
        "--tasks", str(raw_json_path),
        "--skills", str(skills_csv_path),
        "--out", str(out_json_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_repo())
    if proc.returncode != 0:
        return None, (proc.stderr.strip() or proc.stdout.strip())
    try:
        obj = json.loads(out_json_path.read_text(encoding="utf-8"))
        return obj, None
    except Exception as e:
        return None, f"Validator wrote unreadable JSON: {e}"

def _preview_tasks_table(tasks_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert {"tasks":[...]} into rows for st.data_editor.
    required_skills → JSON string; dependencies → comma-separated string.
    """
    rows = []
    for t in (tasks_obj.get("tasks") or []):
        deps = t.get("dependencies") or []
        deps_str = ",".join(str(x) for x in deps) if isinstance(deps, list) else str(deps)
        rs = t.get("required_skills") or []
        try:
            rs_str = json.dumps(rs, ensure_ascii=False)
        except Exception:
            rs_str = "[]"
        rows.append({
            "task_id": t.get("task_id"),
            "title": t.get("title"),
            "estimate_h": t.get("estimate_h"),
            "required_skills_json": rs_str,
            "dependencies_csv": deps_str,
            "due_by": t.get("due_by"),
        })
    return rows

def _rows_to_tasks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert the edited rows back to {"tasks":[...]}.
    - required_skills_json must be a JSON list
    - dependencies_csv is comma-separated task_ids
    """
    tasks = []
    for r in rows:
        try:
            reqs = json.loads(r.get("required_skills_json") or "[]")
            if not isinstance(reqs, list):
                raise ValueError("required_skills_json must be a JSON list")
        except Exception as e:
            raise ValueError(f'Row for task "{r.get("task_id")}": invalid required_skills_json ({e})')
        deps_csv = r.get("dependencies_csv") or ""
        deps = [x.strip() for x in deps_csv.split(",") if x.strip()] if deps_csv else []
        try:
            estf = float(r.get("estimate_h"))
        except Exception:
            raise ValueError(f'Row for task "{r.get("task_id")}": estimate_h must be a number')
        t = {
            "task_id": (r.get("task_id") or "").strip(),
            "title": r.get("title") or "",
            "estimate_h": estf,
            "required_skills": reqs,
            "dependencies": deps,
        }
        due = r.get("due_by")
        if due:
            t["due_by"] = str(due)
        tasks.append(t)
    return {"tasks": tasks}


# ----------------------------
# Page UI
# ----------------------------

st.set_page_config(page_title="AI-PM — Task Builder", page_icon="🧩", layout="wide")
init_app_state()
sidebar_defaults()

st.title("02 — Task Builder (Groq → task_graph.json)")
st.caption("Paste a Brief, generate tasks with Groq, edit, then Validate & Save to runs/<project>/<timestamp>/normalized/task_graph.json")

# Project name
project_name = st.text_input("Project name", value="Demo")

# Inputs: Brief + ontology
st.subheader("Brief")
brief_txt = st.text_area("Paste your project brief here", height=200, placeholder="Describe the project goals, deliverables, and constraints...")
brief_file = st.file_uploader("...or upload a .txt brief", type=["txt"])

st.subheader("Skills Ontology")
col1, col2 = st.columns([2, 3])
with col1:
    skills_upload = st.file_uploader("skills.csv (canonical + synonyms)", type=["csv"])
with col2:
    sheet_url = st.text_input("Or Google Sheet CSV URL (optional)", placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv")

st.caption("Tip: Use **01 — Ingest** first if you need to build or normalize your ontology.")

# Generate from Brief
raw_tasks_state_key = "task_builder_raw_rows"
if st.button("🚀 Generate from Brief", type="primary"):
    # Resolve brief content
    if brief_file is not None:
        try:
            brief_path = _write_uploaded(brief_file, ".txt")
        except Exception as e:
            st.error(f"Failed to read brief file: {e}")
            st.stop()
    else:
        txt = (brief_txt or "").strip()
        if not txt:
            st.error("Provide a brief (paste text or upload a .txt file).")
            st.stop()
        brief_path = _write_temp_text(txt, ".txt")

    # Call LLM extractor (absolute script path + PYTHONPATH + --extract)
    try:
        raw_graph = _run_llm_extractor(brief_path)
    except Exception as e:
        st.error(f"LLM extraction failed: {e}")
        st.stop()

    # Display raw JSON summary + editable grid
    st.success("Tasks generated from brief.")
    st.json({"tasks_count": len(raw_graph.get("tasks", []))})
    st.session_state[raw_tasks_state_key] = _preview_tasks_table(raw_graph)

# Editable grid (if present)
rows = st.session_state.get(raw_tasks_state_key, [])
if rows:
    st.subheader("Edit tasks")
    st.caption("Edit cells below. Use JSON list for `required_skills_json`, e.g. "
               '`[{"name":"react","level":2},{"name":"python","level":3}]` and comma list for `dependencies_csv`, e.g. `T1,T2`.')

    edited = st.data_editor(
        rows,
        key="task_editor",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "task_id": st.column_config.TextColumn("task_id", help="Unique id (e.g., T1)"),
            "title": st.column_config.TextColumn("title"),
            "estimate_h": st.column_config.NumberColumn("estimate_h (hours)", min_value=0.0, step=1.0),
            "required_skills_json": st.column_config.TextColumn("required_skills (JSON list)"),
            "dependencies_csv": st.column_config.TextColumn("dependencies (comma separated task_ids)"),
            "due_by": st.column_config.TextColumn("due_by (YYYY-MM-DD, optional)"),
        },
    )

    # Validate & Save
    st.subheader("Validate & Save")
    st.caption("Select / provide your ontology, then validate and save a normalized `task_graph.json` into a new run folder.")

    # Resolve skills.csv path
    skills_path: Optional[Path] = None
    if skills_upload is not None:
        try:
            skills_path = _write_uploaded(skills_upload, ".csv")
        except Exception as e:
            st.error(f"Failed to read skills.csv upload: {e}")
            st.stop()
    elif (sheet_url or "").strip():
        try:
            skills_path = _download_to_tmp(sheet_url.strip(), ".csv")
        except Exception as e:
            st.error(f"Failed to download sheet: {e}")
            st.stop()
    else:
        st.warning("Please provide an ontology (skills.csv upload or a Google Sheet CSV URL).")

    if st.button("✅ Validate & Save", type="primary", disabled=(skills_path is None)):
        try:
            # Convert edited rows back to a raw tasks JSON
            raw_obj = _rows_to_tasks(edited)
        except Exception as e:
            st.error(f"Parsing error: {e}")
            st.stop()

        # Write raw to temp
        raw_tmp = _write_temp_text(json.dumps(raw_obj, ensure_ascii=False), ".json")
        norm_tmp = Path(tempfile.NamedTemporaryFile("w", delete=False, suffix=".json").name)

        # Run validator → normalized JSON (absolute script path + PYTHONPATH)
        normalized, err = _run_validator(raw_tmp, skills_path, norm_tmp)
        if err:
            st.error(f"Validation failed:\n{err}")
            st.stop()

        # Persist to runs/<project>/<ts>/normalized/task_graph.json + snapshot inputs
        run_dir = Path(new_run_dir(project_name))
        norm_dir = run_dir / "normalized"
        norm_dir.mkdir(parents=True, exist_ok=True)
        dest = norm_dir / "task_graph.json"
        dest.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

        # Snapshot brief + skills
        snap_inputs = [skills_path.as_posix()]
        if brief_file is not None:
            snap_inputs.append(str(_write_uploaded(brief_file, ".txt")))
        elif (brief_txt or "").strip():
            snap_inputs.append(str(_write_temp_text(brief_txt, ".txt")))
        try:
            snapshot_inputs(run_dir, snap_inputs)
        except Exception as e:
            st.warning(f"Snapshot warning: {e}")

        st.success("Normalized task graph saved.")
        st.write(f"**Run folder:** `{run_dir}`")
        st.write(f"**Task graph:** `{dest}`")

# Per your rule: main()
def main() -> None:
    print("This is a Streamlit page. Launch with:\n  streamlit run app_streamlit/02_task_builder.py")

if __name__ == "__main__":
    main()
