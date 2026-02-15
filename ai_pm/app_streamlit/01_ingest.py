# ai_pm/app_streamlit/01_ingest.py
# Phase 1 + Groq-assisted Skill Normalization
#
# What this page provides now:
#   - Tabs: Team (required), Skills (required), Holidays (optional), History (optional), Brief (optional)
#   - Preview and schema validation using core.schemas
#   - NEW: "Skill normalization (Groq)" — detect unknown skills from Team, get LLM suggestions,
#          approve mappings, then persist updated ontology to runs/<project>/<timestamp>/normalized/skills.csv
#   - "Validate & Snapshot" applies approved mappings, writes normalized skills CSV, and snapshots inputs.
#
from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import streamlit as st

# ensure imports work regardless of CWD (done in _state)
from _state import init_app_state, sidebar_defaults
from core import schemas
from core.storage import new_run_dir, snapshot_inputs, is_url
from core.skills import SkillsOntology, _normalize
from core.skills_groq import suggest_with_groq

# ----------------------------
# Utilities (local to page)
# ----------------------------

def _is_google_sheet(url: str) -> bool:
    try:
        u = urlparse(url)
        return "docs.google.com" in u.netloc and "/spreadsheets" in u.path
    except Exception:
        return False

def _to_sheets_csv_export(url: str) -> str:
    """Convert typical Google Sheets URL to a CSV export URL."""
    u = urlparse(url)
    parts = u.path.strip("/").split("/")
    sheet_id = None
    for i, part in enumerate(parts):
        if part == "d" and i + 1 < len(parts):
            sheet_id = parts[i + 1]
            break
    gid = parse_qs(u.fragment).get("gid", ["0"])[0]
    if not sheet_id:
        return url
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def _normalize_source(path_or_url: str) -> str:
    if is_url(path_or_url) and _is_google_sheet(path_or_url):
        return _to_sheets_csv_export(path_or_url)
    return path_or_url

def _read_csv_from_upload(file) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Accepts a Streamlit UploadedFile, returns (header, rows)."""
    raw = file.read()
    text = raw.decode("utf-8-sig", errors="replace")
    sio = io.StringIO(text)
    rd = csv.DictReader(sio)
    header = rd.fieldnames or []
    rows = [dict(r) for r in rd]
    return header, rows

def _read_csv_from_url(url: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    norm = _normalize_source(url)
    r = requests.get(norm, timeout=30)
    r.raise_for_status()
    text = r.text
    sio = io.StringIO(text)
    rd = csv.DictReader(sio)
    header = rd.fieldnames or []
    rows = [dict(r) for r in rd]
    return header, rows

def _preview_table(rows: List[Dict[str, Any]], title: str, max_rows: int = 50) -> None:
    st.caption(title)
    if not rows:
        st.info("No rows.")
        return
    st.dataframe(rows[:max_rows], use_container_width=True, hide_index=True)

def _require_columns_or_message(kind: str, header: Sequence[str]) -> bool:
    if kind == "team":
        req = schemas.TEAM_REQUIRED_COLUMNS
    elif kind == "skills":
        req = schemas.SKILLS_REQUIRED_COLUMNS
    elif kind == "holidays":
        req = schemas.HOLIDAYS_REQUIRED_COLUMNS
    elif kind == "history":
        req = schemas.HISTORY_REQUIRED_COLUMNS
    else:
        req = []
    missing = [c for c in req if c not in header]
    if missing:
        st.error(f"{kind}.csv is missing required columns: {missing}")
        return False
    return True

def _save_upload_to_temp(upload, filename: str, tmpdir: str) -> Path:
    """Persist an uploaded file to a temp path (so snapshot can copy it)."""
    dest = Path(tmpdir) / filename
    dest.write_bytes(upload.getvalue())
    return dest

def _extract_skill_terms_from_team_rows(rows: List[Dict[str, Any]]) -> List[str]:
    """Pull all skill 'name' tokens from the team rows' JSON 'skills' cell."""
    terms: List[str] = []
    for row in rows:
        cell = str(row.get("skills", "")).strip()
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
            continue
    return terms

def _build_ontology_from_skills_rows(rows: List[Dict[str, Any]]) -> SkillsOntology:
    """Create a SkillsOntology from in-memory skills rows by writing a temp CSV."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_skill", "synonyms"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"canonical_skill": r.get("canonical_skill", ""), "synonyms": r.get("synonyms", "")})
        tmp = Path(f.name)
    onto = SkillsOntology.load_from_csv(tmp)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return onto

# ----------------------------
# Page init
# ----------------------------

st.set_page_config(page_title="AI-PM — Ingest", page_icon="📥", layout="wide")
init_app_state()
sidebar_defaults()

st.title("01 — Data Ingest & Validation")
st.write(
    "Provide Team and Skills (required), and optionally Holidays, History, and a Brief. "
    "Use **Skill normalization (Groq)** to resolve unknown skills before **Validate & Snapshot**."
)

# Project name + simple state bucket
with st.container():
    project_name = st.text_input("Project name", value="Demo", help="Used for runs/<project>/<timestamp>/")
    st.caption("Project folders use a safe slug of this name.")
    st.session_state.setdefault("skill_suggestions", {})  # term -> {canonical, candidates, reason}
    st.session_state.setdefault("skill_mappings", {})     # term -> canonical (approved)

# State for collected sources/rows
state: Dict[str, Dict[str, Any]] = {
    "team": {"rows": None, "source": None, "header": None},
    "skills": {"rows": None, "source": None, "header": None},
    "holidays": {"rows": None, "source": None, "header": None},
    "history": {"rows": None, "source": None, "header": None},
    "brief": {"text": None, "upload": None, "name": None},
}

# Tabs for each input
tab_team, tab_skills, tab_holidays, tab_history, tab_brief = st.tabs(
    ["Team (required)", "Skills (required)", "Holidays (optional)", "History (optional)", "Brief (optional)"]
)

with tab_team:
    st.subheader("Team")
    mode = st.radio("Input method", ["Upload CSV", "Sheet/URL"], horizontal=True, key="team_mode")
    if mode == "Upload CSV":
        team_upload = st.file_uploader("Upload team.csv", type=["csv"], accept_multiple_files=False, key="team_upload")
        if team_upload:
            header, rows = _read_csv_from_upload(team_upload)
            if _require_columns_or_message("team", header):
                state["team"]["rows"], state["team"]["header"] = rows, header
                state["team"]["source"] = f"uploaded:{team_upload.name}"
                _preview_table(rows, "Preview: team.csv")
    else:
        team_url = st.text_input("Team CSV URL (Google Sheet link accepted)", key="team_url")
        if team_url:
            try:
                header, rows = _read_csv_from_url(team_url)
                if _require_columns_or_message("team", header):
                    state["team"]["rows"], state["team"]["header"] = rows, header
                    state["team"]["source"] = team_url
                    _preview_table(rows, "Preview: team.csv (from URL)")
            except Exception as e:
                st.error(f"Failed to load team from URL: {e}")

with tab_skills:
    st.subheader("Skills")
    mode = st.radio("Input method", ["Upload CSV", "Sheet/URL"], horizontal=True, key="skills_mode")
    if mode == "Upload CSV":
        f = st.file_uploader("Upload skills.csv", type=["csv"], accept_multiple_files=False, key="skills_upload")
        if f:
            header, rows = _read_csv_from_upload(f)
            if _require_columns_or_message("skills", header):
                state["skills"]["rows"], state["skills"]["header"] = rows, header
                state["skills"]["source"] = f"uploaded:{f.name}"
                _preview_table(rows, "Preview: skills.csv")
    else:
        u = st.text_input("Skills CSV URL (Google Sheet link accepted)", key="skills_url")
        if u:
            try:
                header, rows = _read_csv_from_url(u)
                if _require_columns_or_message("skills", header):
                    state["skills"]["rows"], state["skills"]["header"] = rows, header
                    state["skills"]["source"] = u
                    _preview_table(rows, "Preview: skills.csv (from URL)")
            except Exception as e:
                st.error(f"Failed to load skills from URL: {e}")

with tab_holidays:
    st.subheader("Holidays (optional)")
    mode = st.radio("Input method", ["None", "Upload CSV", "Sheet/URL"], horizontal=True, key="holidays_mode")
    if mode == "Upload CSV":
        f = st.file_uploader("Upload holidays.csv", type=["csv"], accept_multiple_files=False, key="holidays_upload")
        if f:
            header, rows = _read_csv_from_upload(f)
            if _require_columns_or_message("holidays", header):
                state["holidays"]["rows"], state["holidays"]["header"] = rows, header
                state["holidays"]["source"] = f"uploaded:{f.name}"
                _preview_table(rows, "Preview: holidays.csv")
    elif mode == "Sheet/URL":
        u = st.text_input("Holidays CSV URL (Google Sheet link accepted)", key="holidays_url")
        if u:
            try:
                header, rows = _read_csv_from_url(u)
                if _require_columns_or_message("holidays", header):
                    state["holidays"]["rows"], state["holidays"]["header"] = rows, header
                    state["holidays"]["source"] = u
                    _preview_table(rows, "Preview: holidays.csv (from URL)")
            except Exception as e:
                st.error(f"Failed to load holidays from URL: {e}")

with tab_history:
    st.subheader("History (optional)")
    mode = st.radio("Input method", ["None", "Upload CSV", "Sheet/URL"], horizontal=True, key="history_mode")
    if mode == "Upload CSV":
        f = st.file_uploader("Upload history.csv", type=["csv"], accept_multiple_files=False, key="history_upload")
        if f:
            header, rows = _read_csv_from_upload(f)
            if _require_columns_or_message("history", header):
                state["history"]["rows"], state["history"]["header"] = rows, header
                state["history"]["source"] = f"uploaded:{f.name}"
                _preview_table(rows, "Preview: history.csv")
    elif mode == "Sheet/URL":
        u = st.text_input("History CSV URL (Google Sheet link accepted)", key="history_url")
        if u:
            try:
                header, rows = _read_csv_from_url(u)
                if _require_columns_or_message("history", header):
                    state["history"]["rows"], state["history"]["header"] = rows, header
                    state["history"]["source"] = u
                    _preview_table(rows, "Preview: history.csv (from URL)")
            except Exception as e:
                st.error(f"Failed to load history from URL: {e}")

with tab_brief:
    st.subheader("Brief (optional)")
    brief_text = st.text_area("Enter a short project brief (optional)", height=160)
    brief_upload = st.file_uploader("…or upload a brief (.txt/.md)", type=["txt", "md"], accept_multiple_files=False, key="brief_upload")
    if brief_text:
        state["brief"]["text"] = brief_text
        state["brief"]["name"] = "brief.txt"
        st.caption("Will include inline text as brief.txt")
    if brief_upload:
        state["brief"]["upload"] = brief_upload
        state["brief"]["name"] = brief_upload.name
        st.caption(f"Will include uploaded file: {brief_upload.name}")

# ----------------------------
# Skill normalization (Groq)
# ----------------------------

st.divider()
st.subheader("Skill normalization (Groq)")

if state["team"]["rows"] and state["skills"]["rows"]:
    onto = _build_ontology_from_skills_rows(state["skills"]["rows"])
    terms = _extract_skill_terms_from_team_rows(state["team"]["rows"])
    unknowns = onto.find_unknown_terms(terms)

    if unknowns:
        st.warning("Unknown skills detected:", icon="⚠️")
        st.write(", ".join([f"`{u}`" for u in unknowns]))

        # Trigger suggestions (stores in session_state)
        if st.button("🔎 Run Groq suggestions for unknowns", help="Calls Groq to pick best canonical for each unknown"):
            suggestions = {}
            for term in unknowns:
                sug = suggest_with_groq(onto, term)
                suggestions[term] = {
                    "canonical": sug.canonical,
                    "reason": sug.reason,
                    "candidates": sug.candidates,
                }
            st.session_state["skill_suggestions"] = suggestions
            st.success("Suggestions ready. Review below and approve mappings.")

        # If we have suggestions, render decision form
        if st.session_state.get("skill_suggestions"):
            st.markdown("**Suggestions** (edit or skip):")
            with st.form("skill_map_form", clear_on_submit=False):
                choices: Dict[str, str] = {}
                for term, sug in st.session_state["skill_suggestions"].items():
                    cands = sug["candidates"]
                    default = sug["canonical"] if sug["canonical"] in cands else "Skip"
                    options = ["Skip"] + cands
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        picked = st.selectbox(
                            f"Map `{term}` →",
                            options=options,
                            index=options.index(default) if default in options else 0,
                            key=f"pick_{term}",
                        )
                    with col2:
                        st.caption(f"Reason: {sug['reason'] or '—'}")
                    choices[term] = picked
                submitted = st.form_submit_button("💾 Save mapping decisions")
                if submitted:
                    accepted = {t: c for t, c in choices.items() if c != "Skip"}
                    st.session_state["skill_mappings"] = accepted
                    if accepted:
                        st.success(f"Saved {len(accepted)} mapping(s). They will be applied on snapshot.")
                    else:
                        st.info("No mappings selected.")
    else:
        st.success("No unknown skills. You're good to snapshot.")

else:
    st.info("Provide Team and Skills to enable normalization.")

# ----------------------------
# Validate & Snapshot
# ----------------------------

st.divider()
colA, colB = st.columns([1, 3])
with colA:
    validate_and_snapshot = st.button("✅ Validate & Snapshot", type="primary")
with colB:
    st.caption("Validates datasets, applies approved skill mappings, writes normalized skills CSV, and snapshots inputs with hashes.")

if validate_and_snapshot:
    # 1) Required presence
    if not state["team"]["rows"] or not state["skills"]["rows"]:
        st.error("Team and Skills are required. Please provide them via upload or URL.")
        st.stop()

    # 2) Row-level validations
    ok = True
    tv, te = schemas.validate_team_rows(state["team"]["rows"], "team.csv")
    if te:
        ok = False
        st.error(f"[team] {len(te)} error(s) — first: {te[0].message}")
    sv, se = schemas.validate_skills_rows(state["skills"]["rows"], "skills.csv")
    if se:
        ok = False
        st.error(f"[skills] {len(se)} error(s) — first: {se[0].message}")

    hv = he = yv = ye = 0
    if state["holidays"]["rows"]:
        hv, he = schemas.validate_holidays_rows(state["holidays"]["rows"], "holidays.csv")
        if he:
            ok = False
            st.error(f"[holidays] {len(he)} error(s) — first: {he[0].message}")
    if state["history"]["rows"]:
        yv, ye = schemas.validate_history_rows(state["history"]["rows"], "history.csv")
        if ye:
            ok = False
            st.error(f"[history] {len(ye)} error(s) — first: {ye[0].message}")

    if not ok:
        st.stop()

    # 3) Build ontology and apply approved mappings (if any)
    onto = _build_ontology_from_skills_rows(state["skills"]["rows"])
    approved: Dict[str, str] = st.session_state.get("skill_mappings", {})
    for term, canonical in approved.items():
        onto.add_synonym(canonical, term)

    # 4) Create run directory up-front so we can persist normalized CSV
    try:
        run_dir = new_run_dir(project_name)
    except Exception as e:
        st.error(f"Failed to create run directory: {e}")
        st.stop()

    # 5) Persist normalized ontology
    normalized_dir = Path(run_dir) / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv = normalized_dir / "skills.csv"
    try:
        onto.persist_csv(normalized_csv, json_synonyms=True)
        st.success(f"Normalized skills written: `{normalized_csv}`")
    except Exception as e:
        st.error(f"Failed to write normalized skills.csv: {e}")
        st.stop()

    # 6) Prepare inputs list (files or URLs). For uploads, write temp CSVs so snapshot can copy them.
    snapshot_sources: List[str] = []
    tempdir = tempfile.TemporaryDirectory()

    def _ensure_csv_from_rows(field_key: str, filename: str) -> None:
        rows = state[field_key]["rows"] or []
        if not rows:
            return
        path = Path(tempdir.name) / filename
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        snapshot_sources.append(path.as_posix())

    # Team
    if state["team"]["source"] and str(state["team"]["source"]).startswith("http"):
        snapshot_sources.append(state["team"]["source"])
    else:
        _ensure_csv_from_rows("team", "team.csv")

    # Skills (raw) — keep original ontology as input; normalized version is persisted separately
    if state["skills"]["source"] and str(state["skills"]["source"]).startswith("http"):
        snapshot_sources.append(state["skills"]["source"])
    else:
        _ensure_csv_from_rows("skills", "skills.csv")

    # Optional inputs
    if state["holidays"]["rows"]:
        if state["holidays"]["source"] and str(state["holidays"]["source"]).startswith("http"):
            snapshot_sources.append(state["holidays"]["source"])
        else:
            _ensure_csv_from_rows("holidays", "holidays.csv")
    if state["history"]["rows"]:
        if state["history"]["source"] and str(state["history"]["source"]).startswith("http"):
            snapshot_sources.append(state["history"]["source"])
        else:
            _ensure_csv_from_rows("history", "history.csv")

    # Brief
    if state["brief"]["text"] or state["brief"]["upload"]:
        brief_name = state["brief"]["name"] or "brief.txt"
        brief_path = Path(tempdir.name) / brief_name
        if state["brief"]["upload"] is not None:
            brief_path.write_bytes(state["brief"]["upload"].getvalue())
        else:
            brief_path.write_text(state["brief"]["text"], encoding="utf-8")
        snapshot_sources.append(brief_path.as_posix())

    # 7) Snapshot inputs into this run (normalized CSV already stored under run_dir/normalized)
    try:
        result = snapshot_inputs(Path(run_dir), snapshot_sources)
    except Exception as e:
        st.error(f"Snapshot failed: {e}")
        st.stop()

    st.success("Validation passed and snapshot created.")
    st.write(f"**Run folder:** `{result.run_dir}`")
    st.write(f"**Inputs:** `{result.inputs_dir}`")
    st.write(f"**Log:** `{result.logs_dir}/hashes.json`")
    try:
        manifest = json.loads((Path(result.logs_dir) / "hashes.json").read_text(encoding="utf-8"))
        st.json(manifest)
    except Exception:
        pass

# Optional non-Streamlit main()
def main() -> None:
    print("This is a Streamlit page. Launch via:\n  streamlit run app_streamlit/01_ingest.py")

if __name__ == "__main__":
    main()
