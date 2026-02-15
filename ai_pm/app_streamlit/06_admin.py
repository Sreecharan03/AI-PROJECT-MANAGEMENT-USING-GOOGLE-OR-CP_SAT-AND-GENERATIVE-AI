# ai_pm/app_streamlit/06_admin.py
# Phase 6/7 — Admin: preferences (θ) + Export Run Package
#
# WHAT THIS PAGE DOES
# -------------------
# • Shows current θ from runs/<Project>/preferences.json (if present), else defaults.
# • Lets you upload a preferences.json to replace the current one.
# • Lets you reset θ to defaults and saves back to runs/<Project>/preferences.json.
# • NEW: Lets you select a completed run (runs/<Project>/<timestamp>/) and:
#     - optionally generate a summary.pdf into plan/
#     - export a portable ZIP under runs/<Project>/<timestamp>/exports/<Project>_<timestamp>.zip
# • No /tmp writes; all artifacts live in the repo under runs/.
# • Includes a tiny main().
#
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# --- Robust import shim so 'core.*' resolves under Streamlit ---
BASE_DIR = Path(__file__).resolve().parents[1]  # .../ai_pm
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# ----------------------------------------------------------------

from _state import init_app_state, sidebar_defaults

# Exporters (for ZIP/PDF)
try:
    from core.exporters import export_run_zip, export_pdf_summary
except Exception as e:
    export_run_zip = None
    export_pdf_summary = None
    _EXPORT_IMPORT_ERR = e
else:
    _EXPORT_IMPORT_ERR = None

DEFAULT_THETA = {"skill_fit": 0.5, "fairness": 0.2, "continuity": 0.2, "deadline_risk": 0.1}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _prefs_path(project: str) -> Path:
    return _repo_root() / "runs" / project / "preferences.json"

def _load_prefs(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"theta": dict(DEFAULT_THETA), "updated_at": None, "last_comparison": None}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        th = data.get("theta") or {}
        for k, v in DEFAULT_THETA.items():
            th.setdefault(k, v)
        data["theta"] = th
        return data
    except Exception:
        return {"theta": dict(DEFAULT_THETA), "updated_at": None, "last_comparison": None}

def _save_prefs(p: Path, payload: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _runs_root(project: str) -> Path:
    return _repo_root() / "runs" / project

def _list_runs(project: str) -> List[Path]:
    root = _runs_root(project)
    if not root.exists():
        return []
    # pick only those that look complete: have plan/plan.json
    candidates: List[Path] = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if (d / "plan" / "plan.json").exists():
            candidates.append(d)
    return candidates

def _zip_out_path(run_dir: Path, project: str) -> Path:
    (run_dir / "exports").mkdir(parents=True, exist_ok=True)
    return run_dir / "exports" / f"{project}_{run_dir.name}.zip"

# ---------- UI ----------
st.set_page_config(page_title="AI-PM — Admin", page_icon="🛠️", layout="wide")
init_app_state()
sidebar_defaults()

st.title("06 — Admin (Preferences & Export)")
st.caption("Manage θ (coactive learning) and export run packages (ZIP, optional summary.pdf).")

project = st.text_input("Project name", value="Demo")

# ===== Left: Preferences =====
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Preferences (θ)")
    prefs_path = _prefs_path(project)
    prefs = _load_prefs(prefs_path)
    st.json({"file": str(prefs_path), **prefs})

    up = st.file_uploader("Replace preferences.json", type=["json"], key="prefs_up")
    if up and st.button("Upload & replace", type="secondary"):
        try:
            newp = json.loads(up.getvalue().decode("utf-8"))
            if "theta" not in newp or not isinstance(newp["theta"], dict):
                st.error("Invalid file: missing 'theta' object.")
            else:
                newp["updated_at"] = datetime.now().isoformat(timespec="seconds")
                _save_prefs(prefs_path, newp)
                st.success(f"Replaced: {prefs_path}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    if st.button("Reset θ to defaults", type="primary"):
        payload = {
            "theta": dict(DEFAULT_THETA),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "last_comparison": None
        }
        _save_prefs(prefs_path, payload)
        st.success(f"Reset to defaults at: {prefs_path}")

# ===== Right: Export Run Package =====
with colB:
    st.subheader("Export Run Package (ZIP)")
    if _EXPORT_IMPORT_ERR:
        st.error(f"Exporters unavailable: {_EXPORT_IMPORT_ERR}")
    else:
        runs = _list_runs(project)
        if not runs:
            st.info(f"No runs found under `runs/{project}`. Generate a plan first on page 03.")
        else:
            # Choose a run (latest first)
            labels = [d.name for d in runs]
            idx_default = 0
            sel = st.selectbox("Run (timestamp folder)", options=labels, index=idx_default)
            run_dir = runs[labels.index(sel)]

            # Optional PDF
            gen_pdf = st.checkbox("Generate summary.pdf before zipping", value=True,
                                  help="Creates plan/summary.pdf (KPIs + mini table). Requires reportlab.")

            # Preview targets
            plan_path = run_dir / "plan" / "plan.json"
            kpis_path = run_dir / "plan" / "kpis.json"
            out_zip = _zip_out_path(run_dir, project)

            st.write({
                "run_dir": str(run_dir),
                "plan": str(plan_path),
                "kpis": str(kpis_path) if kpis_path.exists() else "(missing)",
                "zip_out": str(out_zip),
            })

            # Action: Export
            if st.button("Build ZIP"):
                try:
                    # Optional PDF
                    if gen_pdf and export_pdf_summary is not None:
                        try:
                            plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
                            kpis_obj = json.loads(kpis_path.read_text(encoding="utf-8")) if kpis_path.exists() else None
                            pdf_path = run_dir / "plan" / "summary.pdf"
                            export_pdf_summary(plan_obj, kpis_obj, pdf_path)
                            st.success(f"summary.pdf written → {pdf_path}")
                        except Exception as e:
                            st.warning(f"summary.pdf failed: {e}")

                    # ZIP
                    export_run_zip(project, run_dir, out_zip)
                    st.success(f"ZIP written → {out_zip}")

                    # Download button (reads ZIP bytes to stream)
                    try:
                        data = out_zip.read_bytes()
                        st.download_button(
                            "Download ZIP",
                            data=data,
                            file_name=out_zip.name,
                            mime="application/zip"
                        )
                    except Exception:
                        st.info("ZIP created on disk; manual download from path above if streaming fails.")

                except Exception as e:
                    st.error(f"Export failed: {e}")

# Rule: include a main()
def main() -> None:
    print("This is a Streamlit page. Launch with:\n  streamlit run app_streamlit/06_admin.py")

if __name__ == "__main__":
    main()
