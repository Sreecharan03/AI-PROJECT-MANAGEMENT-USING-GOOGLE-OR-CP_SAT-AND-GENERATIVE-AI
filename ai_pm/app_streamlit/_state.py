from __future__ import annotations
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
import streamlit as st
# --- make repo root importable as a package (so `core/...` works) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # .../ai_pm
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _load_yaml(path: Path, default: dict) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else default
    except FileNotFoundError:
        return default

def init_app_state() -> None:
    if st.session_state.get("_app_init_done"):
        return

    root = _repo_root()
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path.as_posix(), override=True)

    settings_default = {
        "timezone": "Asia/Kolkata",
        "workweek": ["Mon","Tue","Wed","Thu","Fri"],
        "workday_hours": 8,
        "data_dir": "runs",
    }
    models_default = {
        "task_extractor": "llama-3.1-8b-instant",
        "safety": "",
    }
    settings = _load_yaml(root / "config" / "settings.yaml", settings_default)
    models = _load_yaml(root / "config" / "models.yaml", models_default)

    st.session_state["settings"] = settings
    st.session_state["models"] = models
    st.session_state["env"] = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
        "APP_MODE": os.environ.get("APP_MODE", "dev"),
        "DATA_DIR": os.environ.get("DATA_DIR", settings.get("data_dir", "runs")),
    }
    st.session_state["_app_init_done"] = True

def sidebar_defaults() -> None:
    s = st.session_state.get("settings", {})
    with st.sidebar:
        st.header("App Defaults")
        st.caption("Loaded from config/settings.yaml")
        st.write(f"**Timezone:** {s.get('timezone', 'Asia/Kolkata')}")
        ww = s.get("workweek", [])
        st.write(f"**Workweek:** {', '.join(ww) if ww else 'Mon–Fri'}")
        st.write(f"**Workday hours:** {s.get('workday_hours', 8)}")
        st.divider()
        st.caption("Phase 0 shell — no business logic yet.")
