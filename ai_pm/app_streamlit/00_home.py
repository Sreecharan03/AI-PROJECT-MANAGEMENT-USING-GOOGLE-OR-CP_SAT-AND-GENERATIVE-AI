# ai_pm/app_streamlit/00_home.py
# Home page with robust navigation:
# - If Streamlit supports st.page_link AND the pages are in the multipage registry, show clickable links.
# - Else, fall back to showing shell commands the user can copy to open each page directly.
from pathlib import Path
import streamlit as st
from _state import init_app_state, sidebar_defaults, _repo_root

# ----- Page config & shared state -----
st.set_page_config(page_title="AI-PM — Home", page_icon="🧠", layout="wide")
init_app_state()
sidebar_defaults()  # shows tz / workweek / workday defaults in the sidebar

# ----- Try navigation (safe) -----
def try_sidebar_nav() -> bool:
    """
    Try to render clickable page links. Returns True if links were rendered,
    False if we should fall back to command hints.
    """
    ok = False
    with st.sidebar:
        st.markdown("### Navigation")
        # st.page_link can raise if the target file isn't registered in the multipage app
        def _safe_link(path: str, label: str, icon: str = "📄") -> bool:
            try:
                st.page_link(path, label=label, icon=icon)
                return True
            except Exception:
                return False

        # Attempt links for each page
        results = [
            _safe_link("app_streamlit/00_home.py", "00 — Home", "🏠"),
            _safe_link("app_streamlit/01_ingest.py", "01 — Ingest", "⬆️"),
            _safe_link("app_streamlit/02_task_builder.py", "02 — Task Builder", "🧩"),
            _safe_link("app_streamlit/03_optimizer.py", "03 — Optimizer", "🧮"),
            _safe_link("app_streamlit/04_review.py", "04 — Review", "📝"),
            _safe_link("app_streamlit/05_analytics.py", "05 — Analytics", "📊"),
            _safe_link("app_streamlit/06_admin.py", "06 — Admin", "⚙️"),
        ]
        ok = all(r is not False for r in results) or any(results)
        if not ok:
            st.caption("⚠️ Streamlit page links aren’t available in this environment. Use the commands below.")
    return ok

links_work = try_sidebar_nav()

# ----- Main content -----
st.title("AI-PM — Home")
st.success("Bootstrap is live. Use the left sidebar if links show, or the commands below.")

st.markdown(
    "This app reads defaults from `config/settings.yaml` (timezone, workweek, workday). "
    "Upload data on **01 — Ingest**, build tasks on **02 — Task Builder**, and "
    "generate a plan on **03 — Optimizer**."
)

# If links aren’t available, show copy-paste commands
if not links_work:
    st.subheader("How to open each page (copy/paste these commands)")
    st.code(
        "streamlit run app_streamlit/01_ingest.py\n"
        "streamlit run app_streamlit/02_task_builder.py\n"
        "streamlit run app_streamlit/03_optimizer.py\n"
        "streamlit run app_streamlit/04_review.py\n"
        "streamlit run app_streamlit/05_analytics.py\n"
        "streamlit run app_streamlit/06_admin.py",
        language="bash",
    )
    st.caption("Tip: you can run multiple pages in different terminals if that’s easier.")

with st.expander("Debug / Context"):
    root = _repo_root()
    st.write({"repo_root": str(root)})
    st.write({"config_dir": str(root / "config")})
    st.write({"runs_dir": str(root / "runs")})

# Per your rule: provide a main(), even though Streamlit runs this file directly.
def main():
    print("Launch with:\n  streamlit run app_streamlit/00_home.py\n"
          "If the sidebar links don't appear, run pages directly, e.g.:\n"
          "  streamlit run app_streamlit/03_optimizer.py")

if __name__ == "__main__":
    main()
