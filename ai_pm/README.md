# AI-PM (Phase 0 — Bootstrap)

Minimal multipage Streamlit shell with shared sidebar state (timezone, workweek, hours).
No business logic yet (LLM/CP-SAT comes later).

## Quickstart
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # (optional) then edit
streamlit run app_streamlit/00_home.py

## Defaults
- Timezone: Asia/Kolkata
- Workweek: Mon–Fri
- Workday: 8 hours
- Data dir: runs/
