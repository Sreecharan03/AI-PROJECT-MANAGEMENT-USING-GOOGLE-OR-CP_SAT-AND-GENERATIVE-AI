# ai_pm/scripts/export_run.py
# Phase 7 — Export Run Package (CLI wrapper)
#
# WHY THIS CODE
# -------------
# Provides a small, dependable command-line entrypoint to package a completed run:
#   - Optionally generate a 1-page PDF summary into run_dir/plan/summary.pdf
#   - Always build a full ZIP using core.exporters.export_run_zip(...)
#
# USAGE
# -----
# From the repo root (ai_pm/):
#   python -m scripts.export_run \
#     --project Demo \
#     --run-ts 20251025_094720 \
#     --out /tmp/Demo_20251025_094720.zip \
#     --pdf
#
# Or if you already know the absolute run dir:
#   python -m scripts.export_run \
#     --project Demo \
#     --run-dir runs/Demo/20251025_094720 \
#     --out /tmp/Demo_20251025_094720.zip
#
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Local imports
# We import from core.exporters to reuse export_run_zip and export_pdf_summary.
# This script is meant to be executed from the repo root so that 'core' is importable.
try:
    from core.exporters import export_run_zip, export_pdf_summary
except Exception as e:
    print(f"[ERROR] Failed to import exporters: {e}", file=sys.stderr)
    sys.exit(2)


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_run_dir(project: str, run_ts: Optional[str], run_dir_opt: Optional[Path]) -> Path:
    """
    Prefer --run-dir if provided, otherwise build runs/<project>/<run_ts>.
    """
    if run_dir_opt:
        rd = Path(run_dir_opt).resolve()
    else:
        if not (project and run_ts):
            raise ValueError("When --run-dir is not provided, both --project and --run-ts are required.")
        base = Path.cwd() / "runs" / project
        rd = (base / run_ts).resolve()
    if not rd.exists():
        raise FileNotFoundError(f"Run dir not found: {rd}")
    return rd


def _maybe_build_pdf(run_dir: Path) -> Optional[Path]:
    """
    If plan.json exists, generate summary.pdf under plan/.
    Returns the pdf path if created, else None.
    """
    plan = run_dir / "plan" / "plan.json"
    if not plan.exists():
        print("[WARN] plan.json not found; skipping PDF summary.", file=sys.stderr)
        return None
    kpis = run_dir / "plan" / "kpis.json"
    plan_obj = _read_json(plan)
    kpis_obj = _read_json(kpis) if kpis.exists() else None
    out_pdf = run_dir / "plan" / "summary.pdf"
    export_pdf_summary(plan_obj, kpis_obj, out_pdf)
    return out_pdf


def main() -> int:
    ap = argparse.ArgumentParser("Export AI-PM run package")
    ap.add_argument("--project", type=str, help="Project name (for README and path resolution).")
    ap.add_argument("--run-ts", type=str, help="Timestamp folder name under runs/<project>/ (e.g., 20251025_094720).")
    ap.add_argument("--run-dir", type=Path, help="Full path to run folder. Overrides --project/--run-ts.")
    ap.add_argument("--out", type=Path, required=True, help="Output ZIP path.")
    ap.add_argument("--pdf", action="store_true", help="Also generate a summary.pdf into run_dir/plan/ before zipping.")

    args = ap.parse_args()

    try:
        run_dir = _resolve_run_dir(args.project, args.run_ts, args.run_dir)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Optional PDF summary
    if args.pdf:
        try:
            pdf_path = _maybe_build_pdf(run_dir)
            if pdf_path:
                print(f"[OK] PDF summary -> {pdf_path}")
        except Exception as e:
            print(f"[WARN] PDF summary failed: {e}", file=sys.stderr)

    # Always build the ZIP
    try:
        out = export_run_zip(args.project or run_dir.parent.name, run_dir, args.out)
        print(f"[OK] ZIP written -> {out}")
        return 0
    except Exception as e:
        print(f"[ERROR] ZIP export failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
