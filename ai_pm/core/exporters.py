# ai_pm/core/exporters.py
# Phase 7 — Analytics & Exports (Foundational utilities)
#
# WHAT THIS MODULE DOES
# ---------------------
# 1) export_assignments_csv(plan_json, out_csv): writes a flat CSV of task assignments.
# 2) export_plan_json(plan_json, out_json): pretty JSON copy of the plan.
# 3) export_pdf_summary(plan_json, kpis_json, out_pdf): 1-page PDF summary (KPIs + mini table).
# 4) export_run_zip(project, run_dir, out_zip): bundles inputs/ → normalized/ → plan/ → logs/ plus README.md.
#
# USAGE (CLI)
# -----------
# • Export CSV & PDF beside your plan:
#   python -m core.exporters \
#     --plan runs/Demo/20251025_094720/plan/plan.json \
#     --kpis runs/Demo/20251025_094720/plan/kpis.json \
#     --csv /tmp/assignments.csv \
#     --pdf /tmp/summary.pdf \
#     --json /tmp/plan_copy.json
#
# • Export a full run package ZIP:
#   python -m core.exporters \
#     --export-zip \
#     --project Demo \
#     --run-dir runs/Demo/20251025_094720 \
#     --out /tmp/Demo_20251025_094720.zip
#
from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# reportlab is in requirements; we keep a friendly error if missing.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
except Exception as _e:
    # Defer raising until someone calls the PDF exporter.
    _REPORTLAB_ERR = _e
else:
    _REPORTLAB_ERR = None


# ----------------------------
# Small helpers
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _friendly_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class PlanRefs:
    plan: Dict[str, Any]
    kpis: Optional[Dict[str, Any]] = None


# ----------------------------
# Core exports
# ----------------------------

def export_assignments_csv(plan: Dict[str, Any], out_csv: Path) -> Path:
    """
    Write a flat assignment table from a plan.json.

    Columns:
      task_id,title,member_id,member_name,estimate_h,start,end,due_by,due_violation
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for t in plan.get("tasks", []):
        rows.append({
            "task_id": t.get("task_id"),
            "title": t.get("title"),
            "member_id": t.get("member_id"),
            "member_name": t.get("member_name"),
            "estimate_h": _safe_float(t.get("estimate_h")),
            "start": t.get("start"),
            "end": t.get("end"),
            "due_by": t.get("due_by"),
            "due_violation": bool(t.get("due_violation") or False),
        })

    cols = ["task_id","title","member_id","member_name","estimate_h","start","end","due_by","due_violation"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return out_csv


def export_plan_json(plan: Dict[str, Any], out_json: Path) -> Path:
    """Pretty write the plan JSON."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def export_pdf_summary(plan: Dict[str, Any], kpis: Optional[Dict[str, Any]], out_pdf: Path) -> Path:
    """
    Build a simple one-page PDF summary (title, KPIs, and first ~12 tasks as a table).
    Requires reportlab.
    """
    if _REPORTLAB_ERR is not None:
        raise RuntimeError(
            "PDF export requires reportlab. Install it (already in requirements): "
            f"{_REPORTLAB_ERR}"
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: List[Any] = []

    project = plan.get("project") or "AI-PM"
    title = f"{project} — Plan Summary"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Generated: {_friendly_ts()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # KPIs block (if available)
    if kpis:
        story.append(Paragraph("KPIs", styles["Heading2"]))
        k = kpis
        data = [
            ["coverage", str(k.get("coverage"))],
            ["capacity_violations", str(k.get("capacity_violations"))],
            ["avg_skill_fit", str(k.get("avg_skill_fit"))],
            ["utilization_stddev", str(k.get("utilization_stddev"))],
            ["critical_path_hours", str(k.get("critical_path_hours"))],
            ["due_by_violations", str(k.get("due_by_violations"))],
        ]
        tbl = Table(data, hAlign="LEFT", colWidths=[160, 200])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("BOX", (0,0), (-1,-1), 0.25, colors.gray),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # Assignment table (truncate to keep 1-page friendly)
    tasks = plan.get("tasks", [])
    story.append(Paragraph("Assignments (first 12 rows)", styles["Heading2"]))
    header = ["task_id", "title", "member", "start", "end"]
    data = [header]
    for t in tasks[:12]:
        data.append([
            t.get("task_id"),
            (t.get("title") or "")[:40],
            f'{t.get("member_name")} ({t.get("member_id")})',
            t.get("start") or "",
            t.get("end") or "",
        ])
    tbl = Table(data, hAlign="LEFT", colWidths=[60, 180, 140, 80, 80])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("BOX", (0,0), (-1,-1), 0.25, colors.gray),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
    ]))
    story.append(tbl)

    doc.build(story)
    return out_pdf


def export_run_zip(project: str, run_dir: Path, out_zip: Path) -> Path:
    """
    Bundle a complete run folder into a portable ZIP:
      - inputs/, normalized/ (if present), plan/, logs/, preferences.json (if present), README.md (generated)
    """
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # Synthesize a small README with pointers
    plan_path = run_dir / "plan" / "plan.json"
    kpis_path = run_dir / "plan" / "kpis.json"
    readme = [
        f"# AI-PM Run Package — {project}",
        "",
        f"- Exported: {_friendly_ts()}",
        f"- Run folder: {run_dir.name}",
        "",
        "## Contents",
        "- `inputs/` original uploads & configs",
        "- `normalized/` cleaned artifacts (if present)",
        "- `plan/` plan.json + kpis.json",
        "- `logs/` UI/solver logs",
        "- `preferences.json` (if present)",
        "",
        "## Quick start",
        "Open `plan/plan.json` to inspect assignments. CSV in `plan/assignments.csv` if present.",
    ]
    (run_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    # Also emit a CSV beside the plan if present
    try:
        if plan_path.exists():
            plan = _read_json(plan_path)
            export_assignments_csv(plan, run_dir / "plan" / "assignments.csv")
    except Exception:
        # non-fatal
        pass

    # Build the ZIP
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def _add_rel(p: Path):
            arc = p.relative_to(run_dir)
            zf.write(p, arcname=str(arc))

        # add directories if present
        for sub in ["inputs", "normalized", "plan", "logs"]:
            d = run_dir / sub
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file():
                        _add_rel(p)

        # add root files we care about
        for root_file in ["README.md", "preferences.json"]:
            p = run_dir / root_file
            if p.exists():
                _add_rel(p)

    return out_zip


# ----------------------------
# CLI entrypoint
# ----------------------------

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser("AI-PM exporters")
    ap.add_argument("--plan", type=Path, help="path to plan.json")
    ap.add_argument("--kpis", type=Path, help="path to kpis.json (optional)")
    ap.add_argument("--csv", type=Path, help="path to write assignments.csv")
    ap.add_argument("--pdf", type=Path, help="path to write summary.pdf")
    ap.add_argument("--json", type=Path, help="path to write plan copy.json")

    ap.add_argument("--export-zip", action="store_true", help="export a full run package")
    ap.add_argument("--project", type=str, help="project name (for ZIP readme)")
    ap.add_argument("--run-dir", type=Path, help="runs/<Project>/<timestamp> directory")
    ap.add_argument("--out", type=Path, help="output zip path (for --export-zip)")

    args = ap.parse_args()

    if args.export_zip:
        if not (args.project and args.run_dir and args.out):
            print("[ERROR] --export-zip requires --project, --run-dir, --out")
            return 2
        out = export_run_zip(args.project, args.run_dir, args.out)
        print(f"[OK] ZIP written -> {out}")
        return 0

    # Non-zip exports require a plan
    if not args.plan:
        print("[ERROR] Provide --plan (and optional --kpis/--csv/--pdf/--json).")
        return 2

    plan = _read_json(args.plan)
    kpis = _read_json(args.kpis) if args.kpis and args.kpis.exists() else None

    if args.csv:
        p = export_assignments_csv(plan, args.csv)
        print(f"[OK] CSV -> {p}")
    if args.json:
        p = export_plan_json(plan, args.json)
        print(f"[OK] JSON -> {p}")
    if args.pdf:
        p = export_pdf_summary(plan, kpis, args.pdf)
        print(f"[OK] PDF -> {p}")

    if not (args.csv or args.pdf or args.json):
        print("[INFO] Nothing to do (no --csv/--pdf/--json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
