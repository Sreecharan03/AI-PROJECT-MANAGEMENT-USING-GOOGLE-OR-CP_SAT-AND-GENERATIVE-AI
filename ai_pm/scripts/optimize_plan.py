# ai_pm/scripts/optimize_plan.py
# Phase 4 — headless runner: optimize → plan.json + kpis.json (and snapshot inputs)
#
# Usage example (from repo root):
#   python -m scripts.optimize_plan \
#     --project "Demo" \
#     --team samples/team.csv \
#     --tasks /tmp/task_graph.normalized.json \
#     --settings config/settings.yaml \
#     --history samples/history.csv \
#     --theta 0.5 0.2 0.2 0.1 \
#     --horizon-weeks 4
#
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

from core.optimizer import solve_plan
from core.kpis import compute_kpis
from core.storage import new_run_dir, snapshot_inputs

def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entrypoint. Creates a fresh run folder and writes:
      runs/<project>/<timestamp>/
        plan/plan.json
        plan/kpis.json
        inputs/ (snapshots of team, tasks, settings, history)
        logs/hashes.json
    """
    ap = argparse.ArgumentParser(description="CP-SAT optimizer → plan.json + KPIs, with input snapshotting.")
    ap.add_argument("--project", required=True, help="Project name for runs/<project>/<timestamp>/")
    ap.add_argument("--team", type=Path, required=True, help="Path to team.csv")
    ap.add_argument("--tasks", type=Path, required=True, help="Path to normalized task_graph.json (Phase 3 output)")
    ap.add_argument("--settings", type=Path, default=Path("config/settings.yaml"), help="settings.yaml path")
    ap.add_argument("--history", type=Path, help="Optional history.csv path")
    ap.add_argument("--theta", type=float, nargs=4, metavar=("SKILL","FAIR","CONT","DEADL"),
                    help="Weights for (skill_fit, fairness, continuity, deadline_risk)")
    ap.add_argument("--horizon-weeks", type=int, help="Capacity horizon in weeks (if omitted, auto-computed)")
    args = ap.parse_args(argv)

    # 1) Solve plan (assignment + greedy schedule)
    theta = None
    if args.theta:
        theta = {"skill_fit": args.theta[0], "fairness": args.theta[1], "continuity": args.theta[2], "deadline_risk": args.theta[3]}

    try:
        plan = solve_plan(args.team, args.tasks, args.settings,
                          history_csv=args.history, theta=theta, horizon_weeks=args.horizon_weeks)
    except Exception as e:
        print(f"[ERROR] Optimizer failed: {e}", file=sys.stderr)
        return 2

    # 2) Create a new run directory and persist artifacts
    run_dir = Path(new_run_dir(args.project))
    plan_dir = run_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    plan_path = plan_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) KPIs
    horizon = args.horizon_weeks or 4  # for utilization checks if user didn't specify
    kpis = compute_kpis(args.team, args.tasks, plan_path, horizon_weeks=horizon)
    kpis_path = plan_dir / "kpis.json"
    kpis_path.write_text(json.dumps(kpis, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Snapshot inputs for provenance
    srcs = [args.team.as_posix(), args.tasks.as_posix(), args.settings.as_posix()]
    if args.history and args.history.exists():
        srcs.append(args.history.as_posix())
    try:
        snapshot_inputs(run_dir, srcs)
    except Exception as e:
        # Not fatal; the plan is already written
        print(f"[WARN] Snapshot failed: {e}", file=sys.stderr)

    # 5) Report
    print(f"[OK] Run folder: {run_dir}")
    print(f"[OK] Plan:        {plan_path}")
    print(f"[OK] KPIs:        {kpis_path}")
    print(f"[OK] Inputs:      {run_dir}/inputs")
    print(f"[OK] Hashes log:  {run_dir}/logs/hashes.json")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
