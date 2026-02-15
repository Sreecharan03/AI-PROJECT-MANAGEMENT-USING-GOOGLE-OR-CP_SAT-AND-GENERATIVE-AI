# ai_pm/scripts/update_theta.py
"""
Phase 6 — Coactive Learning (θ updates that persist)

WHAT:
  - Compare Scenario B (preferred) against Scenario A (rejected) using real plan features.
  - Compute per-feature improvements and nudge θ toward features where B dominates.
  - Persist updated preferences JSON.

FEATURES:
  * skill_fit (↑ better)     — average of per-task skill_match from core.explain
  * fairness (↓ better)      — utilization_stddev from core.kpis (lower is better)
  * continuity (↑ better)    — average of per-task continuity from core.explain
  * deadline_risk (↓ better) — avg lateness hours = avg(max(-slack_hours,0)) from core.explain

USAGE:
  python -m scripts.update_theta \
    --team samples/team.csv \
    --tasks scenarios/task_graph.reopt.json \
    --preferred runs/Demo/<ts_B>/plan/plan.json \
    --rejected  runs/Demo/<ts_A>/plan/plan.json \
    --prefs-out runs/Demo/preferences.json \
    [--prefs-in runs/Demo/preferences.json] \
    [--eta 0.1] \
    [--history samples/history.csv] \
    [--horizon-weeks 4]

OUTPUT:
  Writes JSON with:
    {
      "theta": {"skill_fit":..., "fairness":..., "continuity":..., "deadline_risk":...},
      "updated_at": "...",
      "last_comparison": { "eta":..., "deltas": {...}, "metrics_A": {...}, "metrics_B": {...} }
    }

NOTE:
  Keep code minimal; no external deps beyond existing core modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Core modules (already in your repo)
from core.explain import explain_plan
from core.kpis import compute_kpis


DEFAULT_THETA = {"skill_fit": 0.5, "fairness": 0.2, "continuity": 0.2, "deadline_risk": 0.1}


def _load_prefs(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {"theta": dict(DEFAULT_THETA)}
    if not path.exists():
        return {"theta": dict(DEFAULT_THETA)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        theta = data.get("theta") or {}
        # Ensure all 4 keys exist; fill from defaults if missing
        for k, v in DEFAULT_THETA.items():
            theta.setdefault(k, v)
        data["theta"] = theta
        return data
    except Exception:
        return {"theta": dict(DEFAULT_THETA)}


def _norm_theta(theta: Dict[str, float]) -> Dict[str, float]:
    # Clip [0,1], then renormalize to sum to 1 (if total > 0).
    clipped = {k: min(1.0, max(0.0, float(v))) for k, v in theta.items()}
    s = sum(clipped.values())
    if s <= 1e-12:
        return dict(DEFAULT_THETA)
    return {k: v / s for k, v in clipped.items()}


def _aggregate_explain(team_csv: Path, tasks_json: Path, plan_json: Path, history_csv: Optional[Path], horizon_weeks: int) -> Dict[str, float]:
    """
    Use core.explain to get per-task metrics; return aggregated features:
      - skill_fit_avg
      - continuity_avg
      - deadline_risk_avg_lateness (avg of max(-slack,0))
    """
    res = explain_plan(team_csv, tasks_json, plan_json, history_csv=history_csv, horizon_weeks=float(horizon_weeks))
    rats = (res or {}).get("rationales") or {}
    if not rats:
        return {"skill_fit_avg": 0.0, "continuity_avg": 0.0, "deadline_risk_avg_lateness": 0.0}

    n = 0
    skill_sum = 0.0
    cont_sum = 0.0
    lateness_sum = 0.0
    for r in rats.values():
        n += 1
        skill_sum += float(r.get("skill_match") or 0.0)
        cont_sum += float(r.get("continuity") or 0.0)
        slack = r.get("slack_hours")
        # lateness is positive only if slack < 0
        if slack is None:
            # treat as no lateness
            pass
        else:
            try:
                s = float(slack)
                if s < 0:
                    lateness_sum += (-s)
            except Exception:
                pass
    if n == 0:
        return {"skill_fit_avg": 0.0, "continuity_avg": 0.0, "deadline_risk_avg_lateness": 0.0}
    return {
        "skill_fit_avg": skill_sum / n,
        "continuity_avg": cont_sum / n,
        "deadline_risk_avg_lateness": lateness_sum / n
    }


def _metrics_for_plan(team_csv: Path, tasks_json: Path, plan_json: Path, history_csv: Optional[Path], horizon_weeks: int) -> Dict[str, float]:
    """
    Collect comparable metrics for A vs B:
      - skill_fit_avg (↑)
      - continuity_avg (↑)
      - fairness_util_std (↓)    -> from core.kpis
      - deadline_risk_avg_late (↓) -> from core.explain slack
    """
    agg = _aggregate_explain(team_csv, tasks_json, plan_json, history_csv, horizon_weeks)
    # fairness via KPIs (utilization_stddev)
    kpis = compute_kpis(team_csv, tasks_json, plan_json, horizon_weeks=horizon_weeks) or {}
    fairness_util_std = float(kpis.get("utilization_stddev") or 0.0)
    return {
        "skill_fit_avg": float(agg["skill_fit_avg"]),
        "continuity_avg": float(agg["continuity_avg"]),
        "fairness_util_std": fairness_util_std,
        "deadline_risk_avg_late": float(agg["deadline_risk_avg_lateness"]),
    }


def _delta_vector(metrics_A: Dict[str, float], metrics_B: Dict[str, float]) -> Dict[str, float]:
    """
    Compute deltas in the direction of "B better than A".
    For ↑ features: diff = B - A
    For ↓ features: diff = A - B (since lower is better)
    """
    return {
        "skill_fit":         (metrics_B["skill_fit_avg"] - metrics_A["skill_fit_avg"]),
        "continuity":        (metrics_B["continuity_avg"] - metrics_A["continuity_avg"]),
        "fairness":          (metrics_A["fairness_util_std"] - metrics_B["fairness_util_std"]),  # lower better
        "deadline_risk":     (metrics_A["deadline_risk_avg_late"] - metrics_B["deadline_risk_avg_late"])  # lower better
    }


def _nudge_theta(theta: Dict[str, float], deltas: Dict[str, float], eta: float) -> Dict[str, float]:
    """
    Only reward positive improvements:
      theta_new = theta + eta * (d_pos / sum(d_pos))   if any d_pos>0
    Then clip and renormalize.
    """
    pos = {k: max(0.0, float(v)) for k, v in deltas.items()}
    s = sum(pos.values())
    if s > 1e-12:
        step = {k: (pos[k] / s) * eta for k in pos}
        updated = {k: float(theta.get(k, 0.0)) + step[k] for k in theta}
    else:
        updated = dict(theta)  # no improvement; leave as-is
    return _norm_theta(updated)


def _save_prefs(out_path: Path, theta: Dict[str, float], eta: float, metrics_A: Dict[str, float], metrics_B: Dict[str, float], deltas: Dict[str, float]) -> None:
    payload = {
        "theta": theta,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "last_comparison": {
            "eta": eta,
            "deltas": deltas,
            "metrics_A": metrics_A,
            "metrics_B": metrics_B
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Coactive update of θ using preferred (B) over rejected (A).")
    ap.add_argument("--team", required=True, help="team.csv")
    ap.add_argument("--tasks", required=True, help="task_graph.json (normalized or split)")
    ap.add_argument("--preferred", required=True, help="Scenario B plan.json (chosen)")
    ap.add_argument("--rejected", required=True, help="Scenario A plan.json (not chosen)")
    ap.add_argument("--prefs-in", default=None, help="existing preferences.json (optional)")
    ap.add_argument("--prefs-out", required=True, help="output preferences.json (will be created/overwritten)")
    ap.add_argument("--eta", type=float, default=0.1, help="learning rate (default: 0.1)")
    ap.add_argument("--history", default=None, help="history.csv (optional)")
    ap.add_argument("--horizon-weeks", type=int, default=4, help="horizon used for KPI comparability (default: 4)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    team = Path(args.team)
    tasks = Path(args.tasks)
    plan_B = Path(args.preferred)
    plan_A = Path(args.rejected)
    prefs_in = Path(args.prefs_in) if args.prefs_in else None
    prefs_out = Path(args.prefs_out)
    hist = Path(args.history) if args.history else None
    horizon = int(args.horizon_weeks)
    eta = float(args.eta)

    # 1) Load prior prefs (theta)
    prefs = _load_prefs(prefs_in)
    theta0 = _norm_theta(prefs.get("theta", DEFAULT_THETA))

    # 2) Compute metrics for A and B
    metrics_A = _metrics_for_plan(team, tasks, plan_A, hist, horizon)
    metrics_B = _metrics_for_plan(team, tasks, plan_B, hist, horizon)

    # 3) Deltas in "B better than A" direction
    deltas = _delta_vector(metrics_A, metrics_B)

    # 4) Nudge theta
    theta1 = _nudge_theta(theta0, deltas, eta)

    # 5) Persist
    _save_prefs(prefs_out, theta1, eta, metrics_A, metrics_B, deltas)

    # 6) Console summary
    print("[OK] Preferences updated.")
    print("Old theta:", theta0)
    print("New theta:", theta1)
    print("Deltas (B vs A):", deltas)
    print(f"Written -> {prefs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
