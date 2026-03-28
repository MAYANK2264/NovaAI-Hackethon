"""
Deterministic graders for all three supply chain tasks.
Each grader returns a score in [0.0, 1.0] with a detailed breakdown.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from env.models import Observation, PurchaseOrder, Supplier


@dataclass
class GradeResult:
    task_id: str
    final_score: float           # 0.0 – 1.0
    passed: bool                 # True if score >= 0.6
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _disrupted_ids(obs: Observation) -> List[str]:
    return [sid for d in obs.disruptions for sid in d.affected_supplier_ids]


def _supplier_map(obs: Observation) -> Dict[str, Supplier]:
    return {s.supplier_id: s for s in obs.suppliers}


def _at_risk_orders(obs: Observation) -> List[PurchaseOrder]:
    dids = set(_disrupted_ids(obs))
    return [o for o in obs.pending_orders if o.original_supplier_id in dids]


def _resolved_orders(obs: Observation) -> List[PurchaseOrder]:
    dids = set(_disrupted_ids(obs))
    return [
        o for o in obs.pending_orders
        if o.original_supplier_id in dids
        and o.status == "allocated"
        and o.current_supplier_id not in dids
    ]


# ---------------------------------------------------------------------------
# Task 1 grader — single supplier failure (easy)
# ---------------------------------------------------------------------------

def grade_single_supplier_failure(final_obs: Observation) -> GradeResult:
    """
    Scoring:
    - Resolution rate: 50%  (at-risk orders successfully re-routed)
    - Budget compliance: 20% (stayed within $50k)
    - Lead time compliance: 20% (allocated supplier lead_time <= required_by_day)
    - No disrupted supplier used: 10%
    """
    at_risk = _at_risk_orders(final_obs)
    resolved = _resolved_orders(final_obs)
    sup_map = _supplier_map(final_obs)
    dids = set(_disrupted_ids(final_obs))
    notes = []

    # Resolution rate
    resolution = len(resolved) / max(1, len(at_risk))
    notes.append(f"Resolved {len(resolved)}/{len(at_risk)} at-risk orders.")

    # Budget compliance
    budget_ok = 1.0 if final_obs.budget_remaining >= 0 else max(0.0, 1 + final_obs.budget_remaining / final_obs.total_budget)
    notes.append(f"Budget remaining: ${final_obs.budget_remaining:,.0f}")

    # Lead time compliance
    lt_checks = [
        o for o in resolved
        if sup_map.get(o.current_supplier_id) and
           sup_map[o.current_supplier_id].lead_time_days <= o.required_by_day
    ]
    lt_score = len(lt_checks) / max(1, len(resolved))
    notes.append(f"Lead time compliant: {len(lt_checks)}/{len(resolved)}")

    # No disrupted supplier penalty
    bad_alloc = [o for o in resolved if o.current_supplier_id in dids]
    no_bad = 1.0 if not bad_alloc else max(0.0, 1.0 - len(bad_alloc) * 0.25)

    score = (
        resolution * 0.50
        + budget_ok * 0.20
        + lt_score * 0.20
        + no_bad * 0.10
    )

    return GradeResult(
        task_id="task_single_supplier_failure",
        final_score=round(min(1.0, max(0.0, score)), 4),
        passed=score >= 0.6,
        components={
            "resolution_rate": round(resolution, 4),
            "budget_compliance": round(budget_ok, 4),
            "lead_time_compliance": round(lt_score, 4),
            "no_disrupted_supplier": round(no_bad, 4),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Task 2 grader — port congestion cascade (medium)
# ---------------------------------------------------------------------------

def grade_port_congestion_cascade(final_obs: Observation) -> GradeResult:
    """
    Scoring:
    - Urgent order resolution: 35%
    - Overall resolution rate: 25%
    - Cost efficiency (vs baseline): 20%
    - Split orders used where appropriate: 10%
    - Budget compliance: 10%
    """
    at_risk = _at_risk_orders(final_obs)
    resolved = _resolved_orders(final_obs)
    sup_map = _supplier_map(final_obs)
    dids = set(_disrupted_ids(final_obs))
    notes = []

    # Urgent resolution
    urgent_at_risk = [o for o in at_risk if o.priority == "urgent"]
    urgent_resolved = [o for o in resolved if o.priority == "urgent"]
    urgent_score = len(urgent_resolved) / max(1, len(urgent_at_risk))
    notes.append(f"Urgent: {len(urgent_resolved)}/{len(urgent_at_risk)} resolved")

    # Overall resolution
    overall = len(resolved) / max(1, len(at_risk))
    notes.append(f"Overall: {len(resolved)}/{len(at_risk)} resolved")

    # Cost efficiency
    baseline = sum(o.unit_cost * o.quantity for o in at_risk)
    current = sum(o.unit_cost * o.quantity for o in resolved)
    if baseline > 0 and current > 0:
        ratio = current / baseline
        cost_score = 1.0 if ratio <= 1.15 else max(0.0, 1.0 - (ratio - 1.15) * 2.5)
        notes.append(f"Cost ratio: {ratio:.2f}x baseline")
    else:
        cost_score = 0.5
        notes.append("No baseline cost")

    # Split orders — were large orders (qty > 200) split?
    large_at_risk = [o for o in at_risk if o.quantity > 200]
    # Split orders show up as SPLIT in their IDs
    split_created = [o for o in final_obs.pending_orders if "SPLIT" in o.order_id]
    split_score = min(1.0, len(split_created) / max(1, len(large_at_risk)))
    notes.append(f"Splits: {len(split_created)} split segments from {len(large_at_risk)} large orders")

    # Budget
    budget_ok = 1.0 if final_obs.budget_remaining >= 0 else max(0.0, 1 + final_obs.budget_remaining / final_obs.total_budget)

    score = (
        urgent_score * 0.35
        + overall * 0.25
        + cost_score * 0.20
        + split_score * 0.10
        + budget_ok * 0.10
    )

    return GradeResult(
        task_id="task_port_congestion_cascade",
        final_score=round(min(1.0, max(0.0, score)), 4),
        passed=score >= 0.55,
        components={
            "urgent_resolution": round(urgent_score, 4),
            "overall_resolution": round(overall, 4),
            "cost_efficiency": round(cost_score, 4),
            "split_usage": round(split_score, 4),
            "budget_compliance": round(budget_ok, 4),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Task 3 grader — multi-shock crisis (hard)
# ---------------------------------------------------------------------------

def grade_multi_shock_crisis(final_obs: Observation) -> GradeResult:
    """
    Scoring:
    - Stockout prevention (SKU-level): 30%
    - Urgent + critical order resolution: 25%
    - Overall resolution rate: 20%
    - Budget hard constraint (0 if over budget): 15%
    - Lead time hard constraint (max 14 days): 10%
    """
    at_risk = _at_risk_orders(final_obs)
    resolved = _resolved_orders(final_obs)
    sup_map = _supplier_map(final_obs)
    dids = set(_disrupted_ids(final_obs))
    notes = []

    # Stockout prevention
    risk_skus = set(final_obs.stockout_risk_skus)
    covered_skus = set(
        o.sku for o in resolved
        if o.sku in risk_skus
    )
    stockout_score = len(covered_skus) / max(1, len(risk_skus)) if risk_skus else 1.0
    notes.append(f"Stockout SKUs covered: {len(covered_skus)}/{len(risk_skus)}")

    # Urgent + critical resolution (priority=urgent orders)
    critical_at_risk = [o for o in at_risk if o.priority == "urgent"]
    critical_resolved = [o for o in resolved if o.priority == "urgent"]
    critical_score = len(critical_resolved) / max(1, len(critical_at_risk))
    notes.append(f"Critical/urgent: {len(critical_resolved)}/{len(critical_at_risk)}")

    # Overall resolution
    overall = len(resolved) / max(1, len(at_risk))
    notes.append(f"Overall: {len(resolved)}/{len(at_risk)}")

    # Budget — hard constraint
    if final_obs.budget_remaining >= 0:
        budget_score = 1.0
    else:
        budget_score = 0.0   # Hard failure for over-budget
        notes.append("HARD FAIL: Over budget.")

    # Lead time constraint — max 14 days
    over_lt = [
        o for o in resolved
        if sup_map.get(o.current_supplier_id) and
           sup_map[o.current_supplier_id].lead_time_days > 14
    ]
    lt_score = 1.0 - (len(over_lt) / max(1, len(resolved)))
    if over_lt:
        notes.append(f"Lead time violations (>14d): {len(over_lt)}")

    score = (
        stockout_score * 0.30
        + critical_score * 0.25
        + overall * 0.20
        + budget_score * 0.15
        + lt_score * 0.10
    )

    return GradeResult(
        task_id="task_multi_shock_crisis",
        final_score=round(min(1.0, max(0.0, score)), 4),
        passed=score >= 0.45,  # Hard task — lower passing bar
        components={
            "stockout_prevention": round(stockout_score, 4),
            "critical_resolution": round(critical_score, 4),
            "overall_resolution": round(overall, 4),
            "budget_constraint": round(budget_score, 4),
            "lead_time_constraint": round(lt_score, 4),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

GRADERS = {
    "task_single_supplier_failure": grade_single_supplier_failure,
    "task_port_congestion_cascade": grade_port_congestion_cascade,
    "task_multi_shock_crisis": grade_multi_shock_crisis,
}


def grade(task_id: str, final_obs: Observation) -> GradeResult:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task_id: {task_id}")
    return GRADERS[task_id](final_obs)
