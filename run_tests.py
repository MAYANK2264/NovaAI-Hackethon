"""
Self-contained test runner — no pytest, no pydantic, no network.
Validates: reset(), step(), state(), graders, reward range, determinism.
"""
from __future__ import annotations
import sys, os, copy, random
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch env.models to use stdlib dataclass version
import env.models_compat as _m
import env.models as _real
# replace attributes in the real module namespace
for attr in dir(_m):
    if not attr.startswith('__'):
        setattr(_real, attr, getattr(_m, attr))

from env.environment import SupplyChainEnv, TASK_CONFIGS
from graders.graders import grade, GRADERS

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []

def check(name, condition, detail=""):
    icon = PASS if condition else FAIL
    msg = f"  {icon} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append(condition)
    return condition

print("\n══════════════════════════════════════════")
print("  Supply Chain OpenEnv — Test Suite")
print("══════════════════════════════════════════\n")

for task_id in TASK_CONFIGS:
    print(f"── Task: {task_id}")

    env = SupplyChainEnv(task_id)
    cfg = TASK_CONFIGS[task_id]

    # 1. state() before reset returns None
    check("state() before reset returns None", env.state() is None)

    # 2. step() before reset raises
    try:
        env.step(_real.Action())
        check("step() before reset raises RuntimeError", False)
    except RuntimeError:
        check("step() before reset raises RuntimeError", True)

    # 3. reset() returns clean Observation at step=0
    r = env.reset()
    obs = r.observation
    check("reset() returns Observation", obs is not None)
    check("step==0 after reset", obs.step == 0)
    check("task_id matches", obs.task_id == task_id)
    check("has disruptions", len(obs.disruptions) > 0)
    check("has pending_orders", len(obs.pending_orders) > 0)
    check("has suppliers", len(obs.suppliers) > 0)
    check("has inventory", len(obs.inventory) > 0)
    check("budget_remaining > 0", obs.budget_remaining > 0)
    check("task_description not empty", obs.task_description != "")

    # 4. disrupted suppliers are flagged
    disrupted_ids = {sid for d in obs.disruptions for sid in d.affected_supplier_ids}
    flagged_correctly = all(
        s.is_disrupted for s in obs.suppliers if s.supplier_id in disrupted_ids
    )
    check("disrupted suppliers flagged", flagged_correctly)

    # 5. state() matches last obs
    s = env.state()
    check("state() returns Observation after reset", s is not None)
    check("state().step == 0", s.step == 0)

    # 6. noop step — reward in [0,1]
    sr = env.step(_real.Action())
    r2 = sr.reward
    check("noop reward in [0,1]", 0.0 <= r2.total <= 1.0,
          f"total={r2.total:.4f}")
    check("breakdown components in [0,1]",
          all(0.0 <= v <= 1.0 for v in [
              r2.breakdown.stockout_avoidance,
              r2.breakdown.cost_efficiency,
              r2.breakdown.lead_time_score,
              r2.breakdown.budget_adherence,
          ]))
    check("step incremented", sr.observation.step == 1)

    # 7. smart reallocation earns >= noop reward
    env2 = SupplyChainEnv(task_id)
    obs2 = env2.reset().observation
    at_risk = [o for o in obs2.pending_orders if o.original_supplier_id in disrupted_ids]
    reallocations = []
    for order in at_risk:
        alt = next(
            (s for s in sorted(obs2.suppliers, key=lambda x: x.cost_per_unit)
             if not s.is_disrupted and order.sku in s.available_skus),
            None
        )
        if alt:
            reallocations.append(_real.ReallocationAction(
                order_id=order.order_id,
                new_supplier_id=alt.supplier_id,
                quantity=order.quantity,
                priority=order.priority,
            ))
    smart_action = _real.Action(reallocations=reallocations, reasoning="Cheapest available alt supplier")
    smart_result = env2.step(smart_action)
    check("smart action reward >= noop",
          smart_result.reward.total >= r2.total,
          f"smart={smart_result.reward.total:.4f} noop={r2.total:.4f}")

    # 8. Allocating to disrupted supplier adds penalty
    env3 = SupplyChainEnv(task_id)
    obs3 = env3.reset().observation
    at_risk3 = [o for o in obs3.pending_orders if o.original_supplier_id in disrupted_ids]
    if at_risk3 and disrupted_ids:
        bad_action = _real.Action(
            reallocations=[_real.ReallocationAction(
                order_id=at_risk3[0].order_id,
                new_supplier_id=list(disrupted_ids)[0],
                quantity=at_risk3[0].quantity,
            )]
        )
        bad_result = env3.step(bad_action)
        check("bad alloc generates penalty",
              len(bad_result.reward.penalties) > 0,
              f"penalties={bad_result.reward.penalties}")
    else:
        check("bad alloc generates penalty", True, "skipped — no at-risk orders")

    # 9. cancellation sets order status
    env4 = SupplyChainEnv(task_id)
    obs4 = env4.reset().observation
    target_id = obs4.pending_orders[0].order_id
    cancel_result = env4.step(_real.Action(cancel_orders=[target_id]))
    cancelled = next((o for o in cancel_result.observation.pending_orders if o.order_id == target_id), None)
    check("cancel_orders sets status=cancelled",
          cancelled is not None and cancelled.status == "cancelled")

    # 10. Episode terminates within max_steps
    env5 = SupplyChainEnv(task_id)
    env5.reset()
    done_seen = False
    for _ in range(cfg["max_steps"] + 2):
        sr5 = env5.step(_real.Action())
        if sr5.done:
            done_seen = True
            break
    check(f"episode terminates within max_steps({cfg['max_steps']})", done_seen)

    # 11. Determinism — same seed same orders
    envA = SupplyChainEnv(task_id)
    envB = SupplyChainEnv(task_id)
    idsA = sorted(o.order_id for o in envA.reset().observation.pending_orders)
    idsB = sorted(o.order_id for o in envB.reset().observation.pending_orders)
    check("deterministic: same seed → same order IDs", idsA == idsB)

    # 12. Double reset produces clean state
    env6 = SupplyChainEnv(task_id)
    env6.reset()
    env6.step(_real.Action())
    env6.reset()
    check("double reset → step==0", env6.state().step == 0)

    # ── Grader ──
    print(f"\n  Grader: {task_id}")

    check("grader exists", task_id in GRADERS)

    env7 = SupplyChainEnv(task_id)
    env7.reset()
    last_sr = None
    for _ in range(cfg["max_steps"]):
        last_sr = env7.step(_real.Action())
        if last_sr.done:
            break
    grade_result = grade(task_id, last_sr.observation)
    check("grader score in [0,1]",
          0.0 <= grade_result.final_score <= 1.0,
          f"score={grade_result.final_score:.4f}")
    check("grader has components", len(grade_result.components) >= 3)
    check("grader has notes", len(grade_result.notes) >= 1)
    check("grader.passed is bool", isinstance(grade_result.passed, bool))

    # Grader determinism
    g1 = grade(task_id, last_sr.observation)
    g2 = grade(task_id, last_sr.observation)
    check("grader is deterministic", g1.final_score == g2.final_score)

    # Smart vs noop grader comparison
    env8 = SupplyChainEnv(task_id)
    obs8 = env8.reset().observation
    smart_relos = []
    dis8 = {sid for d in obs8.disruptions for sid in d.affected_supplier_ids}
    for order in [o for o in obs8.pending_orders if o.original_supplier_id in dis8]:
        alt = next(
            (s for s in sorted(obs8.suppliers, key=lambda x: x.cost_per_unit)
             if not s.is_disrupted and order.sku in s.available_skus), None
        )
        if alt:
            smart_relos.append(_real.ReallocationAction(
                order_id=order.order_id,
                new_supplier_id=alt.supplier_id,
                quantity=order.quantity,
            ))
    if smart_relos:
        smart_sr8 = env8.step(_real.Action(reallocations=smart_relos))
        smart_g = grade(task_id, smart_sr8.observation)

        env9 = SupplyChainEnv(task_id)
        env9.reset()
        noop_sr9 = env9.step(_real.Action())
        noop_g = grade(task_id, noop_sr9.observation)

        check("smart grade >= noop grade",
              smart_g.final_score >= noop_g.final_score,
              f"smart={smart_g.final_score:.4f} noop={noop_g.final_score:.4f}")

    print()

# ── Summary ──
total = len(results)
passed = sum(results)
failed = total - passed
print("══════════════════════════════════════════")
print(f"  Results: {passed}/{total} passed  |  {failed} failed")
print("══════════════════════════════════════════")
if failed:
    sys.exit(1)
