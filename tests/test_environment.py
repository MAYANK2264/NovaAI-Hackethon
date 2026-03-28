"""
Test suite — validates OpenEnv spec compliance and grader logic.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import SupplyChainEnv, TASK_CONFIGS
from env.models import Action, ReallocationAction, SplitOrderAction
from graders.graders import grade, GRADERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(TASK_CONFIGS.keys()))
def task_id(request):
    return request.param


@pytest.fixture
def easy_env():
    return SupplyChainEnv("task_single_supplier_failure")

@pytest.fixture
def medium_env():
    return SupplyChainEnv("task_port_congestion_cascade")

@pytest.fixture
def hard_env():
    return SupplyChainEnv("task_multi_shock_crisis")


# ---------------------------------------------------------------------------
# Spec compliance
# ---------------------------------------------------------------------------

class TestOpenEnvSpec:

    def test_reset_returns_observation(self, task_id):
        env = SupplyChainEnv(task_id)
        result = env.reset()
        assert result.observation is not None
        assert result.observation.step == 0
        assert result.observation.task_id == task_id

    def test_reset_clean_state(self, task_id):
        env = SupplyChainEnv(task_id)
        env.reset()
        env.reset()   # Second reset should produce clean state
        obs = env.state()
        assert obs.step == 0

    def test_state_returns_observation(self, task_id):
        env = SupplyChainEnv(task_id)
        assert env.state() is None   # Before reset
        env.reset()
        assert env.state() is not None

    def test_step_returns_result(self, task_id):
        env = SupplyChainEnv(task_id)
        env.reset()
        action = Action()
        result = env.step(action)
        assert result.observation is not None
        assert 0.0 <= result.reward.total <= 1.0
        assert isinstance(result.done, bool)

    def test_step_before_reset_raises(self, task_id):
        env = SupplyChainEnv(task_id)
        with pytest.raises(RuntimeError):
            env.step(Action())

    def test_observation_has_required_fields(self, task_id):
        env = SupplyChainEnv(task_id)
        result = env.reset()
        obs = result.observation
        assert len(obs.disruptions) > 0
        assert len(obs.pending_orders) > 0
        assert len(obs.suppliers) > 0
        assert len(obs.inventory) > 0
        assert obs.budget_remaining > 0
        assert obs.task_description != ""

    def test_disrupted_suppliers_flagged(self, task_id):
        env = SupplyChainEnv(task_id)
        result = env.reset()
        disrupted_ids = {sid for d in result.observation.disruptions for sid in d.affected_supplier_ids}
        for sup in result.observation.suppliers:
            if sup.supplier_id in disrupted_ids:
                assert sup.is_disrupted, f"{sup.supplier_id} should be flagged disrupted"

    def test_reward_range(self, task_id):
        env = SupplyChainEnv(task_id)
        env.reset()
        for _ in range(3):
            result = env.step(Action())
            assert 0.0 <= result.reward.total <= 1.0
            assert 0.0 <= result.reward.breakdown.stockout_avoidance <= 1.0
            assert 0.0 <= result.reward.breakdown.cost_efficiency <= 1.0
            assert 0.0 <= result.reward.breakdown.lead_time_score <= 1.0
            assert 0.0 <= result.reward.breakdown.budget_adherence <= 1.0

    def test_episode_terminates(self, task_id):
        env = SupplyChainEnv(task_id)
        env.reset()
        cfg = TASK_CONFIGS[task_id]
        done = False
        for _ in range(cfg["max_steps"] + 2):
            result = env.step(Action())
            if result.done:
                done = True
                break
        assert done, "Episode should terminate within max_steps + 2"


# ---------------------------------------------------------------------------
# Action processing
# ---------------------------------------------------------------------------

class TestActionProcessing:

    def test_valid_reallocation(self, easy_env):
        result = easy_env.reset()
        obs = result.observation

        # Find an at-risk order and a valid alternate supplier
        disrupted_ids = {sid for d in obs.disruptions for sid in d.affected_supplier_ids}
        at_risk = [o for o in obs.pending_orders if o.original_supplier_id in disrupted_ids]
        assert at_risk, "Should have at-risk orders"

        order = at_risk[0]
        alt_supplier = next(
            (s for s in obs.suppliers
             if not s.is_disrupted and order.sku in s.available_skus),
            None
        )
        if alt_supplier is None:
            pytest.skip("No alternate supplier for this SKU in test scenario")

        action = Action(
            reallocations=[
                ReallocationAction(
                    order_id=order.order_id,
                    new_supplier_id=alt_supplier.supplier_id,
                    quantity=order.quantity,
                    priority=order.priority,
                )
            ]
        )
        result = easy_env.step(action)
        assert result.reward.total > 0.0, "Valid reallocation should earn positive reward"

    def test_allocating_to_disrupted_supplier_penalized(self, easy_env):
        result = easy_env.reset()
        obs = result.observation

        disrupted_ids = {sid for d in obs.disruptions for sid in d.affected_supplier_ids}
        at_risk = [o for o in obs.pending_orders if o.original_supplier_id in disrupted_ids]
        if not at_risk:
            pytest.skip()

        order = at_risk[0]
        action = Action(
            reallocations=[
                ReallocationAction(
                    order_id=order.order_id,
                    new_supplier_id=list(disrupted_ids)[0],
                    quantity=order.quantity,
                )
            ]
        )
        result = easy_env.step(action)
        assert result.reward.penalties, "Should have penalties for allocating to disrupted supplier"

    def test_cancel_order(self, easy_env):
        result = easy_env.reset()
        obs = result.observation
        order_id = obs.pending_orders[0].order_id
        action = Action(cancel_orders=[order_id])
        result = easy_env.step(action)
        cancelled = next((o for o in result.observation.pending_orders if o.order_id == order_id), None)
        assert cancelled is not None
        assert cancelled.status == "cancelled"

    def test_noop_action(self, task_id):
        env = SupplyChainEnv(task_id)
        env.reset()
        result = env.step(Action())
        assert result.observation is not None
        assert 0.0 <= result.reward.total <= 1.0


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

class TestGraders:

    def test_all_tasks_have_grader(self):
        for task_id in TASK_CONFIGS:
            assert task_id in GRADERS

    def test_grader_score_range(self, task_id):
        env = SupplyChainEnv(task_id)
        env.reset()
        # Run to completion with noop
        obs = None
        for _ in range(TASK_CONFIGS[task_id]["max_steps"]):
            result = env.step(Action())
            obs = result.observation
            if result.done:
                break
        assert obs is not None
        grade_result = grade(task_id, obs)
        assert 0.0 <= grade_result.final_score <= 1.0

    def test_grader_with_good_actions(self):
        """An agent that correctly resolves orders should score higher than noop."""
        env = SupplyChainEnv("task_single_supplier_failure")
        result = env.reset()
        obs = result.observation

        disrupted_ids = {sid for d in obs.disruptions for sid in d.affected_supplier_ids}
        at_risk = [o for o in obs.pending_orders if o.original_supplier_id in disrupted_ids]

        reallocations = []
        for order in at_risk:
            alt = next(
                (s for s in obs.suppliers
                 if not s.is_disrupted and order.sku in s.available_skus),
                None
            )
            if alt:
                reallocations.append(ReallocationAction(
                    order_id=order.order_id,
                    new_supplier_id=alt.supplier_id,
                    quantity=order.quantity,
                ))

        if reallocations:
            action = Action(reallocations=reallocations)
            step_result = env.step(action)
            good_grade = grade("task_single_supplier_failure", step_result.observation)

            # Compare against noop baseline
            env2 = SupplyChainEnv("task_single_supplier_failure")
            env2.reset()
            noop_result = env2.step(Action())
            noop_grade = grade("task_single_supplier_failure", noop_result.observation)

            assert good_grade.final_score >= noop_grade.final_score, \
                "Resolving orders should score >= noop"

    def test_grader_deterministic(self, task_id):
        """Same episode + same actions = same grade."""
        scores = []
        for _ in range(2):
            env = SupplyChainEnv(task_id)
            env.reset()
            for _ in range(3):
                result = env.step(Action())
                if result.done:
                    break
            g = grade(task_id, result.observation)
            scores.append(g.final_score)
        assert scores[0] == scores[1], "Grader must be deterministic"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_orders(self, task_id):
        env1 = SupplyChainEnv(task_id)
        env2 = SupplyChainEnv(task_id)
        obs1 = env1.reset().observation
        obs2 = env2.reset().observation
        ids1 = sorted(o.order_id for o in obs1.pending_orders)
        ids2 = sorted(o.order_id for o in obs2.pending_orders)
        assert ids1 == ids2

    def test_budget_decreases_on_costly_reallocation(self):
        env = SupplyChainEnv("task_single_supplier_failure")
        result = env.reset()
        obs = result.observation
        initial_budget = obs.budget_remaining

        disrupted_ids = {sid for d in obs.disruptions for sid in d.affected_supplier_ids}
        at_risk = [o for o in obs.pending_orders if o.original_supplier_id in disrupted_ids]
        if not at_risk:
            return

        order = at_risk[0]
        # Find a MORE expensive supplier
        alt = next(
            (s for s in sorted(obs.suppliers, key=lambda x: -x.cost_per_unit)
             if not s.is_disrupted and order.sku in s.available_skus),
            None
        )
        if alt and alt.cost_per_unit > order.unit_cost:
            action = Action(reallocations=[
                ReallocationAction(
                    order_id=order.order_id,
                    new_supplier_id=alt.supplier_id,
                    quantity=order.quantity,
                )
            ])
            result = env.step(action)
            assert result.observation.budget_remaining < initial_budget
