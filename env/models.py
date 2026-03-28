"""
Typed models for the Supply Chain Disruption Triage environment.
Uses stdlib dataclasses for zero-dependency portability.
Production deployment can swap to Pydantic BaseModel with no logic changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import copy


@dataclass
class Supplier:
    supplier_id: str
    name: str
    location: str
    capacity_per_day: int
    cost_per_unit: float
    lead_time_days: int
    reliability_score: float
    available_skus: List[str]
    is_disrupted: bool = False
    disruption_reason: Optional[str] = None

    def dict(self):
        return copy.copy(self.__dict__)


@dataclass
class PurchaseOrder:
    order_id: str
    sku: str
    quantity: int
    required_by_day: int
    original_supplier_id: str
    unit_cost: float
    current_supplier_id: Optional[str] = None
    status: str = "pending"   # pending | allocated | at_risk | cancelled | fulfilled
    priority: str = "normal"  # urgent | normal | deferrable

    def dict(self):
        return copy.copy(self.__dict__)


@dataclass
class DisruptionEvent:
    disruption_id: str
    event_type: str            # supplier_failure | port_delay | price_spike | shortage | bankruptcy
    affected_supplier_ids: List[str]
    affected_skus: List[str]
    severity: str              # low | medium | high | critical
    description: str
    day_occurred: int
    delay_days: int = 0
    price_multiplier: float = 1.0

    def dict(self):
        return copy.copy(self.__dict__)


@dataclass
class InventoryLevel:
    sku: str
    current_stock: int
    safety_stock: int
    reorder_point: int

    def dict(self):
        return copy.copy(self.__dict__)


@dataclass
class Observation:
    step: int
    task_id: str
    task_description: str
    disruptions: List[DisruptionEvent]
    pending_orders: List[PurchaseOrder]
    suppliers: List[Supplier]
    inventory: List[InventoryLevel]
    demand_forecast: Dict[str, List[int]]
    budget_remaining: float
    total_budget: float
    days_elapsed: int
    stockout_risk_skus: List[str]
    done: bool = False
    info: Dict = field(default_factory=dict)

    def model_dump(self):
        d = copy.copy(self.__dict__)
        d['disruptions'] = [x.dict() for x in self.disruptions]
        d['pending_orders'] = [x.dict() for x in self.pending_orders]
        d['suppliers'] = [x.dict() for x in self.suppliers]
        d['inventory'] = [x.dict() for x in self.inventory]
        return d


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

@dataclass
class ReallocationAction:
    order_id: str
    new_supplier_id: str
    quantity: int
    priority: str = "normal"


@dataclass
class SplitOrderAction:
    order_id: str
    splits: List[ReallocationAction]


@dataclass
class Action:
    reallocations: List[ReallocationAction] = field(default_factory=list)
    cancel_orders: List[str] = field(default_factory=list)
    split_orders: List[SplitOrderAction] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    stockout_avoidance: float
    cost_efficiency: float
    lead_time_score: float
    budget_adherence: float

    def dict(self):
        return copy.copy(self.__dict__)


@dataclass
class Reward:
    total: float
    breakdown: RewardBreakdown
    penalties: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def model_dump(self):
        return {
            'total': self.total,
            'breakdown': self.breakdown.dict(),
            'penalties': self.penalties,
            'explanation': self.explanation,
        }


# ---------------------------------------------------------------------------
# Step / Reset results
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    observation: Observation
    reward: Reward
    done: bool
    info: Dict = field(default_factory=dict)

    def model_dump(self):
        return {
            'observation': self.observation.model_dump(),
            'reward': self.reward.model_dump(),
            'done': self.done,
            'info': self.info,
        }


@dataclass
class ResetResult:
    observation: Observation
    info: Dict = field(default_factory=dict)

    def model_dump(self):
        return {
            'observation': self.observation.model_dump(),
            'info': self.info,
        }
