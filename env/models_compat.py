"""
Stdlib-only version of models for local validation.
The real models.py uses Pydantic for the HF Space deployment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal


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


@dataclass
class PurchaseOrder:
    order_id: str
    sku: str
    quantity: int
    required_by_day: int
    original_supplier_id: str
    unit_cost: float
    current_supplier_id: Optional[str] = None
    status: str = "pending"   # pending/allocated/at_risk/cancelled/fulfilled
    priority: str = "normal"  # urgent/normal/deferrable


@dataclass
class DisruptionEvent:
    disruption_id: str
    event_type: str
    affected_supplier_ids: List[str]
    affected_skus: List[str]
    severity: str
    delay_days: int
    price_multiplier: float
    description: str
    day_occurred: int


@dataclass
class InventoryLevel:
    sku: str
    current_stock: int
    safety_stock: int
    reorder_point: int


@dataclass
class Observation:
    step: int
    task_id: str
    task_description: str
    disruptions: List[DisruptionEvent]
    pending_orders: List[PurchaseOrder]
    suppliers: List[Supplier]
    inventory: List[InventoryLevel]
    demand_forecast: Dict
    budget_remaining: float
    total_budget: float
    days_elapsed: int
    stockout_risk_skus: List[str]
    done: bool = False
    info: Dict = field(default_factory=dict)


@dataclass
class ReallocationAction:
    order_id: str
    new_supplier_id: str
    quantity: int
    priority: str = "normal"


@dataclass
class SplitSegment:
    order_id: str
    new_supplier_id: str
    quantity: int
    priority: str = "normal"


@dataclass
class SplitOrderAction:
    order_id: str
    splits: List[SplitSegment]


@dataclass
class Action:
    reallocations: List[ReallocationAction] = field(default_factory=list)
    cancel_orders: List[str] = field(default_factory=list)
    split_orders: List[SplitOrderAction] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class RewardBreakdown:
    stockout_avoidance: float
    cost_efficiency: float
    lead_time_score: float
    budget_adherence: float


@dataclass
class Reward:
    total: float
    breakdown: RewardBreakdown
    penalties: Dict = field(default_factory=dict)
    explanation: str = ""


@dataclass
class StepResult:
    observation: Observation
    reward: Reward
    done: bool
    info: Dict = field(default_factory=dict)


@dataclass
class ResetResult:
    observation: Observation
    info: Dict = field(default_factory=dict)
