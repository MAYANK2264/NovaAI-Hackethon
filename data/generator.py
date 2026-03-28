"""
Deterministic synthetic data generator for supply chain scenarios.
Seed-controlled so baseline scores are fully reproducible.
"""

from __future__ import annotations
import random
from typing import List, Dict, Tuple

from env.models import (
    Supplier, PurchaseOrder, DisruptionEvent,
    InventoryLevel
)

# Fixed SKU catalog
SKUS = [
    "SKU-CHIP-A1", "SKU-CHIP-B2", "SKU-MOTOR-X",
    "SKU-SENSOR-7", "SKU-CABLE-USB", "SKU-BATTERY-L",
    "SKU-DISPLAY-4K", "SKU-FRAME-AL", "SKU-PCB-BASE",
    "SKU-FAN-12V",
]

SUPPLIER_TEMPLATES = [
    {"name": "Apex Components Ltd",     "location": "Shenzhen, CN",   "cost_base": 12.5,  "lead": 7,  "rel": 0.95, "cap": 500},
    {"name": "NordSupply GmbH",         "location": "Hamburg, DE",    "cost_base": 18.0,  "lead": 5,  "rel": 0.97, "cap": 300},
    {"name": "PacificParts Co",         "location": "Osaka, JP",      "cost_base": 15.0,  "lead": 9,  "rel": 0.92, "cap": 400},
    {"name": "Delta Sourcing Inc",      "location": "Dallas, TX",     "cost_base": 22.0,  "lead": 3,  "rel": 0.98, "cap": 250},
    {"name": "Meridian Electronics",    "location": "Seoul, KR",      "cost_base": 14.0,  "lead": 8,  "rel": 0.93, "cap": 450},
    {"name": "Atlas Industrial",        "location": "Mumbai, IN",     "cost_base": 10.5,  "lead": 11, "rel": 0.88, "cap": 600},
    {"name": "Zenith Supply Chain",     "location": "Rotterdam, NL",  "cost_base": 19.5,  "lead": 4,  "rel": 0.96, "cap": 350},
    {"name": "Coastal Components",      "location": "Taipei, TW",     "cost_base": 13.0,  "lead": 8,  "rel": 0.91, "cap": 480},
    {"name": "Summit Manufacturing",    "location": "Toronto, CA",    "cost_base": 20.0,  "lead": 3,  "rel": 0.99, "cap": 200},
    {"name": "Horizon Parts Ltd",       "location": "Bangalore, IN",  "cost_base": 11.0,  "lead": 12, "rel": 0.85, "cap": 700},
]


def make_suppliers(rng: random.Random, num: int = 10) -> List[Supplier]:
    suppliers = []
    for i, t in enumerate(SUPPLIER_TEMPLATES[:num]):
        # Each supplier handles a random subset of SKUs
        sku_count = rng.randint(3, 7)
        available = rng.sample(SKUS, k=sku_count)
        suppliers.append(Supplier(
            supplier_id=f"SUP-{i+1:02d}",
            name=t["name"],
            location=t["location"],
            capacity_per_day=t["cap"],
            cost_per_unit=round(t["cost_base"] + rng.uniform(-1.5, 1.5), 2),
            lead_time_days=t["lead"],
            reliability_score=t["rel"],
            available_skus=available,
        ))
    return suppliers


def make_orders(
    rng: random.Random,
    suppliers: List[Supplier],
    num_orders: int,
    disrupted_ids: List[str],
) -> List[PurchaseOrder]:
    orders = []
    disrupted_set = set(disrupted_ids)

    for i in range(num_orders):
        sup = rng.choice(suppliers)
        sku = rng.choice(sup.available_skus)
        qty = rng.randint(50, 400)
        required_by = rng.randint(5, 20)
        priority_choices = ["urgent", "normal", "normal", "deferrable"]
        priority = rng.choice(priority_choices)
        status = "at_risk" if sup.supplier_id in disrupted_set else "pending"
        orders.append(PurchaseOrder(
            order_id=f"PO-{i+1:04d}",
            sku=sku,
            quantity=qty,
            required_by_day=required_by,
            original_supplier_id=sup.supplier_id,
            current_supplier_id=sup.supplier_id,
            status=status,
            unit_cost=sup.cost_per_unit,
            priority=priority,
        ))
    return orders


def make_inventory(rng: random.Random) -> List[InventoryLevel]:
    levels = []
    for sku in SKUS:
        current = rng.randint(20, 300)
        safety = rng.randint(30, 80)
        levels.append(InventoryLevel(
            sku=sku,
            current_stock=current,
            safety_stock=safety,
            reorder_point=safety + rng.randint(10, 40),
        ))
    return levels


def make_demand_forecast(rng: random.Random) -> Dict[str, List[int]]:
    forecast = {}
    for sku in SKUS:
        daily = [rng.randint(10, 60) for _ in range(14)]
        forecast[sku] = daily
    return forecast


def compute_stockout_risk(
    inventory: List[InventoryLevel],
    orders: List[PurchaseOrder],
    forecast: Dict[str, List[int]],
) -> List[str]:
    at_risk = []
    for inv in inventory:
        projected = inv.current_stock
        incoming = sum(o.quantity for o in orders if o.sku == inv.sku and o.status not in ("cancelled",))
        demand_14d = sum(forecast.get(inv.sku, []))
        if projected + incoming < demand_14d * 0.6:
            at_risk.append(inv.sku)
        elif projected < inv.safety_stock:
            at_risk.append(inv.sku)
    return list(set(at_risk))
