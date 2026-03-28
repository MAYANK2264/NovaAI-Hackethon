# Supply Chain Disruption Triage — OpenEnv

> **An agent environment for real-world supply chain crisis management.**
> Agents triage live disruption signals and re-allocate purchase orders across alternate
> vendors to minimize stockouts, cost overruns, and delivery delays.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.ai)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

---

## Why This Problem?

Supply chain disruptions cost the global economy **$4 trillion+ annually** (McKinsey, 2023).
When a supplier fails, a port closes, or a commodity price spikes, procurement teams
manually scramble across spreadsheets and email chains to re-route hundreds of orders.
This process is slow (hours to days), error-prone, and requires deep domain expertise.

No off-the-shelf AI tool reliably solves this. It demands:
- Multi-objective optimization (cost vs. speed vs. stockout risk vs. budget)
- Constraint satisfaction (capacity limits, SKU availability, lead time windows)
- Cascading impact reasoning (one disruption ripples through multiple orders)
- Prioritization under uncertainty (urgent orders first, partial coverage is better than none)

This environment models exactly that workflow, with deterministic graders that reward
partial progress — making it suitable for both RL training and LLM evaluation.

---

## Environment Description

The agent operates a procurement desk. At each step it receives an **Observation**
describing the current supply chain state and must issue an **Action** (re-allocations,
cancellations, or order splits) to minimize a multi-objective cost function.

### Observation Space

| Field | Type | Description |
|---|---|---|
| `disruptions` | `List[DisruptionEvent]` | Active events (failures, delays, price spikes, shortages) |
| `pending_orders` | `List[PurchaseOrder]` | Orders and their current allocation status |
| `suppliers` | `List[Supplier]` | All suppliers with capacity, cost, lead time, and disruption flag |
| `inventory` | `List[InventoryLevel]` | Current stock per SKU vs. safety stock threshold |
| `demand_forecast` | `Dict[str, List[int]]` | 14-day daily demand forecast per SKU |
| `budget_remaining` | `float` | Remaining procurement budget (USD) |
| `stockout_risk_skus` | `List[str]` | SKUs currently at risk of stockout |
| `step` | `int` | Current episode step |
| `task_description` | `str` | Natural language task objective |

### Action Space

```json
{
  "reallocations": [
    {
      "order_id": "PO-0001",
      "new_supplier_id": "SUP-04",
      "quantity": 250,
      "priority": "urgent"
    }
  ],
  "cancel_orders": ["PO-0007"],
  "split_orders": [
    {
      "order_id": "PO-0003",
      "splits": [
        {"order_id": "PO-0003", "new_supplier_id": "SUP-02", "quantity": 150, "priority": "normal"},
        {"order_id": "PO-0003", "new_supplier_id": "SUP-09", "quantity": 150, "priority": "normal"}
      ]
    }
  ],
  "reasoning": "Rerouted urgent chip orders to Dallas supplier; split large motor order."
}
```

### Reward Function

Reward is a **multi-objective weighted score** in `[0.0, 1.0]` computed at every step:

| Component | Weight | Measures |
|---|---|---|
| Stockout avoidance | **40%** | Fraction of at-risk orders resolved; bonus for urgent-first |
| Cost efficiency | **30%** | Cost vs. baseline; penalizes >10% premium harshly |
| Lead time compliance | **20%** | Allocated supplier lead time ≤ required-by day |
| Budget adherence | **10%** | Remaining budget ratio; hard penalty for overspend |

Penalties (subtracted from total) are applied for: allocating to disrupted suppliers,
unknown order/supplier IDs, SKU mismatches, and split quantity errors.

---

## Tasks

### Task 1 — Single Supplier Failure *(easy)*
**ID:** `task_single_supplier_failure`

A factory fire has taken `SUP-01` (Apex Components, Shenzhen) offline. 3 of 8 orders
are now at risk. Re-route them to alternate suppliers within a $50,000 budget.

- Max steps: 5
- Passing grade: ≥ 0.60
- Key challenge: Find a supplier that carries the affected SKUs at acceptable cost

### Task 2 — Port Congestion Cascade *(medium)*
**ID:** `task_port_congestion_cascade`

Shanghai port congestion causes +9 day delays across all CN/TW shipments. Additionally,
PacificParts (SUP-03) is short on chip components (+15% cost, +5 day delay). 12 of 18
orders are impacted. Split large orders, prioritize urgent ones, stay within $120,000.

- Max steps: 8
- Passing grade: ≥ 0.55
- Key challenge: Optimal use of order splitting + urgency-weighted triage

### Task 3 — Multi-Shock Supply Crisis *(hard)*
**ID:** `task_multi_shock_crisis`

Simultaneous events:
- `SUP-06` (Atlas Industrial) files for **bankruptcy** — all orders void
- Rotterdam port **closed** (strike) — +12 day delay on NL shipments
- Global **chip + battery shortage** — +35% price spike, 60% allocation cap reduction
- 20+ orders affected, hard budget ($200k) and lead time (≤14 days) constraints

- Max steps: 12
- Passing grade: ≥ 0.45
- Key challenge: Multi-constraint optimization under severe resource scarcity

---

## Baseline Scores

Run against `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router:

| Task | Difficulty | Avg Reward | Final Step Reward |
|---|---|---|---|
| task_single_supplier_failure | Easy | ~0.55 | ~0.65 |
| task_port_congestion_cascade | Medium | ~0.45 | ~0.55 |
| task_multi_shock_crisis | Hard | ~0.30 | ~0.38 |
| **Overall** | | **~0.43** | |

*Scores vary by model. Frontier models (GPT-4o, Claude Sonnet) typically score 0.65–0.80 on easy, 0.50–0.65 on medium, 0.35–0.50 on hard.*

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- A HuggingFace account + API token

### Local Development

```bash
# Clone and install
git clone <your-repo-url>
cd supply-chain-disruption-triage
pip install -r requirements.txt

# Start the environment server
python server.py
# → Running at http://localhost:8080

# In another terminal, run the inference agent
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py --task task_single_supplier_failure
python inference.py --all-tasks
```

### Running Tests

```bash
# With pydantic installed (full suite):
pytest tests/ -v

# Without pydantic (stdlib-only, works in any env):
python run_tests.py
```

### Docker

```bash
# Build
docker build -t supply-chain-env .

# Run
docker run -p 8080:8080 supply-chain-env

# Verify
curl http://localhost:8080/validate
```

### OpenEnv Validation

```bash
openenv validate --url http://localhost:8080
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{"task_id": "..."}` |
| `/step` | POST | Submit an action. Body: `{"action": {...}}` |
| `/state` | GET | Get current observation |
| `/validate` | GET | Health check + spec compliance ping |
| `/tasks` | GET | List all tasks with metadata |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
supply-chain-disruption-triage/
├── inference.py              # Baseline agent (OpenAI client)
├── server.py                 # FastAPI HTTP server
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Container definition
├── requirements.txt
├── run_tests.py              # Self-contained test runner (no pytest needed)
├── env/
│   ├── models.py             # Pydantic typed models (Observation, Action, Reward)
│   ├── models_compat.py      # Stdlib dataclass version (local testing without pydantic)
│   └── environment.py        # Core env logic: reset(), step(), state()
├── graders/
│   └── graders.py            # Deterministic graders for all 3 tasks
├── data/
│   └── generator.py          # Synthetic supply chain data generator
└── tests/
    └── test_environment.py   # pytest test suite
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select **Docker** as the SDK
3. Push this repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/supply-chain-env
   git push space main
   ```
4. The Space will build and expose the environment at your Space URL
5. Submit that URL to the OpenEnv competition platform

---

## License

MIT
