"""
Inference Script — Supply Chain Disruption Triage
===================================================
Baseline agent that uses an LLM to triage supply chain disruptions.

Environment variables required:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face / API key

Usage:
    python inference.py
    python inference.py --task task_port_congestion_cascade
    python inference.py --all-tasks
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import textwrap
from typing import List, Dict, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8080")

MAX_STEPS = 12
TEMPERATURE = 0.1
MAX_TOKENS = 1500
FALLBACK_ACTION = {"reallocations": [], "cancel_orders": [], "split_orders": [], "reasoning": "No action — fallback."}

ALL_TASKS = [
    "task_single_supplier_failure",
    "task_port_congestion_cascade",
    "task_multi_shock_crisis",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert supply chain manager triaging disruptions in real-time.

You will receive a JSON observation describing:
- Active disruption events (supplier failures, port delays, price spikes, etc.)
- Pending purchase orders, their status, and which are at risk
- Available alternate suppliers with capacity, cost, and lead time
- Current inventory levels and demand forecast
- Remaining budget

Your job is to output a JSON action that re-allocates at-risk orders to stable suppliers.

RULES:
1. Never allocate orders to disrupted suppliers (is_disrupted: true).
2. Only allocate a SKU to a supplier that lists it in available_skus.
3. Respect the budget — total reallocation cost must not exceed budget_remaining.
4. Prioritize "urgent" orders first.
5. Split large orders (qty > 200) across two suppliers when single-supplier capacity is insufficient.
6. Choose the supplier with lowest cost that still meets the lead time requirement.
7. Cancel only truly deferrable orders when budget is critically constrained.

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown, no explanation outside the JSON:
{
  "reallocations": [
    {
      "order_id": "PO-0001",
      "new_supplier_id": "SUP-02",
      "quantity": 100,
      "priority": "urgent"
    }
  ],
  "cancel_orders": [],
  "split_orders": [
    {
      "order_id": "PO-0003",
      "splits": [
        {"order_id": "PO-0003", "new_supplier_id": "SUP-04", "quantity": 150, "priority": "normal"},
        {"order_id": "PO-0003", "new_supplier_id": "SUP-09", "quantity": 150, "priority": "normal"}
      ]
    }
  ],
  "reasoning": "Brief explanation of decisions made."
}
""").strip()


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def reset(self, task_id: str) -> dict:
        r = requests.post(f"{self.base}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = requests.post(f"{self.base}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return r.json()

    def validate(self) -> dict:
        r = requests.get(f"{self.base}/validate", timeout=10)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# LLM action generation
# ---------------------------------------------------------------------------

def build_user_prompt(obs: dict, step: int, history: List[Dict]) -> str:
    # Trim observation for token efficiency
    trimmed = {
        "step": obs.get("step"),
        "task_description": obs.get("task_description"),
        "budget_remaining": obs.get("budget_remaining"),
        "stockout_risk_skus": obs.get("stockout_risk_skus"),
        "disruptions": obs.get("disruptions"),
        "pending_orders": [
            {k: v for k, v in o.items() if k in
             ("order_id", "sku", "quantity", "required_by_day",
              "current_supplier_id", "status", "priority", "unit_cost")}
            for o in obs.get("pending_orders", [])
            if o.get("status") in ("pending", "at_risk")
        ],
        "suppliers": [
            {k: v for k, v in s.items() if k in
             ("supplier_id", "name", "cost_per_unit", "lead_time_days",
              "capacity_per_day", "available_skus", "is_disrupted")}
            for s in obs.get("suppliers", [])
            if not s.get("is_disrupted")
        ],
        "inventory": obs.get("inventory"),
    }

    lines = [
        f"=== Step {step} ===",
        f"Observation:\n{json.dumps(trimmed, indent=2)}",
    ]
    if history:
        lines.append("\nPrevious actions and rewards:")
        for h in history[-3:]:
            lines.append(f"  Step {h['step']}: reward={h['reward']:.4f} | {h['reasoning'][:120]}")

    lines.append("\nRespond with ONLY the JSON action object.")
    return "\n".join(lines)


def parse_action(text: str) -> dict:
    if not text:
        return FALLBACK_ACTION
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        action = json.loads(text)
        # Validate required keys
        for key in ("reallocations", "cancel_orders", "split_orders"):
            if key not in action:
                action[key] = []
        if "reasoning" not in action:
            action["reasoning"] = ""
        return action
    except json.JSONDecodeError:
        print(f"  [WARN] Failed to parse action JSON. Using fallback.")
        return FALLBACK_ACTION


def get_action(client: OpenAI, obs: dict, step: int, history: List[Dict]) -> dict:
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [WARN] LLM call failed: {exc}. Using fallback.")
        return FALLBACK_ACTION
    return parse_action(text)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: EnvClient,
    llm: OpenAI,
    task_id: str,
    verbose: bool = True,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    reset_result = env.reset(task_id)
    obs = reset_result["observation"]

    if verbose:
        print(f"Goal: {obs['task_description'][:200]}")
        print(f"Budget: ${obs['budget_remaining']:,.0f}")
        print(f"Orders: {len(obs['pending_orders'])} | Disruptions: {len(obs['disruptions'])}")

    history: List[Dict] = []
    cumulative_reward = 0.0
    final_obs = obs

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            print(f"  Environment signalled done at step {step-1}.")
            break

        action = get_action(llm, obs, step, history)

        if verbose:
            print(f"\n  Step {step}:")
            print(f"    Reallocations: {len(action.get('reallocations', []))}")
            print(f"    Cancellations: {len(action.get('cancel_orders', []))}")
            print(f"    Splits: {len(action.get('split_orders', []))}")
            print(f"    Reasoning: {action.get('reasoning', '')[:150]}")

        try:
            step_result = env.step(action)
        except Exception as exc:
            print(f"  [ERROR] Step failed: {exc}")
            break

        reward_info = step_result.get("reward", {})
        reward = reward_info.get("total", 0.0)
        cumulative_reward += reward
        obs = step_result["observation"]
        final_obs = obs
        done = step_result.get("done", False)

        if verbose:
            bd = reward_info.get("breakdown", {})
            print(f"    Reward: {reward:.4f} | stockout={bd.get('stockout_avoidance', 0):.2f} "
                  f"cost={bd.get('cost_efficiency', 0):.2f} lead={bd.get('lead_time_score', 0):.2f} "
                  f"budget={bd.get('budget_adherence', 0):.2f}")
            if reward_info.get("penalties"):
                print(f"    Penalties: {reward_info['penalties']}")

        history.append({
            "step": step,
            "reward": reward,
            "reasoning": action.get("reasoning", ""),
        })

        if done:
            print(f"  Episode complete at step {step}.")
            break

    avg_reward = cumulative_reward / max(1, len(history))
    final_reward = history[-1]["reward"] if history else 0.0
    print(f"\nEpisode summary: avg_reward={avg_reward:.4f} | final_step_reward={final_reward:.4f}")

    return {
        "task_id": task_id,
        "steps": len(history),
        "avg_reward": round(avg_reward, 4),
        "final_reward": round(final_reward, 4),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Disruption Triage — Inference")
    parser.add_argument("--task", default="task_single_supplier_failure",
                        choices=ALL_TASKS, help="Task to run")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 3 tasks")
    parser.add_argument("--env-url", default=ENV_BASE_URL, help="Environment base URL")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY env var not set.")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME env var not set.")
        sys.exit(1)

    env = EnvClient(base_url=args.env_url)
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Health check
    try:
        v = env.validate()
        print(f"Environment health: {v['status']} | tasks: {v['tasks']}")
    except Exception as exc:
        print(f"WARNING: Could not reach environment at {args.env_url}: {exc}")
        print("Make sure the server is running: python server.py")
        sys.exit(1)

    tasks = ALL_TASKS if args.all_tasks else [args.task]
    results = []

    for task_id in tasks:
        result = run_episode(env, llm, task_id, verbose=not args.quiet)
        results.append(result)
        time.sleep(0.5)

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print(f"{'='*60}")
    print(f"{'Task':<40} {'Steps':>6} {'Avg Reward':>12} {'Final':>8}")
    print(f"{'-'*40} {'-'*6} {'-'*12} {'-'*8}")
    for r in results:
        print(f"{r['task_id']:<40} {r['steps']:>6} {r['avg_reward']:>12.4f} {r['final_reward']:>8.4f}")

    overall = sum(r["avg_reward"] for r in results) / len(results)
    print(f"\nOverall average reward: {overall:.4f}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "overall_avg": overall}, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
