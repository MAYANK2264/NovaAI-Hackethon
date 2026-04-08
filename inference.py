"""
Inference Script — Supply Chain Disruption Triage
===================================================
Baseline agent for supply chain disruption triage.
Follows strict OpenEnv pre-submission checklist.

Environment variables:
    API_BASE_URL: (Defaulted) 
    MODEL_NAME:   (Defaulted)
    HF_TOKEN:     (Required for LLM mode, no default)

Usage:
    python inference.py
    python inference.py --task task_port_congestion_cascade
    python inference.py --all-tasks --mode llm
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Pre-submission Checklist Compliance
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") # Checklist: No default allowed for tokens

# OpenAI client initialization - we use this for the LLM mode
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-token",
)

# Local environment server URL
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8080")

ALL_TASKS = [
    "task_single_supplier_failure",
    "task_port_congestion_cascade",
    "task_multi_shock_crisis",
]

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
# Logging & Format Compliance (MANDATORY)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", 
        flush=True
    )

def get_llm_action(obs: dict) -> dict:
    """
    LLM-based reasoning agent. Uses OpenAI client to decide reallocations.
    """
    prompt = f"""
You are a Supply Chain Triage Agent. A disruption has occurred.
Current State: {json.dumps(obs, indent=2)}

Decide which orders to reallocate to alternate suppliers to minimize stockouts and costs.
Return ONLY a JSON object matching the action space:
{{
  "reallocations": [{{ "order_id": "...", "new_supplier_id": "...", "quantity": ..., "priority": "..." }}],
  "reasoning": "Explain your choice"
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        # Fallback to heuristic if LLM fails (e.g. rate limit)
        return get_heuristic_action(obs)


def get_heuristic_action(obs: dict) -> dict:
    """
    Reroute to cheapest valid supplier, avoid disrupted, respect lead time.
    """
    reallocations = []
    
    disrupted_suppliers = {s["supplier_id"] for s in obs.get("suppliers", []) if s.get("is_disrupted")}
    suppliers = [s for s in obs.get("suppliers", []) if not s.get("is_disrupted")]
    
    # Sort suppliers by cost (cheapest first)
    suppliers.sort(key=lambda x: x.get("cost_per_unit", 999999))
    
    at_risk_orders = [o for o in obs.get("pending_orders", []) if o.get("status") in ("at_risk", "pending")]
    
    budget_remaining = obs.get("budget_remaining", 0)
    
    for order in at_risk_orders:
        sku = order.get("sku")
        required_by = order.get("required_by_day", 0)
        
        # Find cheapest supplier that has SKU and meets lead time
        best_sup = None
        for sup in suppliers:
            if sku in sup.get("available_skus", []) and sup.get("lead_time_days", 999) <= required_by:
                # Check if we have budget (simplified)
                cost_diff = (sup.get("cost_per_unit", 0) - order.get("unit_cost", 0)) * order.get("quantity", 0)
                if budget_remaining >= cost_diff:
                    best_sup = sup
                    budget_remaining -= cost_diff
                    break
        
        if best_sup:
            reallocations.append({
                "order_id": order.get("order_id"),
                "new_supplier_id": best_sup.get("supplier_id"),
                "quantity": order.get("quantity"),
                "priority": order.get("priority", "normal")
            })

    return {
        "reallocations": reallocations,
        "cancel_orders": [],
        "split_orders": [],
        "reasoning": "Heuristic: Reroute to cheapest valid supplier meeting lead time."
    }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: EnvClient,
    task_id: str,
    mode: str = "heuristic"
) -> float:
    benchmark_name = "supply-chain-env"
    
    log_start(task=task_id, env=benchmark_name, model=MODEL_NAME)

    reset_result = env.reset(task_id)
    obs = reset_result["observation"]

    rewards_list = []
    final_score = 0.0
    steps_taken = 0
    success = False

    while not obs.get("done"):
        step_num = obs.get("step", 0) + 1
        steps_taken = step_num
        
        # Decide action based on mode
        if mode == "llm" and HF_TOKEN:
            action = get_llm_action(obs)
        else:
            action = get_heuristic_action(obs)

        try:
            step_result = env.step(action)
        except Exception as exc:
            log_step(step=step_num, action="unknown", reward=0.0, done=True, error=str(exc))
            break

        reward_info = step_result.get("reward", {})
        reward = reward_info.get("total", 0.0)
        obs = step_result["observation"]
        
        rewards_list.append(reward)
        final_score = reward # Final reward is the grade [0, 1]
        
        log_step(
            step=step_num, 
            action=action.get("reasoning", "action executed"), 
            reward=reward, 
            done=obs.get("done", False), 
            error=None
        )

        if obs.get("done"):
            success = final_score > 0.1 # Threshold for success
            break

    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards_list)
    return final_score


def main():
    parser = argparse.ArgumentParser(description="Supply Chain Disruption Triage — Inference")
    parser.add_argument("--task", default="task_single_supplier_failure",
                        choices=ALL_TASKS, help="Task to run")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 3 tasks")
    parser.add_argument("--env-url", default=ENV_BASE_URL, help="Environment base URL")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic", help="Agent mode")
    args = parser.parse_args()

    env = EnvClient(base_url=args.env_url)

    # Health check
    try:
        env.validate()
    except Exception as exc:
        print(f"CRITICAL: Could not reach environment at {args.env_url}: {exc}")
        sys.exit(1)

    tasks = ALL_TASKS if args.all_tasks else [args.task]

    for task_id in tasks:
        run_episode(env, task_id, mode=args.mode)


if __name__ == "__main__":
    main()
