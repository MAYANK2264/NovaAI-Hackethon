"""
FastAPI server — exposes OpenEnv HTTP interface:
  POST /reset
  POST /step
  GET  /state
  GET  /validate
  GET  /tasks
"""

from __future__ import annotations
from typing import Optional, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import SupplyChainEnv, TASK_CONFIGS
from env.models import Action

app = FastAPI(
    title="Supply Chain Disruption Triage — OpenEnv",
    description="OpenEnv environment for real-world supply chain disruption triage.",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session; for multi-session use thread-local or session IDs)
_env: Optional[SupplyChainEnv] = None


# ---------------------------------------------------------------------------
# Request / response schemas for the HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_single_supplier_failure"


class StepRequest(BaseModel):
    action: Action


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Body, Request

@app.post("/reset")
async def reset(request: Request):
    global _env
    # Fail-safe: Manually parse the body to avoid any FastAPI validation errors
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    # Gracefully default to the primary task if body or task_id is missing
    task_id = body.get("task_id", "task_single_supplier_failure") if isinstance(body, dict) else "task_single_supplier_failure"
    
    if task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    
    _env = SupplyChainEnv(task_id=task_id)
    result = _env.reset()
    return result.model_dump()


@app.post("/step")
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    result = _env.step(req.action)
    return result.model_dump()


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    s = _env.state()
    if s is None:
        raise HTTPException(status_code=400, detail="No state available.")
    return s.model_dump()


@app.get("/validate")
def validate():
    """Health check + OpenEnv spec compliance ping."""
    return {
        "status": "ok",
        "name": "supply-chain-disruption-triage",
        "version": "1.0.0",
        "tasks": list(TASK_CONFIGS.keys()),
        "openenv_compliant": True,
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
                "total_budget": cfg["total_budget"],
                "num_orders": cfg["num_orders"],
                "num_disruptions": cfg["num_disruptions"],
            }
            for tid, cfg in TASK_CONFIGS.items()
        ]
    }


@app.get("/")
def root():
    return {"message": "Supply Chain Disruption Triage OpenEnv. See /docs for API."}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
