---
title: Supply Chain Disruption Triage RL Environment
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🌍 Supply Chain Disruption Triage AI

> **A production-ready autonomous environment for real-time supply chain resilient triage.**

**Project Lead:** MAYANK CHOUHAN  
**Scenario Architect:** Sunoy Roy  
**UI/UX Lead:** Aman Khan  
**Corpus:** Supply Chain Resiliency & Logistics Optimization
**Live Demo (Vercel):** [https://supply-chain-env.vercel.app/](https://supply-chain-env.vercel.app/)  
**Hugging Face Space:** [https://huggingface.co/spaces/MAYANK22/supply-chain-env](https://huggingface.co/spaces/MAYANK22/supply-chain-env)

---

## 🎯 Project Overview
This project simulates the high-stakes world of global procurement. During a disruption (e.g., natural disasters, factory fires, or port strikes), an AI agent must navigate complex trade-offs to ensure continuity. The system evaluates agents based on:
- **Stockout avoidance (40%)**: Maintaining 100% service levels.
- **Cost efficiency (30%)**: Minimizing procurement cost variance.
- **Lead time (20%)**: Respecting critical delivery windows.
- **Budget adherence (10%)**: Staying within operational limits.

This is an **OpenEnv-compliant** multi-objective constraint-satisfaction environment.

---

## 🏗️ Architecture
- **Environment**: Python-based RL environment with Pydantic typed models.
- **Server**: FastAPI exposing the OpenEnv HTTP API.
- **Data**: Seed-based synthetic generator producing suppliers, SKUs, and orders.
- **Tasks**: 3 predefined tasks (Easy, Medium, Hard) plus a Live Realworld Crisis mode.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11
- Docker (optional)

### 2. Local Installation
```bash
git clone https://github.com/MAYANK2264/NovaAI-Hackethon-openenv.git
cd NovaAI-Hackethon-openenv
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python server.py
# Server will be running at http://localhost:8080
```

### 4. Run Baseline Inference
```bash
# Set environment variables for LLM (if using LLM agent)
# export HF_TOKEN=your_token
# export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

python inference.py --all-tasks
```

---

## 🔌 API Usage

### `POST /reset`
Initialize/Reset the environment for a specific task.
```json
{
  "task_id": "task_single_supplier_failure"
}
```

### `POST /step`
Submit an action and receive the next observation and reward.
```json
{
  "action": {
    "reallocations": [
      {
        "order_id": "PO-0001",
        "new_supplier_id": "SUP-02",
        "quantity": 100,
        "priority": "urgent"
      }
    ],
    "cancel_orders": [],
    "split_orders": [],
    "reasoning": "Rerouting due to supplier failure."
  }
}
```

### `GET /validate`
Check environment health and task availability.

---

## 🐋 Docker Deployment
```bash
docker build -t supply-chain-env .
docker run -p 8080:8080 supply-chain-env
```

## 🏆 Hackathon Submission Details
- **Project**: Supply Chain Disruption Triage
- **Author**: MAYANK CHOUHAN
- **Compliance**: OpenEnv v1.0
