# Supply Chain Agent Operations Dashboard

An advanced, dark-themed operations dashboard built with React, TailwindCSS v4, and Recharts. Designed to serve as the live command-center UI for the **Supply Chain Disruption Triage OpenEnv**.

## 🎯 How It Works

The dashboard functions as the visualization layer for the underlying Python environment `server.py` API, reading multi-objective states and submitting allocation actions.

1. **State Ingestion**: Upon initialization, the dashboard connects to `localhost:8080/state`. If the Python server is offline, it seamlessly falls back into a **Local Demo** mode driven by a rich automated JS mocking engine.
2. **Action Dispatch**: Users (or AI agents) evaluate open Purchase Orders against available tracking Data (unit cost, reliability). Clicking a healthy supplier instantly drafts a JSON `reallocation` action in the **Reasoning Terminal**.
3. **Execution Loop**: The "Step Emit" button fires a `POST /step` to the backend and fetches the next `Observation` frame and `Reward` scores.
4. **Reward Plotting**: Complex multi-objective rewards (Stockout Avoidance, Cost Efficiency, Lead Time, Budget Adherence) are tracked progressively in the Live Sparkline chart as the episode steps towards completion.

## 🚀 How to Use It

### **1. Running the Back-end Environment**
Ensure your local API is running to supply real data:
```bash
# In the repository root
python server.py
# Server runs on http://localhost:8080
```

### **2. Running the Frontend Dashboard**
```bash
# In the `supply-chain-ui` directory
npm install
npm run dev
```

### **3. UI Interactions**
- **Task Selector Selector**: Switch between three OpenEnv crisis levels: Easy (Single Supplier Failure), Medium (Port Cascade), Hard (Multi-Shock).
- **Supplier Directory (Right Panel)**: Cards display live metrics. A Red 'Disrupted' overlay locks you out of using inactive vendors. Clicking an active vendor stages an action string.
- **Budget Gauge (Left Panel)**: Tracks total procurement spend vs allocated hardcap in real-time. Turns red when funds slip below 20%.
- **Reasoning Terminal (Bottom Panel)**: Watch simulated command-line outputs confirming allocations and reasoning paths.
- **AutoRun**: Found under the step emitter, clicking AutoRun will rapidly fire step transitions so you can review episode outcomes dynamically.

## 🌟 Where it Excels

- **Instant Fallback Resiliency**: The dashboard excels at maintaining operational integrity. If the Python API drops or is unreachable, the system automatically patches to the Local Demo state ensuring UX demonstrations can continue unimpeded.
- **Data Density**: Mashing "Industrial Ops" aesthetics with Tailwind provides a high-density, highly legible UI pattern capable of displaying hundreds of unique order properties (SKU, Cost, Status) on a single screen without scrolling fatigue.
- **Agent Interactivity**: Built specifically for Reinforcement Learning (RL) and LLM agents, the "Agent Reasoning" ticker provides unparalleled granular insight into *why* the supply chain triage AI is making routing choices.
