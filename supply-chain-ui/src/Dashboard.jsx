import { useState, useEffect, useRef, useMemo } from 'react';
import Globe from 'react-globe.gl';
import { AreaChart, Area, XAxis, Tooltip, ResponsiveContainer } from 'recharts';
import {
  AlertTriangle, CheckCircle2, Activity, Clock, Box, ShieldAlert,
  Terminal, ShieldCheck, Truck, RefreshCcw, HandCoins, AlertOctagon, User, Factory, ChevronDown, ChevronUp, Pause
} from 'lucide-react';

const TASKS = [
  { id: 'task_single_supplier_failure', label: 'Easy: Single Failure' },
  { id: 'task_port_congestion_cascade', label: 'Medium: Port Congestion' },
  { id: 'task_multi_shock_crisis', label: 'Hard: Multi-Shock' },
  { id: 'task_live_realworld_crisis', label: '🔴 Live: Web API Data' }
];

// Backend URL - defaults to local for dev, but can be overridden
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

const MOCK_STATE = {
  step: 1,
  task_id: "demo_mode",
  task_description: "Initialize backend to see real openenv state. Running demo mode.",
  disruptions: [
    { type: "supplier_failure", affected_supplier_ids: ["SUP-04"], severity: "high", description: "Apex Components offline." }
  ],
  pending_orders: [
    { order_id: "PO-0001", sku: "CHIP-x86", quantity: 2500, original_supplier_id: "SUP-04", current_supplier_id: "SUP-04", status: "at_risk", priority: "urgent", unit_cost: 15.5 },
    { order_id: "PO-0002", sku: "MOTOR-DC", quantity: 500, original_supplier_id: "SUP-01", current_supplier_id: "SUP-01", status: "allocated", priority: "normal", unit_cost: 45.0 }
  ],
  suppliers: [
    { supplier_id: "SUP-04", name: "Apex Components", is_disrupted: true, cost_per_unit: 14.5, lead_time_days: 7, reliability_score: 0.2 },
    { supplier_id: "SUP-01", name: "Global Motors", is_disrupted: false, cost_per_unit: 45.0, lead_time_days: 14, reliability_score: 0.95 },
    { supplier_id: "SUP-02", name: "TechFab Taiwan", is_disrupted: false, cost_per_unit: 16.5, lead_time_days: 5, reliability_score: 0.88 }
  ],
  budget_remaining: 35000,
  total_budget: 50000,
  reward: {
    total: 0.55,
    breakdown: { stockout_avoidance: 0.6, cost_efficiency: 0.5, lead_time_score: 0.7, budget_adherence: 0.4 },
    explanation: "Re-allocated 1 order. Penalties for cost overrun."
  }
};

export default function Dashboard() {
  const [activeTask, setActiveTask] = useState(TASKS[0].id);
  const [activeView, setActiveView] = useState('globe');
  const [useMock, setUseMock] = useState(false);
  const [sysState, setSysState] = useState(MOCK_STATE);
  const [history, setHistory] = useState([{ step: 0, reward: 0 }]);
  const [globeSize, setGlobeSize] = useState({ width: 800, height: 600 });
  const [countries, setCountries] = useState({ features: [] });
  const containerRef = useRef(null);
  const [agentReasoning, setAgentReasoning] = useState("Agent initialized. Awaiting reset...");
  const [reasoningExpanded, setReasoningExpanded] = useState(true);
  const [autoRun, setAutoRun] = useState(false);
  const [stagedAction, setStagedAction] = useState(null);
  const [backendAlive, setBackendAlive] = useState(false);
  const [backendStatus, setBackendStatus] = useState('Disconnected');
  const autoRunInterval = useRef(null);

  // Connection watchdog - checks if FastAPI is responsive every 30s
  useEffect(() => {
    let interval;
    const pingBackend = async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      try {
        const r = await fetch(`${API_BASE}/validate`, { signal: controller.signal });
        if (r.ok) {
          setBackendStatus('Connected');
          setBackendAlive(true);
        } else {
          setBackendStatus('Offline');
          setBackendAlive(false);
          if (!useMock) setUseMock(true);
        }
      } catch (err) {
        setBackendStatus('Offline');
        setBackendAlive(false);
        if (!useMock) setUseMock(true);
      } finally {
        clearTimeout(timeoutId);
      }
    };
    
    pingBackend();
    interval = setInterval(pingBackend, 30000);
    return () => clearInterval(interval);
  }, [useMock]);

  const handleReset = async () => {
    setHistory([{ step: 0, reward: 0 }]);
    setAgentReasoning(`Environment reset requesting ${activeTask}...`);
    setAutoRun(false);

    if (useMock || !backendAlive) {
      setSysState({ ...MOCK_STATE, task_id: activeTask, step: 0 });
      setAgentReasoning(`[DEMO] Reset ${activeTask} complete.`);
      return;
    }

    try {
      const resp = await fetch(`${API_BASE}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: activeTask })
      });
      const data = await resp.json();
      setSysState(data.observation || data); // handle diff openenv returns
      setAgentReasoning("Reset complete: environment clean.");
    } catch (e) {
      setAgentReasoning("Reset failed: " + e.message);
      setUseMock(true);
    }
  };

  const handleStep = async (actionBody = { reason: "noop" }) => {
    let act = stagedAction || { reasoning: "Requested no-op pass." };
    if (!stagedAction && actionBody) act = actionBody;

    setAgentReasoning("Evaluating step action...\n" + JSON.stringify(act, null, 2));

    if (useMock || !backendAlive) {
      // Simulation mode for when backend is down/unreachable
      setTimeout(() => {
        const nextStep = sysState.step + 1;
        const newReward = Math.min(1.0, (sysState.reward?.total || 0) + 0.1);
        setSysState(s => ({
          ...s,
          step: nextStep,
          budget_remaining: s.budget_remaining - 500,
          pending_orders: s.pending_orders.map(o => o.status === 'at_risk' ? { ...o, status: 'allocated', current_supplier_id: 'SUP-02' } : o),
          reward: {
            ...s.reward,
            total: newReward,
            breakdown: { stockout_avoidance: 0.8, cost_efficiency: 0.6, lead_time_score: 0.8, budget_adherence: 0.5 }
          }
        }));
        setHistory(h => [...h, { step: nextStep, reward: newReward }]);
        setStagedAction(null);
        setAgentReasoning(`[DEMO ACTUATOR]\nApplied action. Simulated reward: ${newReward.toFixed(2)}`);
      }, 400);
      return;
    }

    try {
      const resp = await fetch(`${API_BASE}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: act })
      });
      const data = await resp.json();
      setSysState(data.observation || data);
      const r_tot = data.reward?.total || data.observation?.reward?.total || 0;
      setHistory(h => [...h, { step: (data.observation?.step || 0), reward: r_tot }]);
      setAgentReasoning(`> API returns reward: ${r_tot.toFixed(2)}\n${data.reward?.explanation || "No notes."}`);
      setStagedAction(null);
    } catch (e) {
      setAgentReasoning("Step failed: " + e.message);
      setAutoRun(false);
    }
  };

  useEffect(() => {
    if (autoRun) {
      autoRunInterval.current = setInterval(() => handleStep({ reasoning: "Auto-run fallback." }), 2000);
    } else {
      clearInterval(autoRunInterval.current);
    }
    return () => clearInterval(autoRunInterval.current);
  }, [autoRun, sysState]);

  const onSupplierClick = (sup) => {
    if (sup.is_disrupted) return;
    const atRisk = sysState.pending_orders.find(o => o.status === 'at_risk');
    if (atRisk) {
      setStagedAction({
        reallocations: [{ order_id: atRisk.order_id, new_supplier_id: sup.supplier_id, quantity: atRisk.quantity }]
      });
      setAgentReasoning(`Drafting reallocation for ${atRisk.order_id} to ${sup.name}...`);
    } else {
      setAgentReasoning(`No at_risk orders to reallocate to ${sup.name}.`);
    }
  };

  const getStatusColor = (status) => {
    if (status === 'at_risk') return 'text-brand-red border-brand-red/50 bg-brand-red/10 shadow-[0_0_10px_rgba(255,68,68,0.3)] pulse-red';
    if (status === 'allocated') return 'text-brand-teal border-brand-teal/50 bg-brand-teal/10';
    if (status === 'cancelled') return 'text-gray-500 border-gray-600 bg-gray-800';
    return 'text-brand-amber border-brand-amber/50 bg-brand-amber/10';
  };

  const currentReward = sysState.reward || { breakdown: {} };

  const arcsData = useMemo(() => {
    return (sysState.pending_orders || []).map(po => {
      const sup = sysState.suppliers?.find(s => s.supplier_id === po.current_supplier_id);
      if (!sup) return null;
      return {
        startLat: sup.lat || 0,
        startLng: sup.lng || 0,
        endLat: po.dest_lat || 34.0522,
        endLng: po.dest_lng || -118.2437,
        color: ['rgba(0, 212, 170, 0)', po.status === 'at_risk' ? '#ff4444' : '#00d4aa'],
        stroke: 1.5,
      };
    }).filter(Boolean);
  }, [sysState.pending_orders, sysState.suppliers]);

  const labelsData = useMemo(() => {
    return (sysState.suppliers || []).map(sup => ({
      lat: sup.lat || 0,
      lng: sup.lng || 0,
      text: sup.name,
      color: sup.is_disrupted ? '#ff4444' : '#cbd5e1',
      size: sup.is_disrupted ? 1.0 : 0.6,
    }))
  }, [sysState.suppliers]);

  const floatingNodes = useMemo(() => {
    return Array(60).fill().map(() => ({
      lat: (Math.random() - 0.5) * 180,
      lng: (Math.random() - 0.5) * 360,
      text: '',
      color: Math.random() > 0.5 ? 'rgba(234, 0, 217, 0.4)' : 'rgba(0, 229, 255, 0.4)',
      size: 0.8,
      altitude: Math.random() * 0.8 + 0.1
    }));
  }, []);

  const mergedLabels = useMemo(() => {
    return [
      ...labelsData.map(l => ({ ...l, altitude: 0.05, size: 2.0 })), // Swelled data markers to pop
      ...floatingNodes
    ];
  }, [labelsData, floatingNodes]);

  const ringsData = useMemo(() => {
    return (sysState.suppliers || []).filter(s => s.is_disrupted).map(sup => ({
      lat: sup.lat || 0,
      lng: sup.lng || 0,
      color: '#ff4444',
      maxR: 12,
      propagationSpeed: 2,
      repeatPeriod: 1000
    }))
  }, [sysState.suppliers]);

  const globeEl = useRef();
  
  useEffect(() => {
    if (globeEl.current) {
      globeEl.current.controls().autoRotate = true;
      globeEl.current.controls().autoRotateSpeed = 0.4;
      // Lock center emphasis perfectly on India/Middle East
      globeEl.current.pointOfView({ lat: 20, lng: 75, altitude: 2.2 });
    }
  }, [globeEl.current, activeView]);

  useEffect(() => {
    if (activeView === 'globe' && containerRef.current) {
      const observer = new ResizeObserver(entries => {
        if (entries[0]) {
          setGlobeSize({
            width: entries[0].contentRect.width,
            height: entries[0].contentRect.height
          });
        }
      });
      observer.observe(containerRef.current);
      return () => observer.disconnect();
    }
  }, [activeView]);

  useEffect(() => {
    // Explicit pinned unpkg URL for GeoJSON to ensure CORS resolves and the mesh renders
    fetch('https://unpkg.com/globe.gl@2.30.0/example/datasets/ne_110m_admin_0_countries.geojson')
      .then(res => res.json())
      .then(data => setCountries(data))
      .catch(err => console.error("Failed to load map data.", err));
  }, []);

  return (
    <div className="flex flex-col h-screen text-sm overflow-hidden select-none">
      
      {/* Top Application Bar */}
      <div className="bg-brand-bg/90 border-b border-brand-border h-12 flex justify-between items-center px-4 shrink-0 shadow-sm z-20">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-brand-teal" />
          <span className="text-white font-sans font-medium uppercase tracking-widest text-sm">Supply Chain Matrix</span>
        </div>
        
        {/* Connection Status Indicator */}
        <div className="flex items-center gap-3 bg-brand-card border border-brand-border px-3 py-1.5 rounded pr-4 cursor-default">
          <div className="relative flex h-3 w-3">
            {backendStatus === 'Connected' && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-teal opacity-75"></span>}
            {backendStatus === 'Offline' && <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-brand-red opacity-75"></span>}
            <span className={`relative inline-flex rounded-full h-3 w-3 ${backendStatus === 'Connected' ? 'bg-brand-teal' : backendStatus === 'Offline' ? 'bg-brand-red' : 'bg-gray-500'}`}></span>
          </div>
          <div className="flex flex-col">
            <span className={`text-[10px] font-bold uppercase tracking-widest leading-none mb-0.5 ${backendStatus === 'Connected' ? 'text-brand-teal' : backendStatus === 'Offline' ? 'text-brand-red' : 'text-gray-400'}`}>
              {backendStatus}
            </span>
            <span className="text-[9px] font-mono text-gray-500 leading-none">{API_BASE}</span>
          </div>
        </div>
      </div>

      {/* Alert Strip Top Bar */}
      {sysState.disruptions && sysState.disruptions.length > 0 && (
        <div className="bg-brand-red/90 border-b border-brand-red text-white px-4 py-2 flex items-center gap-3 animate-slide-down shadow-lg shadow-brand-red/20 z-10 shrink-0">
          <AlertOctagon className="w-5 h-5 animate-pulse" />
          <div className="flex-1 font-mono font-medium truncate flex items-center gap-4">
            {sysState.disruptions.map((d, i) => (
              <span key={i} className="flex gap-2 items-center before:content-[''] before:block before:w-1 before:h-4 before:bg-white/50">
                <span className="uppercase text-xs tracking-wider bg-black/30 px-2 py-0.5 rounded italic font-bold">{d.severity} ALERT</span>
                {d.description || `Disruption on ${d.affected_supplier_ids?.join(',')}`}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Main Layout Area */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left Sidebar - Controls */}
        <aside className="w-80 flex flex-col border-r border-brand-border bg-brand-bg/80 backdrop-blur shrink-0 p-4 gap-6 overflow-y-auto custom-scrollbar">
          <div className="space-y-4">
            <h1 className="text-xl font-bold uppercase tracking-widest text-white/90 flex gap-2 items-center">
              <Activity className="text-brand-teal" /> TRIAGE-OPS
            </h1>
            <div className="rounded border border-brand-border bg-brand-card p-3 space-y-2">
              <div className="text-xs uppercase text-gray-400 font-semibold tracking-wider">Connection Source</div>
              <div className="flex gap-2">
                <button
                  className={`flex-1 py-1.5 rounded transition ${!useMock && backendAlive ? 'bg-brand-teal/20 text-brand-teal border-brand-teal/50' : 'bg-transparent text-gray-500 hover:text-white hover:bg-white/5 border-transparent'} border font-mono text-xs`}
                  onClick={() => { setUseMock(false); handleReset(); }}
                >
                  📡 API:8080
                </button>
                <button
                  className={`flex-1 py-1.5 rounded transition ${useMock || !backendAlive ? 'bg-brand-amber/20 text-brand-amber border-brand-amber/50' : 'bg-transparent text-gray-500 hover:text-white hover:bg-white/5 border-transparent'} border font-mono text-xs`}
                  onClick={() => setUseMock(true)}
                >
                  🧬 LOCAL DEMO
                </button>
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs uppercase text-gray-400 font-semibold tracking-wider">Mission Task</div>
              <select
                className="w-full bg-brand-card border border-brand-border text-white text-sm rounded px-3 py-2 outline-none focus:border-brand-teal transition-colors font-mono"
                value={activeTask}
                onChange={e => setActiveTask(e.target.value)}
              >
                {TASKS.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select>
              <div className="text-xs text-gray-500 font-mono leading-relaxed mt-1 p-2 bg-brand-card rounded border border-brand-border border-dashed">
                {sysState.task_description}
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-brand-border">
              <button
                className="w-full bg-transparent border border-gray-600 hover:border-gray-400 text-white rounded py-2 flex justify-center items-center gap-2 hover:bg-white/5 transition-all font-mono uppercase text-xs tracking-wider"
                onClick={handleReset}
              >
                <RefreshCcw className="w-4 h-4" /> Hard Reset
              </button>
              <div className="flex gap-2">
                <button
                  className="flex-1 bg-brand-teal text-black rounded py-2 font-bold flex justify-center items-center gap-2 hover:bg-brand-teal/90 hover:scale-[1.02] shadow-[0_0_15px_rgba(0,212,170,0.3)] transition-all uppercase text-xs tracking-wider"
                  onClick={() => handleStep()}
                  disabled={autoRun}
                >
                  <CheckCircle2 className="w-4 h-4" /> Step Emit
                </button>
                <button
                  className={`px-3 border rounded text-xs uppercase font-mono tracking-widest transition-colors ${autoRun ? 'bg-brand-red/20 text-brand-red border-brand-red font-bold animate-pulse' : 'border-gray-600 text-gray-400 hover:text-white hover:border-gray-400'}`}
                  onClick={() => setAutoRun(!autoRun)}
                >
                  {autoRun ? <Pause className="w-4 h-4" /> : 'AutoRun'}
                </button>
              </div>
            </div>
          </div>

          <div className="space-y-3 pt-4 border-t border-brand-border">
            <div className="flex justify-between items-center text-gray-400 text-xs font-mono uppercase tracking-wider">
              <span>Op Budget</span>
              <span className="text-white">{(sysState.budget_remaining || 0).toLocaleString('en-US', { style: 'currency', currency: 'USD' })}</span>
            </div>
            <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden border border-white/10">
              <div
                className={`h-full transition-all duration-700 ease-out ${((sysState.budget_remaining || 0) / (sysState.total_budget || 1)) < 0.2 ? 'bg-brand-red' : 'bg-brand-teal'}`}
                style={{ width: `${Math.max(0, ((sysState.budget_remaining || 0) / (sysState.total_budget || 1)) * 100)}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-brand-card border border-brand-border rounded p-4 font-mono space-y-4">
            <div className="flex justify-between items-baseline border-b border-brand-border/50 pb-2">
              <span className="text-xs text-gray-500 uppercase">SYS Step Time</span>
              <span className="text-xl text-brand-teal font-light">{sysState.step} / {sysState.max_steps || 12}</span>
            </div>
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-gray-500 uppercase">Total Score</span>
                <span className="text-brand-amber font-bold">{((currentReward.total || 0) * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full mb-2" style={{ height: 80 }}>
                <ResponsiveContainer width="100%" height={80}>
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="colorR" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#aa3bff" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#aa3bff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Tooltip contentStyle={{ backgroundColor: '#161B22', border: '1px solid #30363D' }} />
                    <Area type="monotone" dataKey="reward" stroke="#aa3bff" fillOpacity={1} fill="url(#colorR)" isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 text-xs">
                {['stockout_avoidance', 'cost_efficiency', 'lead_time_score'].map(k => (
                  <div key={k} className="flex justify-between items-center group">
                    <span className="text-gray-500 capitalize">{k.replace(/_/g, ' ')}</span>
                    <div className="w-20 h-1.5 bg-black rounded-full overflow-hidden group-hover:bg-white/10 transition-colors">
                      <div className="h-full bg-gray-300 transition-all duration-500" style={{ width: `${(currentReward.breakdown[k] || 0) * 100}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* Center Panel - Main War Room */}
        <section className="flex-1 flex flex-col min-w-0 border-r border-brand-border bg-brand-bg/50 overflow-hidden relative">
          
          {/* Tab Navigation header */}
          <div className="flex bg-brand-card/30 border-b border-brand-border shrink-0 z-20 shadow-md">
            <button
              onClick={() => setActiveView('globe')}
              className={`flex-1 py-4 text-center font-sans tracking-tight font-bold text-base transition-colors duration-300 relative ${activeView === 'globe' ? 'text-brand-teal' : 'text-gray-500 hover:text-white'}`}
            >
              <div className="flex items-center justify-center gap-2">
                🌍 Global Interactive Map
                {activeView === 'globe' && <div className="absolute bottom-0 left-0 w-full h-[2px] bg-brand-teal shadow-[0_0_10px_rgba(0,212,170,0.8)]"></div>}
              </div>
            </button>
            <div className="w-[1px] bg-brand-border/50"></div>
            <button
              onClick={() => setActiveView('triage')}
              className={`flex-1 py-4 text-center font-sans tracking-tight font-bold text-base transition-colors duration-300 relative ${activeView === 'triage' ? 'text-brand-amber' : 'text-gray-500 hover:text-white'}`}
            >
              <div className="flex items-center justify-center gap-2">
                <Box className="w-5 h-5 pointer-events-none" /> Triage Op Center (POs)
                {activeView === 'triage' && <div className="absolute bottom-0 left-0 w-full h-[2px] bg-brand-amber shadow-[0_0_10px_rgba(255,183,77,0.8)]"></div>}
              </div>
            </button>
          </div>

           {activeView === 'globe' ? (
            /* MASSIVE GLOBE 3D VIEW */
            <div className="flex-1 min-h-0 relative bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#0d1624] via-[#05080c] to-[#000000] overflow-hidden" ref={containerRef}>
              <div className="absolute top-6 left-6 z-10 flex items-center gap-2 pointer-events-none">
                <div className="w-2 h-2 rounded-full bg-brand-teal animate-pulse shadow-[0_0_8px_rgba(0,212,170,1)]"></div>
                <div className="text-white font-mono font-bold text-xs tracking-widest uppercase drop-shadow-lg text-brand-teal/80">Sat-Link Active</div>
              </div>
              <div className="absolute inset-0 cursor-move overflow-hidden flex items-center justify-center">
                {globeSize.width > 0 && (
                  <Globe
                    ref={globeEl}
                    width={globeSize.width}
                    height={globeSize.height}
                    
                    // Core Structure: Transparent Glass Core
                    globeImageUrl="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" // Pure transparent void
                    backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
                    showAtmosphere={true}
                    atmosphereColor="#0055ff" // Deep rich blue base glow
                    atmosphereAltitude={0.4} // Huge bloom radius making it levitate
                    backgroundColor="rgba(0,0,0,0)"

                    // Continent Borders: Laser Sharp Pink Geography
                    polygonsData={countries.features}
                    polygonCapColor={() => 'rgba(0, 0, 0, 0.2)'} // Subtle dark glass interior
                    polygonSideColor={() => 'rgba(0, 0, 0, 0)'}
                    polygonStrokeColor={() => '#ea00d9'} // Unmistakable Magenta outline overriding geography
                    polygonAltitude={0.015}

                    // Network Mesh Layer (Continents replaced with massive pink connectivity dots)
                    hexPolygonsData={countries.features}
                    hexPolygonResolution={4} // Finer grid
                    hexPolygonMargin={0.7} // Very shrunken so they form a dot matrix
                    hexPolygonColor={() => 'rgba(255, 85, 179, 0.85)'} // Intensely bright tracking pink
                    hexPolygonAltitude={0.02}
                    
                    // Cyber Web: Thick Volumetric Floating Lines
                    arcsData={arcsData}
                    arcColor={() => ['rgba(234, 0, 217, 0.6)', 'rgba(0, 229, 255, 1.0)']} // High contrast
                    arcDashLength={0.4}
                    arcDashGap={0.3}
                    arcDashAnimateTime={2500}
                    arcStroke={d => (d.stroke || 0.5) * 1.5} // Thicker, glowing arcs
                    arcAltitudeAutoScale={0.5} // Pushed off the surface to float

                    // Network Nodes & Floating Space Particles
                    labelsData={mergedLabels}
                    labelLat="lat"
                    labelLng="lng"
                    labelText="text"
                    labelSize="size"
                    labelDotRadius={1.0}
                    labelColor="color"
                    labelResolution={4}
                    labelAltitude="altitude"

                    // Disruption Rings
                    ringsData={ringsData}
                    ringColor={() => '#ea00d9'}
                    ringMaxRadius={12}
                    ringPropagationSpeed={2}
                    ringRepeatPeriod={1000}
                    ringResolution={64}
                  />
                )}
              </div>
            </div>
          ) : (
            /* TRIAGE PO DATA VIEW */
            <div className="flex-1 min-h-0 flex flex-col pt-2 animate-fade-in bg-brand-bg/60">
              <div className="px-6 py-4 flex justify-between items-center shrink-0">
                <h2 className="text-xl font-bold font-sans tracking-tight">Active Purchase Orders Tracking</h2>
                <div className="font-mono text-xs text-brand-amber bg-brand-amber/10 px-3 py-1 rounded-full border border-brand-amber/20 shadow-[0_0_10px_rgba(255,183,77,0.1)]">
                  {sysState.pending_orders?.length || 0} Open Tickets
                </div>
              </div>

              <div className="flex-1 overflow-y-auto custom-scrollbar p-6 pt-2 space-y-3">
            {!sysState.pending_orders?.length ? (
              <div className="text-center text-gray-500 font-mono mt-20 opacity-50 flex flex-col items-center">
                <Box className="w-12 h-12 mb-4" />
                No orders localized.
              </div>
            ) : (
              sysState.pending_orders.map((po, idx) => (
                <div key={po.order_id} 
                  className="bg-brand-card border border-brand-border rounded p-4 flex gap-4 transition-all duration-300 transform animate-fade-in items-center hover:bg-white/[0.02]"
                  style={{ animationDelay: `${idx * 50}ms`, animationFillMode: 'both' }}>
                  
                  <div className="w-12 h-12 rounded bg-black/40 border border-brand-border flex items-center justify-center shrink-0">
                    <Truck className="text-gray-400 w-6 h-6" />
                  </div>

                  <div className="flex-1 grid grid-cols-4 gap-4 items-center min-w-0">
                    <div className="col-span-1">
                      <div className="text-white font-mono font-bold truncate">{po.order_id}</div>
                      <div className="text-gray-500 text-xs truncate mt-0.5">{po.sku}</div>
                    </div>
                    
                    <div className="col-span-1 border-l border-brand-border pl-4">
                      <div className="text-xs text-gray-500 uppercase tracking-widest mb-0.5">Vol</div>
                      <div className="font-mono text-gray-300 text-sm">{(po.quantity || 0).toLocaleString()} <span className="text-xs text-gray-600">UNITS</span></div>
                    </div>

                    <div className="col-span-1 border-l border-brand-border pl-4">
                      <div className="text-xs text-gray-500 uppercase tracking-widest mb-0.5">Cost/Unit</div>
                      <div className="font-mono text-gray-300 text-sm">${po.unit_cost?.toFixed(2) || '---'}</div>
                    </div>

                    <div className="col-span-1 border-l border-brand-border pl-4">
                      <div className="text-xs text-gray-500 uppercase tracking-widest mb-0.5">Supplier</div>
                      <div className="font-mono text-gray-300 text-sm truncate" title={po.current_supplier_id}>{po.current_supplier_id || 'unassigned'}</div>
                    </div>
                  </div>

                  {/* Status Chip */}
                  <div className={`px-4 py-1.5 border rounded-full text-xs font-mono font-bold uppercase tracking-widest shrink-0 w-32 text-center transition-colors duration-300 ${getStatusColor(po.status)}`}>
                    {po.status || 'pending'}
                  </div>
                </div>
              ))
            )}
              </div>
            </div>
          )}
        </section>

        {/* Right Panel - Suppliers Directory */}
        <aside className="w-[360px] flex flex-col shrink-0 bg-brand-bg/80">
          <div className="px-6 py-4 border-b border-brand-border shrink-0">
            <h2 className="text-lg font-bold font-sans tracking-tight text-gray-200">Supplier Matrix</h2>
            <p className="text-xs text-gray-500 font-mono mt-1">Vendor Capacity & Risk Graph</p>
          </div>
          <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-4">
            {!sysState.suppliers?.length ? (
              <div className="text-gray-600 font-mono text-xs text-center mt-10 opacity-50">Empty Manifest</div>
            ) : (
              sysState.suppliers.map((sup, idx) => (
                <div key={sup.supplier_id} 
                  onClick={() => onSupplierClick(sup)}
                  className={`relative p-4 rounded border transition-all cursor-${sup.is_disrupted ? 'not-allowed' : 'pointer'} 
                    ${sup.is_disrupted 
                      ? 'border-brand-red/30 bg-brand-red/[0.02]' 
                      : 'border-brand-border bg-brand-card hover:border-gray-500 hover:bg-white/[0.03]'}`}
                >
                  {sup.is_disrupted && (
                    <div className="absolute inset-0 bg-brand-red/10 flex items-center justify-center backdrop-blur-[1px] rounded z-10 px-4">
                      <div className="text-brand-red font-mono font-bold tracking-widest border border-brand-red/50 bg-black/80 px-3 py-1 rounded shadow-lg -rotate-12 border-dashed uppercase text-sm w-full text-center">
                        Disrupted Vendor
                      </div>
                    </div>
                  )}
                  
                  <div className="flex justify-between items-start mb-3 relative z-0">
                    <div>
                      <div className="text-white font-sans font-medium hover:underline decoration-white/20 underline-offset-4">{sup.name}</div>
                      <div className="text-xs text-gray-500 font-mono mt-0.5">{sup.supplier_id}</div>
                    </div>
                    <Factory className={`w-5 h-5 ${sup.is_disrupted ? 'text-brand-red/50' : 'text-gray-500'}`} />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 relative z-0 mb-3">
                    <div className="bg-black/30 rounded p-2 border border-white/5">
                      <div className="flex items-center text-gray-500 text-[10px] uppercase font-bold tracking-wider mb-1">
                        <HandCoins className="w-3 h-3 mr-1" /> Unit Cost
                      </div>
                      <div className="text-white font-mono text-xs">
                        ${sup.cost_per_unit?.toFixed(2) || '0.00'}
                      </div>
                    </div>
                    <div className="bg-black/30 rounded p-2 border border-white/5">
                      <div className="flex items-center text-gray-500 text-[10px] uppercase font-bold tracking-wider mb-1">
                        <Clock className="w-3 h-3 mr-1" /> Lead Time
                      </div>
                      <div className="text-white font-mono text-xs">
                        {sup.lead_time_days || 0} <span className="text-gray-600 text-[10px]">DAYS</span>
                      </div>
                    </div>
                  </div>

                  <div className="relative z-0">
                    <div className="flex justify-between text-[10px] text-gray-500 font-mono mb-1.5 uppercase tracking-widest">
                      <span>Reliability Score</span>
                      <span className="text-white">{(sup.reliability_score || 0.9).toFixed(2)}</span>
                    </div>
                    <div className="w-full h-1 bg-black/50 overflow-hidden relative border-b border-white/5 rounded-full">
                      <div 
                        className={`absolute top-0 left-0 h-full rounded-full ${(sup.reliability_score || 0.9) > 0.8 ? 'bg-brand-teal' : 'bg-brand-amber'}`}
                        style={{ width: `${(sup.reliability_score || 0.9) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>

      </div>

      {/* Reasoning Terminal Bottom Panel */}
      <div className={`mt-auto shrink-0 transition-all duration-500 ease-in-out border-t border-brand-border bg-[#090C10] shadow-[0_-10px_20px_rgba(0,0,0,0.5)] z-20 flex flex-col ${reasoningExpanded ? 'h-64' : 'h-11'}`}>
        <div 
          className="h-11 border-b border-brand-border/50 px-4 flex items-center justify-between cursor-pointer hover:bg-white/[0.02] transition-colors"
          onClick={() => setReasoningExpanded(!reasoningExpanded)}
        >
          <div className="flex items-center gap-3 text-gray-400">
            <Terminal className="w-4 h-4 text-brand-teal" />
            <span className="font-mono text-xs uppercase tracking-widest">Agent Reasoning Output {stagedAction && <span className="text-brand-amber animate-pulse border border-brand-amber px-2 py-0.5 rounded-[3px] ml-2 text-[10px]">Tx Staged</span>}</span>
          </div>
          <button className="text-gray-500 hover:text-white p-1">
            {reasoningExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
        
        {reasoningExpanded && (
          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
            <pre className="text-gray-300 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              <span className="text-gray-600 select-none">agent@triage:~$ cat std.out</span><br/>
              <span className="text-brand-teal/50 select-none">// LOG: Running mission sequence for {activeTask}</span><br/>
              {agentReasoning}
              <span className="inline-block w-2 h-3 bg-brand-teal ml-1 animate-pulse align-baseline"></span>
            </pre>
          </div>
        )}
      </div>

    </div>
  );
}
