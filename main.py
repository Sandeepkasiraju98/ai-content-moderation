from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import uuid
import json

from src.detection.nlp_detector    import QuickToxicityDetector
from src.detection.vision_detector import CLIPImageDetector
from src.scoring.risk_scorer       import RiskScorer
from src.agents.moderation_agent   import ModerationAgent, TriageEngine
from src.actions.action_handler    import ActionHandler
from src.mlops.drift_monitor       import DriftMonitor
from src.mlops.retrainer           import AutoRetrainer
from src.mlops.dashboard           import DashboardDataBuilder


# ── Startup / shutdown ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything on startup
    print("Booting AI Content Moderation System...")
    app.state.nlp     = QuickToxicityDetector()
    app.state.vision  = CLIPImageDetector()
    app.state.scorer  = RiskScorer()
    app.state.agent   = ModerationAgent(model="mistral")
    app.state.triage  = TriageEngine()
    app.state.actions = ActionHandler()
    app.state.monitor = DriftMonitor()
    app.state.trainer = AutoRetrainer()
    app.state.dashboard = DashboardDataBuilder()

    try:
        app.state.scorer.load("models/")
    except:
        app.state.scorer.train()
        app.state.scorer.save("models/")

    # Start background drift checker (every 6 hours)
    app.state.trainer.start_scheduler(interval_hours=6)

    print("All systems ready.")
    yield

    # Shutdown
    app.state.trainer.stop_scheduler()
    print("System shutdown complete.")


app = FastAPI(
    title="AI Content Moderation System",
    version="5.0",
    lifespan=lifespan
)


# ── Helpers ──

class TextInput(BaseModel):
    text: str
    user_id: str
    user_meta: Optional[dict] = None


def run_pipeline(app, user_id, text=None,
                 image_bytes=None, user_meta=None):
    content_id   = str(uuid.uuid4())[:8]
    text_scores  = None
    image_scores = None

    if text:
        text_scores = app.state.nlp.predict(text)
        text_scores['risk_score'] = app.state.nlp.get_risk_score(text)

    if image_bytes:
        image_scores = app.state.vision.predict(image_bytes)

    risk_result = app.state.scorer.score(
        text_scores=text_scores,
        image_scores=image_scores,
        user_meta=user_meta
    )
    risk_score   = risk_result['final_risk_score']
    route        = app.state.triage.triage(risk_score)
    agent_result = None
    final_action = risk_result['action']

    if route == "agent_review":
        agent_result = app.state.agent.review(
            content_text=text,
            text_scores=text_scores,
            image_scores=image_scores,
            risk_result=risk_result,
            user_meta=user_meta
        )
        final_action = agent_result.get(
            'recommended_action', 'flag_for_review'
        )
    elif route == "auto_remove":
        final_action = "auto_remove"
    else:
        final_action = "approved"

    action_result = app.state.actions.execute(
        user_id=user_id,
        content_id=content_id,
        action=final_action,
        risk_score=risk_score,
        agent_result=agent_result,
        content_preview=text
    )

    return {
        "content_id":   content_id,
        "user_id":      user_id,
        "risk_score":   risk_score,
        "severity":     risk_result['severity'],
        "route":        route,
        "final_action": final_action,
        "agent_result": agent_result,
        "text_scores":  text_scores,
        "image_scores": image_scores
    }


# ── Moderation endpoints ──

@app.get("/")
def root():
    return {
        "status":  "running",
        "version": "5.0",
        "endpoints": [
            "/moderate/text",
            "/moderate/image",
            "/moderate/combined",
            "/health",
            "/dashboard",
            "/audit/logs",
            "/mlops/drift-check",
            "/mlops/retrain"
        ]
    }


@app.post("/moderate/text")
def moderate_text(input: TextInput):
    return run_pipeline(
        app, input.user_id,
        text=input.text,
        user_meta=input.user_meta
    )


@app.post("/moderate/image")
async def moderate_image(
    user_id: str,
    file: UploadFile = File(...)
):
    return run_pipeline(
        app, user_id,
        image_bytes=await file.read()
    )


@app.post("/moderate/combined")
async def moderate_combined(
    user_id: str,
    text: str,
    user_meta: str = "{}",
    file: UploadFile = File(None)
):
    return run_pipeline(
        app, user_id,
        text=text,
        image_bytes=await file.read() if file else None,
        user_meta=json.loads(user_meta)
    )


# ── MLOps endpoints ──

@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "models_loaded": True,
        "scheduler":    app.state.trainer.is_running,
        "timestamp":    __import__('datetime').datetime.utcnow().isoformat()
    }


@app.get("/mlops/drift-check")
def run_drift_check():
    report = app.state.monitor.run_check()
    return report


@app.post("/mlops/retrain")
def manual_retrain():
    app.state.trainer._retrain()
    app.state.scorer.load("models/")
    return {
        "status":    "retrained",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


# ── Dashboard ──

@app.get("/dashboard/data")
def dashboard_data(hours: int = 24):
    return {
        "summary":  app.state.dashboard.build_summary(hours),
        "timeline": app.state.dashboard.build_timeline(hours)
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Content Moderation Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family: system-ui, sans-serif;
           background:#0f1117; color:#e2e8f0; padding:24px; }
    h1 { font-size:22px; font-weight:600; margin-bottom:24px; color:#fff; }
    .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }
    .card { background:#1a1d27; border-radius:12px; padding:20px; }
    .card .label { font-size:12px; color:#94a3b8; margin-bottom:8px; }
    .card .value { font-size:28px; font-weight:700; color:#fff; }
    .card .sub   { font-size:12px; color:#64748b; margin-top:4px; }
    .removed  { border-top:3px solid #ef4444; }
    .flagged  { border-top:3px solid #f59e0b; }
    .approved { border-top:3px solid #22c55e; }
    .agent    { border-top:3px solid #818cf8; }
    .chart-card { background:#1a1d27; border-radius:12px;
                  padding:20px; margin-bottom:16px; }
    .chart-card h3 { font-size:14px; color:#94a3b8;
                     margin-bottom:16px; font-weight:500; }
    .badge { display:inline-block; padding:4px 10px; border-radius:6px;
             font-size:12px; font-weight:500; margin-left:8px; }
    .badge-healthy { background:#14532d; color:#4ade80; }
    .badge-warning { background:#451a03; color:#fb923c; }
    button { background:#3b82f6; color:#fff; border:none;
             padding:8px 16px; border-radius:8px; cursor:pointer;
             font-size:13px; margin-right:8px; }
    button:hover { background:#2563eb; }
  </style>
</head>
<body>
  <h1>
    AI Content Moderation
    <span class="badge badge-healthy" id="status-badge">● Live</span>
  </h1>

  <div class="grid">
    <div class="card">
      <div class="label">Total Processed (24h)</div>
      <div class="value" id="total">—</div>
      <div class="sub">content items</div>
    </div>
    <div class="card removed">
      <div class="label">Auto Removed</div>
      <div class="value" id="removed" style="color:#ef4444">—</div>
      <div class="sub">high risk</div>
    </div>
    <div class="card flagged">
      <div class="label">Flagged for Review</div>
      <div class="value" id="flagged" style="color:#f59e0b">—</div>
      <div class="sub">borderline</div>
    </div>
    <div class="card approved">
      <div class="label">Approved</div>
      <div class="value" id="approved" style="color:#22c55e">—</div>
      <div class="sub">clean content</div>
    </div>
  </div>

  <div class="grid" style="grid-template-columns:repeat(3,1fr)">
    <div class="card agent">
      <div class="label">Agent Reviews</div>
      <div class="value" id="agent" style="color:#818cf8">—</div>
      <div class="sub">LLM borderline reviews</div>
    </div>
    <div class="card">
      <div class="label">Avg Risk Score</div>
      <div class="value" id="avg-risk">—</div>
      <div class="sub">0.0 = safe · 1.0 = harmful</div>
    </div>
    <div class="card">
      <div class="label">Human Review Queue</div>
      <div class="value" id="human" style="color:#fb923c">—</div>
      <div class="sub">needs manual review</div>
    </div>
  </div>

  <div class="chart-card">
    <h3>Moderation Actions — Last 24h</h3>
    <canvas id="timelineChart" height="80"></canvas>
  </div>

  <div style="margin-top:16px">
    <button onclick="refresh()">↻ Refresh</button>
    <button onclick="driftCheck()"
            style="background:#7c3aed">Run Drift Check</button>
    <button onclick="manualRetrain()"
            style="background:#065f46">Force Retrain</button>
  </div>

  <div id="drift-result" style="margin-top:16px;
       font-size:13px; color:#94a3b8;"></div>

<script>
let chart = null;

async function refresh() {
  const res  = await fetch('/dashboard/data?hours=24');
  const data = await res.json();
  const s    = data.summary;

  document.getElementById('total').textContent =
    s.total || 0;
  document.getElementById('removed').textContent =
    s.actions?.auto_remove || 0;
  document.getElementById('flagged').textContent =
    s.actions?.flag_for_review || 0;
  document.getElementById('approved').textContent =
    s.actions?.approved || 0;
  document.getElementById('agent').textContent =
    s.agent_reviews || 0;
  document.getElementById('avg-risk').textContent =
    s.avg_risk_score || '0.000';
  document.getElementById('human').textContent =
    s.human_reviews_needed || 0;

  // Timeline chart
  const timeline = data.timeline;
  const labels   = timeline.map(t => t.time.slice(11,16));
  const removed  = timeline.map(t => t.auto_remove || 0);
  const flagged  = timeline.map(t => t.flag_for_review || 0);
  const approved = timeline.map(t => t.approved || 0);

  if (chart) chart.destroy();
  chart = new Chart(
    document.getElementById('timelineChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label:'Removed',  data:removed,
          backgroundColor:'#ef4444' },
        { label:'Flagged',  data:flagged,
          backgroundColor:'#f59e0b' },
        { label:'Approved', data:approved,
          backgroundColor:'#22c55e' }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { stacked:true,
             ticks:{ color:'#94a3b8' },
             grid:{ color:'#1e293b' } },
        y: { stacked:true,
             ticks:{ color:'#94a3b8' },
             grid:{ color:'#1e293b' } }
      },
      plugins: {
        legend:{ labels:{ color:'#94a3b8' } }
      }
    }
  });
}

async function driftCheck() {
  document.getElementById('drift-result').textContent =
    'Running drift check...';
  const res  = await fetch('/mlops/drift-check');
  const data = await res.json();
  const d    = data.drift_analysis;
  document.getElementById('drift-result').textContent =
    d.drift_detected
      ? `⚠️ Drift detected — mean shift: ${d.mean_shift}`
      : `✓ No drift detected — system healthy`;
}

async function manualRetrain() {
  document.getElementById('drift-result').textContent =
    'Retraining in progress...';
  const res  = await fetch('/mlops/retrain', {method:'POST'});
  const data = await res.json();
  document.getElementById('drift-result').textContent =
    `✓ Retrained at ${data.timestamp}`;
}

// Auto refresh every 30 seconds
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


# ── Audit ──

@app.get("/audit/logs")
def get_logs(n: int = 20):
    logs = app.state.actions.get_recent_logs(n)
    return {"total": len(logs), "logs": logs}