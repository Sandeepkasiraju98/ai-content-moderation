# AI Content Moderation System

Production-grade multi-modal content moderation system built with Python, PyTorch, XGBoost, LightGBM, and LLM agents.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## What it does

Automatically detects and moderates harmful content across:

- **Text** — toxicity, hate speech, threats, insults
- **Images** — offensive and NSFW content
- **Combined** — text + image together with user context

---

## System Architecture

```
Content Input
     │
     ▼
┌─────────────────────────────┐
│       DETECTION LAYER       │
│  NLP  (Detoxify + TF-IDF)  │
│  Vision (CLIP + ResNet18)   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│     RISK SCORING LAYER      │
│   XGBoost + LightGBM        │
│   Ensemble meta-classifier  │
└────────────┬────────────────┘
             │
        ┌────┴────┐
        │ Triage  │
        └────┬────┘
   ┌─────────┼─────────┐
   ▼         ▼         ▼
Auto      Agent      Auto
Remove    Review    Approve
         (LLM)
             │
             ▼
┌─────────────────────────────┐
│    ACTION + AUDIT LAYER     │
│  Remove · Flag · Warn · Log │
└─────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│        MLOPS LAYER          │
│  Drift monitor · Retrain    │
│  MLflow · Dashboard         │
└─────────────────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| NLP | Detoxify, HuggingFace, Scikit-learn, TF-IDF |
| Vision | PyTorch, CLIP, ResNet18, torchvision |
| Risk Scoring | XGBoost, LightGBM, feature engineering |
| LLM Agent | Ollama, Mistral (local, free) |
| API | FastAPI, Uvicorn |
| MLOps | MLflow, Evidently, APScheduler |
| Infra | Docker, Docker Compose |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-content-moderation.git
cd ai-content-moderation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama (for LLM agent)

Download from [ollama.com](https://ollama.com) then pull the model:

```bash
ollama pull mistral
```

### 3. Run the system

```bash
# Terminal 1 — API
uvicorn main:app --reload

# Terminal 2 — MLflow
mlflow ui --port 5000
```

### 4. Or run with Docker

```bash
docker-compose up --build
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/moderate/text` | POST | Moderate text content |
| `/moderate/image` | POST | Moderate image content |
| `/moderate/combined` | POST | Moderate text + image |
| `/dashboard` | GET | Live monitoring dashboard |
| `/health` | GET | System health check |
| `/audit/logs` | GET | Full audit trail |
| `/mlops/drift-check` | GET | Run drift detection |
| `/mlops/retrain` | POST | Trigger retraining |

---

## Example Response

```json
{
  "content_id": "a3f1c9b2",
  "risk_score": 0.87,
  "severity": "critical",
  "route": "auto_remove",
  "final_action": "auto_remove",
  "agent_result": null,
  "text_scores": {
    "toxicity": 0.91,
    "threat": 0.76,
    "risk_score": 0.87
  }
}
```

---

## Project Structure

```
content-moderation-ai/
├── src/
│   ├── detection/
│   │   ├── nlp_detector.py       # NLP toxicity models
│   │   └── vision_detector.py    # CLIP + ResNet image models
│   ├── scoring/
│   │   ├── feature_engineer.py   # Feature engineering
│   │   └── risk_scorer.py        # XGBoost + LightGBM ensemble
│   ├── agents/
│   │   └── moderation_agent.py   # LLM triage agent
│   ├── actions/
│   │   └── action_handler.py     # Actions + audit logging
│   └── mlops/
│       ├── drift_monitor.py      # Drift detection
│       ├── retrainer.py          # Auto retraining
│       └── dashboard.py          # Dashboard data
├── notebooks/                    # Training notebooks
├── models/                       # Saved model files
├── data/                         # Datasets and logs
├── main.py                       # FastAPI app entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Pipeline Phases

| Phase | What was built |
|-------|---------------|
| ✅ Phase 1 | NLP toxicity detector — Detoxify + TF-IDF + Logistic Regression |
| ✅ Phase 2 | Image classifier — CLIP (zero-shot) + fine-tuned ResNet18 |
| ✅ Phase 3 | Risk scorer — XGBoost + LightGBM ensemble meta-classifier |
| ✅ Phase 4 | LLM triage agent — Mistral via Ollama + audit logging |
| ✅ Phase 5 | MLOps — drift monitoring + auto retraining + live dashboard + Docker |

---

## Live Dashboard

Visit `http://localhost:8000/dashboard` after starting the server.

Displays real-time:
- Total content processed
- Actions breakdown (removed / flagged / approved)
- Hourly timeline chart
- Agent review count
- Human review queue
- Drift check + manual retrain buttons

---

## Author

**Sandeep Kasiraju**
