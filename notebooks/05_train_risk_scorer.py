import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow
from src.scoring.risk_scorer import RiskScorer

mlflow.set_experiment("risk-scoring")

scorer = RiskScorer()
auc = scorer.train()
scorer.save("models/")

print(f"\nRisk scorer ready. AUC: {auc:.4f}")