import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import mlflow


class DriftMonitor:
    """
    Monitors model input distributions and
    performance metrics over time.
    Triggers retraining when drift is detected.
    """

    def __init__(
        self,
        log_path: str = "data/audit_log.jsonl",
        report_path: str = "data/drift_reports/",
        drift_threshold: float = 0.15
    ):
        self.log_path       = log_path
        self.report_path    = report_path
        self.drift_threshold = drift_threshold
        os.makedirs(report_path, exist_ok=True)

    def load_recent_predictions(
        self, hours: int = 24
    ) -> pd.DataFrame:
        """Load predictions from the last N hours."""
        if not os.path.exists(self.log_path):
            return pd.DataFrame()

        records = []
        cutoff  = datetime.utcnow() - timedelta(hours=hours)

        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    ts = datetime.fromisoformat(
                        record.get('timestamp', '')
                    )
                    if ts >= cutoff:
                        records.append(record)
                except:
                    continue

        return pd.DataFrame(records) if records else pd.DataFrame()

    def compute_score_stats(self, df: pd.DataFrame) -> dict:
        if df.empty or 'risk_score' not in df.columns:
            return {}

        scores = df['risk_score'].dropna()
        return {
            "count":        len(scores),
            "mean":         round(scores.mean(), 4),
            "std":          round(scores.std(), 4),
            "min":          round(scores.min(), 4),
            "max":          round(scores.max(), 4),
            "p50":          round(scores.quantile(0.5), 4),
            "p90":          round(scores.quantile(0.9), 4),
            "p99":          round(scores.quantile(0.99), 4),
            "pct_removed":  round(
                (df['action'] == 'auto_remove').mean() * 100, 2
            ) if 'action' in df.columns else 0,
            "pct_flagged":  round(
                (df['action'] == 'flag_for_review').mean() * 100, 2
            ) if 'action' in df.columns else 0,
            "pct_approved": round(
                (df['action'] == 'approved').mean() * 100, 2
            ) if 'action' in df.columns else 0,
        }

    def detect_drift(
        self,
        reference_stats: dict,
        current_stats: dict
    ) -> dict:
        """
        Simple drift detection: compare mean risk score
        between reference window and current window.
        """
        if not reference_stats or not current_stats:
            return {"drift_detected": False, "reason": "insufficient data"}

        mean_shift = abs(
            current_stats.get('mean', 0) -
            reference_stats.get('mean', 0)
        )
        std_shift = abs(
            current_stats.get('std', 0) -
            reference_stats.get('std', 0)
        )

        drift_detected = mean_shift > self.drift_threshold

        return {
            "drift_detected":   drift_detected,
            "mean_shift":       round(mean_shift, 4),
            "std_shift":        round(std_shift, 4),
            "threshold":        self.drift_threshold,
            "recommendation":   "retrain" if drift_detected else "monitor",
            "checked_at":       datetime.utcnow().isoformat()
        }

    def run_check(self) -> dict:
        print("Running drift check...")

        # Compare last 24h vs previous 24h
        recent   = self.load_recent_predictions(hours=24)
        previous = self.load_recent_predictions(hours=48)

        recent_stats   = self.compute_score_stats(recent)
        previous_stats = self.compute_score_stats(previous)

        drift_result = self.detect_drift(previous_stats, recent_stats)

        report = {
            "timestamp":       datetime.utcnow().isoformat(),
            "recent_stats":    recent_stats,
            "previous_stats":  previous_stats,
            "drift_analysis":  drift_result,
            "total_processed": recent_stats.get('count', 0)
        }

        # Save report
        report_file = os.path.join(
            self.report_path,
            f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Log to MLflow
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_metric(
                "mean_risk_score",
                recent_stats.get('mean', 0)
            )
            mlflow.log_metric(
                "pct_removed",
                recent_stats.get('pct_removed', 0)
            )
            mlflow.log_metric(
                "drift_detected",
                int(drift_result.get('drift_detected', False))
            )

        if drift_result['drift_detected']:
            print(f"DRIFT DETECTED — mean shift: {drift_result['mean_shift']}")
        else:
            print("No drift detected. System healthy.")

        return report