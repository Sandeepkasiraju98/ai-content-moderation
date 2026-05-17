import mlflow
import schedule
import time
import threading
from datetime import datetime
from src.scoring.risk_scorer import RiskScorer
from src.mlops.drift_monitor import DriftMonitor


class AutoRetrainer:
    """
    Watches drift reports and automatically
    retrains the risk scorer when drift is detected.
    """

    def __init__(self):
        self.monitor  = DriftMonitor()
        self.scorer   = RiskScorer()
        self.is_running = False

    def check_and_retrain(self):
        print(f"\n[{datetime.utcnow().isoformat()}] Running scheduled check...")

        report = self.monitor.run_check()
        drift  = report.get('drift_analysis', {})

        if drift.get('drift_detected'):
            print("Drift detected — triggering retraining...")
            self._retrain()
        else:
            print("System healthy — no retraining needed.")

        return report

    def _retrain(self):
        print("Starting retraining pipeline...")

        mlflow.set_experiment("auto-retraining")

        with mlflow.start_run(run_name="auto_retrain"):
            mlflow.log_param(
                "trigger", "drift_detected"
            )
            mlflow.log_param(
                "timestamp", datetime.utcnow().isoformat()
            )

            scorer = RiskScorer()
            auc    = scorer.train()
            scorer.save("models/")

            mlflow.log_metric("retrain_auc", auc)
            print(f"Retraining complete. New AUC: {auc:.4f}")

    def start_scheduler(
        self,
        interval_hours: int = 6
    ):
        """Run drift checks every N hours in background."""
        print(f"Starting scheduler — checks every {interval_hours}h")

        schedule.every(interval_hours).hours.do(
            self.check_and_retrain
        )

        def run():
            self.is_running = True
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        print("Scheduler running in background.")

    def stop_scheduler(self):
        self.is_running = False
        print("Scheduler stopped.")