import json
import os
from datetime import datetime, timedelta
from collections import defaultdict


class DashboardDataBuilder:
    """
    Builds data for the monitoring dashboard
    from the audit log.
    """

    def __init__(self, log_path: str = "data/audit_log.jsonl"):
        self.log_path = log_path

    def load_logs(self, hours: int = 24) -> list:
        if not os.path.exists(self.log_path):
            return []

        records = []
        cutoff  = datetime.utcnow() - timedelta(hours=hours)

        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    r  = json.loads(line.strip())
                    ts = datetime.fromisoformat(
                        r.get('timestamp', '')
                    )
                    if ts >= cutoff:
                        records.append(r)
                except:
                    continue

        return records

    def build_summary(self, hours: int = 24) -> dict:
        logs = self.load_logs(hours)

        if not logs:
            return {
                "total": 0,
                "actions": {},
                "avg_risk_score": 0,
                "hours": hours
            }

        action_counts = defaultdict(int)
        risk_scores   = []

        for r in logs:
            action_counts[r.get('action', 'unknown')] += 1
            if 'risk_score' in r:
                risk_scores.append(r['risk_score'])

        return {
            "total":          len(logs),
            "hours":          hours,
            "actions":        dict(action_counts),
            "avg_risk_score": round(
                sum(risk_scores) / len(risk_scores), 4
            ) if risk_scores else 0,
            "max_risk_score": round(max(risk_scores), 4)
                              if risk_scores else 0,
            "agent_reviews":  sum(
                1 for r in logs if r.get('agent_verdict')
            ),
            "human_reviews_needed": sum(
                1 for r in logs if r.get('human_review')
            ),
            "generated_at":   datetime.utcnow().isoformat()
        }

    def build_timeline(self, hours: int = 24) -> list:
        """Bucket decisions into hourly slots for charting."""
        logs    = self.load_logs(hours)
        buckets = defaultdict(lambda: defaultdict(int))

        for r in logs:
            try:
                ts     = datetime.fromisoformat(r['timestamp'])
                bucket = ts.strftime('%Y-%m-%d %H:00')
                action = r.get('action', 'unknown')
                buckets[bucket][action] += 1
            except:
                continue

        return [
            {"time": t, **counts}
            for t, counts in sorted(buckets.items())
        ]