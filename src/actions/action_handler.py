from datetime import datetime
import json
import os


class ActionHandler:
    """
    Executes moderation actions and logs every
    decision to an audit trail.
    """

    def __init__(self, log_path: str = "data/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def execute(
        self,
        user_id: str,
        content_id: str,
        action: str,
        risk_score: float,
        agent_result: dict = None, 
        content_preview: str = None 
    ) -> dict:

        timestamp = datetime.utcnow().isoformat()

        result = {
            "content_id":      content_id,
            "user_id":         user_id,
            "action":          action,
            "risk_score":      risk_score,
            "timestamp":       timestamp,
            "content_preview": content_preview[:100] if content_preview else None
        }

        # Execute action
        if action == "auto_remove":
            result["status"]  = "removed"
            result["message"] = "Content automatically removed — high risk score"

        elif action == "flag_for_review":
            result["status"]  = "flagged"
            result["message"] = "Content flagged — queued for human review"

        elif action == "warn_user":
            result["status"]  = "warned"
            result["message"] = "User warned — content kept with warning label"

        elif action == "approved":
            result["status"]  = "approved"
            result["message"] = "Content approved — passed moderation"

        else:
            result["status"]  = "unknown"
            result["message"] = f"Unknown action: {action}"

        # Add agent reasoning if available
        if agent_result:
            result["agent_verdict"]   = agent_result.get("verdict")
            result["agent_reasoning"] = agent_result.get("reasoning")
            result["agent_factors"]   = agent_result.get("key_factors", [])
            result["human_review"]    = agent_result.get("needs_human_review", False)

        # Write to audit log
        self._log(result)

        return result

    def _log(self, record: dict):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def get_recent_logs(self, n: int = 20) -> list:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]