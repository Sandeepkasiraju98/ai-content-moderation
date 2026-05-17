import ollama
import json
from datetime import datetime


# ── System prompt — this defines how the agent thinks ──

SYSTEM_PROMPT = """You are a senior content moderation AI agent.

Your job is to review content that automated systems flagged as borderline — 
not clearly safe, not clearly harmful. You make the final call.

You will receive:
- The original content (text and/or image description)
- Scores from multiple ML detectors
- User account history and behavior signals

Your reasoning process:
1. Read the content carefully in full context
2. Review what the detectors found and how confident they are
3. Consider user history — repeat offenders get less benefit of the doubt
4. Think about intent — is this clearly harmful or potentially misunderstood?
5. Make a final decision

You must respond ONLY with a valid JSON object in this exact format:
{
  "verdict": "approve" | "flag" | "remove",
  "confidence": 0.0 to 1.0,
  "severity": "none" | "low" | "medium" | "high" | "critical",
  "reasoning": "Clear explanation of your decision in 2-3 sentences",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "recommended_action": "approved" | "warn_user" | "flag_for_review" | "auto_remove",
  "needs_human_review": true | false
}

Be firm but fair. Context matters. Explain your reasoning clearly."""


class ModerationAgent:
    """
    LLM-powered moderation agent for borderline content.
    Uses Ollama (free, local) with Mistral by default.
    """

    def __init__(self, model: str = "mistral"):
        self.model = model
        self.conversation_history = []
        print(f"Moderation agent initialized with model: {model}")

    def _build_review_prompt(
        self,
        content_text: str = None,
        image_description: str = None,
        text_scores: dict = None,
        image_scores: dict = None,
        risk_result: dict = None,
        user_meta: dict = None
    ) -> str:

        prompt_parts = ["## Content Under Review\n"]

        # Content
        if content_text:
            prompt_parts.append(f"**Text content:**\n\"{content_text}\"\n")
        if image_description:
            prompt_parts.append(f"**Image description:** {image_description}\n")

        # Detector scores
        prompt_parts.append("\n## Detector Scores\n")

        if text_scores:
            prompt_parts.append("**NLP Detector:**")
            prompt_parts.append(
                f"- Toxicity: {text_scores.get('toxicity', 0):.3f}"
            )
            prompt_parts.append(
                f"- Severe toxicity: {text_scores.get('severe_toxicity', 0):.3f}"
            )
            prompt_parts.append(
                f"- Threat: {text_scores.get('threat', 0):.3f}"
            )
            prompt_parts.append(
                f"- Insult: {text_scores.get('insult', 0):.3f}"
            )
            prompt_parts.append(
                f"- Identity attack: {text_scores.get('identity_attack', 0):.3f}"
            )
            prompt_parts.append(
                f"- Overall text risk: {text_scores.get('risk_score', 0):.3f}\n"
            )

        if image_scores:
            prompt_parts.append("**Vision Detector:**")
            prompt_parts.append(
                f"- Image risk score: {image_scores.get('risk_score', 0):.3f}"
            )
            prompt_parts.append(
                f"- Harmful score: {image_scores.get('harmful_score', 0):.3f}"
            )
            prompt_parts.append(
                f"- Label: {image_scores.get('label', 'unknown')}\n"
            )

        if risk_result:
            prompt_parts.append("**Risk Scorer (XGBoost + LightGBM ensemble):**")
            prompt_parts.append(
                f"- Final risk score: {risk_result.get('final_risk_score', 0):.3f}"
            )
            prompt_parts.append(
                f"- Severity: {risk_result.get('severity', 'unknown')}"
            )
            prompt_parts.append(
                f"- Initial action: {risk_result.get('action', 'unknown')}\n"
            )

        # User history
        prompt_parts.append("\n## User History\n")
        if user_meta:
            prompt_parts.append(
                f"- Account age: {user_meta.get('account_age_days', 'unknown')} days"
            )
            prompt_parts.append(
                f"- Previous violations: {user_meta.get('prev_violations', 0)}"
            )
            prompt_parts.append(
                f"- Reports received: {user_meta.get('reports_received', 0)}"
            )
            prompt_parts.append(
                f"- Posts in last 24h: {user_meta.get('posts_last_24h', 0)}"
            )
            prompt_parts.append(
                f"- Verified account: {user_meta.get('is_verified', False)}"
            )
        else:
            prompt_parts.append("- No user history available")

        prompt_parts.append(
            "\n## Your Task\n"
            "This content scored in the borderline range (0.4–0.7). "
            "The automated system was not confident enough to act automatically. "
            "Review all signals above and make the final moderation decision. "
            "Respond ONLY with the JSON format specified."
        )

        return "\n".join(prompt_parts)

    def review(
        self,
        content_text: str = None,
        image_description: str = None,
        text_scores: dict = None,
        image_scores: dict = None,
        risk_result: dict = None,
        user_meta: dict = None
    ) -> dict:

        prompt = self._build_review_prompt(
            content_text=content_text,
            image_description=image_description,
            text_scores=text_scores,
            image_scores=image_scores,
            risk_result=risk_result,
            user_meta=user_meta
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ]
            )

            raw = response['message']['content'].strip()

            # Clean up response — strip markdown if model adds it
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            result = json.loads(raw)
            result['agent_model']   = self.model
            result['reviewed_at']   = datetime.utcnow().isoformat()
            result['review_status'] = "completed"
            return result

        except json.JSONDecodeError:
            # Fallback if LLM doesn't return clean JSON
            return {
                "verdict":            "flag",
                "confidence":         0.5,
                "severity":           "medium",
                "reasoning":          "Agent returned malformed response. Flagging for human review as a precaution.",
                "key_factors":        ["agent_parse_error"],
                "recommended_action": "flag_for_review",
                "needs_human_review": True,
                "review_status":      "parse_error",
                "reviewed_at":        datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "verdict":            "flag",
                "confidence":         0.0,
                "severity":           "unknown",
                "reasoning":          f"Agent error: {str(e)}",
                "key_factors":        ["agent_error"],
                "recommended_action": "flag_for_review",
                "needs_human_review": True,
                "review_status":      "error",
                "reviewed_at":        datetime.utcnow().isoformat()
            }


# ── Triage logic — decides if agent review is needed ──

class TriageEngine:
    """
    Routes content to the right handler based on risk score.
    Saves LLM calls for borderline cases only.
    """

    def __init__(
        self,
        auto_remove_threshold: float = 0.8,
        auto_approve_threshold: float = 0.35,
        agent_low: float = 0.35,
        agent_high: float = 0.8
    ):
        self.auto_remove_threshold  = auto_remove_threshold
        self.auto_approve_threshold = auto_approve_threshold
        self.agent_low  = agent_low
        self.agent_high = agent_high

    def triage(self, risk_score: float) -> str:
        if risk_score >= self.auto_remove_threshold:
            return "auto_remove"
        elif risk_score <= self.auto_approve_threshold:
            return "auto_approve"
        else:
            return "agent_review"