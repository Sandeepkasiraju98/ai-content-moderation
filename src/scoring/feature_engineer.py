import pandas as pd
import numpy as np
from datetime import datetime


class FeatureEngineer:
    """
    Takes raw signals from all detectors and
    engineers a feature vector for the risk classifier.
    """

    def build_features(
        self,
        text_scores: dict = None,
        image_scores: dict = None,
        user_meta: dict = None
    ) -> dict:
        features = {}

        # ── Text features ──
        if text_scores:
            features['text_toxicity']        = text_scores.get('toxicity', 0)
            features['text_severe_toxicity'] = text_scores.get('severe_toxicity', 0)
            features['text_obscene']         = text_scores.get('obscene', 0)
            features['text_threat']          = text_scores.get('threat', 0)
            features['text_insult']          = text_scores.get('insult', 0)
            features['text_identity_attack'] = text_scores.get('identity_attack', 0)
            features['text_risk_score']      = text_scores.get('risk_score', 0)
            # Derived
            features['text_max_score'] = max(
                features['text_toxicity'],
                features['text_severe_toxicity'],
                features['text_threat']
            )
            features['text_any_high'] = int(
                any(v > 0.7 for v in [
                    features['text_toxicity'],
                    features['text_severe_toxicity'],
                    features['text_threat']
                ])
            )
        else:
            # Fill zeros if no text
            for key in ['text_toxicity','text_severe_toxicity','text_obscene',
                        'text_threat','text_insult','text_identity_attack',
                        'text_risk_score','text_max_score','text_any_high']:
                features[key] = 0

        # ── Image features ──
        if image_scores:
            features['image_risk_score']    = image_scores.get('risk_score', 0)
            features['image_harmful_score'] = image_scores.get('harmful_score', 0)
            features['image_safe_score']    = image_scores.get('safe_score', 1)
            features['image_is_harmful']    = int(
                image_scores.get('label', 'safe') != 'safe'
            )
        else:
            features['image_risk_score']    = 0
            features['image_harmful_score'] = 0
            features['image_safe_score']    = 1
            features['image_is_harmful']    = 0

        # ── User behavior features ──
        if user_meta:
            features['user_account_age_days']    = user_meta.get('account_age_days', 365)
            features['user_prev_violations']     = user_meta.get('prev_violations', 0)
            features['user_reports_received']    = user_meta.get('reports_received', 0)
            features['user_posts_last_24h']      = user_meta.get('posts_last_24h', 1)
            features['user_is_verified']         = int(user_meta.get('is_verified', False))
            # Derived — velocity signal
            features['user_high_velocity']       = int(
                user_meta.get('posts_last_24h', 1) > 50
            )
            features['user_repeat_offender']     = int(
                user_meta.get('prev_violations', 0) > 2
            )
        else:
            features['user_account_age_days']  = 365
            features['user_prev_violations']   = 0
            features['user_reports_received']  = 0
            features['user_posts_last_24h']    = 1
            features['user_is_verified']       = 0
            features['user_high_velocity']     = 0
            features['user_repeat_offender']   = 0

        # ── Cross-modal features ──
        # Both text AND image flagged = much higher risk
        features['both_modalities_flagged'] = int(
            features['text_risk_score'] > 0.5 and
            features['image_risk_score'] > 0.5
        )
        features['combined_raw_score'] = round(
            features['text_risk_score'] * 0.5 +
            features['image_risk_score'] * 0.3 +
            min(features['user_prev_violations'] * 0.05, 0.2),
            4
        )

        return features

    def to_dataframe(self, features: dict) -> pd.DataFrame:
        return pd.DataFrame([features])