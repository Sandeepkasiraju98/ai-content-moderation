import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import mlflow
import mlflow.sklearn
import pickle
import os
from src.scoring.feature_engineer import FeatureEngineer


class RiskScorer:
    """
    XGBoost + LightGBM ensemble that takes all
    detector signals and outputs a final risk score.
    """

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.xgb_model = None
        self.lgb_model = None
        self.is_trained = False
        self.feature_columns = None

    def generate_training_data(self, n_samples: int = 10000) -> pd.DataFrame:
        np.random.seed(42)
        rows = []

        for _ in range(n_samples):
            is_harmful = np.random.binomial(1, 0.3)

            if is_harmful:
                # Simulate what Detoxify actually outputs for harmful content
                toxicity        = np.clip(np.random.uniform(0.6, 1.0), 0, 1)
                severe_toxicity = np.clip(np.random.uniform(0.0, 0.5), 0, 1)
                obscene         = np.clip(np.random.uniform(0.0, 0.4), 0, 1)
                threat          = np.clip(np.random.uniform(0.4, 1.0), 0, 1)
                insult          = np.clip(np.random.uniform(0.0, 0.5), 0, 1)
                identity_attack = np.clip(np.random.uniform(0.0, 0.3), 0, 1)

                image_risk      = np.clip(np.random.uniform(0.0, 0.9), 0, 1)
                image_harmful   = image_risk
                image_safe      = 1 - image_risk
                image_label     = 'harmful' if image_risk > 0.4 else 'safe'

                account_age     = int(np.random.uniform(1, 200))
                prev_violations = int(np.random.poisson(2.5))
                reports         = int(np.random.poisson(3))
                posts_24h       = int(np.random.uniform(10, 100))
                is_verified     = False

            else:
                # Simulate what Detoxify actually outputs for safe content
                toxicity        = np.clip(np.random.uniform(0.0, 0.3), 0, 1)
                severe_toxicity = np.clip(np.random.uniform(0.0, 0.05), 0, 1)
                obscene         = np.clip(np.random.uniform(0.0, 0.1), 0, 1)
                threat          = np.clip(np.random.uniform(0.0, 0.15), 0, 1)
                insult          = np.clip(np.random.uniform(0.0, 0.2), 0, 1)
                identity_attack = np.clip(np.random.uniform(0.0, 0.05), 0, 1)

                image_risk      = np.clip(np.random.uniform(0.0, 0.3), 0, 1)
                image_harmful   = image_risk
                image_safe      = 1 - image_risk
                image_label     = 'safe'

                account_age     = int(np.random.uniform(100, 2000))
                prev_violations = int(np.random.poisson(0.1))
                reports         = int(np.random.poisson(0.2))
                posts_24h       = int(np.random.uniform(1, 20))
                is_verified     = bool(np.random.binomial(1, 0.4))

            text_scores = {
                'toxicity':        round(toxicity, 4),
                'severe_toxicity': round(severe_toxicity, 4),
                'obscene':         round(obscene, 4),
                'threat':          round(threat, 4),
                'insult':          round(insult, 4),
                'identity_attack': round(identity_attack, 4),
                'risk_score': round(
                    toxicity * 0.4 +
                    severe_toxicity * 0.2 +
                    obscene * 0.1 +
                    threat * 0.15 +
                    insult * 0.1 +
                    identity_attack * 0.05, 4
                )
            }

            image_scores = {
                'risk_score':    round(image_risk, 4),
                'harmful_score': round(image_harmful, 4),
                'safe_score':    round(image_safe, 4),
                'label':         image_label
            }

            user_meta = {
                'account_age_days':  account_age,
                'prev_violations':   prev_violations,
                'reports_received':  reports,
                'posts_last_24h':    posts_24h,
                'is_verified':       is_verified
            }

            features = self.engineer.build_features(
                text_scores, image_scores, user_meta
            )
            features['label'] = is_harmful
            rows.append(features)

        return pd.DataFrame(rows)

    def train(self, df: pd.DataFrame = None):
        if df is None:
            print("Generating synthetic training data...")
            df = self.generate_training_data(10000)
            print(f"Generated {len(df)} samples.")

        feature_cols = [c for c in df.columns if c != 'label']
        self.feature_columns = feature_cols

        X = df[feature_cols]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training on {len(X_train)} samples...")

        with mlflow.start_run(run_name="xgb_lgb_risk_scorer"):

            # ── XGBoost ──
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='auc',
                random_state=42
            )
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # ── LightGBM ──
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            self.lgb_model.fit(X_train, y_train)

            # ── Ensemble: average both ──
            xgb_probs = self.xgb_model.predict_proba(X_test)[:, 1]
            lgb_probs = self.lgb_model.predict_proba(X_test)[:, 1]
            ensemble_probs = (xgb_probs + lgb_probs) / 2
            ensemble_preds = (ensemble_probs > 0.5).astype(int)

            auc = roc_auc_score(y_test, ensemble_probs)
            report = classification_report(
                y_test, ensemble_preds,
                target_names=['Safe', 'Harmful'],
                output_dict=True
            )

            mlflow.log_metric("ensemble_auc", auc)
            mlflow.log_metric("precision", report['Harmful']['precision'])
            mlflow.log_metric("recall",    report['Harmful']['recall'])
            mlflow.log_metric("f1",        report['Harmful']['f1-score'])
            mlflow.log_param("xgb_estimators", 200)
            mlflow.log_param("lgb_estimators", 200)

            print(f"\nEnsemble AUC: {auc:.4f}")
            print(classification_report(
                y_test, ensemble_preds,
                target_names=['Safe', 'Harmful']
            ))

            # Feature importance
            importance = pd.Series(
                self.xgb_model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)
            print("\nTop 10 most important features:")
            print(importance.head(10))

        self.is_trained = True
        return auc

    def save(self, path: str = "models/"):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}xgb_risk_scorer.pkl", 'wb') as f:
            pickle.dump(self.xgb_model, f)
        with open(f"{path}lgb_risk_scorer.pkl", 'wb') as f:
            pickle.dump(self.lgb_model, f)
        with open(f"{path}feature_columns.pkl", 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Risk scorer saved to {path}")

    def load(self, path: str = "models/"):
        with open(f"{path}xgb_risk_scorer.pkl", 'rb') as f:
            self.xgb_model = pickle.load(f)
        with open(f"{path}lgb_risk_scorer.pkl", 'rb') as f:
            self.lgb_model = pickle.load(f)
        with open(f"{path}feature_columns.pkl", 'rb') as f:
            self.feature_columns = pickle.load(f)
        self.is_trained = True
        print("Risk scorer loaded.")

    def score(
        self,
        text_scores: dict = None,
        image_scores: dict = None,
        user_meta: dict = None
    ) -> dict:
        if not self.is_trained:
            raise Exception("Model not trained. Call train() or load() first.")

        features = self.engineer.build_features(
            text_scores, image_scores, user_meta
        )
        df = pd.DataFrame([features])[self.feature_columns]

        xgb_prob = self.xgb_model.predict_proba(df)[0][1]
        lgb_prob = self.lgb_model.predict_proba(df)[0][1]
        final_score = round((xgb_prob + lgb_prob) / 2, 4)

        if final_score >= 0.8:
            action = "auto_remove"
            severity = "critical"
        elif final_score >= 0.6:
            action = "flag_for_review"
            severity = "high"
        elif final_score >= 0.4:
            action = "warn_user"
            severity = "medium"
        else:
            action = "approved"
            severity = "low"

        return {
            "final_risk_score": final_score,
            "xgb_score":        round(float(xgb_prob), 4),
            "lgb_score":        round(float(lgb_prob), 4),
            "severity":         severity,
            "action":           action,
            "features_used":    features
        }