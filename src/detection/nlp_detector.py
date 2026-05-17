import pandas as pd
import numpy as np
from detoxify import Detoxify
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
import pickle
import os
import warnings


# ── 1. Quick detection using Detoxify (pretrained, zero training needed) ──

class QuickToxicityDetector:
    """
    Uses a pretrained model — no training required.
    Good for getting results immediately.

    Note: Detoxify returns 'identity_attack'; the Jigsaw CSV uses 'identity_hate'.
    These refer to the same concept but have different key names across classes.
    """
    def __init__(self):
        print("Loading Detoxify model...")
        self.model = Detoxify('original')
        print("Model loaded.")

    def predict(self, text: str) -> dict:
        # FIX: validate input to avoid silent failures on None or non-string input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        results = self.model.predict(text)
        # Convert numpy floats to Python floats
        return {k: round(float(v), 4) for k, v in results.items()}

    def get_risk_score(self, text: str) -> float:
        results = self.predict(text)
        # Weighted risk score — weights sum to 1.0
        score = (
            results['toxicity'] * 0.4 +
            results['severe_toxicity'] * 0.2 +
            results['obscene'] * 0.1 +
            results['threat'] * 0.15 +
            results['insult'] * 0.1 +
            results['identity_attack'] * 0.05
        )
        return round(score, 4)


# ── 2. Custom TF-IDF + Logistic Regression (trainable on your data) ──

class TrainableToxicityDetector:
    """
    Train your own classifier on Jigsaw dataset.
    More control, fully customizable.

    Note: The Jigsaw CSV uses 'identity_hate' (not 'identity_attack').
    This differs from Detoxify's output key name — keep that in mind
    if you compare results between this class and QuickToxicityDetector.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            strip_accents='unicode',
            analyzer='word',
            sublinear_tf=True
        )
        # FIX: changed solver from 'lbfgs' to 'saga'
        # 'lbfgs' only supports L2 penalty and will crash if penalty is changed to 'l1'.
        # 'saga' supports both L1 and L2, making the model more robust to configuration changes.
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='saga'
        )
        self.is_trained = False

    def load_data(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at path: {path}")

        print("Loading dataset...")
        df = pd.read_csv(path)
        df['comment_text'] = df['comment_text'].fillna('')

        # Jigsaw dataset uses 'identity_hate' (not 'identity_attack')
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {missing}")

        df['label'] = (df[label_cols].sum(axis=1) > 0).astype(int)
        print(f"Dataset loaded: {len(df)} rows")
        print(f"Toxic: {df['label'].sum()} | Clean: {(df['label'] == 0).sum()}")
        return df

    def train(self, data_path: str) -> float:
        # FIX: reset model state so re-calling train() doesn't silently use stale vectorizer state
        self.is_trained = False
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            strip_accents='unicode',
            analyzer='word',
            sublinear_tf=True
        )

        df = self.load_data(data_path)
        X = df['comment_text'].values
        y = df['label'].values

        # FIX: moved all steps inside the mlflow run so the run isn't left open on exception
        with mlflow.start_run(run_name="tfidf_logreg_toxicity"):
            mlflow.log_param("max_features", 10000)
            mlflow.log_param("ngram_range", "(1,2)")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("solver", "saga")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print("Vectorizing text...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            print("Training model...")
            self.model.fit(X_train_vec, y_train)

            y_pred = self.model.predict(X_test_vec)
            y_prob = self.model.predict_proba(X_test_vec)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("precision", report['1']['precision'])
            mlflow.log_metric("recall", report['1']['recall'])
            mlflow.log_metric("f1", report['1']['f1-score'])
            mlflow.sklearn.log_model(self.model, "model")

            print(f"\nROC-AUC: {auc:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Clean', 'Toxic']))

        self.is_trained = True
        return auc

    def save(self, path: str = "models/"):
        if not self.is_trained:
            raise Exception("Cannot save: model has not been trained yet.")

        os.makedirs(path, exist_ok=True)

        # FIX: warn if existing model files will be overwritten
        for filename in ["vectorizer.pkl", "toxicity_model.pkl"]:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                warnings.warn(f"Overwriting existing file: {full_path}")

        with open(os.path.join(path, "vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(path, "toxicity_model.pkl"), 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path: str = "models/"):
        vec_path = os.path.join(path, "vectorizer.pkl")
        model_path = os.path.join(path, "toxicity_model.pkl")

        # FIX: check files exist before loading to give a clear error message
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vectorizer not found at: {vec_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print("Model loaded from disk.")

    def predict(self, text: str) -> dict:
        if not self.is_trained:
            raise Exception("Model not trained. Call train() or load() first.")

        # FIX: validate input — empty/None strings cause silent vectorizer failures
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        vec = self.vectorizer.transform([text])
        prob = self.model.predict_proba(vec)[0][1]
        label = "toxic" if prob > 0.5 else "clean"
        return {
            "label": label,
            "confidence": round(float(prob), 4),
            "risk_score": round(float(prob), 4)
        }