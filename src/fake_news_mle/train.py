"""
Entrenamiento del modelo de clasificación fake/real news, con tracking
de parámetros, métricas y artefactos en MLflow.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROCESSED_PATH = Path("../data/processed/news_processed.parquet")
MODEL_DIR = Path("../models")
EXPERIMENT_NAME = "fake-news-classification"


def load_processed_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_model(
    max_features: int = 20_000,
    ngram_max: int = 2,
    C: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Entrena TF-IDF + Regresión Logística y registra todo en MLflow."""
    df = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=test_size,
        random_state=random_state, stratify=df["label"],
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        # --- Parámetros ---
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", f"(1,{ngram_max})")
        mlflow.log_param("C", C)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", test_size)

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_test_vec)
        y_proba = clf.predict_proba(X_test_vec)[:, 1]

        # --- Métricas (tienen sentido para clasificación binaria balanceada) ---
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # --- Artefactos ---
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "model.joblib"
        vectorizer_path = MODEL_DIR / "vectorizer.joblib"
        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(vectorizer_path))
        mlflow.sklearn.log_model(clf, artifact_path="sklearn-model")

        print("Métricas:", metrics)
        return metrics


if __name__ == "__main__":
    train_model()
