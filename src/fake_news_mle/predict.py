"""
Carga el modelo productivo (modelo + vectorizador) y expone una función
simple de predicción, reusada por scripts, el servidor MCP y la app Streamlit.
"""

from pathlib import Path

import joblib

from fake_news_mle.preprocessing import clean_text

MODEL_PATH = Path("models/model.joblib")
VECTORIZER_PATH = Path("models/vectorizer.joblib")

_model = None
_vectorizer = None


def _load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def predict(text: str) -> dict:
    """Predice si una noticia es fake (0) o real (1), con su probabilidad."""
    model, vectorizer = _load_artifacts()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = int(model.predict(vec)[0])
    proba = float(model.predict_proba(vec)[0][label])
    return {
        "label": "real" if label == 1 else "fake",
        "confidence": round(proba, 4),
    }


if __name__ == "__main__":
    sample = "Scientists confirm the earth orbits the sun in newly released study"
    print(predict(sample))
