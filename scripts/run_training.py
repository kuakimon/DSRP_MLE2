"""Ejecuta el entrenamiento del modelo y lo registra en MLflow.

Uso:
    python scripts/run_training.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fake_news_mle.train import train_model  # noqa: E402

if __name__ == "__main__":
    train_model()
