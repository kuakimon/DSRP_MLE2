"""Ejecuta el pipeline de preprocesamiento de datos.

Uso:
    python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fake_news_mle.preprocessing import preprocess  # noqa: E402

if __name__ == "__main__":
    preprocess()
