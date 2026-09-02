"""Ejecuta una predicción sobre un texto dado.

Uso:
    python scripts/run_prediction.py "texto de la noticia a clasificar"
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fake_news_mle.predict import predict  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: python scripts/run_prediction.py "texto de la noticia"')
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    print(predict(text))
