"""
Servidor MCP (Model Context Protocol) para el proyecto de clasificación
de fake news.

Expone 3 tools que un agente puede usar para responder preguntas sobre
el dataset y el modelo:

  1. classify_news(text)      -> clasifica un texto como fake/real
  2. dataset_stats()          -> estadísticas generales del dataset
  3. search_examples(...)     -> busca ejemplos reales del dataset

Requiere: pip install "mcp[cli]" pandas joblib scikit-learn

Ejecutar en modo desarrollo (para probar con MCP Inspector):
    mcp dev mcp_server/server.py

Ejecutar como servidor stdio (lo que usará el cliente de Streamlit):
    python mcp_server/server.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from mcp.server.fastmcp import FastMCP

from fake_news_mle.predict import predict as predict_fn

PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "news_processed.parquet"

mcp = FastMCP("fake-news-mcp")

_dataset_cache = None


def _get_dataset() -> pd.DataFrame:
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = pd.read_parquet(PROCESSED_PATH)
    return _dataset_cache


@mcp.tool()
def classify_news(text: str) -> dict:
    """Clasifica un texto de noticia como 'fake' o 'real', con su confianza.

    Args:
        text: el titular o cuerpo de la noticia a clasificar.
    """
    return predict_fn(text)


@mcp.tool()
def dataset_stats() -> dict:
    """Devuelve estadísticas generales del dataset de entrenamiento:
    número total de artículos, distribución fake/real, y temas (subject)
    más comunes.
    """
    df = _get_dataset()
    return {
        "total_articles": int(len(df)),
        "fake_count": int((df["label"] == 0).sum()),
        "real_count": int((df["label"] == 1).sum()),
        "top_subjects": df["subject"].value_counts().head(5).to_dict(),
        "avg_text_length_words": round(float(df["text_length"].mean()), 1),
    }


@mcp.tool()
def search_examples(label: str = "any", keyword: str = "", limit: int = 3) -> list:
    """Busca ejemplos de artículos del dataset, opcionalmente filtrando por
    etiqueta y por una palabra clave en el texto.

    Args:
        label: 'fake', 'real' o 'any' para no filtrar por etiqueta.
        keyword: palabra o frase a buscar dentro del texto del artículo.
        limit: número máximo de ejemplos a devolver (máx. 10).
    """
    df = _get_dataset()
    limit = max(1, min(limit, 10))

    if label in ("fake", "real"):
        target = 0 if label == "fake" else 1
        df = df[df["label"] == target]

    if keyword:
        df = df[df["clean_text"].str.contains(keyword.lower(), na=False)]

    results = df.head(limit)
    return [
        {
            "title": row["title"],
            "subject": row["subject"],
            "label": "fake" if row["label"] == 0 else "real",
            "excerpt": row["clean_text"][:200],
        }
        for _, row in results.iterrows()
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
