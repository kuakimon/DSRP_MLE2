"""
Funciones de preprocesamiento para el dataset Fake and Real News.

Dataset esperado (Kaggle: clmentbisaillon/fake-and-real-news-dataset):
  - data/raw/Fake.csv  -> columnas: title, text, subject, date
  - data/raw/True.csv  -> columnas: title, text, subject, date

Se etiqueta Fake.csv como 0 y True.csv como 1, se combinan y se limpia
el texto para dejarlo listo para vectorizar.
"""

import re
import string
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carga Fake.csv y True.csv, los etiqueta y los combina en un solo DataFrame."""
    fake_df = pd.read_csv(raw_dir+"/Fake.csv")
    true_df = pd.read_csv(raw_dir+"/True.csv")

    fake_df["label"] = 0  # 0 = fake
    true_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def clean_text(text: str) -> str:
    """Limpieza básica de texto: minúsculas, sin URLs, sin puntuación, sin espacios extra."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la columna de texto final (título + cuerpo) y la limpia."""
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["full_text"] = df["title"] + " " + df["text"]
    df["clean_text"] = df["full_text"].apply(clean_text)
    df["text_length"] = df["clean_text"].str.split().apply(len)
    return df[["title", "subject", "date", "clean_text", "text_length", "label"]]


def preprocess(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Pipeline completo: cargar -> limpiar -> guardar dataset procesado."""
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    df = load_raw_data(raw_dir)
    df = build_features(df)
    output_path = Path(processed_dir) / "news_processed.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Dataset procesado guardado en: {output_path} ({len(df)} filas)")
    return df


if __name__ == "__main__":
    preprocess()
