# Fake News Classification — Proyecto Curso II, Especialización MLE

## a. Problema de ML
Clasificación binaria supervisada de texto: dado el título y cuerpo de un
artículo de noticias, predecir si es **fake (0)** o **real (1)**.

Además del modelo, el proyecto integra un **servidor MCP** y un **agente
conversacional en Streamlit** que permiten consultar el dataset y el modelo
en lenguaje natural (por ejemplo: "¿cuántos artículos fake hay sobre
política?" o "clasifica este titular: ...").

## b. Diagrama de flujo del proyecto
```
data/raw (Fake.csv, True.csv)
        │
        ▼
notebooks/01_preprocesamiento.ipynb ── src/fake_news_mle/preprocessing.py
        │
        ▼
data/processed/news_processed.parquet
        │
        ▼
notebooks/02_machine_learning.ipynb ── src/fake_news_mle/train.py ──► MLflow
        │                                                          (params, metrics, artifacts)
        ▼
models/model.joblib + models/vectorizer.joblib  (modelo productivo)
        │
        ▼
mcp_server/server.py  (tools: classify_news, dataset_stats, search_examples)
        │
        ▼
streamlit_app/app.py  (agente Claude + tool use sobre el MCP)  ◄── usuario final
```
[Reemplazar por una imagen del diagrama si se desea.]

## c. Descripción del dataset y diccionario de datos
**Fuente**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle).
Dos archivos CSV: `Fake.csv` (23,502 artículos) y `True.csv` (21,417 artículos), ~44MB en total.

| Columna | Tipo | Descripción |
|---|---|---|
| `title` | texto | Título del artículo de noticias |
| `text` | texto | Cuerpo del artículo |
| `subject` | categórico | Tema/categoría del artículo (ej. politicsNews, worldnews) |
| `date` | fecha | Fecha de publicación |
| `label` | binario | 0 = fake, 1 = real (agregado en preprocesamiento) |

Columnas derivadas en preprocesamiento: `clean_text` (texto limpio), `text_length` (longitud en palabras).

## d. Model Card
Ver [`docs/model_card_template.md`](docs/model_card_template.md) — completar con las métricas finales tras entrenar.
Referencia: https://www.kaggle.com/code/var0101/model-cards

## e. Resultados — métricas de evaluación offline y online
**Offline** (ver experimentos completos en MLflow, link a agregar tras publicar):
- Accuracy, Precision, Recall, F1-score, ROC-AUC — completar con los valores del mejor run.

**Online** (registradas automáticamente por `streamlit_app/metrics.py` en cada interacción, visibles en la barra lateral de la app y en `data/logs/interactions.jsonl`):

| Métrica online | Qué mide | Valor observado |
|---|---|---|
| Latencia promedio (s) | Tiempo total desde la pregunta hasta la respuesta final del agente | [completar] |
| Tool-calls promedio / turno | Cuánto se apoya el agente en el MCP para responder | [completar] |
| Tasa de éxito | % de turnos donde el agente devolvió una respuesta sin error | [completar] |
| Tasa de error en tools | % de llamadas al MCP que fallaron | [completar] |
| Feedback 👍 / 👎 | Calificación explícita del usuario sobre la utilidad de la respuesta | [completar] |

Para llenar esta tabla: usa la app un rato haciendo preguntas variadas, luego lee `data/logs/interactions.jsonl` (o mira la sidebar) y copia los valores de `load_summary()`.

## f. Conclusiones
[Completar al finalizar: qué tan bien funcionó el modelo, qué mejorarías,
qué tan útil resultó el agente/MCP para explorar el dataset, próximos pasos.]

---

## Estructura del repositorio

```
├── notebooks/
│   ├── 01_preprocesamiento.ipynb
│   └── 02_machine_learning.ipynb
├── data/
│   ├── raw/            # Fake.csv, True.csv (no versionados, ver .gitignore)
│   └── processed/      # dataset limpio en parquet
├── src/fake_news_mle/  # módulo reusable (preprocessing, train, predict)
├── scripts/            # run_preprocessing.py, run_training.py, run_prediction.py
├── mcp_server/         # servidor MCP con tools sobre dataset y modelo
├── streamlit_app/      # app con el agente conversacional
├── models/             # modelo y vectorizador entrenados (no versionados)
├── docs/                # estrategia de git y model card
└── mlruns/              # tracking local de MLflow (no versionado)
```

## Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar Fake.csv y True.csv en data/raw/

# 3. Preprocesar datos
python scripts/run_preprocessing.py

# 4. Entrenar y trackear en MLflow
python scripts/run_training.py
mlflow ui   # http://localhost:5000

# 5. Probar una predicción rápida
python scripts/run_prediction.py "titular de ejemplo a clasificar"

# 6. Levantar el agente con Streamlit (usa el servidor MCP internamente)
export ANTHROPIC_API_KEY=sk-...
streamlit run streamlit_app/app.py
```

## Experimentos en MLflow
Link con evidencia de experimentos y modelo productivo: **[agregar link tras publicar, ej. DagsHub]**

## Estrategia de Git
Ver [`docs/git_strategy.md`](docs/git_strategy.md).

## Versión
Este repositorio corresponde a la **release v1.0.0** del proyecto.
