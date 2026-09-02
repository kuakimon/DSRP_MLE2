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
notebooks/01\\\_preprocesamiento.ipynb ── src/fake\\\_news\\\_mle/preprocessing.py
        │
        ▼
data/processed/news\\\_processed.parquet
        │
        ▼
notebooks/02\\\_machine\\\_learning.ipynb ── src/fake\\\_news\\\_mle/train.py ──► MLflow
        │                                                          (params, metrics, artifacts)
        ▼
models/model.joblib + models/vectorizer.joblib  (modelo productivo)
        │
        ▼
mcp\\\_server/server.py  (tools: classify\\\_news, dataset\\\_stats, search\\\_examples)
        │
        ▼
streamlit\\\_app/app.py  (agente Claude + tool use sobre el MCP)  ◄── usuario final
```

\[Reemplazar por una imagen del diagrama si se desea.]

## c. Descripción del dataset y diccionario de datos

**Fuente**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle).
Dos archivos CSV: `Fake.csv` (23,502 artículos) y `True.csv` (21,417 artículos), \~44MB en total.

|Columna|Tipo|Descripción|
|-|-|-|
|`title`|texto|Título del artículo de noticias|
|`text`|texto|Cuerpo del artículo|
|`subject`|categórico|Tema/categoría del artículo (ej. politicsNews, worldnews)|
|`date`|fecha|Fecha de publicación|
|`label`|binario|0 = fake, 1 = real (agregado en preprocesamiento)|

Columnas derivadas en preprocesamiento: `clean\\\_text` (texto limpio), `text\\\_length` (longitud en palabras).

## d. Model Card

Ver [`docs/model\\\_card\\\_template.md`](docs/model_card_template.md) — completar con las métricas finales tras entrenar.
Referencia: https://www.kaggle.com/code/var0101/model-cards

## e. Resultados — métricas de evaluación offline y online

**Offline** (ver experimentos completos en MLflow, link a agregar tras publicar):

|Métrica|Valor (test)|
|-|-|
|Accuracy|0.9909|
|Precision|0.9879|
|Recall|0.9930|
|F1-score|0.9905|
|ROC-AUC|0.9991|

Estos valores son excelentes y, para este dataset en particular, esperables:
el modelo probablemente está aprendiendo diferencias de *estilo/formato*
entre las fuentes usadas para armar Fake.csv y True.csv (agencias de noticias
reales vs. sitios de fake news), más que "entender" el contenido. Vale la
pena mencionarlo en las conclusiones y en las limitaciones del Model Card.

**Online** (registradas automáticamente por `streamlit\\\_app/metrics.py` en cada interacción, visibles en la barra lateral de la app y en `data/logs/interactions.jsonl`):

|Métrica online|Qué mide|Valor observado|
|-|-|-|
|Latencia promedio (s)|Tiempo total desde la pregunta hasta la respuesta final del agente|88.65ms|
|Tool-calls promedio / turno|Cuánto se apoya el agente en el MCP para responder|0.0|
|Tasa de éxito|% de turnos donde el agente devolvió una respuesta sin error|0.0%|
|Tasa de error en tools|% de llamadas al MCP que fallaron|700.0%|
|Feedback 👍 / 👎|Calificación explícita del usuario sobre la utilidad de la respuesta|0/0|





## f. Conclusiones

Se logró levantar el agente y la conexión exitosa con la key de anthropic, pero el agente tarda mucho tiempo en ejecutar la prediccion (con el modelo .joblib asociado), genera timeout. Se sugiere seguir testeando y midiendo la ejecucion.



\---

## Estructura del repositorio

```
├── notebooks/
│   ├── 01\\\_preprocesamiento.ipynb
│   └── 02\\\_machine\\\_learning.ipynb
├── data/
│   ├── raw/            # Fake.csv, True.csv (no versionados, ver .gitignore)
│   └── processed/      # dataset limpio en parquet
├── src/fake\\\_news\\\_mle/  # módulo reusable (preprocessing, train, predict)
├── scripts/            # run\\\_preprocessing.py, run\\\_training.py, run\\\_prediction.py
├── mcp\\\_server/         # servidor MCP con tools sobre dataset y modelo
├── streamlit\\\_app/      # app con el agente conversacional
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
python scripts/run\\\_preprocessing.py

# 4. Entrenar y trackear en MLflow
python scripts/run\\\_training.py
mlflow ui   # http://localhost:5000

# 5. Probar una predicción rápida
python scripts/run\\\_prediction.py "titular de ejemplo a clasificar"

# 6. Levantar el agente con Streamlit (usa el servidor MCP internamente)
export ANTHROPIC\\\_API\\\_KEY=sk-...
streamlit run streamlit\\\_app/app.py
```

## Experimentos en MLflow

Link con evidencia de experimentos y modelo productivo: **\[agregar link tras publicar, ej. DagsHub]**

## Estrategia de Git

Ver [`docs/git\\\_strategy.md`](docs/git_strategy.md).

## Versión

Este repositorio corresponde a la **release v1.0.0** del proyecto.

