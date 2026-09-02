# Model Card — Clasificador de Fake News

Basado en el estándar de Model Cards (ver referencia en el enunciado del proyecto).

## Detalles del modelo
- **Tipo de modelo**: TF-IDF + Regresión Logística (baseline). [Actualizar si se prueban otros modelos]
- **Versión**: v1.0.0
- **Fecha de entrenamiento**: [completar]
- **Entrenado por**: [tu nombre]

## Uso previsto
- **Uso principal**: clasificar titulares/artículos de noticias en español o inglés como "fake" o "real", con fines educativos.
- **Fuera de alcance**: no debe usarse como única fuente de verdad para decisiones editoriales o de moderación de contenido real.

## Datos de entrenamiento
- **Dataset**: Fake and Real News Dataset (Kaggle, clmentbisaillon/fake-and-real-news-dataset).
- **Tamaño**: ~44,900 artículos (23,502 fake / 21,417 real).
- **Split**: 80% train / 20% test, estratificado por label.

## Métricas de evaluación
| Métrica | Valor (test) |
|---|---|
| Accuracy | 0.9909 |
| Precision | 0.9879 |
| Recall | 0.9930 |
| F1-score | 0.9905 |
| ROC-AUC | 0.9991 |

## Consideraciones éticas y limitaciones
- El dataset combina fuentes de EE.UU. de un período histórico específico (2016-2017); el modelo puede no generalizar bien a noticias actuales o de otros países.
- El estilo de escritura, no el contenido factual, es lo que el modelo realmente aprende a distinguir — puede fallar ante fake news bien escritas o noticias reales con tono sensacionalista.
- Las métricas obtenidas (F1 ≈ 0.99, ROC-AUC ≈ 0.999) son inusualmente altas para un problema de NLP real; esto sugiere que el modelo está explotando diferencias sistemáticas de formato/fuente entre `Fake.csv` y `True.csv` (por ejemplo, estilo de redacción de Reuters vs. sitios de fake news) en lugar de razonar sobre veracidad. No debe interpretarse como que el modelo "detecta mentiras" en un sentido general.

## Métricas online (agente + MCP en la demo de Streamlit)
Registradas automáticamente por `streamlit_app/metrics.py` (`data/logs/interactions.jsonl`):

| Métrica | Valor |
|---|---|
| Latencia promedio de respuesta | [completar] |
| Tool-calls promedio por turno | [completar] |
| Tasa de éxito (sin errores) | [completar] |
| Tasa de error en tool-calls | [completar] |
| Feedback de usuarios (👍/👎) | [completar] |

Estas métricas evalúan la *experiencia del agente en uso real*, no la calidad
del clasificador en sí — complementan las métricas offline de arriba.
