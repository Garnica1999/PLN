# KAGGLE_AGE_2026_PUJ

Predicción del rango de edad (`age_range`) de una persona a partir del texto de un tweet.
Competencia académica privada del curso **Procesamiento de Lenguaje Natural** (Maestría en
Inteligencia Artificial, Pontificia Universidad Javeriana).

- **Competencia Kaggle**: `tweet-age-2026-puj` (id 167385) — host `NLP_Course_PUJ`.
- **Cierre**: 2026-10-19 23:00 UTC.
- **Métrica oficial**: F1 Score (leaderboard). 70% de la nota = LB privado, 30% = notebook.
- **Presupuesto de envíos**: 10 submissions puntuadas (se conserva la mejor).

## Restricciones de la competencia

- **Prohibido**: transformers/LLMs (BERT, RoBERTa, GPT, T5, ...) y **cualquier modelo de
  vectorización** (embeddings preentrenados o entrenados desde cero por nosotros:
  Word2Vec, FastText, GloVe, Doc2Vec, SBERT).
- **Permitido**: TF-IDF/CountVectorizer, features artesanales, NLTK, spaCy, scikit-learn,
  TensorFlow, PyTorch, lexicones y clasificadores clásicos/ensembles.
- **El conjunto de test no se usa para entrenar, ni para construir vocabularios, ni para
  seleccionar features/hiperparámetros/umbrales.**

## Datos

| Archivo | Filas | Columnas | Separador |
|---|---|---|---|
| `data/raw/age_train.csv` | 169.336 | `id;text;age_range` | `;` |
| `data/raw/age_test.csv` | 112.892 | `id;text` | `;` |
| `data/raw/age_submission.csv` | 112.892 | `id,age_range` | `,` |

6 clases balanceadas: `13-17`, `18-24`, `25-34`, `35-49`, `50-64`, `65-xx`.

## Estructura

```
data/{raw,interim,processed,external}   datos crudos, intermedios y procesados
notebooks/                              notebooks por fase (01..09) + 99 consolidado
src/                                    módulos reutilizables del pipeline
scripts/                                lanzadores de fase
features/                               matrices de features cacheadas
models/                                 modelos serializados
experiments/                            logs por experimento
reports/{figures,tables}                figuras y tablas del informe
submissions/                            age_submission_vNN.csv (validados)
```

Documentos de memoria del proyecto: `PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md`,
`experiments.csv`, `FINAL_REPORT.md`.

## Entorno

- Intérprete obligatorio: `/home/carlos/venv/bin/python` (venv del proyecto).
- GPU: AMD Radeon RX 9070 XT (ROCm 7.2). Uso opcional y **gated** (ver `src/gpu.py` y
  la decisión D-007): antes de usar la GPU se verifica que esté libre; si está ocupada
  se devuelve el control al usuario y se entra en modo espera. Todo el pipeline es
  CPU-first.

## Ejecución

```bash
/home/carlos/venv/bin/python -m src.data --download --verify     # descarga y valida
/home/carlos/venv/bin/python scripts/run_notebook.py 01_data_audit  # ejecuta un notebook
```
