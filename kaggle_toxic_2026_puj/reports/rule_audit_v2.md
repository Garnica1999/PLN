# Auditoría de reglas (FASE 87) — KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-25. Alcance: verificar el cumplimiento de las prohibiciones de la
especificación v2 (`docs/advanced_research_v2_spec.md`, secciones 6–7, 62–64, 86)
sobre el código y los artefactos del proyecto. Método: búsquedas dirigidas
(`grep -rn`) sobre `experiments/*.py`, `src/*.py` y los cuadernos, más la revisión
de hashes, configuración de seed y requisitos.

## 1. Modelos, embeddings y clasificadores preentrenados

| Regla | Comando (resumen) | Resultado |
|---|---|---|
| Sin transformers/BERT/LLM | `grep -rniE "transformers\|llm" experiments/*.py src/*.py` | 0 usos reales. Coincidencias solo en comentarios que documentan la prohibición: `experiments/11_dl_desde_cero.py:2058` y `experiments/20_emocion_nbsvm_ortografia_v2.py:41` (nota sobre VADER-es descartado por requerir transformers). |
| Sin modelos preentrenados | `grep -rnE "from_pretrained\|AutoModel\|AutoTokenizer"` | 0 coincidencias. |
| Sin embeddings preentrenados | `grep -rnE "sentence_transformers\|gensim\|KeyedVectors\|fasttext\|word2vec\|glove"` | 0 coincidencias. |
| Sin clasificadores preentrenados | Revisión de dependencias en `requirements.txt` | Ninguna dependencia aporta clasificadores; el paquete de análisis es spaCy (uso simbólico, ver §2). |

**Veredicto**: cumplido, sin violaciones.

## 2. spaCy: uso exclusivamente simbólico

- Cargas del modelo con NER excluido:
  `spacy.load(..., exclude=["ner"])` en `experiments/01_eda_linguistico.py:3306`,
  `03_variantes_representaciones.py:423`, `06_error_analysis.py:692`,
  `08_features_densas_h11_h12.py:627`, `16_targeting_v2.py:582` y
  `19_sintaxis_cita_negacion_v2.py:439`.
- `grep -rnE "\.vector\b|\.tensor\b|\.similarity\("`: las coincidencias son
  (a) docstrings que documentan la prohibición (D-003) y
  (b) `torch.tensor(...)` de PyTorch en `experiments/11_dl_desde_cero.py`. No hay
  uso de vectores, tensores ni similitud de spaCy en ninguna parte.

**Veredicto**: cumplido. Solo POS, lemas, dependencias y morfología.

## 3. Corpus, etiquetas y lexicones externos

- Sin descargas de corpus: `data/external/lexicons/` está vacío y no existe
  `reports/lexicon_catalog.csv` (consistente con D-007: sin lexicones externos).
- Los léxicos usados son inducidos fold-local desde el propio train
  (`experiments/16_targeting_v2.py`, `experiments/21_backlog_restante_v2.py`).
- No se introdujeron etiquetas externas ni el esquema HatEval público.

**Veredicto**: cumplido.

## 4. Uso del conjunto de test

- El test solo se usa con `transform` + `predict` para generar los CSV de envío.
- Umbrales, selección de modelo y calibración se ajustan sobre OOF/desarrollo;
  el reajuste de umbral de fase 1 quedó marcado con
  `advertencia_sesgo_optimista` en `reports/fase13a_refit_umbral_metricas.json`.
- Sin pseudo-labeling, sin inferencia de prevalencias del test, sin fit sobre test.

**Veredicto**: cumplido.

## 5. Reproducibilidad

| Elemento | Evidencia |
|---|---|
| Seeds | 42 en modelos clásicos; 42/43/44 en DL (`experiments/11_dl_desde_cero.py:280-284`), con `use_deterministic_algorithms(False)` documentado (ROCm). |
| Congelamiento de V03 | `experiments/baseline_v03_frozen/hashes.txt`: `mod03b_lr_union.joblib` sha256 `263ed5bd…`, submission `7d7478f2…` (misma huella que `submissions/toxic_submission_v03_lr_union_thr.csv`). Reconstrucción byte a byte verificada. |
| Holdout v2 | `config/holdout_v2.json`: 248 tweets, pliegue 0 congelado, sha256 `e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd`; 2/5 evaluaciones usadas. |
| Entorno | `requirements.txt` con versiones (Python 3.14.7, scikit-learn 1.8.0, spaCy 3.8.13, torch 2.14.0+rocm7.2 aparte). |
| Cuaderno | 42 celdas, 26 comprobaciones `[OK]` / 0 fallos; contenido de salidas reproducible (digesto normalizado `c3aa4198983c6aa7` estable entre ejecuciones; el kernel puede dividir los mensajes de stdout sin cambiar el texto). |

## Veredicto final

**0 violaciones de reglas y 0 accesos indebidos al test.** Las prohibiciones de la
especificación (secciones 6–7) se cumplen en todo el código auditado; el uso de
spaCy es simbólico (D-003) y no existen corpus, etiquetas ni lexicones externos.
