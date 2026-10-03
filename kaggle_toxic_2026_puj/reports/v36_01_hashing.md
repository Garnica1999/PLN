# V36-01: Hashing trick con TF-IDF sobre la representacion de V03

Generado por `experiments/v36_01_hashing.py` (V36-01 (Hashing trick con TF-IDF sobre la representacion de V03)).

**Estado `OOF_EVALUATED`. Alcance: solo cambia la representacion de vocabulario explicito a hashing. No se introducen nuevas n-grams, BM25, features, URL, embeddings ni kernels**

## 1. Configuracion

- HashingVectorizer(analyzer=word, ngram_range=(1,2), n_features=2**18, alternate_sign=False)
- HashingVectorizer(analyzer=char_wb, ngram_range=(3,5), n_features=2**18, alternate_sign=False)
- TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True, norm=l2)
- dentro de cada training fold, nunca con validacion

## 2. Controles de implementacion

| control | valor |
|---|---|
| dimension_por_bloque | 262144 |
| dimension_total | 524288 |
| no_negativo_por_bloque | [True, True] |
| sin_nan | True |
| sin_inf | True |
| reproducible | True |
| sparsity_muestra | 99.95514742533366 |
| nnz_muestra | 70547 |
| porcentaje_features_no_vacias | 0.04485257466634115 |

## 3. Auditoria de colisiones

| bloque | n-grams | buckets unicos | colisiones | proporcion |
|---|---:|---:|---:|---:|
| word | 3435 | 3408 | 27 | 0.0079 |
| char | 21211 | 20363 | 848 | 0.0400 |

## 4. Resultado OOF

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0.7772114525 | 0 |
| Hashing 2^18 | 0.7484267738 | -0.0287846787 |

| etiqueta | V03 | hashing | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7430 | -0.0187 |
| TR | 0.8157 | 0.7678 | -0.0479 |
| AG | 0.7542 | 0.7344 | -0.0198 |

train F1 0.9603 | gap +0.2119

## 5. Complementariedad contra V03

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 46 | 89 | -43 | 135 |
| TR | 24 | 62 | -38 | 86 |
| AG | 61 | 87 | -26 | 148 |

**rescates 131 | danios 238 | neto -107 | correlacion 0.9171**

## 6. Efecto de la dimensionalidad

| dimension | corrected_cv | delta vs V03 |
|---|---:|---:|
| 2^18 | 0.7484267738 | -0.0287846787 |

## 7. Fairness: que cambia exactamente

| componente | V03 | Hashing |
|---|---|---|
| representacion | vocabulario explicito | hashing |
| word_ngrams | (1,2) | (1,2) |
| char_ngrams | (3,5) | (3,5) |
| tfidf | si | si |
| sublinear_tf | si | si |
| min_df | 2 | no aplica |
| dimensionalidad | ~27k | 2x2^18 |
| vocabulario | si | no |
| colisiones | no | si |

**DECISION: OVERFIT**

## 8. Lo que este script no hace

- No envia submission.
- No introduce nuevas n-grams, BM25, features, URL, embeddings ni kernels.
- No hace grid search de dimensionalidad.
- No modifica V03.

Coste: 7.3 s, memoria 1417 MB, 3 modelos, 0 submissions.

