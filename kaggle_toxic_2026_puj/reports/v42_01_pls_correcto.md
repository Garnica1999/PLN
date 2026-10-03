# V42-01: PLS correcto sobre document x term

Generado por `experiments/v42_01_pls_correcto.py` (V42-01 (PLS correcto sobre document x term, con control SVD matched)).

**Estado `OOF_EVALUATED`. V40 construyo m = (V x 3) y aplico PLS sobre (n_training, 3). Este script usa X = document x term TF-IDF (la representacion exacta de V03) y Y = [HS, TR, AG]**

## 1. La correccion respecto a V40

V40 construyo `m = (V x 3)` y aplico PLS sobre `(n_training, 3)`. Este script usa
`X = document x term TF-IDF` (la representacion exacta de V03) y `Y = [HS, TR, AG]`.

## 2. Benchmark de memoria y tiempo

| magnitud | valor |
|---|---:|
| X sparse | (1186, 20441) |
| X dense | 193.9 MB |
| tiempo matriz | 0.15 s |
| tiempo PLS | 5.21 s |
| memoria antes | 228.9 MB |
| memoria tras PLS | 778.9 MB |
| folds en paralelo | False |

Los cinco folds se procesaron secuencialmente.

## 3. Resultado OOF

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0.7772114525 | 0 |
| SVD matched | 0.6632953248 | -0.1139161278 |
| PLS | 0.7634690854 | -0.0137423671 |

| etiqueta | V03 | PLS | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7419 | -0.0198 |
| TR | 0.8157 | 0.8028 | -0.0130 |
| AG | 0.7542 | 0.7458 | -0.0084 |

train F1 0.9888 | gap +0.2254

## 4. Complementariedad contra V03

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 38 | 63 | -25 | 101 |
| TR | 18 | 26 | -8 | 44 |
| AG | 36 | 46 | -10 | 82 |

**rescates 92 | danios 135 | neto -43 | correlacion 0.9490**

## 5. Analisis de memorization

| magnitud | valor |
|---|---:|
| train F1 | 0.9888 |
| OOF F1 | 0.7635 |
| gap | +0.2254 |
| depende de terminos raros | True |

**DECISION: OVERFIT**

## 6. Lo que este script no hace

- No usa PLSSVD ni PLSCanonical.
- No envia submission.
- No introduce hashing, BM25, features nuevas, embeddings ni kernels.
- No hace grid de n_components.
- No modifica V03.

Coste: 32.7 s, memoria 863 MB, 3 modelos, 0 submissions.

