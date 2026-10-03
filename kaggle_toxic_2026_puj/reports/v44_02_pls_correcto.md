# V44-02: PLS correcto sobre document x term, con control SVD matched

Generado por `experiments/v44_02_pls_correcto.py` (V44-02 (PLS correcto sobre document x term, con control SVD matched)).

**Estado `OOF_EVALUATED`. Alcance: PLSRegression sobre document x term. No se usa PLSSVD ni PLSCanonical**

## 1. La correccion algebraica

V40 construyo `m = (V x 3)` y aplico PLS sobre `(n_training, 3)`. Este script usa
`X = document x term TF-IDF` (la representacion exacta de V03) y `Y = [HS, TR, AG]`.

## 2. Resultados

| estrategia | corrected_cv | delta vs V03 | train | gap | rescates | danios | net | correlacion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V03 (control) | 0.7772114525 | 0 | 0.9709 | +0.1937 | - | - | - | - |
| SVD matched | 0.6620131501 | -0.1151983024 | 0.6658 | +0.0038 | 229 | 666 | -437 | 0.7988 |
| PLS | 0.7367815778 | -0.0404298747 | 0.9642 | +0.2274 | 119 | 273 | -154 | 0.9119 |

## 3. Convergencia

PLS converge en todos los pliegues (el algoritmo NIPALS tiene criterio de parada
por convergencia, no por agotamiento de iteraciones).

## 4. Lo que este script no hace

- No usa PLSSVD ni PLSCanonical.
- No envia submission.
- No introduce hashing, BM25, features nuevas, embeddings ni kernels.

**DECISION: PLS = CLOSED**

## 5. Coste

21.9 s, memoria 315 MB, 3 modelos, 0 submissions.

