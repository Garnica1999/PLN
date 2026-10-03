# V38-01: embedding supervisado por factorizacion termino x etiqueta

Generado por `experiments/v38_01_embedding_supervisado.py` (V38-01 (embedding supervisado por factorizacion termino x etiqueta)).

**Estado `OOF_EVALUATED`. Alcance: factorizacion termino x etiqueta de baja capacidad. No es una red neuronal completa: es una representacion distribuida derivada de asociaciones termino-etiqueta**

## 1. El metodo

matriz termino x etiqueta (conteos de co-ocurrencia con cada etiqueta), normalizada con BM25 por fila, factorizada con SVD truncado de baja dimension.

Dimension **16**, k1 1.2, b 0.75.

Por que no es equivalente a SVD sobre documento x termino: SVD sobre documento x termino es
no supervisado, no usa las etiquetas. La factorizacion termino x etiqueta si las usa, porque la
matriz se construye contando co-ocurrencias con cada etiqueta por separado.

## 2. Vocabulario y parametros

| pliegue | vocabulario | dimension | varianza explicada | parametros |
|---|---:|---:|---:|---:|
| 1 | 5899 | 2 | 0.9025 | 11798 |
| 2 | 6012 | 2 | 0.9012 | 12024 |
| 3 | 5967 | 2 | 0.9007 | 11934 |
| 4 | 6022 | 2 | 0.9001 | 12044 |
| 5 | 6022 | 2 | 0.9032 | 12044 |

## 3. Resultado OOF

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0.7772114525 | 0 |
| embedding supervisado | 0.5954655712 | -0.1817458814 |

| etiqueta | V03 | embedding | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.6149 | -0.1468 |
| TR | 0.8157 | 0.6114 | -0.2044 |
| AG | 0.7542 | 0.5601 | -0.1941 |

train F1 0.9143 | gap +0.3188

## 4. Complementariedad contra V03

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 146 | 554 | -408 | 700 |
| TR | 49 | 316 | -267 | 365 |
| AG | 126 | 595 | -469 | 721 |

**rescates 321 | danios 1465 | neto -1144 | correlacion 0.5986**

## 5. Analisis de memorization y terminos raros

| magnitud | valor |
|---|---:|
| terminos con df <= 5 | 6681 (94.09 %) |
| correlacion proporcion rara y error | -0.0394 |
| train F1 | 0.9143 |
| OOF F1 | 0.5955 |
| gap | +0.3188 |
| depende de terminos raros | False |

## 6. Evidencia historica

los embeddings probados en EBD-01 a EBD-25 fueron NO supervisados (FastText, word2vec). No existe ninguna implementacion donde la capa embedding sea entrenada con las etiquetas HS/TR/AG. La familia esta abierta.

**DECISION: OVERFIT**

## 7. Lo que este script no hace

- No envia submission.
- No construye una red neuronal completa.
- No usa el test para nada.
- No modifica V03.

Coste: 9.8 s, memoria 229 MB, 3 modelos, 0 submissions.

