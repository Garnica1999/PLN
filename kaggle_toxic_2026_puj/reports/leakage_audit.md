# Auditoría de fuga de datos - KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-22 (Fase 0). Fuente: solo `data/raw/` oficial.

## 1. Duplicados y near-duplicates

| comparacion   |   pares_ge_0.90 |   pares_0.80_0.90 |   similitud_maxima |
|:--------------|----------------:|------------------:|-------------------:|
| train-train   |               3 |                 4 |                  1 |
| train-test    |               6 |                10 |                  1 |

Umbral de near-duplicate: similitud coseno >= 0.90 sobre TF-IDF de
n-gramas de carácter (char_wb 4-4) del texto normalizado.

## 1b. Grupos de near-duplicados (umbral 0.85, train+test)

Grupos con más de un texto: 4.

|   grupo |   test |   train |   total |
|--------:|-------:|--------:|--------:|
|    1899 |      6 |       0 |       6 |
|     112 |      2 |       3 |       5 |
|     338 |      1 |       3 |       4 |
|    1272 |      1 |       1 |       2 |

Estos grupos se guardan en `data/processed/grupos_near_duplicados.csv` y
se usan como variable de agrupación en la validación cruzada para no
repartir copias casi idénticas entre train y validación.

## 2. Fuga por orden de `id`

| etiqueta   |   spearman_rho |   spearman_p |   chi2 |   chi2_p |   cramers_v |
|:-----------|---------------:|-------------:|-------:|---------:|------------:|
| HS         |         0.0166 |     0.490351 | 12.201 | 0.202201 |      0.084  |
| TR         |        -0.0019 |     0.937956 |  8.082 | 0.52593  |      0.0683 |
| AG         |        -0.007  |     0.772355 | 10.595 | 0.304475 |      0.0782 |

Si `spearman_p` y `chi2_p` no son significativos, el orden de los ids no
codifica la etiqueta y un split aleatorio estratificado es defendible.

## 3. Pseudo-usuarios (primera mención)

Pseudo-autores con >= 3 tweets: 17.

| mencion         |   n_tweets |   HS |   TR |   AG |   share_HS |
|:----------------|-----------:|-----:|-----:|-----:|-----------:|
| youtube         |         14 |    1 |    1 |    3 |      0.071 |
| barbijaputa     |          7 |    3 |    2 |    2 |      0.429 |
| sanchezcastejon |          6 |    2 |    0 |    2 |      0.333 |
| anaisbernal     |          5 |    0 |    0 |    0 |      0     |
| leticiadolera   |          5 |    0 |    0 |    0 |      0     |
| lortegadiaz     |          4 |    4 |    4 |    4 |      1     |
| relatofeminista |          4 |    2 |    1 |    1 |      0.5   |
| zurine3         |          4 |    0 |    0 |    0 |      0     |
| drodriguezven   |          4 |    3 |    3 |    3 |      0.75  |
| lopezdoriga     |          3 |    3 |    3 |    3 |      1     |

## 4. Regla anti-fuga externa (HatEval)

El corpus corresponde al esquema de anotación de HatEval (SemEval-2019
Task 5, subconjunto español). Los textos y etiquetas originales circulan
públicamente: queda PROHIBIDO consultarlos, cruzarlos o usarlos para
entrenar, corregir o validar. Solo se usan las etiquetas de
`toxic_train.csv`. Decisión registrada como D-002.

## 5. Acciones derivadas

- Validación primaria: `StratifiedGroupKFold` con grupos = near-duplicados
  (umbral 0.85) y estratificación por combinación HS x TR x AG.
- Control de robustez adicional con pseudo-autor (primera mención) como
  grupo, para el subconjunto afectado.
- Ninguna transformación se ajusta con datos de test.