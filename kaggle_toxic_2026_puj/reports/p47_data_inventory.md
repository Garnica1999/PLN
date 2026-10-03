# P47 - Auditoria de disponibilidad de datos

Generado por `experiments/v47_00_auditoria_datos.py` (V47-00 (P0 auditoria de datos y expansion legal del train)).

Fecha 2026-09-28T00:29:46+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

> El holdout y el test se han abierto **solo para contar filas y calcular hashes**. Ninguna etiqueta se ha leido para ninguna decision.

Estructura fisica verificada: `/home/carlos/Git/PLN-v25/kaggle_toxic_2026_puj/data` es un **directorio real** (False). Dentro, `processed, raw` son enlaces simbolicos que apuntan fuera del worktree, y el resto son directorios reales:

- `processed` -> `/home/carlos/Git/PLN/kaggle_toxic_2026_puj/data/processed`
- `raw` -> `/home/carlos/Git/PLN/kaggle_toxic_2026_puj/data/raw`

## 1. Inventario de fuentes oficiales y derivadas

| path | clase | filas | labels | sup. | emb. | val. | pred. |
|---|---|---:|---|:-:|:-:|:-:|:-:|
| `data/raw/toxic_train.csv` | OFFICIAL_LABELED_TRAIN | 1731 | HS,TR,AG | si | si | si | si |
| `data/raw/toxic_test.csv` | OFFICIAL_TEST | 1154 | ninguna | no | no | no | si |
| `data/processed/holdout_v2_ids.csv` | DERIVED_FEATURE | 1731 | HS,TR,AG (copia) | no | no | no | no |
| `data/processed/holdout2_ids.csv` | DERIVED_FEATURE | 247 | ninguna | no | no | no | no |
| `data/processed/grupos_near_duplicados.csv` | DERIVED_FEATURE | 2885 | ninguna | no | no | si | no |
| `features/oof_v03_final.npz` | DERIVED_FEATURE | 1483 | y_true embebido | no | no | si | no |
| `features/v2_structured_features.parquet` | DERIVED_FEATURE | 2885 | ninguna | no | no | si | si |

## 2. Corpus externos

**Ninguno es legal para entrenamiento supervisado.** D-030 los autoriza solo para aprender la representacion distribuida, y la familia de embeddings esta cerrada.

| id | origen | licencia | documentos | tokens | sup. | emb. |
|---|---|---|---:|---:|:-:|:-:|
| `E-CORP-01` | wikipedia_es | CC BY-SA 4.0 (atribucion obligator | 2847 | 10,002,400 | no | si |
| `E-CORP-06` | m_c4_es | ODC-BY 1.0 (declarada en la ficha  | 16738 | 10,000,239 | no | si |
| `E-CORP-08` | wikipedia_es | CC BY-SA 4.0 (atribucion obligator | 21529 | 50,000,028 | no | si |
| `E-CORP-09` | wikipedia_es | CC BY-SA 4.0 y texto de MediaWiki  | 1694 | 5,011,404 | no | si |
| `E-CORP-07` | wikipedia_10M + web_10M | mezcla de ambas |  | 20,000,045 | no | si |

Artefactos de embeddings en disco: **19** modelos fastText. Son `EMBEDDING_ARTIFACT`: salidas derivadas, no fuentes de filas.

## 3. Reconciliacion de las cifras citadas

| magnitud | afirmado | real | veredicto |
|---|---:|---:|---|
| dev | 1483 | **1483** | OK |
| holdout | 519 | **248** | CORREGIDO |
| unlabeled_rows | 44000 | **0** | CORREGIDO |

- **519 NO es el holdout: es el numero de positivos TR en train (519). El holdout real tiene 248 filas**
- **No existe ninguna fuente de ~44k filas. Los corpus externos se miden en TOKENS, no en filas. El '44' que circula en el repositorio son las 44 features del meta-modelo E2-02-META**

## 4. Pregunta central: hay train etiquetado sin usar?

| magnitud | valor |
|---|---:|
| train oficial etiquetado | 1731 |
| dev (usado por V03) | 1483 |
| holdout (agotado) | 248 |
| test oficial | 1154 |
| test tiene labels | False |
| **train sin asignar** | **0** |
| holdout2 (segunda particion) | 247 |
| holdout2 dentro del holdout | 0 |
| documentos externos | 42,808 |
| tokens externos | 95,014,116 |

El train esta **completamente asignado**: 1483 en dev y 248 en holdout, sin resto. `holdout2` son 247 filas dibujadas **de dev**, no del holdout, asi que no anaden etiquetas nuevas.

Contaminacion de test en el corpus: NINGUNA: el manifiesto declara que ningun texto de toxic_test.csv entra en el corpus, y la auditoria de fuga de E-CORP-01 da 0 textos con coincidencia de 8-gramas.

## 5. Veredicto

**CASE = A+D**

- `DATA_EXPANSION` = **CLOSED**
- `TRAIN_AUGMENTATION` = **CLOSED**
- experimento nuevo ejecutado: **False**

El train oficial tiene 1731 filas y estan todas asignadas: 1483 en dev y 248 en holdout. No queda ni una fila etiquetada oficial sin usar. La recomendacion de 'mas datos etiquetados' no es operacionalmente posible con este corpus.

V6-08 ya ejecuto exactamente las tres transformaciones que la directiva propone (espaciado, elongacion, repeticion de puntuacion), con la augmentation dentro del training fold y los grupos preservados. J1 dio -0.0061 y J2 tambien negativo. Repetirlo bajo otra denominacion seria duplicar un experimento cerrado.

## 6. Correcciones de cifras que esta auditoria deja fijadas

- 519 es el numero de positivos TR; el holdout tiene 248.
- las fuentes externas se miden en tokens; ningun dataset del proyecto tiene 44k filas.

