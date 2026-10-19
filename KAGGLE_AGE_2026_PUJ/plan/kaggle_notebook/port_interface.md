# Interfaz de porte — cadena AGE (`repro/cadena/`)

**Objetivo:** reproducir la cadena completa del campeon V48 (TweetAge_2026_PUJ) desde los CSV crudos,
como modulos Python verificables y luego embebibles en el notebook de Kaggle.

## Reglas de oro (obligatorias para todo el paquete)

1. **Porte literal.** Cada funcion reproduce EXACTAMENTE el script original citado (mismo modelo, mismos
   hiperparametros, mismos datos de entrada, mismas transformaciones). NO se "mejora", NO se reordena, NO se
   reinterpreta. En el docstring: referencia `archivo:lineas` del original.
2. **Sin dependencias del proyecto.** Solo `numpy`, `pandas`, `scipy`, `sklearn`, stdlib. Prohibido
   `import src` / `import scripts`. Se permite `from cadena.comun import ...` (o import relativo).
3. **ASCII puro.** Todo el codigo en ASCII (sin tildes/emojis). Comentarios y docstrings en espanol sin tildes.
4. **Sin efectos en import.** Nada de I/O ni computo a nivel de modulo.
5. **Determinismo.** Si el original usa una semilla, se porta esa semilla tal cual. No anadir aleatoriedad.
6. **Formas y dtypes EXACTOS** de las salidas (los verifica un spike contra los artefactos guardados).

## Modulos y firmas

### `comun.py` (ya escrito; interfaz fija)
Constantes `CLASSES`, `SEED`, `CHAR_300K`, `CHAR_700K`, `MENTION`, `HASHTAG`, `WORD`, `CLASS_MIDPOINTS`,
`ADJACENT_PAIRS`; helpers `collapse_spaces`, `tokens_of`, `cargar_train`, `cargar_test`, `cargar_folds`,
`to_scores`, `softmax_ordinal`.

### `dense.py`
- `extraer_textometria(df: pd.DataFrame) -> pd.DataFrame`
  - 38 columnas en el orden `TEXTOMETRY_FEATURES + STYLE_FEATURES` de `src/textometry.py`.
  - Indice alineado con `df`. Porte literal de `src/textometry.py` (+ `_EMOJI_RE`/`_URL_RE` de `src/preprocessing.py`).

### `structured.py`
- `extraer_estructuradas(df: pd.DataFrame) -> pd.DataFrame`
  - 87 columnas, 6 familias en orden (distribution 21, char_distribution 10, lexical_richness 9,
    word_shape 10, position 19, structure_entropy 18) de `src/advanced_features.py`.
  - **Con la correccion de `scripts/phase3b_recorregido.py:42-51`** (`ALNUM_RE` y recalculo de
    `shape_alphanumeric_ratio`).

### `supervised.py`
- `supervisadas_infold(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray`
  - (169336, 73) float64; LOFO estricto: para cada fold, ajustar con el train de los otros 4 y transformar
    la validacion (porte de `scripts/advanced_02_features.py:205-234` + `src/advanced_features.py:326-490`).
  - Orden de columnas: mention 17 + hashtag 17 + rt_source 17 + rarity 12 + lexicon 10.
- `supervisadas_test(textos_train, y, textos_test) -> np.ndarray` (misma funcion con train completo; usado por el test).

### `miembros_oof_a.py`
- `oof_e013(textos, y, fold_idx) -> (n,6) float32` — margenes LinearSVC reordenados a CLASSES.
- `oof_e016(...) -> (n,6) float32` — margenes.
- `oof_e009(...) -> (n,6) float32` — probabilidades OvR reordenadas.
- `oof_e011(...) -> (n,6) float32` — probabilidades.
- `oof_e010(...) -> (n,6) float32` — probabilidades.
- `oof_e014(...) -> (n,6) float32` — margenes.
- `oof_e015_ordinal(...) -> (n,) float32` — salida continua (Ridge sobre CLASS_MIDPOINTS).
- `oof_e015_softmax(...) -> (n,6) float32` — `softmax_ordinal` de la salida anterior (via comun).
- Folds: `fold_idx` externo (protocolo congelado); entrenar con `fold_idx != fold`, predecir `== fold`.

### `miembros_oof_b.py`
- `oof_e017(...) -> (n,6) float32` — probabilidades.
- `oof_e019b(textos, y, fold_idx, dense: pd.DataFrame) -> (n,6) float32` — MLP(256,128) sobre [35 densas + SVD64 in-fold].
- `oof_e021a(...) -> (n,6) float32` — Frank & Hall acumulativo.
- `oof_e021b(...) -> (n,6) float32` — Rocchio + softmax T=8.
- `oof_e028a(...) -> (n,5) float32` — margenes de 5 SVM por par adyacente (columnas = ADJACENT_PAIRS).
- `oof_adv05(...) -> (n,6) float32` — NB-SVM OvR sobre char300+menciones (margenes).

### `nb_svm.py`
- `nb_log_count_ratio(x_train, binary_target) -> np.ndarray` (porte literal).
- `nb_svm_margins(textos, y, train_idx, valid_idx, config) -> (len(valid_idx),6) float32` — helper comun
  para ADV05/ADV07 (log-count ratio + LinearSVC(C=0.2, max_iter=3000) por clase, margenes OvR).

### `test_side.py`
- `representaciones_test(train: pd.DataFrame, test: pd.DataFrame) -> dict` — porte de
  `src/pipeline.py:157-198` (`build_representations`).
- `base_models_test(reps, y, dense_train, dense_test, textos_train, textos_test) -> list[np.ndarray]`
  — porte de `src/pipeline.py:199-288` (`fit_base_models`), 13 bloques en el orden de `OOF_FILES`.
- `nb_word_test(textos_train, y, textos_test) -> (m,6) float32` — ADV07 lado test
  (`scripts/phase25_submission_fresca.py:92-103`).
- `matriz_test_fresca(bloques_72, nb_test, densas_test, estructuradas_test, supervisadas_test, nb_word) -> (m,279)`
  — ensambla en el orden exacto: `[72] + [nb_test 6] + [35] + [87] + [73] + [6]` con `assert shape[1] == 279`.

## Verificacion (fuera del alcance de los agentes)

Los spikes (`repro/spike*.py`, en el servidor) comparan cada salida contra los artefactos guardados:
- `oof_E013_margenes.npy`, `oof_E016_margenes.npy`, `oof_E-009.npy`, `oof_E-011.npy`, `oof_E-010.npy`,
  `oof_E014_margenes.npy`, `oof_E015_ordinal.npy`, `oof_E015_softmax.npy`, `oof_E017_probabilidades.npy`,
  `oof_E019b_densas_svd.npy`, `oof_E021a_ordinal.npy`, `oof_E021b_rocchio.npy`, `oof_E028a_pares.npy`,
  `oof_ADV05_nbsvm.npy`, `oof_ADV07_nbsvm_word.npy`.
- `dense_train.parquet` (35 de 38 columnas usadas), `age_structured_features.parquet` (87),
  `age_supervised_features.npy` (73), `meta_features_train_v11.npy` (273).
