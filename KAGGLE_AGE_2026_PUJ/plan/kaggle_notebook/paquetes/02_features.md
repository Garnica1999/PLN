# Paquete 02 — Features densas / estructuradas / supervisadas / lexicos

**Estado:** extraccion Fase 1 (2026-10-17). Solo lectura del repo; sin ejecutar scripts.
**Nota:** todo el codigo del proyecto es ASCII puro; los fragmentos se citan tal cual.

## 0. Tabla resumen

| # | Feature (train / test) | Generador (script:linea) | Dependencias | Determinismo | Tiempo doc. | Forma final | Estado |
|---|---|---|---|---|---|---|---|
| 1 | `dense_train.parquet` / `dense_test.parquet` | `scripts/fase02_eda.py:68-79` -> `src/textometry.py:83-173` | `re`, numpy, pandas; `src.preprocessing._EMOJI_RE/_URL_RE` | total (sin RNG, sin fit) | no registrado en `experiments/02_eda.log` | (169336, 38) / (112892, 38); se usan 35 | ADOPTADO (35 de 38 en 279) |
| 2 | `age_structured_features.parquet` / `_test.parquet` | `scripts/phase3b_recorregido.py:78-88` (recalculo corregido); original `scripts/advanced_02_features.py:154-166` | `src/advanced_features.py`, `scipy.stats`, `zlib` | total | 935.9 s (recalculo); 388.1 s solo locales | (169336, 87) / (112892, 87) | ADOPTADO (87 en 279) |
| 3 | `age_supervised_features.npy` + `age_supervised_columns.txt` | `scripts/advanced_02_features.py:205-234` | `src/advanced_features.py:326-490`; folds de `cv.make_folds` | total dado el fold (sin RNG) | dentro de los 2689.4 s de ADV-02 | (169336, 73) float64 | ADOPTADO (73 en 279, in-fold) |
| 4 | `lexicon_features_{registro,vida,temas}.npy` + `_nombres.json` | `scripts/phase38_bloques_lexicos.py:74-81` -> `src/lexicons.py:94-204` | `lexicons/*.txt`, `unicodedata`, `hashlib` | total (funcion determinista del texto) | 314.1 s (json); 300.0 s (CSV) | (169336, 19 / 16 / 37) float32 | NO adoptado (D-039) |
| 5 | `classify_tweet_type` | `scripts/phase1_diagnostico.py:58-88` | `dense_train.parquet` (6 columnas) | total | dentro de PH1-L1 (403.0 s) | vector de etiquetas (8 tipos) | diagnostico / segmentacion |
| 6 | `length_residuals.npy` + `_columns.txt` | `scripts/phase5_residuales.py:87-120` | dense + structured; folds | total dado el fold | PH5-L2 363.6 s | (169336, 46) float64 | CERRADO (delta -0.0019) |
| 7 | `oof_ADV05_nbsvm.npy` (miembro 14) | `scripts/advanced_05_nbsvm_seleccion.py:46-49,76-88,146` | sklearn TF-IDF + LinearSVC | in-fold, sin RNG | 1467.0 s (ADV-05) | (169336, 6) float32 | ADOPTADO (miembro 279) |
| 8 | `oof_ADV07_nbsvm_word.npy` (cols 273-278) | `scripts/phase7_nbsvm_canal_meta.py:53-54,57-78,130` | idem, word(1,2) | in-fold, sin RNG | 931.5 s (PH7-L7) | (169336, 6) float32 | ADOPTADO (279) |
| 9 | `dup_clusters.parquet` (soporte) | `scripts/fase01_auditoria.py:252-255` -> `src/cv.py:118-229` | union-find; sin RNG | total | Fase 1 | 282228 filas x 3 col | soporte de folds (D-003) |
| 10 | `pos_features.parquet` / `v14_linguistico.npy` | `scripts/phase10_dos_vias.py` / `scripts/v14a_linguistico.py:231` | spaCy 3.8.13 + es_core_news_md/lg 3.8.0 | total | PH10-B / V14B | 26 / 33 col | NO en 279 (cerrados) |

## 1. Dense (textometria) — ADOPTADO

- Generacion: `scripts/fase02_eda.py:68-79` con cache; llamada `textometry.extract_textometry(train/test)`
  y `to_parquet(... dense_train.parquet / dense_test.parquet)`. Entrada: `data.load_train()/load_test()`
  (`src/data.py:248-270`, CSV crudo con `sep=";"` y `clean_text` = strip del padding de 140).
- Definicion de columnas: `src/textometry.py:51-62` (`TEXTOMETRY_FEATURES` 15 + `STYLE_FEATURES` 23 = 38).
  El bloque usado en el meta es:
  `DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]`
  (35 columnas; los 3 emojis se excluyen porque el corpus no tiene ninguno).
- Algoritmo (`src/textometry.py:99-172`): texto en minusculas; `words = lowered.str.split()`; n_chars,
  n_words, unicos, hapax, entropia de Shannon base 2 de los conteos de token; TTR, CTTR, hapax_ratio,
  stopword_ratio (lista fija `SPANISH_STOPWORDS`, lineas 39-48), content_ratio, long_word_ratio (>=8);
  estilometria por regex: `EMOTICON_RE` (26), `ELONGATED_RE` (28), `MULTI_PUNCT_RE` (30),
  `OPEN_PUNCT_RE` (32), `ALL_CAPS_RE` (34), `SENTENCE_SPLIT_RE` (36); `_EMOJI_RE`/`_URL_RE` vienen de
  `src.preprocessing`. Detalle fino: los emoticonos se cuentan sobre el texto **sin URLs**
  (`text_without_urls`, linea 104) porque `http://` contaminaba el conteo.
- Dependencias: solo stdlib + numpy/pandas; sin modelos, sin hashing, sin spaCy.
- Determinismo: total; features locales al documento sin parametros aprendidos (docstring 6-9).
- Test: `dense_test.parquet`, misma funcion sobre `age_test.csv`.

## 2. Estructuradas de segunda generacion — ADOPTADO

- Generacion vigente: `scripts/phase3b_recorregido.py:78-88` (recalculo tras el bug de `TOKEN_RE`).
  Version original: `scripts/advanced_02_features.py:154-166`. Ambas escriben
  `features/age_structured_features.parquet` y `features/age_structured_features_test.parquet`.
- Algoritmo: 6 familias locales de `src/advanced_features.py`, concatenadas en ese orden:
  1. `distribution_features` (75-111): 21 cols `wlen_*` (media/mediana/std/cv/skew/kurtosis/iqr/mad/range,
     p10-p90, tail/head_ratio, prop_len1/2/3/ge8/ge10). Skew/kurtosis = 0 si varianza nula.
  2. `char_distribution_features` (117-153): 10 cols `char_*` (ratios vocal/consonante/alfa/espacio/simbolo,
     entropia, Simpson, HHI, dominante, unicos).
  3. `lexical_richness_features` (159-189): 9 cols (mattr_25, root_ttr, log_ttr, maas, yule_k, simpson_d,
     dis_legomena_ratio, repeated_token_ratio, top_token_concentration).
  4. `word_shape_features` (195-226): 10 cols `shape_*` + `repeat_*`/`adjacent_repeat_ratio`; tokenizador
     `TOKEN_RE` (36-37) y `WORD_RE` (32) que ahora aceptan mayusculas.
  5. `position_features` (232-267): 19 cols `pos_*`, `first/last_token_is_*`, `first/last_word_*`.
  6. `structure_entropy_features` (273-306): 18 cols `punct_*_density` (9), punct_type_diversity,
     punct_transitions, token_entropy, punct_entropy, entity_count, entity_entropy,
     `compression_ratio`/`compression_gain` (zlib level 6, lineas 302-305).
- Correccion 3b: `ALNUM_RE = re.compile(r"\b(?=\w*\d)(?=\w*[A-Za-z])\w+\b")` (`phase3b_recorregido.py:42`)
  recalcula `shape_alphanumeric_ratio` (45-51) porque el tokenizador separa letras y digitos.
  Resultado documentado: 87 features, **0 columnas con varianza nula** (`experiments/phase3b_recorregido.json:1-5`).
- Dependencias: stdlib (`re`, `zlib`, `collections.Counter`) + numpy/pandas/scipy. Sin spaCy, sin hashing.
- Determinismo: total; tests de robustez de casos limite en `advanced_02_features.py:69-91`.
- Test: `age_structured_features_test.parquet` (misma funcion, textos de `age_test.csv`).

## 3. Supervisadas in-fold — ADOPTADO

- Generacion: `scripts/advanced_02_features.py:205-234`. Para cada fold LOFO
  (`cv.make_folds(y, groups=dup_clusters, n_splits=5, n_repeats=1, scheme="group")`, lineas 189-191) se
  calculan los 5 bloques con train del fold y se transforma la validacion; el resultado se reensambla por
  indice y se guarda: `np.save(... age_supervised_features.npy)` y
  `(cfg.FEATURES / "age_supervised_columns.txt").write_text("\n".join(...))`.
- Bloques (implementacion en `src/advanced_features.py`):
  - `supervised_entity_features` (326-393) x3: `mention`, `hashtag`, `rt_source`. Aprende
    `P(clase|entidad)` con suavizado `smoothing=20.0` sobre el prior del train del fold; 17 columnas por
    bloque: `{kind}_mean_c0..c5`, `{kind}_best_c0..c5`, `margin`, `entropy`, `known_count`,
    `unknown_ratio`, `dominance`. Sin entidad conocida: `[prior, prior, 0,0,0,0,0]` (linea 377).
  - `lexical_rarity_features` (396-434): frecuencias solo del train del fold; 12 cols `rarity_*`,
    `rare_token_ratio`, `very_rare_token_ratio`, `oov_ratio`, `hapax_token_ratio`,
    `top_frequency_ratio`, `lexical_novelty`, `lexical_familiarity`; OOV con `log10` centinela `-6.0` (419).
  - `class_lexicon_scores` (437-490): log-odds Monroe et al. 2008, `alpha0=100.0, min_count=5,
    top_k=3000`; 10 cols `lexicon_c0..c5`, `lexicon_margin`, `lexicon_entropy`, `lexicon_best_class`,
    `lexicon_std`.
- Orden exacto de columnas en el fichero: mention (17) + hashtag (17) + rt_source (17) + rarity (12) +
  lexicon (10) = 73 (verificado en `features/age_supervised_columns.txt`).
- Dependencias: numpy/pandas + `collections.Counter`; **sin spaCy** (todo regex: `MENTION_RE`,
  `HASHTAG_RE`, `RT_SOURCE_RE`, `WORD_RE`, lineas 28-32). Sin hashing.
- Determinismo: total dado el fold (usa `np.bincount`/`Counter`; no hay RNG). La aleatoriedad esta solo
  en el splitter: `StratifiedGroupKFold(shuffle=True, random_state=42)` (`src/cv.py:364-366`).
- Test: **no existe `.npy` de test**. El lado test se calcula fresco con el train completo en
  `scripts/phase25_submission_fresca.py:69-75` (misma funcion, `train` completo como fuente).

## 4. Lexicos (registro / vida / temas) — NO adoptado

- Generacion: `scripts/phase38_bloques_lexicos.py:74-81`:
  `matriz, nombres = lx.matriz_lexica(textos, lexicon, familias=(familia,))`; guarda
  `features/lexicon_features_<familia>.npy` y `..._nombres.json`. Corpus: solo train (no hay version test).
- Carga y agregados (`src/lexicons.py`): `carga_lexicones` (94-150) lee `lexicons/{familia}/*.txt`
  (24 ficheros: registro 6, vida 6, temas 12; una subcategoria se retira al cargar), translitera a ASCII (61-66),
  tokeniza `TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")` (48); resuelve ambiguedad DD-5 (un token en dos
  subcategorias se elimina) y retira subcategorias con <5 entradas (`MIN_ENTRADAS_SUBCATEGORIA`, 51):
  unico retiro documentado `vida/hogar` (2 entradas). `matriz_lexica` (159-204) emite por subcategoria
  `count`, `ratio = count/n_palabras` (n_palabras = max(len,1)), `presence`, mas `lex_<familia>_ratio_medio`
  por familia; todo float32.
- Columnas: registro 19 = 6x3+1; vida 16 = 5x3+1; temas 37 = 12x3+1 (total 72). Nombres exactos en los
  `_nombres.json` (p.ej. `lex_registro_abreviaciones_internet_count`, `lex_temas_religion_ratio`, ...).
- Dependencias: stdlib (`unicodedata`, `hashlib`, `re`) + numpy/pandas. SHA256 por fichero en
  `experiments/phase38_bloques_lexicos.json:116-140`. Sin spaCy, sin RNG.
- Determinismo: total; la transformacion texto -> categorias no usa etiquetas ni parametros (docstring
  `src/lexicons.py:3-6`).
- Resultado/cierre: control lexicon-only 0.2033 (temas) / 0.1520 (registro) / 0.0731 (vida) vs azar
  0.1667; 0 de 72 columnas con |rho|>=0.8 contra V12; fusion V12+bloques de -0.0001 a -0.0018 ->
  **CERRADA (D-039)**, artefactos conservados como recurso (`DECISIONS.md:813-834`,
  `experiments.csv:71-72`).

## 5. classify_tweet_type — diagnostico/segmentacion

- `scripts/phase1_diagnostico.py:58-88`. Entrada: `dense_train.parquet` con `n_chars`, `n_mention`,
  `n_hashtag`, `n_url`, `is_rt`, `n_words` (lineas 71-76).
- Logica exacta: asignaciones secuenciales sobre un array inicial `solo_texto`; **cada asignacion
  sobrescribe la anterior**, de modo que la prioridad efectiva es el inverso del listado:
  `muy_corto` (length<=15) > `rt_puro`/`rt_con_comentario` (is_rt==1; <80 / >=80 chars) >
  `mencion_alta` (mentions>=3) > `hashtag_alto` (hashtags>=3) > `url_alta` (urls>=1 & words>3) >
  `solo_mencion` (mentions>=2 & words<=3) > `solo_url` (urls>=1 & words<=3) > `solo_texto`.
  Consecuencia a retener: un tweet de 3+ menciones con <=3 palabras cae en `mencion_alta`, no en
  `solo_mencion`; un RT corto cae en `muy_corto`.
- Dependencias: numpy/pandas; sin spaCy. Determinismo: total (umbrales fijos).
- Uso: segmentacion descriptiva y `length_band = pd.qcut(n_chars, 4)` (`phase2_ponderacion.py:80`).

## 6. Otras derivadas y bloques de la matriz 279

- **Composicion 279** (docs: `INFORME_COMPLETO.md:80`; codigo: `scripts/v19_b1_boosting.py:97-108`):
  78 scores de 14 miembros (E013, E016, E-009, E-011, E-010, E014, E015_ordinal[1], E015_softmax,
  E017, E019b, E021a, E021b, E028a[5], ADV05) + 35 densas + 87 estructuradas + 73 supervisadas +
  6 NB-SVM de palabra (ADV07). `matriz_279()` = `meta_features_train_v11.npy` (273, guardada en
  `scripts/phase3_holdout_v11.py:148`) + `oof_ADV07_nbsvm_word.npy`.
- **ADV05** (miembro char+menciones): `CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6),
  "min_df": 5, "sublinear_tf": True, "max_features": 300_000, "lowercase": True}` y
  `MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "sublinear_tf": True,
  "lowercase": True}` (`advanced_05_nbsvm_seleccion.py:46-49`); `nb_log_count_ratio` (66-68) y
  LinearSVC(C=0.2, max_iter=3000) uno-vs-rest in-fold; OOF a `oof_ADV05_nbsvm.npy` (146).
- **ADV07** (NB-SVM de palabra): `WORD_CONFIG = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3,
  "sublinear_tf": True, "max_features": 300_000, "lowercase": True}` (`phase7_nbsvm_canal_meta.py:53-54`);
  mismo log-count ratio + LinearSVC; OOF a `oof_ADV07_nbsvm_word.npy` (130). Test fresco: vectorizador
  ajustado en train completo (`phase25_submission_fresca.py:92-102`).
- **length_residuals**: `phase5_residuales.py:47-66` selecciona features con |corr|>=0.25 con `n_chars`
  (46 finales); `conditional_mean` (69-84) estima `E[feature|n_chars]` con 30 bins de cuantiles del train
  del fold; residuo = feature - esperado; guarda `length_residuals.npy` (46 col float64) y
  `_columns.txt` (nombres `dense::*` / `struct::*`). NO adoptado: delta -0.0019 con MLP (`experiments.csv:42`).
- **Vista estructural** (`phase21_vista_estructural.py`): 6 margenes de LinearSVC sobre
  TF-IDF word(1,3) de tipos; candidatos locales NO adoptados (D-032; `phase25_submission_fresca.py:120-121`).
- **Soporte de folds**: `dup_clusters.parquet` se construye sobre train+test concatenados
  (`fase01_auditoria.py:252-255`); `make_folds` agrupa por `cluster` con
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` (`src/cv.py:332-375`). Es la unica
  fuente de aleatoriedad de este paquete y debe replicarse bit a bit en el notebook.
- **spaCy (fuera del paquete)**: solo lo usan las ramas cerradas POS/V14 (`phase10_dos_vias.py:107-109`,
  `v14a_linguistico.py:66-67,78,231`); versiones `spacy==3.8.13` y modelos `es_core_news_md/lg 3.8.0`
  (`requirements.txt:49-50,257`). Ninguna de las 4 familias de este paquete lo necesita.

## 7. Verificacion de la adopcion ADV-01 / AGE_ADVANCED_V01

El nombre "ADV-01" corresponde en `experiments.csv:32` a la **auditoria de brechas** (gap analysis, sin
metrica). La adopcion con los deltas citados es la de las features de segunda generacion, registrada como
**ADV-02 + ADV-03** y adoptada con el nombre de candidato **`AGE_ADVANCED_V01`** (submission v09):

- CV: `experiments.csv:33` -> macro-F1 0.58967 (std 0.00428), "macro-F1 0.5897 vs baseline 0.5866
  (delta +0,0031)", decision "Adoptar como AGE_ADVANCED_V01."; artefacto
  `experiments/advanced_02_resultados.json` (baseline 0.5865636; con_todo 0.5896715; 267 features;
  2689.4 s).
- Holdout: `experiments.csv:34` -> "holdout: base 0.5914 vs candidato 0.5966 (delta +0,0052)";
  artefacto `experiments/advanced_03_holdout.json` (share 0.15, seed 2026, 25204 filas, 267 features,
  delta_macro_f1 0.0052, 789.9 s).
- Decision D-017: `DECISIONS.md:332-336` ("CV 0,5866 -> 0,5897 (+0,0031), positiva en 5/5 folds;
  holdout congelado 0,5914 -> 0,5966 (+0,0052), confirmado... adoptar el candidato
  `AGE_ADVANCED_V01` (267 features del meta)").
- Leaderboard: `RUNLOG.md:290` (subida de v09, F1 0.60807, "+0,0043 sobre v08") y `RUNLOG.md:294`
  ("delta CV (LOFO): +0,0031 · delta holdout congelado: +0,0052 · delta leaderboard: +0,0043").
- Fuente exacta de la frase conjunta: `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:114`
  ("AGE_ADVANCED_V01 | Features supervisadas de entidad. CV +0.0031, holdout +0.0052, leaderboard
  +0.0043."); tambien `reports/age_complete_experiment_history.csv:33-34` y
  `experiments/auditoria_v1v37_inventario.json:865-890`.
- Aportes desglosados (D-017): menciones +0.0016, fuente RT +0.0007, rareza +0.0003, lexico por clase
  +0.0001, hashtags -0.0001 (coinciden con `advanced_02_resultados.json.aporte_supervisado`).

## 8. Notas de porte para el notebook

1. Orden de dependencias en el notebook: CSV crudos -> `dup_clusters` (soporte) -> dense -> structured ->
   folds -> supervised (in-fold) -> lexicos (opcionales, no adoptados).
2. Rutas crudas: `data/raw/age_train.csv` (169336 filas, sep `;`, cols id/text/age_range) y
   `data/raw/age_test.csv` (112892 filas). `clean_text` (strip) es obligatorio antes de textometria.
3. Los generadores son ASCII puro y se pueden copiar literalmente desde `src/textometry.py`,
   `src/advanced_features.py`, `src/lexicons.py`; el unico cambio de contexto es `cfg.FEATURES` -> cache local.
4. Version de entorno de referencia (`requirements.txt`): numpy 1.26.4, pandas 2.3.3, scipy 1.17.1,
   scikit-learn 1.8.0, pyarrow 24.0.0; spaCy NO se requiere para este paquete.
5. Determinismo: estas cuatro familias no consumen RNG (excepto el splitter de folds, seed 42). Los
   `set_seed()` de los scripts (`src/config.py:75-85`) existen por los modelos, no por las features.
