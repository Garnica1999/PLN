# Paquete 01 — Miembros del meta-modelo y meta-modelo (cadena V11/V12)

> Fuente: repo `KAGGLE_AGE_2026_PUJ` @ `e43c594`, 2026-10-17. Solo lectura; sin ejecutar scripts.
> Comun a los 14 miembros: `src/config.py:62` `CLASSES = ["13-17","18-24","25-34","35-49","50-64","65-xx"]`
> y `src/config.py:70` `SEED = 42`. Salidas OOF en `features/*.npy` (float32), salvo la matriz del meta (float64).

## 1. `scripts/phase2_ponderacion.py` (definicion del meta-modelo)

### 1.1 MEMBER_FILES — 14 entradas (fichero, transformacion) — lineas 49-55
```
MEMBER_FILES = [("oof_E013_margenes.npy", "raw"), ("oof_E016_margenes.npy", "raw"),
                ("oof_E-009.npy", "log"), ("oof_E-011.npy", "log"), ("oof_E-010.npy", "log"),
                ("oof_E014_margenes.npy", "raw"), ("oof_E015_ordinal.npy", "raw"),
                ("oof_E015_softmax.npy", "log"), ("oof_E017_probabilidades.npy", "log"),
                ("oof_E019b_densas_svd.npy", "log"), ("oof_E021a_ordinal.npy", "log"),
                ("oof_E021b_rocchio.npy", "log"), ("oof_E028a_pares.npy", "raw"),
                ("oof_ADV05_nbsvm.npy", "raw")]
```
- `log` = `np.log(np.clip(matrix, 1e-12, None))` (linea 73); `raw` = sin transformacion.
- Vectores 1-D se reescalan: `matrix = matrix.reshape(-1, 1)` (lineas 71-72).
- Bloques de columnas (0-77): E013 0-5, E016 6-11, E-009 12-17, E-011 18-23, E-010 24-29, E014 30-35,
  E015_ordinal 36, E015_softmax 37-42, E017 43-48, E019b 49-54, E021a 55-60, E021b 61-66, E028a 67-71, ADV05 72-77.

### 1.2 DENSE_FEATURES (56-57) y MIN_SEGMENT (58)
```
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
MIN_SEGMENT = 800
```
- 15 textometria + 23 estilo − 3 emoji = **35 columnas** (`src/textometry.py:51-62`).
- `MIN_SEGMENT` solo lo usa `merge_small_segments` (84-89) en `evaluate_segmented` (no toca la matriz).

### 1.3 `build_features()` (61-81) — concatenacion exacta
```
features = np.concatenate([np.concatenate(parts, axis=1),
                           dense[DENSE_FEATURES].to_numpy(float), structured, supervised],
                          axis=1)
```
- Fuentes: `data.load_train()`; `features/dense_train.parquet` (35 col); `features/age_structured_features.parquet`
  (87 col); `features/age_supervised_features.npy` (73 col). Total 78+35+87+73 = **273**.
- Orden final: `[14 miembros (78)] [densas (35)] [estructuradas (87)] [supervisadas (73)]`.
- Metadatos que NO entran a la matriz: `tweet_type = classify_tweet_type(dense)` (import de
  `scripts/phase1_diagnostico.py`, linea 63) y `length_band = qcut(n_chars, 4)` (80).

### 1.4 Modelos
- `EncodedMLP` (106-125): LabelEncoder + StandardScaler + `MLPClassifier(hidden_layer_sizes=(64,),
  alpha=1e-3, batch_size=512, learning_rate_init=1e-3, max_iter=40, early_stopping=True,
  n_iter_no_change=5, random_state=cfg.SEED)`.
- `make_model` (128-136): `"logreg"` -> `Pipeline(StandardScaler, LogisticRegression(C=1.0, max_iter=3000,
  solver="lbfgs", random_state=cfg.SEED))`; `"mlp"` -> `EncodedMLP()`; otro -> `ValueError`.
- `evaluate_global` (92-103): LOFO sobre los folds; entrena con `train_idx = concat(val_idx de los otros 4)`
  y evalua macro-F1 con `labels=cfg.CLASSES`.
- Folds del meta (179-181): `cv.make_folds(y, groups=dup_clusters["cluster"][:len(y)], n_splits=5, n_repeats=1, scheme="group")`.

## 2. Miembros: tabla resumen

Folds (los 14): `cv.make_folds(y, groups=dup_clusters["cluster"][:169336], n_splits=5, n_repeats=1,
scheme="group")` -> `StratifiedGroupKFold(shuffle=True, random_state=42)` (`src/cv.py:332-374`; seed 42).
`dup_clusters` lo genera `scripts/fase01_auditoria.py:254-255` con `cv.build_dup_clusters` (Jaccard >= 0.8
sobre texto normalizado; `src/cv.py:34-42,118-156`). Los 5 folds son identicos en todos los miembros.

Test: ningun miembro guarda `.npy` de test. El lado test se regenera re-ajustando los mismos modelos sobre
todo el train en `meta_features_test_v11.npy` (phase3) y `meta_features_test_v12_fresca.npy` (phase25).

| # | .npy (forma; dtype) | Generador (script:lineas) | Modelo | Folds | Test | Runtime (fuente) |
|---|---|---|---|---|---|---|
| 1 | oof_E013_margenes (169336,6; f32) | experimento_e013_barrido_c.py:80-92,112 | LinearSVC C=0.2 | 5g/42 | no / pipeline svm300 | 394.4 s (barrido_c_resultados.json) |
| 2 | oof_E016_margenes (169336,6; f32) | experimento_e016_e017.py:76-87,126 | LinearSVC C=0.2 | 5g/42 | no / pipeline svm700 | 271.0 s (e016_e017_resultados.json) |
| 3 | oof_E-009 (169336,6; f32) | experimento_e009_modelos_clasicos.py:55-61,171-175 | LogReg OvR C=4 | 5g/42 | no / pipeline logreg | 541.0 s fit (modelos_clasicos_resultados.json) |
| 4 | oof_E-011 (169336,6; f32) | idem:68-74 | SGD log_loss | 5g/42 | no / pipeline sgd | 21.8 s fit (idem) |
| 5 | oof_E-010 (169336,6; f32) | idem:62-67 | ComplementNB a=0.3 | 5g/42 | no / pipeline cnb | 2.0 s fit (idem) |
| 6 | oof_E014_margenes (169336,6; f32) | experimento_e014_e015_hipotesis_error.py:101-112,151 | LinearSVC C=0.2 + hashtags | 5g/42 | no / pipeline hashtags | 205.3 s (hipotesis_error_resultados.json) |
| 7 | oof_E015_ordinal (169336,; f32) | idem:122-128,152 | Ridge alpha=1.0 | 5g/42 | no / pipeline ridge | 205.3 s (idem) |
| 8 | oof_E015_softmax (169336,6; f32) | idem:129-132,153 | softmax(-|d|/5) | 5g/42 | no / pipeline softmax | 205.3 s (idem) |
| 9 | oof_E017_probabilidades (169336,6; f32) | experimento_e016_e017.py:98-108,127 | LogReg OvR word 1-2 | 5g/42 | no / pipeline word | 271.0 s (idem) |
| 10 | oof_E019b_densas_svd (169336,6; f32) | experimento_e019_modelos_densos.py:113-131,157 | MLP(256,128)+SVD64 | 5g/42 | no / pipeline mlp | 1144.1 s CPU (e019_modelos_densos.json) |
| 11 | oof_E021a_ordinal (169336,6; f32) | experimento_e021_ordinal_centroides.py:57-75,150 | ordinal Frank&Hall | 5g/42 | no / pipeline ordinal | 331.0 s (e021_ordinal_centroides.json) |
| 12 | oof_E021b_rocchio (169336,6; f32) | idem:78-91,151 | centroides Rocchio | 5g/42 | no / pipeline rocchio | 331.0 s (idem) |
| 13 | oof_E028a_pares (169336,5; f32) | experimento_e028_ordinal_fino.py:92-106,147 | 5 SVM de pares | 5g/42 | no / pipeline pares | 169.5 s (e028_ordinal_fino.json) |
| 14 | oof_ADV05_nbsvm (169336,6; f32) | advanced_05_nbsvm_seleccion.py:73-94,136-146 | NB-SVM OvR | 5g/42 | no / nb_test (phase3:121-128; phase25:83-90) | 1466.4 s (advanced_05_nbsvm_seleccion.json) |

### Notas por miembro (columnas en orden CLASSES salvo indicacion)

1. **E013** (`barrido_c`): representacion char_wb 3-6 (min_df=5, max_features=300k) + menciones word 1-1
   (min_df=2); C barrido {0.1,0.2,0.3,0.5,1.0}, mejor C=0.2 (`oof_E013_bestC.npy`). Margenes
   `decision_function` reordenados a CLASSES (91-92). Tambien guarda `oof_E013_labels.npy`. CV 0.5628 (RUNLOG:140).
2. **E016** (`e016_e017`): mismo SVM con char_wb 3-6 (min_df=3, max_features=700k) + menciones. CV 0.5658 (RUNLOG:161).
3. **E-009** (`e009`): `OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
   random_state=42))`; `predict_proba` reordenada a CLASSES (148-151); guarda `oof_E-009_labels.npy`. CV 0.5603.
4. **E-011**: `SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4, random_state=42, n_jobs=32)`. CV 0.5579.
5. **E-010**: `ComplementNB(alpha=0.3)`. CV 0.5370.
6. **E014**: char300 + menciones + hashtags word 1-1 (min_df=2); `LinearSVC(C=0.2, max_iter=3000)`. CV 0.5644 (RUNLOG:160).
7. **E015_ordinal**: `Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=42)` sobre char300+menciones
   con objetivo = `CLASS_MIDPOINTS` (15/21/29.5/42/57/70; lineas 55-57); salida continua 1-D. CV 0.3973 (RUNLOG:160).
8. **E015_softmax**: `exp(-|d|/5.0)` normalizado sobre los 6 puntos medios (129-132). Aporta al stack (RUNLOG:175).
9. **E017**: word 1-2 (min_df=3, 300k) + LogReg OvR C=4.0 (mismos args que E-009). CV 0.5015 (RUNLOG:161).
10. **E019b**: `TruncatedSVD(n_components=64, random_state=42)` ajustada in-fold sobre char300+menciones;
    `[35 densas + SVD(64)]` + StandardScaler; `MLPClassifier(hidden_layer_sizes=(256,128), alpha=1e-3,
    batch_size=512, learning_rate_init=1e-3, max_iter=60, early_stopping=True, n_iter_no_change=6,
    random_state=42)`; `LabelEncoder` con assert de orden CLASSES (98-99). CPU por GPU ocupada (D-007, lineas 13-16). CV 0.4071 (RUNLOG:209).
11. **E021a**: ordinal acumulativo Frank & Hall (57-75): 5 binarios `P(y>k)` con `LogisticRegression(C=1.0,
    max_iter=1000, solver="liblinear", random_state=42)`; `P(y=k)=diff` con clip 1e-12 y normalizacion. CV 0.5527 (RUNLOG:241).
12. **E021b**: Rocchio (78-91): centroide por clase normalizado L2; similitud coseno; softmax temperatura 8.0. CV 0.4017.
13. **E028a**: 5 `LinearSVC(C=0.2, max_iter=3000)` entrenados SOLO con cada par adyacente; margen positivo ->
    clase mayor; columnas = `ADJACENT_PAIRS` [13-17|18-24, 18-24|25-34, 25-34|35-49, 35-49|50-64, 50-64|65-xx]
    (53-54). Precision media en fronteras 0.7428 (RUNLOG:259).
14. **ADV05**: NB-SVM OvR (Wang & Manning): log-count ratio por clase (66-70) sobre char300+menciones +
    `LinearSVC(C=0.2, max_iter=3000)` por clase (85-90); gate fold 0 > 0.54 (dio 0.5672). CV 0.5641 (RUNLOG:304).
    No usar `advanced_06_submission_v10.py` para el orden: alli NB va al final (110-121, cadena v10 de 10 fuentes).

## 3. Matrices del meta: quien las construye y orden exacto

- **`meta_features_train_v11.npy`** (169336,273; f64) — `scripts/phase3_holdout_v11.py:148` (= `build_features()`,
  orden de la seccion 1.3). Verificado reproducible bit a bit: RUNLOG:442 ("169.336 x 273; diferencia maxima
  absoluta 0,0") y `scripts/v19_b1_boosting.py:100-104`.
- **`meta_features_test_v11.npy`** (112892,273; f64) — `phase3_holdout_v11.py:108-148`. Orden exacto:
  `[13 scores del pipeline (72)] + [nb_test (6)] + [densas (35)] + [estructuradas (87)] + [supervisadas (73)]`,
  con `assert x_meta_test.shape[1] == features.shape[1]` (145-146). Las 13 fuentes = MEMBER_FILES[0:13] en el
  mismo orden, re-ajustadas sobre todo el train en `src/pipeline.py:199-288`: svm300, svm700, logreg OvR, sgd,
  cnb, hashtags, ridge ordinal (1 col), softmax ordinal (6; ya en log), word, MLP densas+SVD, ordinal
  acumulativo (log), rocchio (log), pares (5). Configs congeladas en `src/pipeline.py:42-51` (= E013/E016/E014/E017).
- Integridad del test cacheado: `phase24` compara bloque a bloque contra reconstruccion fresca (72/6/35/87/73,
  max|delta| <= 3.1e-05, correlacion 1.0; densas/estructuradas/supervisadas exactas) y `phase25` da 100% de
  acuerdo de predicciones. La submission v12 SUBIDA solo coincide 93.88% con la reconstruccion (D-034,
  trazabilidad del artefacto; RUNLOG:531-533; no es desalineacion).
- **Cadena V12 (279 col)**: train = `meta_features_train_v11.npy` (273) + `oof_ADV07_nbsvm_word.npy` (6)
  (`phase8_holdout_v12.py:63-64`; `phase25:127-129`). OOF de palabras generado en `phase7_nbsvm_canal_meta.py:130`
  (NB-SVM word 1-2 min_df=3 300k, LinearSVC C=0.2; CV 0.5254 RUNLOG:364). Test =
  **`meta_features_test_v12_fresca.npy`** (112892,279; f64), construido en `phase25_submission_fresca.py:57-111`
  y guardado en :137 con orden `test_scores(72) + nb_test(6) + densas(35) + estructuradas(87) + supervisadas(73)
  + nb_word_test(6)`; `assert matriz.shape[1] == 279` (:110). `phase8` habia construido el lado test en proceso
  sin cache (117-171).

## 4. Valores documentados de la cadena (con fuente)

- Meta MLP vs LogReg (tras fix phase3b): **0.5967** vs **0.5906** (+0.0061, 5/5 folds) — phase3b_recorregido.json; RUNLOG:338.
- V11: CV **0.5966 +/- 0.0040**; holdout MLP **0.6024** vs LogReg 0.5981 (delta +0.0043, confirmado) —
  phase3_holdout_v11.json (candidato); RUNLOG:332-341. LB v11 = 0.61744 (RUNLOG:342).
- V12: CV promedio de combinadores **0.6005** — phase7/phase8 (RUNLOG:364); canonico pooled 0.6005 vs media-de-folds
  0.6004 (efecto Jensen; folds 0.6007/0.6050/0.6032/0.6004/0.5928) — RUNLOG:442,733-735. Holdout **0.6077**
  (LogReg 0.5983, MLP 0.6034; delta +0.0043) — phase8_holdout_v12.json. LB enviado **0.62182** (RUNLOG:378;
  phase12_auditoria_estado.json `public_score`); la replica reproducible puntuo 0.62188 (RUNLOG:1088-1089,1179).
  `CHAMPION.md:22`: V12 CV 0.6005000 / LB 0.62182.
- Meta V12 exacto: `0.5 * (predict_proba(LogReg) + predict_proba(EncodedMLP))` + argmax en CLASSES
  (`phase8:80-107,172-182`; `phase25:141-153`).
- Runtime cadena: matrices del meta **796.6 s** (phase3 json; docstring "12 minutos"); phase3b 935.9 s;
  phase7 931.5 s; phase8 932.2 s; phase24 647.9 s; phase25 1723.2 s.

## 5. Notas de porte para el notebook

- Los folds dependen de `features/dup_clusters.parquet` (near-duplicados Jaccard >= 0.8): sin ese cluster no se
  reproducen los OOF. Generarlo con `cv.build_dup_clusters` sobre `concat(train.text, test.text)`.
- `oof_E015_ordinal.npy` es 1-D: reshape a columna antes de concatenar (phase2:71-72).
- El orden de `pipeline.OOF_FILES` (13, sin ADV05) difiere de `MEMBER_FILES` (14); la matriz test usa NB en medio,
  no al final (el orden "NB al final" es de la cadena v10, `advanced_06:110-121`).
- Dependencias: solo sklearn/numpy/scipy/pandas; la SVD de E019b es `TruncatedSVD` in-fold; sin spaCy en esta
  cadena (POS fue una linea cerrada aparte, RUNLOG entrada 89). E019b se documento en CPU.
- Deriva de versiones: el proyecto local uso sklearn 1.8.0 (nota de `EncodedMLP` sobre `np.isnan` en early
  stopping); el servidor/Kaggle puede diferir (plan maestro, Fase 0). Declarar tolerancias si aplica.
