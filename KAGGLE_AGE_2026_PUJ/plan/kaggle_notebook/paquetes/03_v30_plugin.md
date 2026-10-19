# 03 — Paquete de porte: matriz 279, ADV07 (NB-SVM de palabra), V20/V30 y regla plug-in

**Alcance:** reconstruir en el notebook, desde los CSV crudos, la cadena exacta del campeón: matriz 279, miembro V20 (cuatro cuartos), ensemble V30 (V20 + dos LightGBM con configuraciones congeladas), regla plug-in por clase y todos sus checks.
**Fuentes:** `scripts/v19_b1_boosting.py`, `scripts/v13_p1b_fusion_tokens.py`, `scripts/phase2_ponderacion.py`, `scripts/phase3_holdout_v11.py`, `scripts/phase7_nbsvm_canal_meta.py`, `scripts/phase25_submission_fresca.py`, `scripts/v20_f3_umbrales.py`, `scripts/v20_f3c_candidato.py`, `scripts/v30_02_ensemble.py`, `scripts/v30_03_submission.py`, `scripts/v30_05_segunda_particion.py`, `scripts/age_plugin_submission.py`, `experiments/v30_01_optuna_refinamiento.json`, `reports/v29-01_trials.csv`, `reports/age_submission_plugin_manifest.json`. Código citado tal cual (ASCII).

## 0. Constantes comunes

- `cfg.CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]` (`src/config.py`); `cfg.SEED = 42`; `ID_COL = "id"`, `TEXT_COL = "text"`.
- Formas verificadas de los artefactos (cabeceras `.npy`): `meta_features_train_v11.npy` (169336, 273) f8; `oof_ADV07_nbsvm_word.npy` (169336, 6) f4; `meta_features_test_v12_fresca.npy` (112892, 279) f8; `v20_f3b_oof_candidato.npy` / `v29_lgbm_optuna_oof.npy` / `v30_lgbm_refinado_oof.npy` (169336, 6); `age_plugin_test_prob.npy` (112892, 6) f8; `artifacts/p33_scores/ortografia_composicion_discurso/v30_prob.npy` (169336, 6) f8.

## 1. `matriz_279()` — `scripts/v19_b1_boosting.py` (líneas 97-108)

```python
def matriz_279() -> np.ndarray:
    base = np.load(cfg.FEATURES / "meta_features_train_v11.npy")
    extra = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")
    matriz = np.concatenate([base.astype(np.float32), extra.astype(np.float32)], axis=1)
    assert matriz.shape[1] == 279, f"se esperaban 279 columnas y hay {matriz.shape[1]}"
    return matriz
```

**Origen del bloque de 273** — `scripts/phase3_holdout_v11.py` (líneas 141-148):

```python
x_meta_test = np.concatenate(test_scores + [nb_test, dense_test, structured_test,
                                            supervised_test], axis=1)
np.save(cfg.FEATURES / "meta_features_test_v11.npy", x_meta_test)
np.save(cfg.FEATURES / "meta_features_train_v11.npy", features)
```

- Lado train (273): `build_features()` de `scripts/phase2_ponderacion.py` = `MEMBER_FILES` (14 fuentes) + densas + estructuradas + supervisadas. `transform == "log"` aplica `np.log(np.clip(m, 1e-12, None))`; el resto va crudo.
- Desglose exacto por bloques (columnas): 78 miembros (13 fuentes del pipeline = 72: 11 archivos de 6 columnas + `E015_ordinal` 1 + `E028a_pares` 5; más `ADV05_nbsvm` 6) + 35 densas (`DENSE_FEATURES` = 15 textometría + 23 estilo - 3 emoji) + 87 estructuradas (`age_structured_features.parquet`, ADV-02) + 73 supervisadas (`age_supervised_features.npy`) = **273**. + 6 de ADV07 = **279**.
- Orden del bloque de miembros (así lo concatenan train y test): `E013_margenes`, `E016_margenes`, `E-009`, `E-011`, `E-010`, `E014_margenes`, `E015_ordinal`, `E015_softmax`, `E017_probabilidades`, `E019b_densas_svd`, `E021a_ordinal`, `E021b_rocchio`, `E028a_pares`, `ADV05_nbsvm`; luego densas, estructuradas, supervisadas; y al final (solo train-279) ADV07 palabra.
- **Aviso de porte:** `meta_features_test_v11.npy` (112892, 273) es la caché **desalineada** previa a la corrección `phase3b`; para el campeón se usa `meta_features_test_v12_fresca.npy`, construida por `matriz_test_fresca()` de `scripts/phase25_submission_fresca.py` (líneas 57-111), con orden `[13 scores del pipeline] + [NB-SVM char+menciones] + densas + estructuradas + supervisadas + [NB-SVM de palabra]` y `assert matriz.shape[1] == 279`.

## 2. ADV07 — NB-SVM de palabra

**OOF** — `scripts/phase7_nbsvm_canal_meta.py`:

```python
WORD_CONFIG = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3, "sublinear_tf": True,
               "max_features": 300_000, "lowercase": True}

def nb_log_count_ratio(x_train, binary_target):
    positive = np.asarray(x_train[binary_target == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(x_train[binary_target == 0].sum(axis=0)).ravel() + 1.0
    return np.log((positive / positive.sum()) / (negative / negative.sum()))
```

`nb_svm_word_margins(texts, y, train_idx, valid_idx)`: `TfidfVectorizer(**WORD_CONFIG)` ajustado in-fold; por clase `label` (uno-vs-rest) `LinearSVC(C=0.2, max_iter=3000)` sobre `x_train.multiply(ratio).tocsr()`; salida `decision_function` sobre el fold retenido. Se guarda en `features/oof_ADV07_nbsvm_word.npy` (línea 130) como **márgenes** float32, no probabilidades. El vector de márgenes entra **crudo** en la matriz 279.

**Versión de test** — `scripts/phase25_submission_fresca.py` (líneas 92-103; idéntica en `phase8_holdout_v12.py` 153-164): mismo vectorizador (`analyzer="word", ngram_range=(1,2), min_df=3, sublinear_tf=True, max_features=300_000, lowercase=True`) ajustado sobre **todo el train**; `ratio` por clase sobre todo el train; `LinearSVC(C=0.2, max_iter=3000)`; `nb_word_test[:, posicion] = modelo.decision_function(x_test_w.multiply(ratio).tocsr())`. Se concatena como **últimas 6 columnas** de `meta_features_test_v12_fresca.npy` (posiciones 273:279).

## 3. Cabezas lineales y árbol congelado

`logreg()` — `scripts/v13_p1b_fusion_tokens.py` (líneas 79-83):

```python
return Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                            random_state=cfg.SEED))])
```

`HIERBA` — `scripts/v19_b1_boosting.py` (líneas 83-86):

```python
HIERBA = {"objective": "multiclass", "num_class": len(cfg.CLASSES), "n_estimators": 300,
          "learning_rate": 0.05, "num_leaves": 8, "max_depth": 3, "min_child_samples": 200,
          "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0,
          "n_jobs": 32, "random_state": cfg.SEED, "verbose": -1}
```

## 4. V20 — cuatro cuartos, ajustados sobre todo el train

`_v20_en_test` — `scripts/v30_03_submission.py` (líneas 93-117). Es **0.25·LogReg + 0.25·MLP(64) + 0.25·árbol HIERBA + 0.25·cadena de 5 clasificadores binarios**, todos reajustados sobre el train completo:

```python
modelo_lr = logreg().fit(matriz, y)
red = EncodedMLP().fit(matriz, y)
modelo_arbol = lgb.LGBMClassifier(**HIERBA).fit(matriz, y)
acumuladas = np.zeros((n_test, N_CLASES - 1))
for k in range(1, N_CLASES):
    objetivo = (clase <= k - 1).astype(int)
    modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz, objetivo)
    acumuladas[:, k - 1] = modelo.predict_proba(test_matriz)[:, 1]
```

Monotonía `acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])`; distribución de 6 clases por diferencias (`seis[:,0]=acum[:,0]`, `seis[:,k]=acum[k]-acum[k-1]`, `seis[:,5]=1-acum[4]`), clip a >=0 y normalización. **Sí: cinco binarios**, todos con `HIERBA`. `EncodedMLP` = MLPClassifier(hidden (64,), alpha 1e-3, batch 512, lr 1e-3, max_iter 40, early_stopping, random_state 42) con `StandardScaler` y `LabelEncoder` (`scripts/phase2_ponderacion.py` 106-121). CV del candidato V20: **0.6097782** (`CV_V20` en `v30_02`/`v30_03`). Su OOF es `features/v20_f3b_oof_candidato.npy`, creado por `scripts/v20_f3_umbrales.py` (línea 213). La receta completa del test está también en `scripts/v20_f3c_candidato.py` (líneas 107-135).

## 5. Ensemble V30

**Miembros y pesos** — `experiments/v30_02_ensemble.json` → `manifest`:

| Miembro | Peso congelado | Qué es |
|---|---|---|
| `v20_f3b_oof_candidato` | 0.3548 | los cuatro cuartos de V20, intactos |
| `v29_lgbm_optuna` | 0.3354 | LightGBM de V29, mejor trial por fold |
| `v30_lgbm_refinado` | 0.3098 | LightGBM de V30, mejor trial por fold |

**De dónde salen los pesos** — `scripts/v30_02_ensemble.py`: por fold se optimizan con Optuna (300 trials, `theta` en [-4,4] por miembro, `w = softmax(theta)`) sobre un **holdout interior** estratificado del 20 % del train del fold (el exterior nunca se toca); la combinación se aplica al fold retenido. Los pesos congelados son la **media de los cinco vectores por fold, renormalizada** (líneas 218-220: `medios = np.mean([...], axis=0); medios = medios / medios.sum()`), guardada redondeada a 4 decimales. Candidato principal: `"E1 Optuna macro-F1"`. Pesos por fold no persistidos; solo la media.

| Método | corrected_cv (mezcla por fold) | Pesos |
|---|---|---|
| E1 Optuna macro-F1 (elegido) | 0.6130051 | 0.3548 / 0.3354 / 0.3098 |
| E2 macro-F1 menos lambda·entropía | 0.6121866 | 0.2164 / 0.3150 / 0.4685 |

CV individuales: V20 0.6097782; V29-LGBM 0.6121698; V30-LGBM 0.6122387 (`v29_04_optuna_lightgbm.json`, `v30_01_optuna_refinamiento.json`).

**Configuraciones congeladas** — `_configs_del_miembro(nombre)` (`scripts/v30_05_segunda_particion.py` 119-143): para V29 lee `reports/v29-01_trials.csv` (la fila con `es_el_mejor` de cada fold; el artefacto de V29 no persistió los params) y para V30 lee `experiments/v30_01_optuna_refinamiento.json` → `detalle_por_fold[*].params_mejores`, en orden de fold; claves `p_` sin prefijo e int si el valor es entero. Tablas (valores tal como los usa el refit):

| V29 fold | num_leaves | max_depth | learning_rate | min_child_samples | subsample | colsample_bytree | reg_alpha | reg_lambda | n_estimators |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 87 | 5 | 0.04301669500288421 | 76 | 0.6581570543882859 | 0.70766450002428 | 0.6147385859030499 | 9.792145504680492e-05 | 816 |
| 1 | 55 | 6 | 0.03352698655336303 | 54 | 0.8941120537508466 | 0.6299160105578635 | 4.026902217415037e-07 | 6.6850377528309e-07 | 1372 |
| 2 | 57 | 8 | 0.022124342973521553 | 94 | 0.9225205673898446 | 0.52058407816027 | 1.8772169842126896 | 0.018194045769254683 | 912 |
| 3 | 43 | 9 | 0.03000286836410436 | 22 | 0.6998254389554196 | 0.5250455092128341 | 1.0090819656249398e-05 | 7.87982213180386e-06 | 863 |
| 4 | 56 | 7 | 0.020811077189914432 | 29 | 0.7762118181212365 | 0.6242271327287489 | 1.3833851148235308e-05 | 0.0004885693686069363 | 1084 |

| V30 fold | num_leaves | max_depth | learning_rate | min_child_samples | subsample | colsample_bytree | reg_alpha | reg_lambda | n_estimators |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 41 | 10 | 0.015612823179143214 | 79 | 0.775913545060191 | 0.5969526009433974 | 1.0941449983677585e-05 | 0.00036896192791107643 | 1011 |
| 1 | 32 | 7 | 0.017387326275059966 | 23 | 0.8215355497227395 | 0.8291548707315466 | 0.8747434883808647 | 2.053077365511584e-05 | 1486 |
| 2 | 63 | 7 | 0.02867292541012883 | 110 | 0.7807106452264457 | 0.4021908926823267 | 1.1486360263326448e-05 | 0.00031823134119350724 | 1023 |
| 3 | 42 | 7 | 0.015729268043126833 | 41 | 0.999685646683389 | 0.596195315949217 | 0.004978399116554409 | 1.8436108646384413e-08 | 1477 |
| 4 | 64 | 6 | 0.03818854692388361 | 52 | 0.8731483359860639 | 0.46934472778342573 | 1.927060050589197e-08 | 0.004339321331656464 | 1000 |

Notas: V29 fija `subsample_freq = 1` en el CSV; en V30 lo aporta el constructor del refit. `_lgbm_en_test` (v30_03, 147-168) arma `p = {"objective": "multiclass", "num_class": 6, "verbose": -1, "n_jobs": 28, "subsample_freq": 1, **params}`, saca `n_estimators` y ajusta **sin early stopping** sobre todo el train; promedia las **5 predicciones** (una por fold) por miembro.

**Mezcla y test** — `v30_03_submission.py` (232-240): `vector = pesos / suma` y `prob_candidata = np.tensordot(vector, np.stack([...]), axes=(0, 0))`; cada miembro pasa por `a_clases(...)` (v30_03, 72-90), que reordena a `cfg.CLASSES` (tolera `classes_` string o enteros 0-5). Matriz de test: `meta_features_test_v12_fresca.npy` (112892, 279).

- Probabilidades de test del pipeline plug-in: `age_plugin_submission.py` (253-254) guarda la mezcla V30 **sin** plug-in en `features/age_plugin_test_prob.npy` (112892, 6), columnas en orden `cfg.CLASSES`.
- Submission V30 (sin plug-in): `submissions/age_submission_v30_final_candidate.csv`, SHA256 `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be`, 10/10 checks, cambia 7253 filas respecto del envío V20 (`experiments/v30_03_submission_construccion.json`).

## 6. Regla plug-in — `scripts/age_plugin_submission.py`

**Pesos exactos** (línea 79): `PESOS_CONGELADOS = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])` — **con 4 decimales**, no `[1.00, 1.17, 1.27, 1.16, 1.00, 0.81]`. La variante redondeada a 2 decimales está documentada como la tercera cifra distinta: `(C) redondeados 0.6159908`.

**De dónde salen** (`scripts/age_umbrales_por_clase.py`): ascenso por coordenadas en log-espacio (`optimiza_pesos`), ajustado **dentro** de cada fold externo con los otros cuatro como validación interna sobre el OOF canónico `v30_prob.npy`; el vector congelado es la **media de los 5 vectores por fold** (`age_umbrales_validacion.json` → `pesos.media = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`; `pesos.por_fold` con 5 filas). Congelado en el pre-registro commit `bd08aba`; `age_plugin_submission.py` no contiene optimizador.

**La regla** (258-261):

```python
score = prob * PESOS_CONGELADOS
pred_base = np.array(cfg.CLASSES)[np.argmax(prob, axis=1)]
pred_plug = np.array(cfg.CLASSES)[np.argmax(score, axis=1)]
```

**Verificación diagonal** (`verifica_orden_de_clases`, 121-147): sobre `artifacts/p33_scores/ortografia_composicion_discurso/{y_true.npy, v30_prob.npy}` calcula `E[p[:, j] | y = i]` y exige `argmax` de cada fila en la diagonal. Resultado registrado (`reports/age_submission_plugin_manifest.json`): diagonal 0.5913 / 0.4071 / 0.3979 / 0.4676 / 0.5667 / 0.6401, `argmax_por_fila [0,1,2,3,4,5]`, `orden_verificado true`. Si falla: STOP.

**Sanity checks OOF** (`sanity_check`, 150-185; hay TRES números y solo uno es la operación de la submission):
- (A) nested 0.6154643: cada fold con SUS pesos (estimación honesta del mecanismo).
- (B) congelados globales **0.6159732** ← lo que ejecuta la submission; es el que verifica el sanity check.
- (C) redondeados a 2 decimales 0.6159908.
- Canónica: V30 0.6134471 → nested 0.6154643 → frozen-global 0.6159732. Alternativa (semilla 987654, `age_plugin_confirmacion.json`): V30 0.6144120 → congelado 0.6164700 (+0.0020580, 5/5 folds) y nested 0.6167003 (+0.0022884).
- `reproduce = |v30-0.6134471|<1e-6 and |frozen_global-0.6159732|<1e-6 and cambios_alta_confianza==0` (umbral alta confianza 0.60). Si falla: STOP. Registrado: `v30 0.6134471`, `frozen_global 0.6159732`, `filas_que_cambian 11344`, `cambios_en_alta_confianza 0`, `reproduce true`.

**Todos los checks previos a generar la submission** (flujo [0/6]..[6/6]): (0) orden de clases + STOP; (1) sanity check + STOP; (2) esquema: `n_test == 112892` y `test_matriz.shape[1] == matriz.shape[1]` + STOP; (3) tres miembros con receta congelada; (4) regla y diagnóstico; (5) diez checks del fichero (filas, columnas exactas `[id, age_range]`, ids únicos, ids iguales al test, mismo orden, sin NaN, clases válidas, sin columnas auxiliares, todas las clases presentes, sin duplicados) + STOP si falla alguno; (6) manifest. En test: `filas_que_cambian_respecto_de_v30 = 7408`, `cambios_en_alta_confianza = 0`, bootstrap IC95 canónica [0.0010143, 0.002948] y alternativa [0.0011204, 0.0030622].

**Artefactos sellados:** submission `submissions/age_submission_plugin_candidate.csv`, SHA256 `29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0`; manifest `reports/age_submission_plugin_manifest.json` (pesos, regla, hashes de `v30_02_ensemble.json`, `v30_03_submission_construccion.json`, `v30_lgbm_refinado_oof.npy`, `meta_features_test_v12_fresca.npy`, hashes de textos train/test, `git_commit`); resultado `experiments/age_plugin_submission.json`.

## 7. Números documentados (con fuente)

| Magnitud | Valor | Fuente |
|---|---|---|
| corrected_cv V30 (canónico, pesos globales sobre OOF) | 0.6134471 (full: 0.6134471410431589) | `CHAMPION.md`; `scripts/p33_00_scores_url.py` (`CV_V30`); `experiments/v35_00g_techo_analisis.json` (`base.macro_f1_argmax_v30`) |
| V30 mezcla por fold (E1, contexto) | 0.6130051 | `experiments/v30_02_ensemble.json` (`tabla`) |
| plug-in frozen-global canónico | 0.6159732 (full: **0.6159732487657944**) | `experiments/v32_01_surrogate_macrof1.json`, `v35_00g_techo_analisis.json` (`base.macro_f1_con_plugin`), `reports/age_submission_plugin_manifest.json` |
| plug-in nested canónico | 0.6154643 | `experiments/age_umbrales_por_clase.json`, `age_plugin_submission.json` |
| plug-in alternativa (987654) | 0.6164700 (V30 0.6144120) | `experiments/age_plugin_confirmacion.json`; docstring `age_plugin_submission.py` |
| LB V30 | 0.63302 | `CHAMPION.md`; `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md`; `reports/age_plugin_external_transfer.md` |
| LB plug-in | 0.63575 | ídem |
| SHA256 submission plug-in | `29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0` | `reports/age_submission_plugin_manifest.json` |
| SHA256 submission V48 | `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62` | `experiments/v48_01_submission_construccion.json` |

## 8. Notas de porte (asserts para el notebook)

1. Reconstruir el OOF de cada miembro in-fold con los folds canónicos (`make_folds` agrupado por `dup_clusters`, semilla 42) y las 279 columnas; reconstruir el lado test con `meta_features_test_v12_fresca.npy` o rehacerlo con la secuencia de §1-§2.
2. Asserts: matriz train `(169336, 279)`; test `(112892, 279)`; `age_plugin_test_prob.npy` `(112892, 6)`; OOF V30 canónico `(169336, 6)`; media fila-a-fila del test de la mezcla ~1.0; columnas en orden `cfg.CLASSES`.
3. El `v30_prob.npy` canónico se reproduce con `p33_00_scores_url.py` (132-146): tensordot de `manifest.pesos / suma` sobre `[v20, v29, v30]` (v20/v29 desde `experiments/v29_pool_oof.npz` o `features/*.npy`, v30 desde `features/v30_lgbm_refinado_oof.npy`). Sirve de ancla del sanity check plug-in.
4. La submission se decide con `argmax(prob * PESOS_CONGELADOS)`; `prob` es la mezcla V30 sin transformar. No calibrar, no normalizar después del producto.
