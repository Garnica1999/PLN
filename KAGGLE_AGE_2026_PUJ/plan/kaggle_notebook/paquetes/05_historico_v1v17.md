# 05 — Paquete de porte: fase historica V1–V17 (fundacional, PH y busqueda residual)

**Alcance:** documentar la fase V1–V17 (los experimentos que van de la primera baseline al cierre de la busqueda residual) para el notebook independiente de Kaggle, que re-ejecuta desde los CSV crudos (`data/raw/age_train.csv` = 169.336 filas; `data/raw/age_test.csv` = 112.892 filas).
**Convencion de versiones:** "Vn" = version de submission (v01..v12); V13–V17 = fases de busqueda residual (bloques 7.0–10.0 de `experiments.csv`). La clasificacion por familia del inventario de `reports/auditoria_experimentos_v1_v37.md` es orientativa (ver §6, trampa 6).
**Fuentes primarias:** `experiments.csv` (216 filas), `RUNLOG.md` (sesiones 1–20), `notebooks/99_entrega_consolidado.ipynb` (celdas 9–23 y 37–44), `FINAL_REPORT.md` §9, `reports/` (v13_*.md, v14_*.md, v15_*.md, v16_*.md, v17_*.md, auditoria_experimentos_v1_v37.md), `experiments/*.json` y `submissions/`/`reports/submissions_registry.csv`. Codigo citado tal cual (ASCII puro).

## 0. Denominadores comunes (obligatorios en toda re-ejecucion)

- **Folds (primarios):** `src/cv.py::make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")`; `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` agrupado por clúster de near-duplicados. `cfg.SEED = 42`, `cfg.N_SPLITS = 5`, `cfg.CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]`.
- **Clusters:** `build_dup_clusters` sobre train+test (sin etiquetas): texto normalizado identico **o** Jaccard de tokens `>= 0.8` con blocking por 3 tokens raros (df<=200); cache `features/dup_clusters.parquet`. Test cubierto por un cluster de train: **4,2554%** (`experiments/audit_01_resumen.json`).
- **Metrica:** macro-F1 sobre las 6 clases; CV = media de 5 folds (los scripts viejos) o pooled (fases modernas; §6 trampa 1). Brecha CV↔LB calibrada: **+0,0177 ± 0,0023** en 10 mediciones (RUNLOG, notebook 99 c21).
- **Holdout congelado (cuando se cite):** 15% de clústeres, semilla 2026, 25.204 filas / 24.602 clústeres (ADV-03 en adelante).
- **Leyenda Re-ej:** `[R]` re-ejecutable barato desde CSV crudos (vectorizadores + lineales, minutos); `[R2]` re-ejecutable solo reconstruyendo antes la cadena completa de OOF (horas); `[X]` no re-ejecutable barato / depende de artefactos o modelos externos: documentar el numero, no re-correr.

## 1. V1–V4 fundacional (representaciones, baselines, modelos, meta-modelo)

| Exp | Script | Configuracion exacta | CV | LB (v) | Fuente | Re-ej |
|---|---|---|---|---|---|---|
| E-001 | `scripts/experimento_e001_wordchar_linearsvc.py` | TF-IDF `word(1,2)` min_df=3 + `char_wb(3,5)` min_df=5, `sublinear_tf=True`, `max_features=300_000` c/u; `LinearSVC(C=1.0, max_iter=3000)`; 336.493 features; 232 s | **0,5344 ± 0,0050** | **0,55235** (v01) | experiments.csv; RUNLOG S1 L71/L92; notebook 99 c21 | R |
| E-002a/b/c | `scripts/experimentos_representaciones.py` | `word 1 / 1-2 / 1-3` (min_df=3, sublinear, 300k) + LinearSVC C=1.0 | 0,5025 / 0,4971 / 0,4941 | — | experiments.csv | R |
| E-003a/b/c | idem | `char_wb 2-4 / 3-5 / 3-6` min_df=5, 300k + LinearSVC C=1.0; 224.628 (3-5) y 300.000 (3-6) features | 0,5261 / **0,5408** / **0,5413** | — | experiments.csv; notebook 99 c10 | R |
| E-004b | idem | `char_wb 3-6 + word 1-gram` (339.409 features) | 0,5354 | — | experiments.csv | R |
| E-005a..e | idem | politica de texto: normalizado 0,3902 · marcadores `<url>/<user>` 0,4107 · quitar URLs 0,5277 · quitar menciones 0,4094 · quitar puntuacion 0,5264 | ver celdas | — | experiments.csv; RUNLOG S1 L119-131 | R |
| E-007a | idem | union sin IDF (TF puro) | 0,5339 | — | experiments.csv | R |
| E-008 | idem | `char_wb 3-6` + **canal TF-IDF de menciones** (`@cuenta`, word 1, min_df=2); 314.420 features | **0,5557 ± 0,0052** | **0,57018** (v02) | experiments.csv; RUNLOG S1 L113/L137 | R |
| B0 | `scripts/baselines_b0_b5.py` | Dummy de clase mayoritaria (prior): accuracy 0,1704 (frecuencia de 13-17) | **0,0485** (suelo) | — | experiments.csv; baselines_resultados.json | R |
| B1 | idem | TF-IDF `word(1,2)` min_df=3 300k + `OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear"))`; 130 s | 0,5015 ± 0,0049 | — | experiments.csv | R |
| B5 | idem | 35 features densas + `LGBMClassifier` (textometria+estilo); 377 s | 0,3567 ± 0,0036 | — | experiments.csv | R |
| E-009..E-012 | `scripts/experimento_e009_modelos_clasicos.py` | base comun `CHAR_CONFIG` `char_wb(3,6)` min_df=5 300k + `MENTION_CONFIG` word(1,1) min_df=2; LogReg OvR liblinear C=4 · ComplementNB alpha=0.3 · SGD log_loss alpha=1e-6 · LinearSVC C=0.3 | 0,5603 / 0,5370 / 0,5579 / **0,5624** | — | experiments.csv; modelos_clasicos_resultados.json | R |
| E-013 | `scripts/experimento_e013_barrido_c.py` | barrido `C in {0.1, 0.2, 0.3, 0.5, 1.0}` sobre char 3-6+menciones; optimo **C=0.2**; margenes OOF -> `features/oof_E013_margenes.npy` | **0,5628 ± 0,0032** | — | experiments.csv; reports/tables/barrido_c.csv | R |
| Fase 20 | `scripts/fase20_ensamble.py` | stacking LOFO: `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")` sobre OOF de 4 modelos (24 cols); voto suave 0,5628 | **0,5656** (5/5 folds gana) | **0,58081** (v03) | fase20_ensamble.json; RUNLOG S1 L141 | R2 |
| E-014 / E-015 | `scripts/experimento_e014_e015_hipotesis_error.py` | + canal hashtags (SVM C=0.2) / `Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000)` sobre el punto medio ordinal | 0,5644 / 0,3973 | — | experiments.csv; RUNLOG S2 L160 | R |
| E-016 / E-017 | `scripts/experimento_e016_e017.py` | `char_wb(3,6)` min_df=3 **700k** + menciones, SVM C=0.2 / `word(1,2)` LogReg OvO (C=4, liblinear) | **0,5658** / 0,5015 | — | e016_e017_resultados.json | R |
| Fase 20b | `scripts/fase20b_ensamble_extendido.py` | meta LogReg (C=1.0, lbfgs, max_iter=3000) + `StandardScaler` sobre 9 fuentes (49 cols); LOFO elige 5 base + hashtags + ordinal + word | **0,5748 ± 0,0048** | **0,59178** (v04) | fase20b_ensamble_extendido.json; RUNLOG S2 L162 | R2 |
| E-018 | `scripts/experimento_e018_meta_enriquecido.py` | 9 fuentes + 35 densas (textometria+estilo) en meta LogReg (C=1.0, lbfgs) | **0,5784 ± 0,0044** (+0,0036) | **0,59518** (v05) | e018_meta_enriquecido.json; RUNLOG S2 L182 | R2 |
| E-019a/b | `scripts/experimento_e019_modelos_densos.py` | `MLPClassifier(hidden_layer_sizes=(128,64)/(256,128), alpha=1e-3, batch_size=512, learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=5)` sobre 35 densas / 35 densas + `SVD(64)`; CPU por GPU ocupada (D-007) | 0,3381 / 0,4071 | — | e019_modelos_densos.json; notebook 99 c12 | R |
| E-020 | `scripts/experimento_e020_mlp_en_ensamble.py` | + MLP densas+SVD como miembro del meta (10 fuentes) | **0,5851** (+0,0066) | **0,60273** (v06) | e020_mlp_en_ensamble.json; RUNLOG S2 L210 | R2 |
| E-021a/b | `scripts/experimento_e021_ordinal_centroides.py` | ordinal acumulativo Frank&Hall: 5 binarios `LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")` con `P(y=k)=P(y>k-1)-P(y>k)` / centroides Rocchio coseno sobre char 3-6 + menciones | 0,5527 ± 0,0037 / 0,4017 ± 0,0056 | — | e021_ordinal_centroides.json | R2 |
| E-022 | `scripts/experimento_e022_ensamble_final.py` | 10 fuentes + ordinal + centroides (102 cols) con meta LogReg / meta LightGBM | **0,5860 ± 0,0045** / 0,5858 | **0,60372** (v07) | e022_ensamble_final.json; RUNLOG S3 L242 | R2 |
| E-023 / E-024 | `scripts/experimento_e023_especialistas_longitud.py`, `..._e024_especialistas_en_ensamble.py` | especialistas SVM por longitud (umbral 93 chars) y su entrada al ensamble | **+0,0000** | — | e023/e024 json; RUNLOG S3 L243 | R2 |
| E-025..E-027 | `scripts/experimento_e025_e026_vistas_nuevas.py` | BM25(k1=1.2, b=0.75) char 3-6+menciones+SVM C=0.2 / kNN sobre SVD / aporte al ensamble | 0,4688 / 0,3088 / **−0,0002** | — | e025_e026_vistas_nuevas.json; e027 | R2 |
| E-028a/b | `scripts/experimento_e028_ordinal_fino.py` | 5 SVM binarios de **pares adyacentes** (`LinearSVC(C=0.2, max_iter=3000)`, margenes) / clasificador grueso 3 grupos; en ensamble pares +0,0006 (grupos −0,0003) | 0,7428 / 0,7128 (exactitud en fronteras) | **0,60376** (v08) | e028_ordinal_fino.json; RUNLOG S3 L259-271 | R2 |

**Lecturas EDA que alimenta:** el corpus es ASCII puro; las menciones son la senal individual mas fuerte (quitarlas −0,1251; canal dedicado +0,0144, D-010); `char_wb 3-6` domina a palabra (0,5413 vs 0,4971); limpiar el texto destruye senal (normalizado 0,3902). El suelo uniforme documentado es **B0 = 0,04854** (experiments.csv). Decisiones: D-010, D-011 (pesos por clase +0,0006, no adoptado), D-016 (saturacion), D-019 (frontera 18-24|25-34).

## 2. V5–V12: fases PH (los hitos que producen el v12)

| Exp | Script | Configuracion exacta | CV | LB (v) | Fuente | Re-ej |
|---|---|---|---|---|---|---|
| ADV-02 | `scripts/advanced_02_features.py` + `src/advanced_features.py` | 160 features: 87 locales + 73 **supervisadas in-fold** (`supervised_entity_features`: `P(clase|mencion/hashtag/fuente_rt)` con suavizado Dirichlet, rareza, lexico); meta LogReg sobre 267; 2.689 s | **0,5897** (+0,0031, 5/5) | **0,60807** (v09) | advanced_02_resultados.json; RUNLOG S4 L281 | R2 |
| ADV-03 | `scripts/advanced_03_holdout_candidato.py` | holdout congelado 15% semilla 2026: baseline 107 features vs candidato 267 | 0,5914 -> **0,5966** (+0,0052) | — | advanced_03_holdout.json | R2 |
| ADV-04 | `scripts/advanced_04_incertidumbre.py` | transformaciones no lineales de scores + acuerdo | +0,0002 (cerrada) | — | RUNLOG S5 L303 | R2 |
| ADV-05 | `scripts/advanced_05_nbsvm_seleccion.py` | NB-SVM char: TF-IDF reponderado por log-count ratio (Wang&Manning), SVM C=0.2; seleccion ANOVA F top-150 | 0,5641; top-150 0,5896 (no aporta); en ensamble +0,0007 | — | advanced_05_nbsvm_seleccion.json | R2 |
| ADV-06 | `scripts/advanced_06_submission_v10.py` | candidato v10 = 273 features (v09 + NB-SVM) | 0,5904 ± 0,0047 | no enviada | RUNLOG S5 L305; FINAL_REPORT §9 | R2 |
| PH1-L1 | `scripts/phase1_diagnostico.py` | interseccion de errores del ensamble OOF | 25,94% de filas rescatables; rescatadores: softmax ordinal 36%, centroides 31%, MLP 31% | — | phase1_diagnostico.json | R2 |
| PH2-L6 | `scripts/phase2_ponderacion.py` | meta global MLP vs LogReg; `EncodedMLP` = `MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, batch_size=512, learning_rate_init=1e-3, max_iter=40, early_stopping=True, n_iter_no_change=5)` + `StandardScaler`; LogReg `(C=1.0, max_iter=3000, solver="lbfgs")`; segmentado por tipo/longitud tambien medido | **0,5966** vs 0,5906 (**+0,0061, 5/5**); segmentado −0,0007/−0,0015 | — | phase2_ponderacion.json; RUNLOG S7 L337 (D-020) | R2 |
| PH3-V11 | `scripts/phase3_holdout_v11.py` | holdout: meta MLP vs meta LogReg; criterio Δ>=+0,003 y >=4/5 folds | LogReg 0,5981 -> MLP **0,6024** (+0,0043) | **0,61744** (v11) | phase3_holdout_v11.json; RUNLOG S8 L345 | R2 |
| PH5-L2 | `scripts/phase5_residuales.py` | residuales de longitud como features del meta | −0,0019 (2/5) | — | RUNLOG S9 L362 | R2 |
| PH6-L3 | `scripts/phase6_contenido_estilo.py` | estilo solo vs contenido solo (char/menciones) | 0,2793 / 0,5537; aporte +0,0008; hipotesis refutada | — | RUNLOG S9 L363 | R2 |
| PH7-L7 | `scripts/phase7_nbsvm_canal_meta.py` | **NB-SVM palabra** in-fold: `WORD_CONFIG word(1,2) min_df=3 sublinear 300k`, SVM C=0.2, log-count ratio; y **promedio de combinadores** `0.5*MLP + 0.5*LogReg` | NB palabra 0,5254 (+0,0009, 3/5); promedio **0,6005** (+0,0027) | — | phase7_nbsvm_canal_meta.json; RUNLOG S9 L364 | R2 |
| PH8-V12 | `scripts/phase8_holdout_v12.py` | holdout: LogReg 0,5983 · MLP 0,6034 · **promedio 0,5/0,5 = 0,6077** (+0,0043); criterio `>= 0,6024 + 0,003 = 0,6054` -> confirmado | 0,6005 (pooled) | **0,62182** (v12) | phase8_holdout_v12.json; RUNLOG S10 L378 | R2 |
| PH9-COMB | `scripts/phase9_combinadores.py` | individuales LogReg 0,5915 / Ridge(alpha=1.0) 0,5836 / SGD(modified_huber, alpha=1e-5) 0,4774 / MLP 0,5978; promedios | MLP+LogReg **0,6005** (optimo); +Ridge 0,5999; +SGD 0,5676; 4 combinadores 0,5702 | — | phase9_combinadores.json (D-025) | R2 |
| PH10-A/B | `scripts/phase10_dos_vias.py` | retrieval train->test / POS spaCy `es_core_news_md` | −0,0009 / −0,0032; cobertura de clúster in-fold 0,00%, duplicados exactos 1,36% (acierto 0,6151) | — | phase10_dos_vias.json; RUNLOG S11 L394 | X |
| PH11-RUIDO | `scripts/phase11_ruido_etiqueta.py` | exclusion de filas contradictorias (643) / duplicadas (2.123) | +0,0001 / +0,0016 (3/5), bajo el gate | — | phase11_ruido_etiqueta.json | R2 |
| PH12-AUD | `scripts/phase12_auditoria_estado.py` | reconstruccion exacta de la matriz + LOFO + end-to-end vs v11 | **0,6004 ± 0,0042** (folds 0,6007/0,6050/0,6032/0,6004/0,5928); 100,0000% vs v11 | — | phase12_auditoria_estado.json; RUNLOG S14 | R2 |
| PH13-ERR | `scripts/phase13_error_desacuerdo.py` | analisis OOF + bloques de votos | 13,87% todos fallan; 26,70% rescatable; quintiles 60,2%->18,1% | — | phase13_error_desacuerdo.json | R2 |
| PH14-CONS | `scripts/phase14_consenso_familias.py` | meta-features de consenso/divergencia JS/gating (46) | −0,0002 (3/5) | — | phase14_consenso_familias.json | R2 |
| PH15-CALIB | `scripts/phase15_calibracion_scores.py` | calibracion y blending en logits (T in [0.25, 4.0]) | +0,0003 (3/5) | — | phase15_calibracion_scores.json | R2 |
| PH18-LIMPIEZA | `scripts/phase18_limpieza_selectiva.py` | F2 un representante por clúster / F3 clúster selectivo | −0,0001 / +0,0009 (3/5) | — | phase18_limpieza_selectiva.json | R2 |
| PH16-ORD-GREEDY | `scripts/phase16_ordinal_greedy.py` | edad implícita por familia (G1) y media greedy 10 pasos (G2) | −0,0003 (1/5) / −0,0234 | — | phase16_ordinal_greedy_g1g2.json | R2 |
| PH30-35 | `scripts/phase30..35_*.py` | 33 capas de decision (proyeccion, coste ordinal, transicion, Luce, correccion, corrector minimo) sobre OOF del v12 | mejor G (tau 0,05) **+0,0011 (4/5, neto −117)**; 0/33 adoptadas | — | phase31/32/33 json; RUNLOG S21 (D-036/D-037) | R2 |
| PH36-40 | `scripts/phase36..40_*.py` | lexicones (1.369 entradas; registro/vida/temas) en control y fusion | control 0,2033/0,1520/0,0731; mejor fusion +0,0001; cerrada | — | RUNLOG S22 (D-038/D-039) | X |
| PH21/22/19 | ver §4 | vista estructural y de extremos (secuencia de tipos de token) | ver §4 | — | phase21/22/19 json | R2 |

**Combinador de produccion del v12 (definicion exacta, re-ejecutable tras reconstruir los 14 OOF):**
```python
# 14 archivos OOF (78 cols: 12 de 6 cols + oof_E015_ordinal de 1 + oof_E028a_pares de 5):
# E013_margenes, E016_margenes, E-009, E-011, E-010, E014_margenes, E015_ordinal,
# E015_softmax, E017_probabilidades, E019b_densas_svd, E021a_ordinal, E021b_rocchio,
# E028a_pares, ADV05_nbsvm + 35 densas + 87 estructuradas + 73 supervisadas = 273
# (meta_features_train_v11.npy); + oof_ADV07_nbsvm_word (6) = 279 (matriz del meta; ver paquete 03).
# Combinador: 0.5 * MLP(64) + 0.5 * LogReg(C=1.0, max_iter=3000, solver="lbfgs")
```
Referencia canonica: **CV 0,6005 pooled / 0,6004 media de folds; holdout 0,6077; LB 0,62182** (notebook 99 c21-23; FINAL_REPORT §9; v17_p0_reproduccion.json).

## 3. V13–V17: canales residuales, fronteras y cierre (bloques 7.0–10.0)

| Exp | Script | Configuracion exacta | Resultado | Fuente | Re-ej |
|---|---|---|---|---|---|
| V13-P0A | `scripts/v13_p0a_auditoria_idioma.py` | auditoria de idioma/ASCII sobre 251 ficheros | 6 CJK + 260 emoji corregidos; 0 errores | v13_p0a_auditoria_idioma.json | X |
| V13-P0B | `scripts/v13_p0b_colinealidad.py` | colinealidad de la matriz completa de produccion (279 variables, sin reducir; muestra 20.000 filas; Spearman con grama exacta, D-040) | 401 pares \|rho\|>=0,80; 229 redundantes; 15 duplicados exactos; 54 clusters; 1 constante (`supervisada_49`); rango efectivo 263/278; 87 direcciones para 95% var | v13_p0b_colinealidad.json | R2 |
| V13-P1A | `scripts/v13_p1a_evidencia_tokens.py` | **evidencia por token** `P(clase\|token)` in-fold: 36 features (18 por banda; bandas [20,100] y [100,None]; df_min=20; suavizado 20.0); control LogReg/LinearSVC block-only; 368 s | LogReg **0,3997 ± 0,0042** / LinearSVC 0,3917 (suelo azar 0,1667; 1/36 redundante) | v13_p1a_evidencia_tokens.json; RUNLOG V13 | R2 |
| V13-P1C | `scripts/v13_p1c_identidades.py` | separa lexico de identidad de cuenta: completo / sin cuentas / sin cuentas ni hashtags | 0,3997 / 0,3410 / 0,2992 (LogReg); menciones = 25,2% de la mejora sobre el azar | v13_p1c_identidades.json | R2 |
| V13-P1B | `scripts/v13_p1b_fusion_tokens.py` | fusion de las 36 cols con el v12 + 3 controles permutados; 5.740 s | **delta −0,0002 (3/5)**; controles 0,6000/0,6003/0,6000; IC95 [−0,0015; +0,0010]; neto −60 -> **NO ADOPTAR** | v13_p1b_fusion_tokens.json; RUNLOG V13 | R2 |
| V13-P16 | `scripts/v13_p16_suficiencia.py`, `..._p16b_activacion.py`, `..._p16c_reconstruccion.py`, `..._p16d_puerta.py` | por que falla toda fusion: suficiencia (R2), activacion por franjas, reconstruccion fuera de muestra, rejilla de 90 puertas | R2 mediano P1 −0,219 -> P16c +0,522; AUC confianza 0,7173; mejor puerta = no usar el bloque (**0,6005**, 0 filas desviadas) | v16_p16_suficiencia.json; v16c/v16d; RUNLOG V13 | R2 |
| V13-P17A/B | `scripts/v13_p17a_pronombres.py`, `..._p17b_pronombres.py` | 26 features de pronombres de persona (conteo/ratio/presencia por categoria) | control 0,1918/0,1881; P17b rescate 12.043 vs dano 76.118 (**neto −64.075**, umbral 1.014) -> **CERRAR sin fusionar** | v17_p17a/p17b json; RUNLOG V13 | R2 |
| V13-P19 | `scripts/v13_p19_conjunciones.py` | arboles de conjunciones (profundidad 4/6/8) + LogReg/LinearSVC; 36 features | 0,1489/0,1512/0,1539 y LogReg 0,1539: **todos bajo el azar 0,1667**; 0,64 conjunciones/doc -> **CERRAR** | v19_p19_conjunciones.json | R2 |
| V13-P20 | `scripts/v13_p20_modalidad.py` | 23 features de duda/certeza/modalidad | LogReg 0,1270 / LinearSVC 0,1259 (**bajo azar**); duda en 1,6% de docs -> **CERRAR** | v20_p20_modalidad.json | R2 |
| V13-P3A | `scripts/v13_p3a_especialistas.py` | especialistas binarios LogReg y MLP por par, forma `gate(especialista, V12)` | 18-24\|25-34: 0,4963 -> 0,4944/0,4946 (neto −109/−96); 25-34\|35-49: 0,4974 -> 0,4942/0,4945 (neto −180/−163); global 0,6005 -> 0,5993 -> **CERRAR** | v3_p3a_especialistas.json; RUNLOG V13 | R2 |
| V13-CIERRE | `scripts/v13_cierre.py` | cierre de fase | 16 experimentos, 8 hipotesis, **0/5 candidatos adoptados** (D-041) | v13_final_research.md | X |
| V14A | `scripts/v14a_linguistico.py` | bloque linguistico externo (spaCy `es_core_news_md/lg`, D-043): verbos, genero/numero, diminutivos, NER, dependencias; 33 cols tras podar 4 redundantes | LogReg 0,2409 / LinearSVC 0,2270 (suelo 0,1667) | v14a_linguistico.json | X |
| V14B | `scripts/v14b_fusion_linguistico.py` | fusion con v12 + controles permutados + segunda particion de 4 folds | **delta −0,0015 (1/5)**; 2a particion 0/4 (−0,0031); IC95 [−0,0027; −0,0002] -> **NO ADOPTAR** (cierra la via de relajar §3) | v14b_fusion_linguistico.json | X |
| V15A | `scripts/v15a_correlacion.py` | correlacion de entrada in-fold (Spearman + eta2) sobre 312 cols; indice de conservadas por fold | 402 pares >=0,80; 245 >=0,90; 58 clusters cubren 171/310; eta2 mediana 0,0306/max 0,5599; Jaccard de pares entre folds 1,0000 | v15a_correlacion.json | R2 |
| V15B | `scripts/v15b_poda.py` | poda a 197 variables + gate D-042 + particion de 4 | **0,5965 vs 0,6005; delta −0,0040 (0/5)**; 2a particion 0/4; IC95 [−0,0054; −0,0027] -> **NO ADOPTAR** (D-045) | v15b_poda.json | R2 |
| V16 | `scripts/v16_arquitectura.py` | A1 peso aprendido por fold / A2 combinador logistico sobre las 12 probabilidades (apilamiento en 2 niveles) | A1 0,6002 (−0,0003, 0/5; pesos 0,1/0,5/0,5/0,4/0,5); A2 0,6011 (**+0,0006, 3/5**), IC95 [−0,0003; +0,0015]; fronteras +0,0102/+0,0157 -> **NO ADOPTAR** (D-046) | v16_arquitectura.json | R2 |
| V17-P0 | `scripts/v17_p0_reproduccion.py` | reconciliacion de la referencia (inerte) | pooled **0,6005017473** vs media de folds 0,6004103057 (dif 9,14416e-05); A2 reproducido 0,6011; fronteras 0,5065/0,5131 | v17_p0_reproduccion.json; RUNLOG V17 | X |
| V17-P1 | `scripts/v17_p1_fronteras.py` | especialista de pares E-028a sobre TF-IDF disperso, folds agrupados (inerte) | V12 0,4963/0,4974 vs especialista **0,6864/0,7218**; rescata 14.356/15.571 y rompe 3.519/2.872; techo 0,7481/0,7726 | v17_p1_fronteras.json | X |
| V17-P1b/P1c | `scripts/v17_p1b_curva_compuerta.py`, `..._p1c_regla_conjunta.py` | curva de la compuerta de confianza (6 umbrales) y regla conjunta (inerte) | max **+0,0002** / +0,0001; negativa desde umbral 0,20; mejor conjunta +0,0000; techo con arbitro perfecto +0,0107/+0,0077 | v17_p1b/p1c json | X |
| V17-P8 | `scripts/v17_p8_perfil_confianza.py` | perfil de confianza (AUC Mann-Whitney) de rescatables vs danables | V12 no sirve (separacion 0,578/0,5268; entropia 0,4323/0,4230 invertida); si sirven acuerdo entre miembros **0,7070/0,7300** y margen del especialista 0,7294/0,7539 | v17_p8_perfil_confianza.json | X |
| V17-P9 | `scripts/v17_p9_arbitro.py` | arbitro binario LogReg (C=0.2) por frontera sobre 47 columnas existentes; protocolo anidado en 3 niveles; umbrales 0,1..0,9 (fraccion 0,25) | **0,6004; delta −0,0001 (0/5)**; IC95 [−0,000172; −5,5e-05]; 8.634 activadas, **3 rescates vs 22 danos** -> **NO ADOPTAR** | v17_p9_arbitro.json | X |
| V17-P5 | `scripts/v17_p5_combinador_local.py` | meta-combinador local LogReg (C=1.0) que redistribuye la masa del par (fuerzas 0..1) | **0,6003; delta −0,0002 (1/5)**; IC95 [−0,000486; +7,6e-05]; fronteras 0,4953/0,4968 -> **NO ADOPTAR** | v17_p5_combinador_local.json | X |
| V17-CIERRE | `scripts/v17_cierre.py` | cierre de las dos formulaciones locales + figuras | ninguna pista PROMISING/FINALIST; V12 congelado (D-047) | v17_final_research.md; v17_master_results.csv | X |

## 4. La "vista estructural" marginal (y su confusion con V13-P1)

- **Vista estructural real (PH21, bloque 5.0):** `scripts/phase21_vista_estructural.py`. Vista de **secuencia de tipos de token, sin contenido**, vectorizada `TF-IDF word(1,3)` + `LinearSVC(C=0.2)`; F1 individual **0,2285**, rescate del 18,17% de los errores del v12, **CV +0,0018 (4/5)**; holdout congelado **0,6077 -> 0,6108 (+0,0031)** (`scripts/phase19_holdout_variante.py --variante VistaEstructura`). No cumple el gate de CV (+0,003) -> **no adoptada, no enviada** (D-032/D-033). Refuerzos (`phase22_estructura_ampliada.py`): V1 +0,0005 (4/5), V2 −0,0003, V3 +0,0006. Candidato marginal: `submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv` (cambia 10,47% de las filas).
- **El par 0,3997 / −0,0002 NO es de la vista estructural:** 0,3997 es el control block-only de **V13-P1 "evidencia por token"** (`v13_p1a_evidencia_tokens.json`, LogReg 0,3997; `v13_p1c_identidades.json` completo 0,3997) y −0,0002 es su fusion (`v13_p1b_fusion_tokens.json`). La vista estructural (PH21) tiene 0,2285 individual y +0,0018/+0,0031. No mezclar ambos pares en el notebook.

## 5. Re-ejecutabilidad: que correr en el notebook y que documentar

| Bloque | Re-ejecutable en Kaggle | Coste aprox. | Que hacer en el notebook |
|---|---|---|---|
| Clusters + folds + EDA (clases, longitud, menciones, 4,2554% de cobertura) | Si (`src/cv.py`, Jaccard 0.8) | minutos | **Re-ejecutar** (base de todo) |
| B0 + matriz de representaciones E-001..E-008 (subconjunto: E-001, E-003b/c, E-008, E-002b, E-005a/d) + B1/B5 | Si (TF-IDF + LinearSVC/LogReg/LGBM) | ~20-40 min | **Re-ejecutar subconjunto** (alimenta EDA y la decision D-010) |
| E-009..E-013 (modelos clasicos + barrido C) | Si | ~10-20 min | Re-ejecutar si el presupuesto alcanza; si no, E-013 solo |
| E-014..E-017 (hashtags, ordinal Ridge, char 700k, word OvO) | Si | ~10-25 min | Opcional (E-016 es el pesado: 533k features) |
| E-018..E-028, fase20/20b, PH1..PH9, ADV-02..06 (meta y combinadores) | Solo tras reconstruir los 14 OOF + NB-SVM palabra de la cadena completa | horas (2-4 h) | **No re-ejecutar**: citar numeros y explicar el mecanismo (diversidad de sesgos; el promedio 0,5/0,5) |
| PH10..PH40 (retrieval, POS spaCy, ruido, consenso, decision, lexicones) | No (spaCy/artefactos) o solo tras la cadena | — | Documentar resultados (lineas cerradas) |
| V13–V17 | No barato: dependen del OOF del v12, del bloque linguistico spaCy (`es_core_news_lg` ni siquiera esta fijado en requirements, defecto N-02) o son inertes | — | **Documentar**: canales residuales, cierre y la vista estructural marginal; no re-correr |
| Vista estructural (token types) como concepto | El **calculo de la vista** si es barato (tipos de token por regex); su integracion con el meta no | minutos / horas | Mostrar la vista en EDA; la integracion se documenta |

## 6. Numeros canonicos y trampas (verificar en QA del notebook)

1. **0,6004 vs 0,6005:** son dos estimadores del mismo v12 (media de folds vs pooled; diferencia 9,14416e-05). Canonico **0,6005 pooled** (V17-P0); el holdout y el IC95 viven en esa escala.
2. **Suelo uniforme = 0,04854** (B0, clase mayoritaria; accuracy 0,1704). No es azar uniforme 1/6=0,1667 (ese es el suelo de los controles de bloque en V13).
3. **LB:** v01 0,55235 · v02 0,57018 · v03 0,58081 · v04 0,59178 · v05 0,59518 · v06 0,60273 · v07 0,60372 · v08 0,60376 · v09 0,60807 · v11 0,61744 · v12 0,62182; **v10 (0,5904) nunca se envio** (FINAL_REPORT §9; notebook 99 c21).
4. **E-019a/b:** 0,3381/0,4071 como clasificadores individuales, pero el MLP aporta +0,0066 al ensamble (E-020, 0,5851): el valor esta en la decorrelacion, no en la exactitud.
5. **v12 no se reconstruye al 100% desde el repo** (93,88% contra el archivo subido, D-034); fresca vs cache = 100,0000% y el holdout base se reproduce en 0,6077 exacto. Artefacto reproducible: `submissions/age_submission_v12_reproducible.csv`.
6. **El campo `familia` del inventario V1–V37 esta desalineado** en las filas modernas (p.ej. filas V19–V22 etiquetadas "V13-V17 busqueda residual"): clasificar por prefijo del `experiment_id`, no por ese campo.
7. **E-021..E-028 no estan en `experiments.csv`** (viven en RUNLOG + `experiments/e0*.json`); E-014..E-020 tienen fila terse ("ver script"). Citar el JSON.
8. **K-02:** el holdout del v12 declaro su criterio sobre si mismo (`confirmado: true`): es puerta de seleccion, no confirmacion independiente. Declararlo asi en el notebook.
9. **Deriva de versiones:** los numeros exactos dependen de sklearn (1.8.0 local vs 1.7.2 servidor vs Kaggle) y de los folds con semilla 42; declarar tolerancia o fijar version (Fase 0 del plan).
10. **Todos los fragmentos de codigo del proyecto son ASCII**; los informes usan tildes. Al portar, conservar la convencion.

## 7. Fuentes por bloque

- (a) V1–V4: `experiments.csv` filas E-001..E-020/B*; `RUNLOG.md` sesiones 1–3; `experiments/E-001_resultados.json`, `representaciones_resultados.json`, `baselines_resultados.json`, `modelos_clasicos_resultados.json`, `barrido_c_resultados.json`, `e016_e017_resultados.json`, `e018..e028*.json`; `notebooks/99_entrega_consolidado.ipynb` celdas 9–12, 21–23; `reports/tables/{representaciones,baselines,modelos_clasicos,modelos_densos}_comparativa.csv`; `reports/submissions_registry.csv`.
- (b) V5–V12: `RUNLOG.md` sesiones 4–21; `experiments/advanced_0*.json`, `phase1..phase40_*.json`; `notebooks/99_entrega_consolidado.ipynb` celdas 15–18; `FINAL_REPORT.md` §9; `reports/auditoria_estado_fase12.md`, `reports/fase_residual_informe.md`.
- (c) V13–V17: `experiments/v13_*.json`, `v14*.json`, `v15*.json`, `v16*.json`, `v17_*.json`; `reports/v13_final_research.md`, `v14_fusion_linguistico.md`, `v15_poda.md`, `v16_arquitectura.md`, `v17_final_research.md`, `v17_master_results.csv`, `v17_candidate_comparison.csv`; `notebooks/99_entrega_consolidado.ipynb` celdas 37–44; `reports/auditoria_experimentos_v1_v37.md`.
