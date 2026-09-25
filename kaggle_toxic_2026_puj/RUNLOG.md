# RUNLOG - KAGGLE_TOXIC_2026_PUJ

Registro cronologico de ejecuciones. Formato:
`fecha hora | fase | accion | comando/script | resultado`.

## 2026-09-22 (Fase 0)

- 12:45 | F0 | Creacion de estructura del proyecto | `mkdir -p kaggle_toxic_2026_puj/{...}`
  | 18 carpetas creadas (data/, notebooks/, src/, features/, models/,
  experiments/, reports/{eda,figuras}, submissions/, docs/, config/).
- 12:46 | F0 | Instalacion de dependencias | `/home/carlos/venv/bin/pip install kaggle emoji tabulate`
  | kaggle 2.2.4, emoji 2.16.0, tabulate (para informes markdown).
- 12:47 | F0 | Rama de trabajo | `git checkout -b proyecto-kaggle-toxic-2026`
  | rama creada desde main (sin commits).
- 12:49 | F0 | Descarga oficial | `kaggle competitions download -c tweet-toxic-2026-puj -p data/raw/`
  | ZIP 160.067 B; extraidos toxic_train.csv (205.417 B), toxic_test.csv
  (127.006 B), submission.csv (13.860 B); tamanos identicos a los metadatos de
  la API.
- 12:50 | F0 | Diagnostico de encoding y parseo | script inline
  | train: UTF-8, sep `;`, 1.732 lineas todas con 5 campos con quoting estandar;
  test: cp1252 (6 bytes 0xBF), 1.155 lineas con 2 campos; submission: coma,
  4 columnas. El quoting QUOTE_NONE rompe 13 filas de train -> se descarta.
- 12:51 | F0 | Escritura de `src/data.py` | modulo de carga/validacion/descarga
  | funciones `read_train`, `read_test`, `read_submission`, `download_competition`,
  `validate_frame`, `sha256_of`.
- 12:52 | F0 | Primera ejecucion de `experiments/00_auditoria_datos.py`
  | error: `read_submission` devolvia etiquetas como texto
  | corregido casteando HS/TR/AG a entero.
- 12:52 | F0 | Segunda ejecucion | error: falta `tabulate` para `to_markdown`
  | instalado tabulate; re-ejecutado.
- 12:53 | F0 | Auditoria completa | `python experiments/00_auditoria_datos.py`
  | 1731/1154 filas; 0 duplicados exactos train; 5 duplicados exactos test;
  0 textos compartidos train-test exactos; 3 pares train-train >=0.90 y 6 pares
  train-test >=0.90; 4 grupos near-duplicados (0.85) con 7 filas de train y 10
  de test; id sin asociacion con etiquetas; 17 pseudo-autores >=3 tweets;
  submission valida (ids == test).
- 12:53 | F0 | Artefactos generados | `reports/auditoria_resumen.json`,
  `reports/dataset_quality.csv`, `reports/leakage_audit.md`,
  `reports/leakage_audit.csv`, `reports/samples_por_clase.csv`,
  `reports/eda/*.csv`, `data/processed/grupos_near_duplicados.csv`.
- 12:55 | F0 | Memoria y documentacion | `PROJECT_STATE.md`, `RUNLOG.md`,
  `DECISIONS.md`, `experiments.csv`, `docs/competition_rules.md`,
  `reports/data_dictionary.csv`, `README.md`, `requirements.txt`, `.gitignore`,
  `config/hardware.json`, `config/gpu_baseline.json` | Fase 0 cerrada.

## 2026-09-22 (Fase 1)

- 13:00 | F1 | Investigacion del estado del arte | delegado a `ml-architect`
  (sesion `ses_f35bf70abffenkSQFLpiiEl953`) | 45 referencias verificadas con
  DOI/URL; `reports/estado_arte.md` (795 lineas), `reports/estado_arte.csv`
  (45 filas, 12 columnas), `references.bib` (45 entradas), `reports/hipotesis.md`
  (H1-H10 con experimentos E01-E10), `reports/contexto.md` (handoff).
- 13:05 | F1 | Verificacion del orquestador | conteo y anti-alucinacion
  | 45 papers unicos sin duplicados; 4 DOI comprobados contra Crossref
  (S19-2079 MineriaUNAM, S19-2072 HATERecognizer, S19-2084 SINAI,
  10.1016/j.eswa.2020.114120) con titulos coincidentes; clasificacion de
  compatibilidad: 37/45 directos, 8 con advertencia. Corregida la cifra de
  "44" a "45" en `reports/contexto.md`.

## 2026-09-22 (Fase 2)

- 13:15 | F2 | EDA, textometria y caracterizacion de TR | delegado a
  `nlp-specialist` (sesion `ses_f35a74177ffe6HLHyLhEZ2sUI2`) | script
  `experiments/01_eda_linguistico.py` (~3.500 lineas); 19 figuras en
  `reports/figuras/`, 21 tablas en `reports/eda/`, `reports/eda_resumen.md`.
- 13:20 | F2 | Cierre de pendientes del especialista (agoto pasos) | el
  orquestador corrigio: conteo real de TR=0 con colectivo, clave
  `imperativo` vs `tiene_imperativo`, integracion de columnas nuevas
  (`integrar_columnas`), determinismo de lexicos inducidos (`sorted`) y
  re-ejecucion doble | idempotencia verificada: 50 artefactos identicos.
- 13:30 | Fase C | Revision de codigo | delegado a `ml-reviewer`
  (sesion `ses_f359ae86cffexD8soIUm6KA20r`) | veredicto APROBADO CON CAMBIOS
  MENORES: 0 criticos, 1 mayor (features inducidas con etiquetas deben
  inducirse dentro del fold en modelado), 5 menores.
- 13:35 | Fase C | Aplicacion de los cambios menores | orquestador
  | encoding_errors="replace" en `src/data.py`; `mascara_deduplicada` ya no
  colapsa ids ausentes en un grupo -1; conteo de deduplicacion interpolado
  (4 filas, no 7); docstrings faltantes; columna redundante `es_hate`
  eliminada. Ruff limpio y doble ejecucion idempotente tras los cambios.
  Pendiente consciente: `rasgos_test`/`fenomenos_test` se calculan sin
  consumirse (menor, solo costo de tiempo).

## 2026-09-22 (Baseline B1 - primera submission)

- 13:45 | B1 | Baseline y submission v01 | `experiments/02_baseline_tfidf_lr.py`
  (escrito por el orquestador a peticion de Carlos para tener un baseline ya)
  | TF-IDF word (1,2) min_df=2 sublinear + LogisticRegression balanced x3;
  OOF con StratifiedGroupKFold(5) agrupado por near-duplicados:
  f1_media_columnas 0.7587 (sin post-proceso) y 0.7578 (con post-proceso);
  por etiqueta HS 0.743, TR 0.803, AG 0.731; exact_match 0.701;
  dimension 4024; 1.05 s.
- 13:46 | B1 | Bug corregido antes de ejecutar | el mapeo id->grupo fallaba por
  dtype (str vs int) y colapsaba todo en el grupo -1 (pliegues vacios);
  arreglado con `astype(int).map(...)` + validacion de ids sin grupo.
- 13:47 | B1 | Submission generada y validada | `submissions/toxic_submission_v01_baseline_tfidf_lr.csv`
  (1154 filas; ids identicos y en orden vs plantilla y test; valores en {0,1};
  0 nulos; 0 duplicados). NO enviada a Kaggle.
- 13:48 | B1 | Fila B1 anadida a `experiments.csv` y metricas en
  `reports/baseline_b1_metricas.json`; modelo en `models/b1_tfidf_lr.joblib`.
- 13:50 | B1 | Observacion para Fase 12 | el modelo subpredice positivos
  (HS 0.373 predicho vs 0.437 train; AG 0.310 vs 0.382) -> el ajuste de umbral
  por etiqueta deberia recuperar F1.

## 2026-09-22 (Envios de Carlos y cierre de E00)

- 14:05 | E00 | Carlos envio la plantilla (v00) | score publico **0.51494**
  | compatible con la familia media column-wise (0.541 esperado) o micro
  (0.543); descarta macro-6 (0.270) y samples (0.284).
- 14:05 | B1 | Carlos envio el baseline (v01) | score publico **0.75677**
  frente a 0.75778 local (diferencia -0.0010) | el protocolo de validacion
  agrupada reproduce el leaderboard; el macro-6 local (0.8097) queda descartado
  como metrica oficial.
- 14:10 | E00 | Registro y cierre | `submissions/leaderboard_log.csv` creado;
  fila E00 de `experiments.csv` actualizada; D-009 cerrada en `DECISIONS.md`
  (metrica primaria = f1_media_columnas).

## 2026-09-22 (Fases 4-5: variantes, representaciones y modelos)

- 14:20 | F4-F5 | Barrido de 25 configuraciones | delegado a `ml-coder`
  (sesion `ses_f358d189fffeuxfcGPC8b5QtyV`) | `experiments/03_variantes_representaciones.py`
  + `src/evaluation.py`; 5-fold agrupado por near-duplicado; 45 s.
- 14:30 | F4-F5 | Resultado | MOD-03b_BEST (union TF-IDF word(1,2)+char_wb(3,5)
  + LR C=2.0 balanced x3) = 0.7927 OOF (+0.0339 vs B1, 5/5 pliegues).
  REP-05 0.7865; REP-06 0.7855; LinearSVC 0.7849. Toda la parte A
  (preprocesamiento) DESCARTADA: mejor PRE-03 (URLs) +0.0035 < umbral 0.005;
  lematizar resta -0.0162; preservar mayusculas resta -0.0229; quitar
  stopwords resta -0.0025. H7 (lematizacion no aporta) APOYADA.
  H8 (post-proceso jerarquico) REFUTADA con esta representacion (-0.0056).
- 14:35 | Fase C | Revision de 03 + evaluation.py | `ml-reviewer`
  (sesion `ses_f35802063ffe82bKUePSC952EW`) | APROBADO CON CAMBIOS MENORES:
  sin fuga, 0 grupos compartidos entre pliegues, PRE-00 reproduce B1 pliegue a
  pliegue; conclusiones de parte A y H8 justificadas. 2 arreglos de
  consistencia aplicados (decision de MOD-04 vs su origen; veredicto de H8
  referido a su origen) y script re-ejecutado.
- 14:45 | F13a | Refit final + umbrales + submissions v02/v03 | delegado a
  `ml-trainer` (sesion `ses_f357c0899ffe76wistbVzBFGjZ`) |
  `experiments/04_refit_submissions_v02_v03.py`; OOF reproducido exacto
  (0.7926588861781015); umbrales optimos OOF HS 0.44 / TR 0.51 / AG 0.46 ->
  f1_media_columnas 0.7989 (+0.0063, 4/5 pliegues) ADOPTADOS con la regla
  pre-registrada; submissions v02 (umbral 0.5) y v03 (umbrales optimos)
  validadas; `features/oof_mod03b.npz`, `models/mod03b_lr_union.joblib`,
  `reports/fase13a_refit_umbral_metricas.json`; filas TH-01/S02/S03 en
  `experiments.csv` y v02/v03 en `leaderboard_log.csv` (pendientes de envio).
- 14:50 | Fase C | Revision de 04 | `ml-reviewer` | pendiente de veredicto en
  el momento de escribir esta linea.

## 2026-09-22 (Fase 13b: ensembles; Fase 12: analisis de error)

- 15:10 | E00/B1 | Scores de Carlos registrados | v02 (umbral 0.5) LB **0.77993**
  (local 0.7927, gap -0.0128); v03 (umbrales) LB **0.78306** (local 0.7989,
  gap -0.0158) | `submissions/leaderboard_log.csv`, filas S02/S03 y D-012
  actualizadas. Hallazgo: los umbrales aportan +0.0031 real en el LB y el gap
  local-LB crece con la capacidad de la representacion (B1 -0.001 vs union
  -0.013/-0.016).
- 15:25 | F13b | Ensembles con evaluacion anidada | delegado a `ml-trainer`
  (sesion `ses_f35620d22ffejblK0va457NrOH`) | `experiments/05_ensembles.py`;
  4 modelos base OOF (M1 word LR 0.7587, M2 union LR 0.7927, M3 union LinearSVC
  0.7849, M4 char LR 0.7567); diversidad real (desacuerdo 4-11 %) pero
  insuficiente: mejor ENS-02 (pesos en rejilla) 0.7949 anidado = +0.0023 vs
  MOD-03b, por debajo del umbral +0.005 -> NO se genera submission v04.
  ENS-04 (M2+M3) con umbrales medianos da 0.7985 (+0.0058) pero es el maximo de
  12 estimaciones anidadas: el propio entrenador recomienda no gastar envio.
- 15:45 | F12 | Analisis de error del mejor modelo | delegado a
  `nlp-specialist` (sesion `ses_f355977b0ffe8m7kdOZFYa54H5`) |
  `experiments/06_error_analysis.py` + `reports/error_analysis.md` (463 lineas)
  + 11 tablas + 5 figuras (fig_20..24) + `error_samples_para_revision.csv`
  (90 casos). Rebanadas mas debiles (n>=40): hashtags F1 0.586 (-0.213),
  preguntas 0.633, referencia colectiva 0.639, cuartil largo Q4 0.645,
  sin mencion 0.697. Patrones: insulto interpersonal en registro de broma
  (40.5 % de los casos insulto+mencion+sin colectivo son (0,0,0)), odio
  implicito contra grupos sin insulto del lexico, regimen noticia/atribucion.
  H8 cerrada como palanca (post-proceso jerarquico -0.0049).
  Nuevas hipotesis H11-H15 con experimentos propuestos.

## 2026-09-22 (Sonda v04 y Fases 6-8: features densas)

- 16:00 | F13c | Sonda v04 (ENS-04 = par M2+M3 con umbrales medianos) | delegado a
  `ml-trainer` (sesion `ses_f3395b7b6ffeTZ4WI8hM76gUGY`) |
  `experiments/07_submission_v04_ens04.py`; F1 anidado reproducido 0.798467;
  umbrales HS 0.44 / TR 0.47 / AG 0.44; submission
  `submissions/toxic_submission_v04_ens04_mediana.csv` validada; filas S04 y
  v04_ens04_mediana registradas (kaggle_f1 pendiente). Predice HS 46.5 % /
  TR 30.2 % / AG 40.5 % de positivos (por encima de la prevalencia de train):
  riesgo de umbral bajo si el test tiene ~7 % menos positivos (E00).
- 16:30 | F6-8 | Features densas H11/H12/H14 y baseline B5 | delegado a
  `ml-coder` (sesion `ses_f3391e391ffeLUMutq8uVaLokn`) |
  `experiments/08_features_densas_h11_h12.py`. RESULTADO NEGATIVO (documentado):
  B5-DENSE 0.6607; HIB-UNION 0.7610 (-0.0316); HIB-H14 0.7630; HIB-H11 0.7713;
  HIB-H12 0.7836 (-0.0091, el menos malo). Control con las 46 columnas densas a
  cero = 0.792659 exacto (la perdida es de las features, no de la concatenacion).
  Rebanadas: solo mejoran referencia colectiva (+0.0137) y Q4 (+0.0135) con
  HIB-H12; empeoran hashtag, pregunta y sin mencion. Los lexicos inducidos
  in-fold funcionan (AG 150 lemas; colectivo: inmigrante/ilegal/refugiado) pero
  concatenarlos degrada. Conclusion: H11, H12 y H14 NO se validan como palanca
  con este contrato; si se insiste, regularizar el bloque denso por separado o
  reajustar C.

## 2026-09-23 (v04 y Fase 10/27: Optuna)

- 00:05 | F13c | Score de la sonda v04 | LB **0.77510** (peor que v02 y v03) | el
  ensemble M2+M3 con umbrales medianos predijo mas positivos (HS 46.5 %) y cayo:
  la ganancia anidada no se traslado; v03 sigue siendo el mejor (0.78306).
  Patron del LB: el optimo esta cerca de la prevalencia estimada del test
  (v02 38.0 % HS < v03 43.9 % > v04 46.5 %).
- 00:15 | F10/27 | Optuna acotado sobre el pipeline ganador | delegado a
  `ml-trainer` (sesion `ses_f336db066ffej2dtEq86Vw1iif`) |
  `experiments/09_optuna_tfidf_lr.py`; 50 trials (28 completos, 22 podados) en
  295 s. Mejor trial #24: word(1,3) min_df=5 max_df=0.9 sublinear + char_wb(3,6)
  min_df=2 sublinear + LR C=3.2575 balanced -> **F1 OOF 0.801225 (+0.0086 vs
  MOD-03b, 5/5 pliegues)**. Top-5 en la misma region (0.7957-0.8012).
  Submission `submissions/toxic_submission_v05_optuna.csv` con umbrales OOF
  0.48/0.51/0.49 (OOF 0.8039; cota optimista, 3/5 pliegues). Positivos
  predichos HS 41.0 % / TR 29.7 % / AG 35.8 % (cerca de la prevalencia estimada
  del test). Filas OPT-01 y S05 registradas; artefactos
  `reports/optuna_trials.csv`, `reports/optuna_estudio.json`,
  `reports/figuras/fig_25_optuna_historia.png`, `models/fase10_optuna_lr_union.joblib`.
- 00:25 | Fase C | Revision de 09 | `ml-reviewer` | en curso al escribir esta
  linea.

- 00:45 | F13d | Sondas de robustez v06/v07 | delegado a `ml-trainer`
  (sesion `ses_f3183d4baffepn0csrIAc409ul`) |
  `experiments/10_sondas_robustez.py`. Control MOD-03b reproducido exacto
  (0.792659, gap train-OOF +0.1818). R1-C1 (C=1.0, min_df=2): OOF 0.7865/0.7929,
  gap +0.1635, pierde nivel local -> mal candidato. R2-MINDF5 (min_df=5, C=1.0):
  OOF 0.7953/0.7993, gap **+0.1272** (-0.0546, la mitad), 10.515 features
  (-62 %), 4/5 pliegues mejoran -> el nivel local se mantiene con la mitad de
  features y de gap. Submissions v06/v07 validadas y registradas (kaggle_f1
  pendiente). Decision por LB: se adopta la que supere 0.78306 (D-013).

## 2026-09-23 (v05/v06/v07 y cierre de la fase competitiva)

- 01:00 | F13 | Scores de v05, v06 y v07 | v05 (Optuna OPT-01) **0.77602**;
  v06 (C=1.0) **0.77227**; v07 (min_df=5) **0.77034** | los tres por debajo de
  v03 (0.78306). La hipotesis de capacidad (D-013) queda REFUTADA: el gap
  train-OOF no predice el gap local-LB.
- 01:10 | F14 | Analisis de ruido y cierre del sondeo (D-014) | los 6 modelos de
  la familia union caen en 0.77034-0.78306 = **0.99 errores estandar**
  (SE binomial 0.0129, n=1154) -> las diferencias entre variantes son ruido de
  muestreo. Unico contraste pareado fiable: v02 vs v03 (mismo modelo, distintos
  umbrales) = +0.0031 a favor de los umbrales OOF.
  Decision: modelo final = MOD-03b + umbrales (v03, LB 0.78306); se detiene el
  sondeo del leaderboard; el esfuerzo pasa a los entregables academicos
  (notebook 30 % de la nota, informe, auditorias) y a la Fase 11 (DL desde cero)
  como requisito metodologico.

## 2026-09-23/24 (Fase 14: auditorias y entregables)

- 23:50 | F11 | DL desde cero | delegado a `dl-specialist`
  (sesion `ses_f2e571f8cffeBa7Ttf3eC37z2P`) | `experiments/11_dl_desde_cero.py`;
  GPU verificada libre (D-008) y registrada en `experiments/gpu_preflight.log`.
  Resultados: DL-01 MLP 0.7949 +/- 0.0262 (3 semillas 0.7947 +/- 0.0011),
  DL-02 CharCNN 0.7410, DL-03 BiLSTM 0.7364 -> el DL desde cero NO supera al
  pipeline clasico (0.7927) fuera del ruido; evidencia metodologica. Bugs
  corregidos y documentados: tokens OOV enmascarados; AMP 2x mas lento en esta
  GPU; coredump con `use_deterministic_algorithms(True)` en ROCm 7.2.
- 00:10 | F14 | Auditorias finales (QA) | delegado a `ml-qa-engineer`
  (sesion `ses_f2e41503bffeORU3YF57QGaPmO`) | veredicto **GO con condiciones**;
  leakage/rules/reproducibility/submission APROBADOS (reproduccion byte a byte
  de B1, MOD-03b y v01-v03); brecha detectada: el control por pseudo-autor de
  D-004 nunca se ejecuto.
- 00:20 | F14 | Cierre de la brecha (D-015) | `experiments/12_control_pseudo_autores.py`
  (orquestador) | con grupos fusionados (near-duplicado + pseudo-autor) el OOF
  baja de 0.7927 a 0.7838 (gap local-LB de -0.0128 a -0.0039) -> el protocolo
  fusionado calibra mucho mejor con el LB. Fila PA-01 registrada.
- 00:30 | F14 | Auditoria de rendimiento | delegado a `ml-performance-auditor`
  (sesion `ses_f2e3c1eafffeifhwRIcj0rrT7o`) | pipeline final: fit 0.30 s,
  inferencia 0.10 s, RSS 233 MB; `models/dl_mlp.pt` 41 MB (64.9 % es el SVD);
  `import torch` +934 MB; recomendaciones P1-P3 para el notebook.
- 00:45 | F14 | Notebook entregable | delegado a `ml-coder`
  (sesion `ses_f2e391c0dffesQWiOqBVIUpTln`) |
  `notebooks/10_final_pipeline.ipynb`: 31 celdas (14 markdown + 17 de codigo),
  ejecutado con nbclient en 1.9 s, 0 errores, idempotente, 9 figuras embebidas;
  cubre las 12 secciones obligatorias y reconstruye v03 exacto desde el joblib.
  Inconsistencias detectadas y documentadas (B2-B4 nunca ejecutados, 10 variantes
  PRE no 12, 3 de 4 grupos near-dup cruzan train-test, scripts 00-12 = 13).
  Se anadio la subseccion de literatura con 11 referencias del estado del arte.
- 01:00 | F14 | Informe final | delegado al agente `general` (el agente
  `scientific-writer` no esta disponible como subagente) |
  `FINAL_REPORT.md`: 27 secciones (25 obligatorias + resumen ejecutivo +
  referencias), ~10.000 palabras, 150 filas de tablas, 42 citas con DOI/URL,
  resultados negativos y limitaciones declarados explicitamente.
- 01:10 | F14 | Cierre | memoria actualizada (PROJECT_STATE, RUNLOG, DECISIONS
  D-015) y verificacion de entregables: 13 scripts, 26 figuras, 53 tablas,
  9 submissions, 51 experimentos registrados.

## 2026-09-25 (Fase v2 - P0: scaffold)

- 08:35 | P0.1 | Congelacion de V03 | `experiments/baseline_v03_frozen/`:
  `model_v03.joblib`, `oof_v03_historical.npz`, `submission_v03.csv`,
  `pipeline_v03.json`, `hashes.txt` | hashes coinciden con los artefactos
  originales (modelo 263ed5bd..., submission 7d7478f2...); la submission de
  respaldo queda inmutable.
- 08:40 | P0.2 | Holdout v2 congelado | `experiments/13_holdout_v2.py` +
  `src/splits.py` (nuevo modulo de protocolo) | corregido un bug logico en la
  construccion del holdout (se usaban los entrenamientos de los pliegues 1-6, que
  incluian el pliegue 0, en vez de sus validaciones); verificacion de que ningun
  grupo cruza la frontera. Desarrollo 1.483 / holdout 248; los 6 combos
  representados en el holdout (000:136, 001:4, 100:13, 101:21, 110:4, 111:70);
  `config/holdout_v2.json` con hash `e5b62e5f...`; regla de maximo 5
  evaluaciones. Fila HOLD-01 registrada.
- 08:45 | P0.3 | OOF corregido de los modelos base | `experiments/14_oof_corregido_v2.py`
  (17 s) | corrected_cv sobre el desarrollo: M2 (V03) 0.7718 (0.7772 con sus
  umbrales), M3 0.7677, M1 0.7486, M4 0.7377; std 0.015-0.023; por etiqueta de
  V03: HS 0.762 / TR 0.816 / AG 0.754. Artefactos
  `features/v2_oof_base_corrected.npz` y `reports/baseline_v03_dev_metrics.json`;
  filas V2-BASE-01..04.
- 08:50 | P0.4 | Reportes de investigacion | `reports/advanced_gap_analysis_v2.md`
  (matriz de 24 brechas fenomeno x representacion) y
  `reports/research_backlog_v2.md` (24 lineas E2-01..E2-24 con hipotesis,
  implementacion, validacion e informacion esperada).
- 08:55 | P0.5 | Cierre de P0 | D-016 (protocolo v2) + tablero `tasks/plan.md` y
  `tasks/todo.md` | GATE P0 SUPERADO: referencia v2 = 0.7772 (V03 con umbrales
  sobre el desarrollo); siguiente P1 (cadenas de etiquetas suaves).

## 2026-09-25 (Fase v2 - P1: dependencia de etiquetas)

- 09:12 | P1 | Cadenas de etiquetas y meta-modelo | delegado a `ml-coder`
  (sesion `ses_f2731ff1dfferZ2uMNWcgFlU9O`) |
  `experiments/15_cadenas_etiquetas_v2.py` (244 s). Resultados sobre el
  desarrollo (referencia 0.7772):
  cadenas E2-01: 0.7726-0.7744 segun orden (+0.002 sobre el base 0.7718,
  IC bootstrap incluye 0, 1/5 pliegues) -> **linea cerrada sin mejora**.
  Meta-modelo E2-02 (4 bases OOF + margenes + entropias + 44 estructurales):
  **0.7853** con anidamiento estricto = **+0.0081**, 4/5 pliegues, IC bootstrap
  [+0.0021, +0.0248], complementariedad +0.0152 -> **adoptado como candidato
  v2-A** (D-017). Cota optimista con umbrales OOF: 0.7888. Holdout intacto
  (0/5 evaluaciones).

## 2026-09-25 (Fase v2 - P2: targeting estructural y reordenamiento)

- 09:40 | P2 | Targeting estructural (E2-05/E2-06) | delegado a `ml-coder`
  (sesion `ses_f27151d15ffe2T5eSPUfUuB9ZV`) | `experiments/16_targeting_v2.py`
  (7.7 s). Resultados: canal abstracto 0.6076; canal+distancias 0.6484;
  union+canal 0.7356; union+n-gramas simbolicos 0.7487 (base 0.7718) -> **linea
  cerrada sin mejora standalone** (D-018). Dato util: el recall de AG en la
  rebanada colectiva sube de 0.452 (V03) a 0.478 con el canal; sus OOF quedan en
  `features/v2_oof_targeting.npz` para el stacking.
- 09:45 | Plan | Reordenamiento aprobado por evidencia | P6 (late fusion /
  stacking) pasa antes que P3-P5: la fusion a nivel meta es la unica via que ha
  mejorado (P1: +0.0081) y la concatenacion degrada en sus dos formas (densa y
  simbolica). Las lineas P3-P5 se justifican como generadoras de canales.

## 2026-09-25 (Fase v2 - P6: stacking)

- 09:48 | P6 | Late fusion / stacking v2 | delegado a `ml-trainer`
  (sesion `ses_f27063508ffeUuVFB7o8jH9E17`) | `experiments/17_stacking_v2.py`
  (71 s). Verificaciones: V03 reproducido 0.7772 y meta de P1 reproducido exacto
  0.7853. Resultados: META-BASES 0.7850; META+CH 0.7807; META+CH+ESTR 0.7844;
  **META-BASES-RIDGE (alpha=3) 0.7874** (3/5 pliegues; IC del delta vs v2-A
  [-0.0054,+0.0093] incluye 0); META+CH+ESTR-RIDGE 0.7822. Los canales de
  targeting NO aportan al meta (delta -0.0006, IC [-0.0077,+0.0062]) -> linea de
  canales cerrada (D-019). Efecto util: recall de AG en la rebanada colectiva
  0.452 -> 0.574 (F1 0.523 -> 0.589). Adoptado por complementariedad
  (`ensemble_gain` +0.0089, IC [+0.0019,+0.0160]). Artefactos:
  `models/v2_meta_candidato.joblib`, `features/v2_meta_oof.npz`,
  `reports/stacking_v2.csv/.json`, `fig_29`.
- 09:55 | Plan | P3-P5 en espera con expectativa baja | producen canales y los
  canales no aportan al meta (evidencia P2+P6); se prioriza P7 (holdout + sonda
  de LB).

## 2026-09-25 (Fase v2 - P7: holdout y submission)

- 10:10 | P7 | Validacion en el holdout congelado | delegado a `ml-trainer`
  (sesion `ses_f26f31a96ffeSGlMBxCUCDFufG`) |
  `experiments/18_p7_holdout_y_submission_v08.py` (3 s). V03 en holdout =
  0.8285 (HS 0.857 / TR 0.849 / AG 0.779); **meta en holdout = 0.8465**
  (HS 0.862 / TR 0.857 / AG 0.820); delta pareado **+0.0180** con IC95
  [+0.0003, +0.0372]. Decision GO (regla meta >= V03 - 0.005). Holdout 2/5
  evaluaciones (`reports/holdout_evaluaciones_v2.csv`).
- 10:15 | P7 | Submission del candidato | `submissions/toxic_submission_v08_meta_v2.csv`
  (1154 filas validadas; positivos HS 44.8 / TR 32.5 / AG 38.6 %); fila S08 y
  entrada `v08_meta_v2` en el leaderboard (kaggle_f1 vacio; la envia Carlos).
  Artefactos: `models/v2_meta_final.joblib`, `reports/p7_holdout_resultados.json`.
- 10:25 | Fase C | Auditoria del pipeline v2 | `ml-reviewer`
  (sesion `ses_f26eaf579ffexCAJBQPJrty64v`) | APROBADO CON OBSERVACIONES:
  0 criticos; **sin fuga en el stacking** (verificado con reajuste independiente
  en /tmp), holdout aislado y contador 2/5 correcto, numeros reproducidos
  (meta dev 0.787357; holdout V03 0.828468 y meta 0.846509), submission v08
  identica al pipeline declarado (0/3462 discrepancias) y nada enviado.
  Defecto menor corregido por el orquestador: fila de cabecera duplicada en
  `experiments.csv` (76 filas, 76 ids unicos).
- 10:30 | Plan | Decision D-020 | se autoriza **una** sonda de LB con v08 con la
  advertencia honesta: el holdout es un pliegue mas facil; aplicando el gap de
  familia (-0.013/-0.016) a la referencia del desarrollo (0.7874), el LB
  esperado cae en ~0.771-0.774, posiblemente por debajo de v03 (0.78306).

## 2026-09-25 (Notebook entregable actualizado con la fase v2)

- 11:15 | Entregables | Notebook final actualizado | delegado a `ml-coder`
  (sesion `ses_f26ba15aeffed82J9rq9nBVysW`) + cierre del orquestador |
  `notebooks/10_final_pipeline.ipynb`: **42 celdas** (20 markdown + 22 de
  codigo), **0 errores**, **2.4 s**, digesto de salidas **identico en dos
  ejecuciones consecutivas** (idempotente). 16 secciones numeradas; las nuevas
  cubren el protocolo v2 (D-016), los paquetes P0-P7 con los numeros exactos
  (incluidos los negativos), las 9 sondas de leaderboard y la verificacion de
  reproducibilidad con **26 comprobaciones [OK]** (V03 dev 0.777211 exacto,
  reconstruccion byte a byte de la submission v03 con sha256 identico, sha256
  del holdout, 100 experimentos / 9 sondas / 21 scripts).
  Backup previo: `/tmp/opencode/10_final_pipeline_BACKUP_pre_v2.ipynb`.
- 11:20 | Pendiente | `FINAL_REPORT.md` aun no incluye la fase v2 (se anadira
  antes de la entrega si Carlos lo pide).

## 2026-09-25 (sondeo LB de los candidatos finales v09/v10/v11)

- 13:45 | Leaderboard | Carlos envio las 3 submissions experimentales.
  Resultados: v09 (C-EMO) **0.77895**; v10 (C-SYN) **0.77978**; v11 (C-EMO-SYN)
  **0.78150**. Los tres por debajo de V03 (0.78306); el mejor (v11) queda a
  -0.00156, dentro del error estandar (~0.013). Gaps local-LB: -0.00199 /
  +0.00047 / +0.00184. El ranking local no predijo el del LB (C-EMO mejor local,
  peor LB).
- 13:50 | Registros | `leaderboard_log.csv` con 12 sondas; estados actualizados
  en `submission_manifest.csv`; resultados en `final_candidate_comparison.md`;
  filas FIN-08..10 con el LB en `experiments.csv`; D-024 (cierre: V03 sigue como
  referencia final).

## 2026-09-25 (fase final de candidatos: screening, metas restringidos y submissions)

- 13:10 | Screening y evaluacion | `experiments/23_candidatos_finales.py`: OOF
  canonico de V03 (`features/oof_v03_final.npz`, corrected_cv 0.777211 verificado)
  y evaluacion anidada de metas restringidos: C-EMO 0.7809, C-EMO-SYN 0.7797,
  C-SYN 0.7793, C-EMO-SYN-TGT-CYT 0.7777 (2-3/5 pliegues); el unico PROMISING
  sigue siendo el meta v08 (0.7874, no transferido). Holdout: C-EMO 0.8364 y
  C-EMO-SYN 0.8336 (IC95 del delta cruza 0); 4/5 evaluaciones usadas.
- 13:25 | Informes | `reports/final_candidate_screening.md` (125 experimentos
  clasificados A-G: A=5, B=9, C=3, D=18, E=67, F=8, G=15),
  `reports/final_candidates.csv` (FIN-01..07) y
  `reports/final_candidate_comparison.md`.
- 13:29 | Submissions | `experiments/25_submissions_candidatos.py`: generadas y
  validadas `submissions/toxic_submission_v09_candidate_a.csv` (C-EMO),
  `v10_candidate_b.csv` (C-SYN) y `v11_candidate_c.csv` (C-EMO-SYN); hashes
  estables en re-ejecucion; `reports/submission_validation_v2.md`,
  `reports/submission_hashes_v2.json` y `submissions/submission_manifest.csv`
  (5 filas: V03, v08 y los 3 candidatos). Decision (D-023): sin evidencia
  suficiente para sonda; V03 sigue como referencia.

## 2026-09-25 (cierre documental: spec v2 archivada, artefactos FASES 88/98 y auditorias)

- 13:30 | Especificacion | Archivada la spec maestra de la fase v2 en
  `docs/advanced_research_v2_spec.md` (113 secciones, FASES 0-99) como contrato de
  investigacion del proyecto.
- 13:35 | Artefactos | Nuevo `experiments/22_artefactos_auditoria_v2.py` (7.1 s,
  sin GPU, sin holdout/test) genero los artefactos exigidos por las FASES
  43/47/50/51/55/88/89: `reports/error_intersection_v2.csv` (17 filas),
  `reports/all_three_errors.csv` (201), `reports/error_by_label_combination_v2.csv`
  (14), `reports/model_complementarity_v2.csv` (496 parejas de 32 canales),
  `reports/feature_catalog_v2.csv` (225 features),
  `features/v2_structured_features.parquet` (2.885 x 123, 0 nulos) y su manifest;
  filas AUD-01..03 en `experiments.csv`. Cifra clave: las-3-mal 98 (V03, 6.61 %)
  frente a 103 (meta, 6.95 %) con IC bootstrap.
- 13:45 | Auditorias | Escritas `reports/spec_compliance_v2.md` (matriz FASE 0-99
  con evidencia; veredicto FASE 99-B "cumplida con observaciones"; mapeo del
  objetivo FASE 112), `reports/rule_audit_v2.md` (0 violaciones de reglas) y
  `reports/target_encoding_audit_v2.md` (sin leakage de aprendizaje).
- 13:55 | Correcciones | `research_backlog_v2.md` con los 24 estados cerrados;
  informe v2 corregido ("0 RT" -> 0 con prefijo canonico y 8 con el token; 100 ->
  118 filas; scripts 13..20 -> 13..22); notebook actualizado a 118 experimentos y
  23 scripts; D-022 registrado.

## 2026-09-25 (P8 y cierre total del backlog v2)

- 12:05 | P8 | Cinco lineas restantes del backlog | delegado a `ml-trainer`
  (sesion `ses_f26a27022ffeUcy5pqi9yXF7PN`) |
  `experiments/21_backlog_restante_v2.py` (27 s). Resultados (referencia 0.7772):
  E2-14 calibracion 0.7857 (mejora Brier 0.1583 -> 0.1199 y log-loss, sin cambio
  de F1) -> cerrada; E2-13 encoder de entidades: cobertura informativa **0.5 %**
  -> cerrada por sparsity; E2-16 umbrales por rebanada: -0.0017 vs global
  anidado (curva monotona) -> cerrada; E2-20 especialistas: 11/21 celdas
  significativas tras BH pero -0.045 de media dentro de su rebanada -> cerrada;
  E2-21 espacios por etiqueta: -0.0253 (0/5) -> cerrada. **Backlog v2: 24/24
  lineas evaluadas y cerradas** (D-021). Holdout intacto (2/5).
- 12:15 | Entregables | Consistencia actualizada | notebook: celdas de backlog y
  alcance reescritas (24/24) y asserts de recuento corregidos (115 experimentos,
  22 scripts 00-21); re-ejecutado: **42 celdas, 0 errores, 26 checks [OK], 0
  fallas**. `FINAL_REPORT.md`: seccion 26.6 actualizada con el cierre del
  backlog. Ambos entregables quedan coherentes con `experiments.csv` (115 filas).

## Pendientes registrados

- E00: sonda de submission para identificar la implementacion exacta de F1 en
  Kaggle (la envia Carlos; no automatizada).
- Fase 1: investigacion del estado del arte (delegada a ml-architect).

## 2026-09-25 (apertura de la fase v4: P0 scaffold)

- 10:05 | Apertura | Rama `proyecto-kaggle-toxic-2026-v4` creada desde `main`
  (bae7612). Spec maestra archivada en `docs/advanced_research_v4_spec.md`
  (108 secciones, numeracion original conservada) y plan ejecutable en
  `tasks/plan_v4.md` + `tasks/todo_v4.md`. D-025 registrada.
- 10:20 | Verificacion | `experiments/26_scaffold_v4.py` (P0, sin entrenar):
  **veredicto OK**. 5/5 hashes de la referencia congelada coinciden;
  `corrected_cv` recalculada desde `features/oof_v03_final.npz` = **0.777211**
  (delta 0 frente a la almacenada); M2 con umbral 0.5 = 0.771778; F1 por
  pliegue [0.7960, 0.7940, 0.7789, 0.7386, 0.7791] (std 0.0161); protocolo v2
  recalculado: 1.731 train = 1.483 desarrollo + 248 holdout, 1.645 grupos,
  **0 grupos compartidos**; presupuesto: holdout 4/5, 12 sondas de LB, mejor
  0.78306 (v03). Informe: `reports/baseline_v03_v4_check.json`.
- 10:35 | Diagnostico | `reports/v4_gap_analysis.md`: cada idea de la spec v4
  clasificada contra la evidencia en disco. Genuinamente nuevas: consistencia
  de etiquetas, dureza/reweighting, weighting supervisado por clase, modelo
  conjunto de etiquetas, calibracion de las entradas del meta, verosimilitud
  class-conditional, sufijos/prefijos, subtipos de HS implicito. Cerradas y
  prohibidas: POS/deps/cita/negacion/targeting/colectivo/emocion como bloque,
  cadenas de etiquetas, concatenacion densa, Optuna, DL desde cero.
- 10:45 | Pre-registro | `reports/research_backlog_v4.md` con 26 lineas
  (V4-A01..V4-X01) y `reports/signal_coverage_v4.csv` (29 fenomenos). Toda linea
  tiene hipotesis, metodo, validacion, senal esperada y criterio de cierre
  **antes** de ejecutarse (control de multiplicidad con 128 experimentos
  previos).
- 10:50 | revision bibliografica | Delegada en paralelo (2022-2026: ruido de
  etiquetas, reweighting, hard example mining, feature weighting supervisado,
  BM25, modelos conjuntos, hate speech clasico, espanol) con restricciones
  duras sin preentrenamiento; salida `reports/estado_arte_v4.md` + `.csv`.
- 11:20 | P1 TRACK A | `experiments/27_label_consistency_v4.py` (3.0 s): clusters de
  consistencia con cuatro criterios (exacto, normalizado, Jaccard >= 0.8,
  cosimilitud TF-IDF >= 0.85). Resultado: **0 duplicados exactos en train**,
  15 clusters con 2+ miembros (84 tweets, 4.85 %), **2 clusters
  contradictorios**; contradiccion por etiqueta HS 48 (2.77 %), TR 48, AG 50
  (2.89 %), concentrada en un unico cluster de 48 tweets con el mismo patron de
  insulto (38 anotados 000 frente a 10 con 110/111). 15 tweets de test tienen
  texto normalizado identico a un tweet de train (conteo, sin usar etiquetas).
  Artefactos: `reports/label_consistency_v4.csv/.json/_ejemplos.md` y
  `features/label_consistency_v4.npz` (con vecindario top-3 reutilizado).
- 11:35 | P1 TRACK B | `experiments/28_hardness_v4.py` (1.7 s): mapa de dureza
  solo-OOF con vecindad, desacuerdo de 20 canales OOF no redundantes y bootstrap
  pareado (2.000 remuestreos). Hallazgo metodologico: la dificultad por etiqueta
  ordena bien (`p_true` vs error r = -0.795/-0.810/-0.801) pero **agregar las tres
  etiquetas rompe el orden** (BCE media r = -0.136; cercania a la frontera
  r = +0.345; violacion del umbral r = +0.811). Ademas el error es **menor**
  dentro de los clusters contradictorios, luego el ruido de anotacion por
  duplicacion no explica el error de V03. D-026 registrada; backlog actualizado.

## 2026-09-25 (cierre de la fase v4: P2-P8)

- 14:10 | P2 TRACK C | `experiments/30_supervised_weighting_v4.py`: 16 variantes de
  geometria y coste. **Todas las reponderaciones de features degradan**: log-odds
  0.7571, chi2 0.6622, MI 0.7378, BM25 0.7748-0.7755, binario 0.7738 (control
  V03 0.777211 exacto). El barrido **anidado** de `class_weight` da 0.7812
  (3/5 pliegues) y la combinacion con downweight por violacion da **0.7854
  (5/5 pliegues, IC95 del delta [-0.0002,+0.0117], el limite inferior toca 0)**.
  V4-C06 (log-odds con C = 2/10/50/200) **refuta** la hipotesis del desajuste con
  la penalizacion L2: 0.7571 -> 0.7435 al subir C.
- 14:25 | P3 TRACK E | `experiments/31_joint_label_v4.py`: Label Powerset sobre 6 y
  4 combinaciones, tres tratamientos de las rarezas y C en {0.5,1,2}. Mejor
  variante 0.7713 (`lr4_mapeo_bal_C2.0`): **el modelo conjunto no supera a los tres
  binarios independientes**. Cerrado como NO_SIGNAL.
- 14:40 | P4 TRACK H | `experiments/32_meta_shift_v4.py`: el shift entre el regimen
  de entrenamiento del meta y el de inferencia es **pequeno** (KS medio 0.0624) y
  **no causa dano** (meta_shift 0.7828 > meta_control 0.7791); la alineacion por
  cuantiles recupera +0.00005 y Platt empeora -0.0171. **Hipotesis abierta de
  v2 §4.3 refutada** (D-028). El meta de 36 features cumple el nivel de D-016.
- 15:00 | P5 TRACK G | `experiments/33_class_conditional_lm_v4.py`: la
  verosimilitud class-conditional (unigram/bigram/char) vale 0.4876 sola pero
  **aporta +0.0041** a la fusion tardia (0.7710 -> 0.7751). Ridge 0.7698, SGD
  0.7454, PassiveAggressive 0.7626. En SVD 300: **RBF SVC 0.7904** (5/5
  pliegues, neto +30, IC fragil), HistGB 0.7297, ExtraTrees 0.7033.
- 15:20 | P6 TRACK D | `experiments/34_structural_sparse_v4.py`: canales de afijos
  (0.7758), interacciones y ventanas locales como tokens, en version hstack y
  standalone. Ninguno aporta. Diagnostico de masa de HS implicito ejecutado
  (`reports/implicit_hate_diagnostic_v4.json`).
- 15:35 | P7 TRACK F | `experiments/35_discovery_v4.py`: 2 de 10 variables no
  supervisadas con shift; V03 rinde **0.7568 en textos unicos** frente a
  **0.9769 en los 68 ejemplos con near-duplicado**; la vecindad discrepante no
  explica el error (delta +0.018 con IC que cruza 0); la propagacion por
  near-duplicado tiene **techo nulo** (0 coincidencias literales test-train); 0 de
  5 canales corrigen mas de lo que rompen.
- 16:00 | P8 candidatos | `experiments/36_candidatos_v4.py`: A 0.7854 (5/5, IC
  toca 0), B 0.7828 (4/5, IC positivo, nivel menor), **C 0.7882 (4/5,
  IC95 [+0.0054,+0.0200], correccion neta +34: unico que pasa los tres gates)**,
  D 0.7904 (5/5, IC toca 0).
- 16:20 | P8 refit y holdout | `experiments/38_refit_final_v4.py`: refit del
  candidato C sobre todo el desarrollo (multiplicadores = mediana de los 5
  pliegues 1.2/1.5/1.2, pesos de ejemplo OOF), prediccion sobre holdout y test.
  **HOLDOUT 0.8163 vs 0.8285 de V03 (delta -0.0122): RECHAZADO**. Submission v12
  generada y validada (sha256 `ec25354c...`) y marcada `DESCARTADO` en el manifest.
  **5/5 evaluaciones de holdout usadas; 0 sondas de LB nuevas** (D-029).
- 16:40 | Cierre documental | `reports/advanced_research_v4.md`,
  `reports/spec_compliance_v4.md`, `experiments/37_auditoria_v4.py` (veredicto
  **GO**: 0 violaciones en 12 scripts, doble ejecucion con SHA256 estable, 41
  artefactos inventariados). Memoria actualizada: D-025..D-029, PROJECT_STATE,
  experiments.csv (143 filas), tasks/todo_v4.md.
- 17:05 | v5 P0 compliance y scaffold | `experiments/embedding_00_scaffold_v5.py`
  (GO): fastText C++ compilado con `cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5` y
  `make -j32` (commit `1142dc4c4ecb`, MIT, sha256 del binario en el manifiesto),
  entrenado y validado (~930k tok/s con 16 hilos); control
  `WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF` verificado con `-minn 0 -maxn 0`
  (el `NaN` inicial era del corpus de juguete de 25 tipos). `D-030` registrada con
  la transcripcion literal de la autorizacion y **sin** inventar evidencia escrita.
  Spec maestra transcrita en `docs/embedding_from_scratch_v5_spec.md`.
  `data/external/corpora/corpus_manifest.json` creado con el esquema de la §35 y
  copia declarativa en `reports/corpus_manifest_v5.json`. Auditoria de recursos:
  0 usos de transformers / sentence-transformers / GloVe / `from_pretrained`.
  `corrected_cv` de V03 reproduce **0.777211** desde el OOF canonico. Trabajo en
  worktree `/home/carlos/Git/PLN-v5` para no interferir con la sesion paralela de
  `KAGGLE_AGE_2026_PUJ`. Fila `EBD-00-SCAFFOLD` en `experiments.csv` (144 filas).
  **0 sondas de LB, 0 usos de holdout.**
- 17:40 | v5 P1 corpus 10M + cobertura TOXIC | `experiments/embedding_01_corpus_wikipedia_v5.py`
  (E-CORP-01): se descargo **una sola parte** del dump de eswiki
  (`eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2`, 305 MB, paginas
  1-159400, sha256 `3248f107...`) en vez de los 5.19 GB completos, porque basta para
  10M, 50M y 100M tokens y hace la fase reproducible con un solo artefacto
  verificable. Pipeline `download -> raw -> clean -> dedup -> normalize ->
  tokenize -> shards -> manifest` en streaming: **2.847 documentos, 10.002.400
  tokens, 271.209 tipos**. **Auditoria de fuga: 0 textos de TOXIC (train y test)
  con coincidencia de 8 o mas tokens**, indexados 9.737.490 8-gramas.
  Cobertura del vocabulario TOXIC (train+test): **0.6839 por tipo**, **0.8833 por
  token**, 2.704 word-OOV, **0 representation-OOV** (todo OOV recibe vector por
  subpalabras). La hipotesis de que la brecha era ortografica por falta de tildes
  quedo **refutada** (8.5% de los OOV son pliegues de acento); el 66.6% son
  deformaciones con perdida de vocales (`cllate`, `violacin`, `espaa`) y el resto
  son formas coloquiales y elongadas ausentes de un corpus enciclopedico
  (elongaciones 0.035 por tipo, coloquial 0.214). Se documento tambien que el
  vecindario de subpalabras con la sonda 50d/1 epoca **no** es interpretable
  (varios vecinos son claramente erroneeos), por lo que la calidad solo se
  juzgara en P2 con el modelo de 200d. **0 sondas de LB, 0 usos de holdout.**
- 18:10 | v5 P2 baseline FastText + control + downstream | `experiments/embedding_02_fasttext_baseline_v5.py`
  (EBD-01..06): tres modelos entrenados **desde cero** con fastText C++ sobre los
  10M tokens de P1 (200d, ws5, neg10, minCount3, 5 epocas, 16 hilos): baseline con
  subpalabras 3-6, control de semilla (7) y control Word2Vec (`minn=0,maxn=0`).
  Se verifico que la normalizacion de P2 coincide token a token con la de P1 sobre
  los 2.885 textos de TOXIC (`src/embedding_utils.py` extrae las funciones comunes y
  deja constancia de que P1 conserva su copia historica).
  **Downstream (corrected_cv v2, 5 pliegues, 3 modelos binarios por etiqueta):**
  `FASTTEXT A_media LR` **0.6603** (delta vs V03 **-0.1169**, IC95 [-0.1397, -0.0942]),
  `FASTTEXT A_media SVC` 0.6584, `WORD2VEC A_media LR` 0.6465, poolings B (0.6348) y
  C (0.6388) peores que la media simple, y **suelo aleatorio 0.5404**.
  **Intrinseca:** los subpalabras cambian la geometria de forma clara (FastText da
  vecinos morfologicos: `violencia` -> `violentas`, `violento`; el control da vecinos
  distribucionales: `violencia` -> `discriminacion`, `delincuencia`), y aportan
  +0.0138 downstream. Entre semillas el coseno centrado es -0.034 (esperable: las
  soluciones de word2vec/fastText solo son identicas salvo rotacion) pero el
  solapamiento de los 10 vecinos es **0.792**, o sea la estructura relativa si se
  reproduce. **Lectura**: el embedding tiene senal real (+0.120 sobre el suelo
  aleatorio) pero aporta la mitad de lo que aporta V03 (+0.237); como sustituto queda
  refutado, y la hipotesis de complementariedad (que exige P6) **sigue sin medir**.
  **0 sondas de LB, 0 usos de holdout, 0 submissions.**
- 18:45 | v5 P3 complementariedad (fusion + meta anidado) | `experiments/embedding_03_fusion_complementariedad_v5.py`
  (EBD-07..09). Todas las variantes se evalúan en el **mismo punto de operación**:
  los umbrales congelados de V03 (0.44/0.51/0.46), para que nadie gane por ajustar
  su propio umbral; el control `V03-TFIDF` reproduce **0.777211 exacto**, lo que
  valida el protocolo celda a celda. Resultados: `V03-TFIDF` 0.7772 | `EMB-SOLO`
  0.6530 | **`FUSION-HSTACK-L2` 0.7818 (delta +0.0046, IC95 [-0.0032, +0.0124])** |
  `FUSION-HSTACK-STANDARD` 0.6901 | `META-ANIDADO` 0.7678 (delta -0.0094, IC95
  negativo). Se detecto y corrigio un bug propio: en la primera version se
  normalizaba L2 y despues se estandarizaba, de modo que el estandarizador
  anulaba el L2 y el bloque denso (varianza unitaria) dominaba al disperso; con
  los mismos datos la version estandarizada pierde 0.09 de F1. Queda fijado L2 por
  documento como convencion de la fase. La ganancia de la fusion L2 se concentra
  en las rebanadas que P1 marco debiles: elongaciones +0.0368, urls +0.0204,
  hashtags +0.0151, colectivos +0.0150, afiliacion +0.0079. Complementariedad
  2x2: 8 celdas solo de la fusion frente a 5 solo de V03. El apilamiento anidado
  empeora y rompe 53 celdas por cada 5 que arregla: quinta vez que el stacking
  generico no aporta. **Estado: CANDIDATE, no adopted** (no cumple A ni B de la
  seccion 29). **0 sondas de LB, 0 usos de holdout, 0 submissions.**
- 19:30 | v5 P4-P5 corpus web mC4-es y revision de la fusion | `experiments/embedding_04_corpus_mc4_v5.py`
  (E-CORP-06/07) y `experiments/embedding_05_fasttext_web_v5.py` (EBD-10). Tramo de
  40 MB del shard 0 de `allenai/c4` config `multilingual/c4-es` (13,9 % del shard,
  sha256 `96ed39f4...`, licencia ODC-BY), con limpieza de boilerplate de Common
  Crawl. Resultado de cobertura: web 10M alcanza **0.8256** por tipo y **0.9512** por
  token frente a 0.6839 y 0.8833 de Wikipedia; la union 20M llega a **0.8413** y
  0.9553, con `elongaciones` de 0.039 a 0.221, `coloquial` de 0.692 a 0.941 e
  `insultos` de 0.778 a 0.983, y 1.346 tipos TOXIC nuevos. **Pero la fusion no
  mejora**: FUSION-WEB 0.7812 (+0.0040), FUSION-MIXTO 0.7776 (+0.0004), frente a
  FUSION-WIKI 0.7818 (+0.0046), todos con IC95 que toca el 0. El embedding **aislado**
  si mejora (0.6530 -> 0.6658), lo que confirma que la cobertura sube pero no es la
  palanca de la complementariedad. Varias variantes ortograficas (`callate`,
  `arabe`, `estn`, `espaa`) pasan a estar en el vocabulario del corpus web.
- 19:50 | v5 auditoria final y cierre | `experiments/embedding_06_auditoria_v5.py`
  (**GO**): 0 usos de recursos prohibidos (la primera version de la auditoria daba
  22 falsos positivos por buscar la palabra `word2vec` en vez del uso de gensim, y
  por penalizar la llamada `holdout_v2()` que **excluye** el holdout; se reescribio
  con comprobaciones semanticas), 0 fugas medidas, holdout 5/5 intacto, submission
  de V03 verificada, los 6 hashes `.vec` coinciden con los informes y las 7 cifras
  afirmadas en D-032 se releen de los informes. Informe de cierre en
  `reports/embedding_phase_closeout_v5.md`. **0 submissions, 0 sondas de LB.**
- 20:40 | v5 P8 curva 10M -> 50M | `experiments/embedding_07_corpus_50M_v5.py` (E-CORP-08) y
  `experiments/embedding_08_fasttext_50M_v5.py` (EBD-13/14), a peticion del usuario.
  `RATIONALE_FOR_CORPUS_EXPANSION-50M` registrado antes de construir, con las
  condiciones 1, 2 y 3 de la §30 y el criterio de parada declarado. El corpus de
  50M se construyo **sin descarga adicional** (misma parte 1 del dump) y con el
  pipeline de P1 **importado**, de modo que los 10M anteriores estan contenido
  identicamente (verificado). 50.000.000 tokens, 21.529 documentos, 646.957 tipos;
  cobertura por tipo 0.6839 -> **0.7703** y por token 0.8833 -> **0.9238**;
  375.748 tipos nuevos (739 relevantes para TOXIC). FastText 200d entrenado en
  577 s (vocab 247.863) y el control Word2Vec en 454 s. **Resultado**:
  `FUSION-WIKI-50M` **0.7803** (delta +0.0031, IC95 [-0.0049, +0.0112]) frente a
  0.7818 con 10M, y `EMB-WIKI-50M` **0.6687** frente a 0.6530 (+0.0157). Es decir,
  5x tokens **mejoran la representacion** y **no** la aportacion complementaria.
  Las subpalabras aportan +0.0030 a 50M frente a +0.0138 a 10M. **D-033**: no se
  escala a 100M ni 500M; el techo no es la representacion sino el tamaño de los
  datos etiquetados y el bloque disperso de V03. **0 submissions, 0 sondas de LB.**
- 21:20 | v5 P9-P10 tramo inferior de la curva (5M) | `experiments/embedding_09_corpus_5M_v5.py`
  (E-CORP-09) y `experiments/embedding_10_fasttext_5M_v5.py` (EBD-15/16), a peticion del
  usuario. `REDUCCION_A_5M` registrada: la §30 aplica a expansiones, no a reducciones.
  Corpus de 5.011.404 tokens verificado como **prefijo exacto** de los 10M, sin descarga
  adicional. FastText 200d entrenado en 43 s (vocabulario 64.839) y control Word2Vec en
  29 s. **Resultado**: `FUSION-WIKI-5M` **0.7757** (delta **-0.0015**, IC95
  [-0.0091, +0.0060]), es decir a 5M la fusion deja de aportar y cae por debajo de V03;
  `EMB-WIKI-5M` 0.6670. La curva del embedding aislado (0.6670 / 0.6530 / 0.6687) es plana:
  los IC95 de 5M y 10M se solapan casi por completo, de modo que la no monotonia del punto
  de 10M no es distinguible del ruido. **D-034** registra el suelo operativo de 10M y
  **corrige** la sobreafirmacion de D-033: la ventaja de las subpalabras (+0.0060 / +0.0138
  / +0.0030) tiene forma de joroba con maximo en 10M, no es monotone. **0 submissions,
  0 sondas de LB.**
- 20:40 | v5.1 P0: migracion del worktree a PLN y auditoria de firmas (EBD-17) | `experiments/embedding_11_auditoria_firmas_v5_1.py`
  **Migracion**: el worktree `proyecto-kaggle-toxic-2026-v5` paso de `/home/carlos/Git/PLN-v5`
  a `/home/carlos/Git/PLN` a peticion del usuario. El WIP sin commitear de
  KAGGLE_AGE_2026_PUJ quedo aparcado con `git stash` (autorizado expresamente; no es un
  commit) y sus datos gitignored no se tocaron. Se copiaron `data/`, `tools/`,
  `experiments/*.log`, `features/` y `models/`; se verificaron los sha256 de los cinco corpus
  y de los diez `.vec`; y el control V03 reproduce **0.777211 exacto** desde la nueva
  ubicacion.
  **P0**: las diez firmas de v5 son distintas y las nueve propuestas de v5.1 no colisionan
  con ninguna, asi que las nueve son Units de entrenamiento nuevas. Presupuesto abierto
  0/9 + 0/1. Dos hallazgos de contabilidad: (1) el `raw_sha256` del manifiesto **no**
  identifica la escala (5M, 10M y 50M de Wikipedia comparten dump), asi que la identidad
  pasa a usar el sha256 del corpus procesado; (2) P5 no guardaba `minn`/`maxn` por modelo, por
  lo que la firma de sus tres modelos se declara en el inventario y se verifica contra el
  artefacto, en lugar de deducirse. Reglas de la seccion 36 limpias, holdout 5/5 intacto.
  **D-035**. P0 no entreno nada:, 0 submissions, 0 sondas de LB.
- 21:20 | v5.1 P1: familia vector_size (EBD-18) | `experiments/embedding_12_vector_size_v5_1.py`
  Entrenados FastText 100d (57 s) y 300d (171 s) sobre Wikipedia 10M con el resto de la
  arquitectura fija; el 200d no se reentrena y se reevalua como control. **El control
  reproduce v5 exacto** (FUSION 0.7818, embedding solo 0.6523 con umbrales congelados).
  FUSION-VS-100 0.7823 (+0.0050, IC95 [-0.0028, +0.0130]), FUSION-VS-300 0.7805 (+0.0033),
  dispersion interna 0.0017. **La dimension no es la palanca**: la familia se cierra por
  inercia y no se exploran 400d, 500d ni 768d. El mejor delta toca el umbral por cinco
  diezmilésimas con el intervalo incluyendo el 0, asi que **D-036** adopta un gate de dos
  niveles (delta >= +0.005 **y** IC95 que excluya el 0). Correccion de metodo registrada: los
  tipos consultados son los 6.059 del train y no los de train+test; con el conjunto
  equivocado 100d salia -0.0065. **Presupuesto 2/9 + 0/1.** 0 submissions, 0 sondas de LB.
