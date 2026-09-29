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
  v2 parrafo 4.3 refutada** (D-028). El meta de 36 features cumple el nivel de D-016.
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
  `data/external/corpora/corpus_manifest.json` creado con el esquema de la parrafo 35 y
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
  (EBD-07..09). Todas las variantes se evaluan en el **mismo punto de operacion**:
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
  condiciones 1, 2 y 3 de la parrafo 30 y el criterio de parada declarado. El corpus de
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
  escala a 100M ni 500M; el techo no es la representacion sino el tamano de los
  datos etiquetados y el bloque disperso de V03. **0 submissions, 0 sondas de LB.**
- 21:20 | v5 P9-P10 tramo inferior de la curva (5M) | `experiments/embedding_09_corpus_5M_v5.py`
  (E-CORP-09) y `experiments/embedding_10_fasttext_5M_v5.py` (EBD-15/16), a peticion del
  usuario. `REDUCCION_A_5M` registrada: la parrafo 30 aplica a expansiones, no a reducciones.
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
  diezmilesimas con el intervalo incluyendo el 0, asi que **D-036** adopta un gate de dos
  niveles (delta >= +0.005 **y** IC95 que excluya el 0). Correccion de metodo registrada: los
  tipos consultados son los 6.059 del train y no los de train+test; con el conjunto
  equivocado 100d salia -0.0065. **Presupuesto 2/9 + 0/1.** 0 submissions, 0 sondas de LB.
- 21:45 | v5.1 P2: familia window (EBD-19) | `experiments/embedding_13_window_v5_1.py`
  Entrenados FastText con ventana 3 (83 s) y 7 (148 s) sobre Wikipedia 10M. El control
  B0 (ventana 5) reproduce v5 exacto (0.7818). FUSION-WS-3 0.7830 (+0.0057, IC95
  [-0.0021, +0.0136]), FUSION-WS-7 0.7806 (+0.0034), dispersion 0.0024. El embedding
  aislado con ventana 3 da **0.6694**, el mejor de toda la fase v5. **Familia CERRADA por
  inercia** (D-037): no se exploran ventanas 1, 2, 9 ni 15. El orden 3 > 5 > 7 es
  contraintuitivo y apunta a que el limite esta en los datos, no en la funcion objetivo.
  El script de P2 define la maquinaria de familia reutilizable que usan P3, P4 y P5.
  **Presupuesto 4/9 + 0/1.** 0 submissions, 0 sondas de LB.
- 22:10 | v5.1 P3: familia min_count (EBD-20) | `experiments/embedding_14_min_count_v5_1.py`
  Entrenados FastText con min_count 2 (128 s, vocabulario 139.479) y 5 (96 s, 69.310). El
  control B0 (min_count 3) reproduce v5 exacto. FUSION-MC-2 0.7771 (-0.0001),
  FUSION-MC-5 0.7792 (+0.0020), el mejor de la familia es el propio baseline 0.7818;
  dispersion 0.0047. **Familia CERRADA por inercia** (D-038). Hallazgo mecanistico: los
  6.059 tipos del train reciben vector en los tres puntos, porque con subpalabras 3-6
  fastText ya compone vector para cualquier tipo consultado; bajar min_count agranda el
  vocabulario general pero no anade ni un tipo de TOXIC con vector. La via de mejora que se
  atribuia a min_count ya esta ocupada por las subpalabras. **Presupuesto 6/9 + 0/1.**
  0 submissions, 0 sondas de LB.
- 22:40 | v5.1 P4: familia subwords (EBD-21) | `experiments/embedding_15_subwords_v5_1.py`
  Intervalos 3-5 y 3-7 con min_n fijo en 3 y bucket 200.000; el control 3-6 reproduce v5
  exacto. FUSION-SW-3-5 0.7791 (+0.0018), FUSION-SW-3-7 0.7767 (-0.0005), el mejor de la
  familia es el propio baseline 0.7818; embedding solo 0.6397 / 0.6523 / 0.6465; dispersion
  0.0051. **D-039**: el intervalo 3-6 ya era el optimo y el orden es monotono (acortar
  empeora porque se reconstruyen menos formas ortograficas; alargar empeora mas por las
  colisiones en el bucket fijo). La familia se cierra con la **regla de dispersion revisada**:
  con el baseline como maximo, la dispersion mide cuanto empeora al alejarse de el y no cuanto
  mejora. La maquinaria compartida no se modifica; la correccion se aplica en P4.
  **Presupuesto 8/9 + 0/1.** 0 submissions, 0 sondas de LB.
- 23:05 | v5.1 P5: familia negative (EBD-22) | `experiments/embedding_16_negative_v5_1.py`
  FastText con 5 negativos (73 s); el control con 10 reproduce v5 exacto. FUSION-NEG-5
  0.7811 (+0.0039, IC95 [-0.0044, +0.0123]) frente a 0.7818 del control; embedding solo
  0.6664 frente a 0.6523. Dispersion interna **0.0006**, la mas baja de las cinco familias.
  **Familia CERRADA** (D-040). Se agotan las **9/9** unidades exploratorias y la **reserva
  queda intacta (0/1)**: no hay ninguna mejora que confirmar con una repeticion, que es lo
  unico para lo que existia la reserva. El informe de P0 pasa a llamarse
  `reports/embedding_architecture_auditoria_v5_1.md` para dejar el nombre que pide la spec
  al informe de resultados de arquitectura (P6). 0 submissions, 0 sondas de LB.
- 23:30 | v5.1 P6: comparacion final y conclusiones (EBD-23) | `experiments/embedding_17_comparacion_final_v5_1.py`
  No entrena: consolida las cinco familias. 14 configuraciones (9 nuevas + el baseline
  reevaluado 5 veces, que reproduce v5 exacto en todas). Embedding aislado 0.6397-0.6694
  (rango 0.0297) frente a fusion 0.7767-0.7830 (rango 0.0063): el rango del embedding
  aislado es 4,7 veces mayor. 0 configura adoptables. **D-041**: conclusion de
  representacion (la arquitectura si cambia la representacion, y el optimo esta en ventana 3
  con subpalabras 3-6, o sea senal local y ortografica) y conclusion de tarea (ninguna
  aporta a V03). **No se escala el corpus**: fallan tres de las cinco condiciones de la
  seccion 12 y las que fallan son de evidencia, no de coste. El informe de P0 se renombra a
  `embedding_architecture_auditoria_v5_1.md`. 0 submissions, 0 sondas de LB.
- 23:55 | v5.1 P7: complementariedad (EBD-24) | `experiments/embedding_18_complementariedad_v5_1.py`
  Reevaluadas cuatro configuraciones sin entrenar. WS-3: salvage 8 filas, damage 2, neta +6;
  el baseline 8/5/+3. **WS-3 no arregla mas: arregla lo mismo y rompe menos**, que es donde
  esta toda su ventaja. La interaccion ortografica da delta +0.0083 dentro (n=1.289) y
  -0.0045 fuera (n=194), diferencia +0.0128 con IC95 que incluye el 0: direccion esperada pero
  sin significacion, y la mascara cubre el 87 % de las filas, asi que el test tiene poco
  poder. **D-042**, que ademas corrige un nombre arrastrado desde v5: `_f1_por_celda`
  devuelve la media de las tres etiquetas de cada fila, de modo que la unidad de analisis son
  las 1.483 filas y no las 4.449 celdas. Los conteos de v5 no cambian; la etiqueta estaba mal.
  0 submissions, 0 sondas de LB.
- 19:45 | v5.1 P8-P11: notebook, documentacion, validacion y auditoria final | `embedding_19`, `embedding_21`, `embedding_20`, `embedding_22`
  **P8 (EBD-25)**: seccion 18 generada por script en el notebook consolidado (44 -> 63
  celdas), con `TRAIN_FROM_SCRATCH=False` y el mecanismo de entrenamiento a la vista.
  **P9 (EBD-27)**: seccion 28 del informe final con las diez entradas de la seccion 38 de la
  spec, generada leyendo los artefactos (173 experimentos, 22 scripts del embedding, 138
  evaluaciones downstream, 42 decisiones).
  **P10 (EBD-26)**: validacion del notebook por **ejecucion completa** con nbconvert. Primer
  intento NO-GO: una celda previa fijaba el numero de filas de experiments.csv en 143, que
  esta fase deja obsoleto; segunda falla por un `NameError` mio en la celda de contabilidad;
  ambas correcciones aplicadas en el generador (que ahora es idempotente con
  `--reemplazar`). Resultado final: **GO**, 32/32 celdas, 0 errores, seccion 18 ejecutada y
  aserciones de presupuesto cumplidas, V03 verificado en 0.777211. El notebook se guarda
  ejecutado con sus salidas.
  **P11 (EBD-28, AUDIT)**: auditoria final, **11/11 comprobaciones OK, veredicto GO**. 19/19
  modelos con hash verificado, 0 recursos prohibidos, 0 fugas, holdout intacto, presupuesto
  9/9 + 0/1, V03 intacto, 0 submissions nuevas, rama correcta y sin push. **D-043**, que cierra
  la fase. 0 submissions, 0 sondas de LB.
- 20:30 | v6 P0: apertura de la fase, gap analysis y diagnostico | `v6_00_gap_analysis.py`, `v6_01_estructura_errores.py`
  Rama `proyecto-kaggle-toxic-2026-v6` creada desde el cierre de v5.1.
  **P0 (V6-00)**: 15 verificaciones programaticas sobre el codigo real (patrones de
  implementacion, con exclusion de la auto-referencia). **C2, C3 y C4 quedan CERRADAS**: ya
  se ejecutaron en V4-D (interacciones 0.7426, ventanas 0.7734) y E2-10 (-0.028 a -0.039).
  C se reduce a C1 (skip-grams), que si es nuevo. J no es duplicado de PRE: PRE fue
  normalizacion, no aumento. D conserva la arquitectura (ya existe en DL-02) pero cambia el
  input (embeddings aprendidos en vez de aleatorios). **Hallazgo de integridad**: 6 de las
  174 filas de experiments.csv tienen las columnas desplazadas; el numero real de
  experimentos es 168, no 170. Las filas no se tocan por ser evidencia historica.
  **P0-bis (V6-01)**: interseccion de errores y estructura de etiquetas. **La interseccion es
  de solo 9 a 12 filas de 1.483 (0.8 %)** frente a 1.133 donde nadie acierta, lo que fija el
  techo de cualquier linea que anada otro modelo en **+0.0027, por debajo del gate de
  +0.003**: **D y E se cierran preventivamente por presupuesto, no por resultado negativo**.
  Y **TR=1 implica HS=1 de forma determinista** (0 casos de TR=1 con HS=0), invariante que
  un clasificador independiente no puede expresar y que **si justifica el track B**. Ademas
  P(AG|HS=0)=0.026 con 22 ejemplos, que obliga a la forma regularizada B2r.
  **D-044**. 0 entrenamientos, 0 submissions, 0 sondas de LB.
- 21:10 | v6 Track B: factorizacion condicional, resultado negativo | `v6_02_label_factorization.py`
  **B0 control reproduce V03 exactamente** (0.777211, delta +0.0000, IC95 [0, 0], 0 salvage,
  0 damage): el pipeline es valido y los deltas son atribuibles. Ese control no es un detalle,
  es lo que hace legitima la comparacion.
  **B1 TR condicional -0.0107** (IC95 [-0.0184, -0.0033], 1/5 pliegues, 1 rescatada / 35
  rotas). B1c y B3 -0.0115. **B2 AG dos ramas -0.0008** con la rama HS=0 regularizada (22
  positivos de 835). **Ninguna variante pasa el Gate 1.**
  Causa medida: V03 solo viola la invariante en **20 filas de prediccion (1.35 %)**. La
  restriccion **si se aplica** (0 violaciones en la variante factorizada), pero el producto
  tambien atenua p(TR) en toda fila con p(HS) intermedia, que es donde V03 acertaba.
  **Track B cerrado por resultado negativo (seccion 54), sin ampliar rejilla.**
  **D-045**. 6 unidades de entrenamiento, 0 submissions, 0 sondas de LB.
- 21:40 | v6 Track A: ponderacion por grupo, resultado negativo | `v6_03_ponderacion_grupo.py`
  **A0 control reproduce V03 exactamente** (0.777211, delta +0.0000, IC95 [0,0]).
  A1 1/sqrt(g) **+0.0006** (IC95 [-0.0029,+0.0040], 3/5), A2 1/g +0.0002 (2/5), A3 con tope
  0.25 **-0.0149** (IC95 que excluye el 0, 1/5). **Ninguna pasa el Gate 1.**
  Hallazgo de estructura que conviene antes que el resultado: **1378 de 1483 filas (92.9 %)
  ya estan en grupos de tamano 1**, asi que solo el 7.1 % de las filas puede cambiar de
  peso. Maximo 7 filas por grupo, 1415 grupos distintos.
  Y el patron es monotono en la agresividad: **cuanto mas se penaliza a los repetidos, peor**
  (A1 > A2 > A3). La lectura es que **un tuit repetido es senal, no ruido de recoleccion**,
  y que no conviene reabrir esta linea ni en variantes mas suaves.
  **Track A cerrado por resultado negativo (seccion 54). D-046.** 0 submissions.
- 21:55 | v6 Track I: ponderacion word/char, resultado negativo | `v6_04_char_weight.py`
  **I0 control reproduce V03 exactamente** (0.777211, delta +0.0000, IC95 [0,0]).
  char_weight 0.5 **-0.0117**, 1.5 **+0.0016** (IC95 [-0.0048,+0.0078], 3/5), 2.0
  **-0.0047**. **Ninguna pasa el Gate 1.**
  El perfil tiene **optimo interior**: el peso heredado de `hstack` no era optimo y la
  respuesta es que el balance deberia inclinarse algo hacia el bloque de caracteres. Pero el
  techo es +0.0016, **menos de la mitad del gate**, con IC95 que incluye el 0 y 3/5 pliegues:
  **la escala no era el cuello de botella**, que era la pregunta. No se afina alrededor de 1.5.
  **Track I cerrado (seccion 54). D-047.** 0 submissions.
- 22:10 | v6 Track C1: skip-grams, resultado negativo neutro | `v6_05_skipgrams.py`
  **El control destino un bug real**: la primera version del tokenizador emitia solo
  unigramas y el control daba 0.7554 en vez de 0.777211 (1407 caracteristicas). Los deltas de
  esa version no eran atribuibles y se descartaron enteros. Corregido para emitir tambien los
  bigramas contiguos: el control da **0.777211 exacto**.
  Resultado real: hueco 1 **-0.0022** (IC95 [-0.0084,+0.0039], 2/5) y huecos 1 y 2
  **-0.0001** (IC95 [-0.0085,+0.0082], 2/5), con el bloque de palabras de 2744 a 5025
  caracteristicas (+83 %). **Los skip-grams son NEUTROS**: ni anaden senal ni restan
  capacidad, a diferencia de las interacciones de V4-D (-0.0346). No es una repeticion de
  V4-D porque los pares compiten con las palabras en la misma matriz, no en un canal aparte.
  Con C cerrada (C2, C3, C4 por duplicacion y C1 aqui), **la sub-linea C queda sin nada
  abierto**, siendo la propuesta central de la spec de v6.
  **Track C1 cerrado (seccion 54). D-048.** 0 submissions.
- 22:35 | v6 Track H: bagging por grupos, cerrado por protocolo | `v6_06_bagging_grupos.py`
  **H0 control reproduce V03 exactamente** (0.777211, delta +0.0000, IC95 [0,0]).
  H1 promedio de 5 **-0.0105** (IC95 [-0.0183,-0.0030] que EXCLUYE el 0, **0/5 pliegues**).
  **La clave: no es por falta de diversidad.** Correlacion media entre pares **0.9573**
  (minima 0.9379), diferencia media absoluta 0.0606: hay cinco modelos genuinamente
  distintos. La hipotesis que el track se probo a si mismo (que el 92.9 % de filas en grupos
  unicos dejaria el bagging sin nada que promediar) queda **refutada por la medicion**, y a
  pesar de eso empeora.
  Explicacion coherente: **efecto de escala**. Promediar contrae las probabilidades y los
  umbrales congelados de V03 estan calibrados para un modelo individual. Reoptimizarlos
  romperia la comparabilidad y reproduciria el sesgo de D-045. **No hay forma limpia de
  medir esta linea con el protocolo actual**, asi que se cierra por protocolo y no por
  mecanismo, dejando constancia de que podria reabrirse con una particion de calibracion
  separada.
  **D-049.** 0 submissions.
- 23:00 | v6 Track F: ruido por par, la deteccion encuentra dificultad | `v6_07_ruido_par.py`
  Deteccion y ponderacion van en **rondas separadas** (sino la sospecha se define a si misma).
  **F0 control reproduce V03 exactamente.** F1 cuatro senales lambda 0.5 **+0.0001**
  (IC95 que incluye el 0, 2/5), F2 solo estructural lambda 1.0 **-0.0010**, F3 lambda 1.0
  **-0.0041**. **Ninguna pasa el Gate 1.**
  **La hipotesis del informe se refuto y se corrigio**: V03 acierta **10.5 %** en las 449
  filas sospechosas frente a **28.3 %** en las 1034 no sospechosas, mas del doble. La
  deteccion SI identifica filas donde el modelo falla, pero como subir lambda **empeora**,
  encuentra **dificultad genuina** y no etiquetado erroneo: **lo que marca es donde hay que
  estar, no donde hay que limpiar**. La senal sirve como marcador de dificultad para
  analizar errores, no para limpiar el entrenamiento.
  Y **el ruido por par es mas escaso que el ruido por duplicado**: la senal estructural
  generaliza el conflicto de V4-A a la similitud y marca **12 pares de 4449** frente a los
  84 tuits del duplicado exacto. **D-050.** 0 submissions.
- 23:25 | v6 Track J: aumento de datos, resultado negativo | `v6_08_aumento.py`
  **J0 control reproduce V03 exactamente** (0.777211, delta +0.0000, IC95 [0,0]), septimo
  control de la fase. J1 una copia (1.186 -> 2.372 filas) **-0.0061** (IC95
  [-0.0141,+0.0018], 1/5). J2 dos copias (-> 3.558 filas) **-0.0063** (IC95
  [-0.0145,+0.0016], 1/5). **Ninguna pasa el Gate 1.**
  El dato que va mas alla del signo: **doblar las copias casi no cambia nada**. Si el
  problema fuera la cantidad, duplicarla deberia empeorar mas. Apunta a **sesgo de las
  copias, no a exceso de ellas**: las variantes deformadas forman un patron coherente que el
  modelo aprende, y da igual que haya una o dos copias.
  Las transformaciones preservan la etiqueta por construccion (espaciado, elongacion,
  puntuacion); no se tocan sinonimos, reordenacion, negacion ni objetivo.
  **Track J cerrado (seccion 54). D-051.**
  **Con esto v6 se queda SIN LINEAS EJECUTABLES**: B, A, C, I, H, F, J cerradas por
  resultado negativo; D y E por techo teorico; G marcada "no ejecutar" desde el gap
  analysis. 0 submissions en toda la fase.
- 23:50 | v6 cierre: entregables y cierre de fase | `v6_09_entregables.py`
  **25 variantes medidas en 7 tracks, 7 controles de reproduccion, 0 superan el Gate 1.**
  De los 14 tracks y sub-tracks de la especificacion: 7 medidos (B, A, I, C1, H, F, J), 2
  cerrados sin gastar (D, E por techo), 1 no ejecutable (G) y C2, C3, C4 por duplicacion.
  **No queda ninguna linea ejecutable.**
  Los mas cercanos al gate: I2-char1.5 (+0.0016, a 0.0014 del gate), A1-sqrt (+0.0006),
  A2-inversa (+0.0002). Ninguno tiene senal estadistica.
  **7 de 24 experimentos gastados.** El presupuesto es un techo, no una obligacion, y dos
  decisiones de D-044 ya habian reducido el gasto antes de empezar.
  **Los 7 controles dieron 0.777211 exacto**, y eso es lo mas importante: la red de seguridad
  funciono. El control C1-0 detecto el bug de tokenizacion y evito que se publicara un delta
  falso.
  Entregables: v6_final_research.md, v6_master_results.csv, v6_hypothesis_registry.csv,
  v6_candidate_comparison.csv, v6_compliance.md, v6_reproducibility.md,
  v6_error_intersection.csv.
  **D-052. V03 congelada. 0 submissions. Rama sin push, pendiente de autorizacion.**
- v7 apertura: worktree, rama y auditoria de brecha
  Worktree en `/home/carlos/Git/PLN-v7` con rama `proyecto-kaggle-toxic-2026-v7` desde v6, para
  no tocar el arbol principal ni el trabajo sin versionar de KAGGLE_AGE. `data/` enlazado.
  Se repara un defecto de v6: `tools/` estaba ignorado entero, asi que
  `verificar_caracteres.py` se anunciaba pero no estaba versionado. Ahora `tools/*` con
  excepcion `!tools/verificar_*.py`, porque git no re-incluye ficheros con el directorio
  padre excluido. Puerta versionada, 0 errores.
  **V7-00 (v7_00_gap_audit.py)**: once tracks auditados con busqueda programatica y lectura de
  cifras. **Cerrados por duplicado con numero**: P0.2 y P0.5 (0.774954), P0.4 (+0.001831),
  P2.1 (10 filas de interseccion), P2.3 (0.761203), P3 (0.741036), P4 (-0.023836), P5
  (-0.250519), P6, P7 (-0.002292). **Abiertos de verdad: P8 y P9.** Con reservas: P0.1 ya
  medido en 0.776613 pero con protocolo mas laxo, y P1.1 cuya parte medible ya esta hecha
  porque el OOF canonico ya es el promedio por pliegue.
  **D-053 parte 1.**

- v7 Track P9: perdida orientada a la F1, resultado negativo | `v7_01_perdida_f1.py`
  Unico cambio respecto a la referencia: la funcion de perdida. Mismas features de V03, mismo
  modelo lineal, mismo protocolo, mismos umbrales, mismo balanceo en las tres perdidas.
  **Control BCE 0.777165 frente a 0.777211: hueco de -0.000046.** Con ese hueco el montaje
  queda verificado y el negativo es atribuible a la perdida.
  **FOCAL -0.0024** (IC95 que incluye el 0, 2/5) y **F1-SUAVE -0.0033** (3/5). Ninguna pasa
  el gate.
  **La mas directa de las dos es la peor.** Optimizar el objetivo literalmente con 1.186 filas
  acopla el descenso a la composicion de esa muestra: es lo que le paso a OPT-01, que tuvo el
  mejor OOF y el peor leaderboard. La conclusion que sobrevive: **en este problema, aproximar
  mejor el objetivo no es mejor**.
  **P9 cerrado. D-053 parte 2.** 0 submissions.
- v7 Track P8: prior de etiqueta del grupo, cerrado negativo | `v7_02_prior_grupo.py`
  El techo se mide ANTES de entrenar y ya acota la linea: **solo 105 de 1483 filas (7.08 %)
  tienen grupo repetido**, y de los pares grupo-etiqueta **80 son uniformes frente a 31
  contradictorios**, o sea que el prior es casi determinista.
  **Control P8-0 reproduce V03 exactamente** (0.777211, hueco +0.000000). P8-1 con las tres
  desviaciones **-0.0011** (IC95 que incluye el 0, 1/5) y P8-2 con soporte **-0.0017**
  (1/5). **Ninguna pasa el gate.**
  El dato que remata el diagnostico: V03 acierta **22.9 %** de las filas con evidencia y
  **22.9 %** del resto, y la caracteristica **no arregla ni una sola fila** de las que tiene
  evidencia. Es inerte exactamente donde se aplica.
  La distincion que sobrevive: la fase v6 concluyo que la repeticion es senal, y es cierto,
  pero **ser senal y ser senal nueva son dos cosas distintas**. Aqui se separan: la
  repeticion es senal que V03 ya captura, porque dos filas del mismo grupo son casi el mismo
  texto y el modelo ya les da la misma prediccion.
  Ademas se corrige una afirmacion del propio informe que los datos refutaban: suponia que
  V03 ya acertaba mejor en las filas con evidencia, y el acierto resulto identico.
  **Track P8 cerrado (seccion 54). D-054.** 0 submissions.
- v7 Track P0.1: umbral anidado estricto, y cierre de toda la capa de decision | `v7_03_umbral_anidado.py`
  Protocolo de **tres niveles** que no se mezclan: ajuste, calibracion, evaluacion. La
  particion interna respeta grupos y estratifica por la combinacion de etiquetas.
  **Tres brazos, porque hay que separar si el codigo esta bien de cuanto cuesta el
  protocolo**:
  - **P0-0** ajusta sobre todo el entrenamiento exterior: **0.777211, delta +0.000000**,
    con lo que el codigo del pipeline queda validado.
  - **P0-1-control** con el pipeline de tres niveles y umbrales congelados: 0.769123,
    **-0.0081**, y ese delta **no es un error: es el coste del protocolo**, que obliga a
    ajustar sobre cinco sextos de las filas.
  - **P0-1-anidado** comparte conjunto de ajuste con el control, asi que el coste se
    cancela y el delta **-0.0019** (IC95 que incluye el 0, 2/5) mide el efecto del umbral.
  **El producto principal es el informe de dispersion**, que era la tarea que v6 dejo
  pendiente. HS: media 0.4120, mediana 0.4100, desviacion 0.0402, rango 0.1000. TR: media
  0.4900, mediana 0.4500, desviacion 0.0652, rango 0.1500. AG: media 0.4560, mediana
  0.4400, desviacion 0.0586, rango 0.1400.
  **Conclusion: los umbrales estan debilmente identificados.** El de TR puede moverse de
  0.37 a 0.60 sin que el dato lo impida. Los tres congelados de V03 caen dentro del rango,
  asi que no son claramente sobreajuste, pero las medianas calibradas son sistematicamente
  mas bajas y el anidado termina peor. Eso explica los dos hechos medidos antes: por que
  E2-16 dio -0.0006 y por que reajustar umbrales infla +0.0036. **No hay informacion en los
  datos que identifique el umbral.**
  **P0.3 y P10 se cierran tambien sin ejecutar, y por evidencia**: la dispersion es alta
  pero eso no implica upside, porque un estimador robusto daria un valor mas cerca de la
  mediana, que esta mas lejos de los congelados que la media.
  El gate detecto un error de diseno propio: la primera version fijaba la tolerancia sobre el
  control de tres niveles y se detuvo al superarla. La deteccion era correcta y la
  tolerancia estaba mal planteada, porque el error era del protocolo y no del codigo.
  **P0 cerrado. D-055. V03 mantiene sus umbrales por evidencia medida y no por inercia.**
  0 submissions.
- v7 cierre: entregables y auditoria de cumplimiento | `v7_04_entregables.py`
  **10 filas en la tabla maestra, 9 variantes medidas, 4 controles de calidad, 0 superan el
  gate.** Los tres mas cercanos al gate: P8-1-desviacion (-0.0011, a 0.0041), P8-2
  (-0.0017) y P0-1-anidado (-0.0019).
  **Auditoria de cumplimiento programatica: 9 prohibiciones, 0 infracciones.** Dos fallos de
  la propia auditoria corregidos: se contaba a si misma (contiene los nombres que busca) y el
  patron del holdout marcaba como infraccion el uso legitimo de `holdout_v2`. El patron se
  redujo a la infraccion real y se verifico con tres casos de prueba, uno de ellos el uso
  legitimo.
  Puerta de caracteres: **0 errores**. Rama con commits locales, **sin push**.
  **D-056. FASE v7 CERRADA sin adopciones.** V03 sigue congelada en 0.777211 con sus tres
  umbrales, por evidencia medida y no por inercia. Queda abierto solo P1 en su formulacion
  de solo prediccion, que no se puede medir en local, y la verificacion de transferencia de
  un finalista.
- v8 apertura: auditoria de brecha y del material de clase | `v8_00_gap_audit.py`
  **Diez pistas experimentales auditadas, cinco sin probar y cinco con precedente; veintidos
  tecnicas del material de clase, nueve genuinamente nuevas y trece cerradas.** El control de
  calidad de la auditoria encontro y corrigio **tres falsos positivos propios**: la asimetria
  casaba con la prosa de un informe anterior, la traduccion con el metodo de conversion de
  jerga, y las idiosincrasias con un comentario sobre cobertura de lexico. Los tres patrones
  se endurecieron para exigir asignacion o llamada. Ademas el conteo simple resulto
  genuinamente usado en la linea ortografica de E2-10 y su cierre se corrigio para citarlo.
  **Dos hallazgos que reencuadran la especificacion**: P2, la ponderacion por razon logaritmica
  de recuentos, **es el estimador de Mosteller y Wallace de 1964** y el proyecto ya la midio en
  cuatro variantes cerradas entre -0.016 y -0.039, de modo que no es una idea nueva sino una
  reejecucion; P3, la estabilidad de features, es el capitulo 8 de ese mismo libro y si tiene
  respaldo doble, con la traduccion de ida y vuelta sustituida por submuestreo.
  Queda escrito que las reglas de la competencia permiten modelos preentrenados y que el
  proyecto mantiene su prohibicion como decision consciente. **D-057. 0 Experiments de
  entrenamiento, 0 submissions.**
- v8 P1: promedio de los cinco modelos al predecir | `v8_01_ensamble_folds.py`
  **Control de identidad perfecto: diferencia maxima por celda de 0.000e+00 frente al OOF
  canonico de V03, y metrica 0.777211 exacta.** Esa comprobacion es mas fuerte que cualquier
  tolerancia de metrica: si las probabilidades coinciden celda a celda, el pipeline es el de
  V03. **La metrica local de esta linea no es informativa y no puede serlo**: la columna de
  desarrollo es el promedio por pliegue, que es la definicion del OOF, asi que tiene que
  coincidir con V03. Lo unico nuevo es el promedio sobre la prueba.
  **Error de control documentado**: la primera version tomaba como control el modelo completo
  medido sobre si mismo y devolvio 0.969085. La causa es que esa cifra es resubstitucion y no
  validacion. El script ahora etiqueta esa cifra como no comparable.
  **Alcance del cambio pequeno y techo pequeno en consecuencia**: contraccion de
  probabilidades de +0.001334, tasas de positivos que se mueven en la tercera cifra decimal, y
  **92 filas de 1154 y 100 celdas de 3462 diferentes de la submission de V03, un 2.9 %**, menos
  que las 126 celdas del cambio v02 a v03 que dio +0.0031 externo. El techo razonable de la
  linea se escribe antes del envio: **orden de milesimas**.
  Dispersion entre pliegues registrada: de 0.7386 a 0.7960 en el rango, de modo que el
  promedio pesa igual un pliegue mas debil.
  **Nueve comprobaciones de integridad del fichero, todas correctas. Fichero escrito y
  PARADA OBLIGATORIA: no se envia automaticamente y no se crean variantes antes del numero.**
  **D-058. 1 Sonda externa preparada, 0 enviadas.**
- v8 P1: resultado de la sonda externa | `v8_01_ensamble_folds.py` y registro
  **LB 0.77470 frente a 0.78306 de V03, diferencia -0.00836. P1 se cierra como linea negativa,
  sin variantes.** El techo de milesimas declarado antes del envio resulto correcto en magnitud
  y equivocado en el signo. La causa mecanica estaba medida: contraccion de +0.001334 y 100
  celdas de 3462. **D-059.** 0 submissions automaticas.
- v8 A2: resolucion del leaderboard | `v8_02_resolucion_liderboard.py`
  **El limite esta en el instrumento y es cuantificable.** Ruido del leaderboard 0.01524 para
  la F1, sensibilidad 7.93e-04 por acierto neto, de modo que **el ruido equivale a 19.2
  aciertos netos**. De las once sondas reales, **10 caen dentro de la banda y ninguna esta por
  encima de V03**. La unica distinguible es v01, con -33.2 netos, que es la peor de todas.
  **Se cae una premisa propia**: el paso de v02 a v03, que el proyecto citaba como el unico
  movimiento positivo del leaderboard y como justificacion del freno de transferencia, son +3.9
  aciertos netos frente a un ruido de 19.2. Fue ruido. El freno sigue valiendo, pero por otra
  razon: no es que la capa de decision tuviera historial positivo, es que el historial entero
  era indistinguible.
  **Se adopta un filtro de detectabilidad previo**: una candidata que cambie C celdas solo es
  resoluble si su tasa de acierto entre ellas supera 0,5 mas 19,2 dividido por C, es decir
  0.60 con cien celdas, 0.55 con doscientas y 0.51 con ochocientas. Ese numero se estima en el
  desarrollo antes de enviar, asi que el filtro es previo y gratis.
  **Reordenacion**: P5, P6 y P7 bajan de prioridad por moverse en la banda de cien a doscientas
  celdas. **P3 sube a la primera posicion.** El objetivo de superar 0.80 queda a 1.1 veces el
  ruido y 21 aciertos netos de ventaja limpia sobre once sondas de las que diez son
  indistinguibles entre si.
- v8 P3: estabilidad de features entre submuestras | `v8_03_estabilidad_features.py`
  **Negativo y monotono en los tres umbrales.** 0,5 deja 1636 rasgos de 20.441 y da
  -0,003308; 0,7 deja 927 y da -0,004628; 0,9 deja 455 y da -0,018890 con el IC95 excluyendo
  cero. Cuanto mas agresiva la seleccion, peor. **El filtro de detectabilidad falla en los tres**
  con tasa de acierto entre celdas modificadas de 0,4402, 0,4436 y 0,3930. **No se genera
  submission y no se gasta sonda.**
  **El hallazgo es como pierde, no que pierde**: acierta del 39 % al 44 % en las celdas que
  toca, por debajo del azar. Un rasgo estable bajo submuestras es de magnitud grande y
  repetida, mas probablemente frecuente y neutro que discriminante. Se conserva el ruido
  frecuente y se pierde la senal minoritaria.
  **Conclusion que sobrevive a la fase**: el capitulo 8 de Mosteller y Wallace mide
  estabilidad para atribuir autoria, donde un rasgo frecuente y neutro si informa porque el
  objetivo es la forma y no la clase. Traducir el criterio a clasificacion no conserva su
  sentido. La idea es buena y la fuente es buena; lo que falla es la traduccion al problema.
  **D-060. 0 submissions.**
- v8 P2: ponderacion por razon logaritmica de recuentos | `v8_04_log_count_ratio.py`
  **Precedente adverso confirmado y ninguna variante admisible.** Con suavizado 1e-6 el ratio
  sobre la TF-IDF da -0,058496; con 1e-3 da +0,004745. **Salto de +0,063241 por un factor de
  mil en un solo parametro.** El filtro de detectabilidad falla en las tres: 0,3266, 0,5281
  frente a un minimo de 0,5317, y 0,3217. **No se genera submission y no se gasta sonda.**
  **Correccion de premisa propia antes de ejecutar**: la auditoria de V8-00 decia que el ratio
  por etiqueta no se habia probado, y era falso; el bucle de E2-15 ya lo hacia por etiqueta. Lo
  no probado era el ratio sobre la TF-IDF de V03 con la constante 2.0.
  **El suavizado sostiene el metodo entero**: con suavizado pequeno el ratio tiene desviacion
  de mas de tres y rango de mas de veinte unidades, porque en la TF-IDF las caracteristicas
  raras ya son pequenas y dividir por una probabilidad casi nula explota.
  **No se barre el suavizado**, porque que la linea pase de -0,058 a +0,005 cambiando un
  parametro sin caracterizar significa que el resultado positivo es un punto de una superficie
  sin explorar, no una senal.
  **El control faltaba en E2-15 y aporta una conclusion**: la constante 2.0 con ratio da
  -0,075 frente a los -0,016 de la constante 1.0, o sea la constante y el ratio no se pueden
  subir a la vez.
  **D-061. 0 submissions.**
- v8 P4: regularizacion conjunta entre las tres etiquetas | `v8_05_regularizacion_conjunta.py`
  **Negativo y monotono, y el mecanismo funciona.** El control interno reproduce V03 con
  diferencia -0,001559. La identidad de la solucion se confirma por el **valor del objetivo**:
  2,928 propio frente a 3,359 de la biblioteca, y con la regularizacion cuadrada la superficie
  es estrictamente convexa, luego la solucion propia es la optima. Peso 0,5 da -0,001471 y
  peso 2,0 da -0,006781. Al tiempo, el coseno entre las tres cabeceras sube de 0,7380 a 0,8265
  a 0,9216: **compartir estructura funciona y por eso empeora.**
  **El hallazgo que unifica la fase**: la tasa de acierto entre las celdas modificadas es de
  0,4186 y 0,3925, por debajo del azar, igual que P3 (0,39 a 0,44) y P1 (0,46). Tres lineas,
  tres mecanismos, mismo sintoma: en las filas que tocan, todas mueven la decision hacia peor.
  **Cinco errores de escala detectados antes de obtener cifra**: factor dos en el gradiente
  (contra diferencias finitas), fallo de dimensiones en la transposicion, normalizacion de la
  perdida por filas efectivas en vez de filas, escala de la penalizacion tres veces y media
  mayor, y recorte de probabilidad. Los dos ultimos se detectaron comparando la **norma de los
  pesos** con la biblioteca, 0,06 frente a 23,3, una comparacion que la metrica sola no
  detecta.
  **D-062. 0 submissions.**
- v8 P5, P6 y P7: las tres lineas de capa de decision | tres scripts
  Ejecutadas por D-063, con autorizacion expresa de gastar envios. **P5 es la primera linea de
  la fase que mueve la decision en la direccion correcta**: elegir la constante por etiqueta con
  validacion anidada da +0,000306 frente a -0,002043 del control, o sea **+0,002349 atribuible a
  la idea**, y una tasa de acierto de **0,5185, la primera por encima del azar**. Submission
  v09, 123 celdas de 3462.
  **P6 vuelve al patron**: -0,005298 por etiqueta, -0,006857 el control, tasa 0,3919. El dato
  duro es el tercer brazo: **sin balanceo de clase la metrica es exactamente 0,000000**, porque
  el modelo predice todo negativo. Todo el proyecto depende de esa sola decision. Submission
  v10, 139 celdas.
  **P7 confirma el patron por cuarta vez**: el corrector sobre el margen da -0,002063 con tasa
  0,3611, y usar solo el desacuerdo es desastroso. Submission v11, 192 celdas.
  **Error de fuga detectado antes de cifras**: con una sola particion, cuatro de los cinco
  modelos habrian entrenado con cada fila, de modo que el desacuerdo estaba contaminado por
  resubstitucion. Se resolvio con cinco repeticiones de la particion con semillas distintas,
  que ademas mide la validacion cruzada repetida del material de clase.
  **EL HALLAZGO DE LA FASE**: el desacuerdo medio entre las cinco particiones es de 0,9175, 0,9671
  y 0,9256, y **solo 2 de 1483 filas** tienen las tres etiquetas por debajo de 0,4. **La
  estimacion es extremadamente inestable entre particiones**, y por eso casi cualquier
  diferencia entre dos modelos es ruido y no senal.
  **Y de ahi sale la mejor linea de la fase**: promediar las cinco repeticiones da
  **+0,005066 con 4 de 5 pliegues** y tasa de acierto **0,5495, que falla el filtro por tres
  milesimas**. Submission v12, 127 celdas. **El riesgo a ponderar antes de enviar: P1
  promediaba cinco modelos de la misma particion y perdio 0,00836 en el externo.**
  **D-064. 0 submissions automaticas, 4 disponibles para envio manual.**
- v8 resultados de las dos sondas y su analisis | registro y dos scripts de analisis
  **P5 dio 0,77781 y el ensemble dio 0,77122, frente a los 0,78306 de V03.** En aciertos netos
  son -6,6 y -14,9, dentro de una banda de ruido de 19,2: **no hay evidencia de que sean peores,
  hay ausencia de efecto.** El ensemble era la mejor linea local de la fase, con +0,005066 y 4
  de 5 pliegues, y su intervalo de confianza incluia cero, que era la senal de aviso ignorada.
  **A2 de la curva de transferencia (V8-09)**: ordenando las catorce sondas comparables por su
  tasa de positivos predicha, la relacion con el leaderboard es una **curva con maximo**:
  cuartil bajo 0,77355, medio-bajo 0,77715, medio-alto **0,77802**, alto 0,77484. **El maximo
  esta en la tasa de V03.** La correlacion lineal de +0,469 es espuria. Eso cierra de raiz la
  familia de calibraciones, umbrales y pesos por etiqueta, y responde a la pregunta que llevaba
  P8: no hay margen en la dimension de la tasa de positivos.
  **A3 de la correccion (V8-10)**: se construye el ensemble de cinco modelos, uno por repeticion,
  que es lo que el OOF promedia, y **da una submission identica a la de veinticinco, cero celdas
  de diferencia**. La hipotesis del error de diseno **queda refutada por medicion directa**: el
  desacople existia en el procedimiento y era inocuo en su consecuencia, y la perdida de
  0,01184 no queda explicada por el efecto de escala. La contraccion real es de +0,000591.
  **Recomendacion: no enviar la v13**, porque predice una tasa de positivos por debajo de V03 y
  la curva dice que ese lado pierde. Quedan 15 sondas, 13 de 14 comparables dentro del ruido, y
  V03 como maximo.
  **D-065 y D-066. 3 sondas en la fase v8, 6 submissions escritas, 0 automaticas.**
- v8 seis tecnicas del material de clase | tres scripts
  **Y un hallazgo que explica la fase entera. El brazo de control que es V03 sin cambiar nada,
  medido con el protocolo de cinco repeticiones, da +0,004309 con 0 de 5 repeticiones que
  mejoran**, y V8-13 lo confirma con +0,002824 con 1 de 3. Es decir: **la diferencia de +0,004 a
  +0,005 que se atribuia a la tecnica de varias lineas de la fase es del promediado, no de la
  tecnica.** Eso explica la sonda del ensemble, que gano +0,005066 local y perdio 0,01184.
  **Criterio nuevo y mas informativo que el intervalo de confianza: cuantas repeticiones de la
  particion mejoran.** La rama de complejidad dio +0,003493, que supera el gate, y **solo mejoro
  en 1 de 5**. Eso es ruido de particion.
  Resultados por linea: asimetria y kurtosis +0,003493 con 1 de 5; idiosincrasias internas
  +0,000378 con 1 de 5 y tasa 0,4484; **seleccion por correlacion -0,183116, demoledora**, porque
  reduce 20.870 rasgos a 251 y deja la metrica en 0,594; seleccion por grupo al 25 % +0,005324 con
  2 de 5, al 10 % -0,008538; discriminante lineal -0,030054; arbol -0,073168; bosque -0,063937.
  **Las tres familias de clasificador alternativas, estructuralmente distintas de la regresion
  logistica, pierden las tres.** El problema no es el clasificador sino la tarea con esta
  representacion.
  **Ninguna de las seis supera al control en numero de repeticiones, y por tanto ninguna se
  envia.** No es decision de presupuesto: es que la evidencia dice que la mejora no existe mas alla
  del promediado, y el promediado ya esta medido en el externo con -0,01184.
  **D-067. 3 submissions mas escritas y verificadas, 0 recomendadas. 10 submissions escritas
  en la fase v8, 0 automaticas.**

## Fase de correcciones de la auditoria forense (2026-09-26)

Aplicadas en este arbol las correcciones B-01..B-14 de la auditoria (rama `proyecto-kaggle-toxic-2026-audit`):

- B-01: bloque EBD-01..06 reconstruido en `experiments.csv` (202 filas, 202 claves unicas, 23 campos por fila). Contenido derivado de los artefactos v5 y marcado como reconstruccion en `notes`.
- B-02: EBD-11 y EBD-12 no existen como experimentos (huecos de numeracion historicos); se documenta aqui y no se renumeran ids.
- B-03: rutas `experiments/experiments/` corregidas en `reports/v6_reproducibility.md`.
- B-04: conteo del holdout actualizado a 5/5 en `reports/target_encoding_audit_v2.md`.
- B-05: conclusiones finales completadas en DECISIONS.md (D-001, D-003, D-004, D-005, D-007, D-008, D-010, D-017).
- B-06/B-07/B-10: notas de auditoria en los scripts (sin cambiar la logica historica).
- B-08: retirada la copia anidada `features/features/` (higiene).
- B-12: columna `definicion_diferencia` en `submissions/leaderboard_log.csv`.
- B-14: notebook corregido y re-ejecutado. Cambios: recuentos actualizados (15 sondas, 207 filas), 3 sondas de v8 anadidas a las tablas, filtro explicito de la familia union, nueva seccion 19 (fase v9 y cierre post-auditoria) con verificacion ejecutable, y artefactos requeridos restaurados en `features/` y `models/` (faltaban en el worktree v8). Ejecucion completa: 65 celdas (33 de codigo), 0 errores, 4.6 s; registro en `reports/notebook_ejecucion_v9.json`.


## Fase v10: calibracion, distribucion y gobierno (2026-09-26)

Worktree `/home/carlos/Git/PLN-v10`, rama `proyecto-kaggle-toxic-2026-v10` desde `23aac93`.
Once experimentos, V10-00 a V10-10. El registro pasa de 207 a **218 filas con 218 claves unicas**.
Sin commit, por indicacion expresa del usuario.

**Pre-registro escrito antes de ejecutar nada**, en `tasks/plan_v10.md`, con los gates de las cinco
sub-vias, la regla de adopcion de cinco condiciones y la regla de envio de una sola sonda. Y un
**addendum de protocolo** escrito despues de P0b y V10-02 pero **antes de la via A**.

- **V10-00 verificacion**: `corrected_cv` reconstruido 0.7772114525165721 con coincidencia exacta,
  `sha256` de la submission v03 identico, 207 filas, 15 sondas. Todo correcto.
- **V10-01 holdout2** (D-069): 247 filas en 235 grupos, dev2 con 1236 en 1180, cero grupos
  compartidos, particion reproducible, cero solape con el holdout v2. **Gate NO superado**:
  la referencia de dev2 quedo en 0.763436, desviacion **-0.013776** frente a la tolerancia de 0.005.
  Se creo **antes** de la via A para que el juez existiera cuando se buscaran candidatas, y se
  escribio antes de ver resultados que **no es virginal**: sus filas salen del desarrollo actual,
  que ya participo en las decisiones de v1 a v9. Su valor es detectar sobreajuste al gate, no
  probar generalizacion.
- **V10-02 ruido de particion** (D-070): la causa de la desviacion de V10-01 queda identificada.
  Control de equivalencia con diferencia **exactamente 0.0**, asi que no era un fallo de codigo.
  Banda de ruido por particion sobre el desarrollo completo: recorrido **0.017184**, desviacion
  tipica **0.005302**. Banda de subconjuntos de 1236 filas: recorrido **0.020248**, desviacion
  tipica **0.007134**, dentro de la cual cae el 0.763436. **Es ruido de tamano y de particion.**
  Y aparece el hallazgo de gobierno: la metrica corregida tiene una banda de mas de 0.017 entre
  particiones equivalentes, lo que confirma por una via independiente el ruido de 0.01524 de V8-02.
- **Addendum de protocolo**: el gate de +0.002 en un solo calculo es **inoperante** con esa banda, y
  es el mismo mecanismo por el que D-067 concluyo que la mejora local era el promediado. Toda
  candidata pasa a medirse con delta medio sobre 5 particiones, intervalo pareado y **recuento de
  particiones que mejoran**, exigiendose las tres a la vez. Los gates por sub-linea no se relajaron.
- **V10-03 arnes** (D-071a): validado en los dos sentidos antes de usarse para juzgar. El control
  nulo da delta exactamente 0.0 y 0 de 5 particiones; el control positivo, una candidata oraculo,
  da **+0.011781 y 5 de 5** con intervalo que no incluye el cero. Reproduce las cifras de V10-02.
- **V10-04 calibracion fuerte** (D-071b): **NO ACCIONABLE**. La isotonica recortada reduce el ECE
  medio un **29.94 por ciento** frente a Platt, muy por encima del 20 exigido, asi que la
  calibracion funciona. Pero **no se traslada a la metrica**: mejor delta medio **+0.000148** con
  **3 de 5** particiones y los tres intervalos pareados incluyendo el cero. Se midio, en vez de
  suponerlo, que la calibracion beta sale peor porque su rejilla **no contiene a Platt**.
- **V10-05 regla F1 optima** (D-071c): **CERRADA**, y la hipotesis era incorrecta. El mecanismo si
  funciono como se derivo, con el shift bajando los umbrales entre 0.0029 y 0.0073, pero los
  umbrales de V03, 0.44, 0.51 y 0.46, son ya **mas altos** que el optimo plug-in natural, 0.3841,
  0.4093 y 0.3789, de modo que la regla se aleja de V03. Delta **-0.003921** con prior simulado,
  1 de 5 particiones, y el gate B fallido porque el shift empeora en vez de ayudar.
- **V10-06 umbrales estables** (D-071d): **CERRADA**. Ninguna parametrizacion mas restringida mantiene
  la ganancia de tres umbrales libres con menor dispersion. La holgura de la capa de decision es
  **irreducible con estos datos**.
- **V10-07 shift de features** (D-073): **VIA CERRADA**. El discriminador da un AUC global fuera de
  muestra de **0.487717**, por debajo de 0.5, con los cinco pliegues entre 0.462749 y 0.525974 y
  exactitud 0.512305. No hay shift de covariables. Diagnostico unicamente, sin entrenar con la
  prueba. 40 caracteristicas mas desplazadas en `reports/v10_07_shift_features.csv`.
- **V10-08 integridad y filtro** (D-074): artefactos de holdout2 con `sha256` intactos, 247 filas
  localizadas, cero solape, particion limpia. **0 de 3 evaluaciones de holdout2 gastadas.** Con el
  filtro de V8-02, una candidata de 124 celdas necesitaria acertar mas del **0.6548** de ellas para
  que el externo la separara del azar, y una de 134 celdas mas del **0.6433**.
- **V10-09 correlacion** (D-075): tres bloques. **Estructura**: las 20441 caracteristicas estan
  **poco correlacionadas**, con correlacion maxima por caracteristica de mediana **0.1704** y solo
  1203 pares por encima de 0.90. **Etiqueta**: AUC mediano de 0.5133, 0.5198 y 0.5147, con solo
  entre el 0.29 y el 0.80 por ciento de caracteristicas fuera de [0.45, 0.55], y mejores de 0.6717,
  0.7569 y 0.6844; el modelo combina **miles de rasgos debiles**. **Eliminacion**: los tres
  umbrales empeoran con **0 de 5** particiones, -0.030626, -0.027661 y -0.027129, y el dano es
  desproporcionado: el 1.38 por ciento de las caracteristicas cuesta 0.0306 de F1.
- **V10-10 regla de conservacion** (D-076): **CERRADA con el control que faltaba en la v8**. La
  sospecha era que la regla de V10-09, quedarse con la mas informativa en solitario, descartaba la
  pieza que desambiguaba. **El control la refuta**: fuerte -0.031252, debil -0.031879, menos
  redundante -0.033885 y **aleatoria -0.028887**, la mejor de las cuatro, con una brecha de 0.005,
  menor que la banda de ruido. **La regla es irrelevante: lo que cuesta 0.03 es cuantas
  caracteristicas se quitan, no cuales.**

**Correccion registrada, D-072**: la cifra de **+0.0036 no es el efecto de optimizar umbrales**. La
fase v6 la atribuyo ahi, y la repetian V10-04 y V10-06. El informe de la v7 ya lo habia
demostrado: el control de tres niveles con umbrales congelados da **-0.0081** solo por el coste del
protocolo, y el efecto aislado del umbral es **-0.0019**. La medicion propia de V10-06 da
**+0.000131**. Las tres cifras miden cosas distintas y las tres coinciden en que el efecto del
umbral es pequeno; lo que no cabe es citar +0.0036 como si lo fuera.

**Balance**: 0 submissions escritas, 0 enviadas, 0 recomendadas. **V03 se mantiene** con 0.777211
local y 0.78306 en el externo. Ninguna candidata cumple mas de dos de las cinco condiciones de
adopcion, y las dos que fallan son las determinantes. El objetivo de superar 0.80 sigue a 1.1
veces el ruido del leaderboard, y la mayor ganancia anidada de toda la fase es de +0.000148.

- **V10-11 P4, BiGRU con afinamiento** (D-077): **H1 CERRADA con dos variantes**. El congelado da
  0.693994, que reproduce el 0.6943 de V9-04 como control interno. El afinado da **0.711639**, con
  **+0.017645**, que es la mejora mas grande entre dos variantes de la misma arquitectura que ha
  producido el proyecto. Pero el resultado que importa es el **control**, una tercera variante con
  una sola tasa de aprendizaje: da 0.699086, solo **+0.005092**. De los +0.017645, **+0.012553 los
  aporta la tasa de aprendizaje diferencial** y no el hecho de mover el embebido. Las mejores
  epocas confirman el mecanismo: la tasa unica para siempre entre 7 y 8, muy por debajo del tope de
  14, porque destruye la representacion preentrenada. Sigue a **-0.065572** de V03. Decision de
  sobreentrenamiento: **early stopping estricto, fijada por el pre-registro antes de ejecutar**, con
  la salvedad anotada de que la mejor epoca se elige por la perdida de validacion. Sin sonda.

**Notebook**: seccion 20 anadida con la sintesis de la fase y una celda de verificacion con
aserciones sobre las 219 filas, V10-00 a V10-11, la huella de V03, la de holdout2, la banda de
particion, los dos controles del arnes, el AUC del shift, la brecha entre reglas de conservacion y
las tres variantes del BiGRU. **Re-ejecutado completo: 67 celdas, 34 de codigo, 0 errores, 4.1 s.**
Habia tres recuentos de 207 que hubo que actualizar a 219 en las celdas 38 y 64, porque si no el
propio helper `verificar` del notebook falla y el entregable no seria ejecutable. Y la primera
version de la celda pedia un JSON del arnes que no existe, porque V10-03 es una libreria y sus
controles viven en la fila del registro; se corrigio antes de ejecutar. Registro en
`reports/notebook_ejecucion_v10.json`.

**Sin push, sin merge, sin rebase, sin commit.**

## Fase v11: shift de prevalencia, calibracion y regla de decision (2026-09-26)

Worktree `/home/carlos/Git/PLN-v11`, rama `proyecto-kaggle-toxic-2026-v11`, desde `a30a467` que es
el commit de la fase v10. v10 no se modifica. Nueve experimentos, V11-00 a V11-08, y el registro
llega a **227 filas con 227 claves unicas y 23 campos, sin filas malformadas**.

**Ninguna submission generada, ninguna enviada, ninguna recomendada. V03 se mantiene** con
0.7772114525165721 local y 0.78306 en el externo. **holdout2 intacto, con 0 de 3 evaluaciones
gastadas.**

### El hallazgo que reencuadra la fase

- **V11-00, P0 reconciliacion**: 25 de 25 superadas, pero la primera ejecucion dio **21 de 25** y
  las tres paradas son el resultado real. Dos eran errores de mi propia comprobacion, que buscaba
  la misma clave en el informe y en el registro. La tercera era un problema de verdad.
- **D-079**: el campo `factor_vs_train` de V9-02 **no** es la prevalencia de test dividida por la
  del train completo, sino por la del **desarrollo**: el codigo define
  `prevalencia_train = y_dev.mean(axis=0)`. Con el train completo el factor queda 0.0002 mas alto
  en HS y TR. **V9-02 no tiene defecto; el nombre del campo no significa lo que dice.**
- **V11-01, P1 reproduccion**: **EXACTA, nivel R3**, con diferencias de -4.51e-07 a +3.29e-07 que
  son el redondeo del JSON, e iteraciones del EM identicas: 26, 16 y 24. **Su resultado queda en
  cuarentena**: el uso de la prueba no alimenta ningun candidato.

### El limite que decidio el alcance

- **V11-02, P1.1 auditoria de legalidad**: clasifica once pasos contra el texto verbatim de
  `docs/competition_rules.md`, con las cuatro citas verificadas una a una. Reparto **5 permitidos,
  1 ambiguo, 5 prohibidos**. El ambiguo es estimar la prevalencia con EM sobre probabilidades no
  etiquetadas, y lo es porque la regla prohibe entrenamiento y aqui no hay entrenamiento, pero la
  intencion de la regla es que el test no participe del desarrollo. **Lo decisivo es que las reglas
  no mencionan la transduccion ni para permitirla ni para prohibirla.**
- Consecuencia: **el uso no se emplea para generar submission**, **v17 transductiva queda en
  cuarentena**, y **P10 no tiene objeto**. La pregunta de transferencia **no tiene respuesta
  verificable dentro de las reglas**. No es falta de calculo: es una limitacion de las reglas.

### La validacion metodologica, que si es nueva

- **V11-06**: los estimadores de prevalencia **funcionan** bajo shift simulado. EM de Saerens con
  error relativo medio **0.0069**; clasificar-y-contar ajustado con error **0.0000**, y ese cero se
  explica antes de aceptarlo: bajo shift **puro** la sensibilidad y la especificidad son
  invariantes, luego el estimador esta **exactamente identificado**. Es un teorema, no una
  propiedad favorable. **Se declara el punto ciego**: el resultado vale dentro del supuesto de shift
  puro y nada mas.
- **V11-07, el mecanismo central, CERRADO**: la correccion de prior **no mejora** la metrica. En el
  factor verdadero, HS da **-0.001032** con **1 de 5** particiones, TR **+0.000000** con **0 de 5**,
  AG **+0.000154** con **1 de 5**; macro medio **-0.000293**, y los tres intervalos incluyen el
  cero. Gates A y B no superados.
- **La direccion del efecto explica el fallo**: la correccion empeora cuando la prevalencia baja y
  ayuda cuando sube, a 1.05 da +0.001095 en TR y +0.001194 en AG. El shift medido es hacia **abajo**,
  luego el mecanismo actua en la direccion contraria. Y la causa de fondo la midio V10-05: los
  umbrales de V03, 0.44, 0.51 y 0.46, estan ya **por encima** de los optimos de F1, 0.3841, 0.4093
  y 0.3789. La capa de decision **ya es conservadora** y la correccion **corrige dos veces**.

### Dos hallazgos metodologicos que valen mas que los resultados

- **La seccion 17 es mas estricta que la herramienta del proyecto.** `verificar_caracteres.py`
  busca alfabetos no latinos; la seccion 17 pide **ASCII puro**.
- **Un escaner de cumplimiento por palabras es inservible.** Daba doce falsos positivos, porque
  encontraba las citas de la prohibicion y su propia lista de patrones. Sustituido por analisis
  estructural de imports.

**Cierre**: 10 de las 12 preguntas de la revision respondidas y 2 con UNKNOWN, que no se rellenan
con una estimacion. Las tres lineas que no dependen del shift, P4, P6 y P8, ya estaban medidas y
cerraron negativas en v10. **La v11 no deja ninguna linea abierta con expectativa razonable.**

**Sin push, sin merge adicional, sin reset, sin rebase, sin modificar otras ramas.**

### V11-09: cierre de entregables (2026-09-26)

**Once entregables de once presentes.** Cuatro se completaron documentando su vacio, y lo
relevante es **lo que no se hizo**: no se copiaron cifras de v10 a tablas de v11, porque habria
sido trazabilidad falsa. Ver D-084 para el detalle entregable por entregable.

`v11_gap_audit.md` es un **puntero** a `v11_00_reconciliacion.md`, no una copia: la auditoria de
brechas si se ejecuto, fue V11-00 con 25 de 25 superadas, y dos copias de la misma observacion
divergen.

Correccion posterior a la ejecucion, documentada: retirado un `import` huerfano de
`v11_01_reproduccion_v9_02.py` que yo mismo introduje al documentar una variable muerta como
comentario. Inerte por construccion, nunca invocado, y la reproduccion re-ejecutada despues del
cambio devuelve **R3 exacto**. Ninguna cifra de ningun informe se modifico.

**Sin push, sin merge adicional, sin reset, sin rebase, sin modificar otras ramas.**

### Correccion de caracteres en el bloque propio de la v11 (2026-09-26)

El segundo escaneo ASCII, el estricto de ASCII puro, distinguisho entre lo heredado y lo propio
en lugar de contar global: **147 caracteres no ASCII antes de D-078** son la desviacion ya
documentada de las fases previas, que **no se toca por decision**, porque modificar esos ficheros rompe la correspondencia entre codigo y resultado historico. **7 em-dashes U+2014 despues de
D-078 si eran mios**, en los titulos de D-078 a D-084, y estan corregidos a guion.

**RUNLOG.md y PROJECT_STATE.mdtienen **0 caracteres no ASCII propios.** La distincion importa
porque un recuento global habria hecho pasar por reformateo masivo lo que era un fallo de siete
guiones, y habria hecho pasar por limpio un bloque mio que no lo estaba.

## Fase v12: frontera, funcion de objetivo y estructura de etiquetas (2026-09-26)

Worktree `/home/carlos/Git/PLN-v12`, rama `proyecto-kaggle-toxic-2026-v12`, desde `039eae4` que es
el estado validado de v11. v11 no se modifica. **Seis experimentos, V12-00 a V12-05**, y el registro
llega a **234 filas con 234 claves unicas y 23 campos, sin filas malformadas**.

**Ninguna submission generada, ninguna enviada, ninguna recomendada. V03 se mantiene** con
0.7772114525165721 local y 0.78306 en el externo. **holdout2 intacto, con 0 de 3 evaluaciones
gastadas.** Segunda particion no ejecutada, porque no hay candidata.

### La asimetria que reencuadro la fase

V12-00, auditoria de brecha, con **sonda estructural sobre el arbol sintactico de 127 modulos** y no
busqueda textual. 24 familias: **16 CLOSED_VALID, 5 PARTIALLY_TESTED, 3 GENUINELY_UNTESTED**.

**17 de las 24 familias cerradas estan en el eje de la representacion y del clasificador, y los tres
huecos que quedan estan todos en el objetivo de entrenamiento.** No se habia tocado nunca la funcion
que se minimiza. Esa asimetria es lo que justificaba abrir la v12, y se sostiene a la vista de los
tres negativos.

La sonda **se autovalido con un positivo conocido**: el unico fichero del proyecto donde aparece la
palabra `contrastive` es el propio script de la auditoria, que la nombra al describir la familia
ausente. Un escaner textual concluiria que el mecanismo **si** se ha probado, con la maxima
confianza, por coincidencia literal. El indice estructural devuelve cero.

### Las tres hipotesis del objetivo de entrenamiento

| ID | mecanismo | mejor delta | pliegues | deteccion | decision |
| --- | --- | --- | --- | --- | --- |
| H1 | regularizacion por etiqueta y grupo | +0.002738 | 3/5 | 0.5196 | CLOSED |
| H2a | perdida propia, entropia cruzada generalizada | +0.001616 | 4/5 | 0.6667 | MARGINAL |
| H2b | restringir el entrenamiento al dominio de la etiqueta | -0.116581 | 0/5 | 0.1420 | CLOSED |
| H3 | objetivo por pares sobre TF-IDF disperso | -0.002469 | 2/5 | 0.4915 | CLOSED |

**H1**: el maximo de la curva del cociente entre bloques cae **sobre el valor neutro** con +0.000006.
No hay nada que extraer: la estructura de penalizacion de V03 ya esta en su optimo. El delta de
+0.002738 lo sostenia un solo pliegue de cinco, y sin el el delta cae a +0.000134.

**H2a**: el techo con **HS verdadero** es de **+0.046271**, con deteccion 0.9286. La premisa se midio
por primera vez: el **80,44 por ciento** de los negativos que ve TR en entrenamiento **no estan
definidos**, porque TR solo existe cuando HS=1. El GCE da +0.001616, una **meseta y no un pico**, y
captura el 3,5 por ciento del techo.

**H2b**: negativo limpio. Descartar el 56 por ciento de los datos desplaza la escala de salida y los
umbrales congelados dejan de tener significado para la subpoblacion con la que se ha entrenado.

**H3**: el mecanismo **mejora la F1 de TR donde la etiqueta esta definida**, de 0.8941 a 0.9091, y
**empeora los falsos positivos en la zona no definida**, de 78 a 97. Hace bien lo que fue disenado
para hacer en local y mal lo que la fase necesita. La causa: un objetivo de **orden** no restringe
el **nivel absoluto**, y los umbrales fijos si necesitan el nivel.

### La auditoria de V6-B, y por que la hipotesis candidata no se abre

V12-04, leyendo el codigo. El encabezado de `v6_02_label_factorization.py` dice literalmente **sin
routing duro**, y el analisis estructural confirma **5 operaciones de producto y 0 condicionales
cuya condicion menciona HS**. B1 es `p(TR|X) = p(HS|X) * p(TR|HS=1,X)`, que es exactamente la
formulacion candidata, y V6-02 tambien restringio el dominio del condicional a HS=1, que es el
brazo M de H2. **Los cuatro resultados son negativos y el cierre tiene el mecanismo identificado.**

La reconciliacion que cierra el diagnostico: **20 filas** donde V03 predice TR=1 con HS **predicho**
igual a 0, que es lo que la formulacion suave alcanza, frente a **78 filas** donde V03 predice TR=1 y
en realidad son HS=0, que es el techo. **Las 58 de diferencia son las que `p(HS)` no puede
identificar**, porque HS se predice mal en 143 filas.

### La conclusion, con la correccion conceptual que hay que conservar

**El techo condicionado por HS verdadera demuestra que hay senal estructural abundante. El cuello de
botella no es la falta de capacidad predictiva para TR y AG, cuya F1 donde la etiqueta esta definida
es de 0.8941: es la inferencia de la condicion.**

**Y H2 y H3 NO demuestran que la funcion de perdida no sea el problema en sentido absoluto.** Un
unico mecanismo positivo no autoriza a descartar la familia. Lo que si se dice es mas preciso: **las
variaciones de objetivo evaluadas tienen un impacto pequeno comparado con el techo que introduce el
error de la condicion de HS**, que es del 3,5 por ciento en el mejor caso.

### Verificacion

Cuatro fallos propios en H3 los detectaron las comprobaciones internas **antes de mirar un
resultado**: la penalizacion L2 ausente del valor del objetivo, el recorte del margen rompiendo la
regla de la cadena, el vertice de la bisagra, y la busqueda lineal demasiado corta. Ademas un **fallo
de diseno**: el termino por pares era una media y la entropia cruzada una suma, luego lambda era
irreconciliable con una razon de 1 a 3500, y el barrido daba el mismo resultado de 0 a 0,5.
**El gradiente verifica con residuo maximo de 3,6e-07** frente a 1e-06, en tres valores de lambda y
dos puntos.

**Sin push, sin merge adicional, sin reset, sin rebase, sin modificar otras ramas.**

## Fase v13: estimacion de HS e inferencia estructurada de etiquetas (2026-09-26)

Worktree `/home/carlos/Git/PLN-v13`, rama `proyecto-kaggle-toxic-2026-v13`, desde `ce774bc` que es
el estado validado de v12. v12 y v11 no se modifican. **Dos experimentos, V13-00 y V13-01**, mas el
informe final V13-02, y el registro llega a **237 filas con 237 claves unicas y 23 campos, sin filas
malformadas**.

**Ninguna submission generada, ninguna enviada, ninguna recomendada. V03 se mantiene** con
0.7772114525165721 local y 0.78306 en el externo. **holdout2 intacto, con 0 de 3 evaluaciones
gastadas.** Segunda particion no ejecutada, porque no hay candidata. Test no utilizado.

### La auditoria va primera y salva la fase antes de gastar nada

El enunciado exigia ejecutar la auditoria historica antes de gastar presupuesto, y el resultado es
que **tres de los cuatro mecanismos propuestos no eran nuevos**. P1 es V4-E literal, cerrado como
NO_SIGNAL con 0.771289. **P2 ya se habia ejecutado y adoptado**: E2-02 dio 0.7853 y E2-18 dio 0.7874,
ambos por encima del gate de candidato, y ninguno se envio porque el leaderboard fue 0.78163 frente
a 0.78306 de V03. P3 es el mismo meta y V8-08. P4 tiene su version dura en H2 y su version de ranking
en H3. La auditoria inventaria 11 mecanismos: 8 ya cerrados, 3 genuinamente nuevos.

**Un error mio, corregido y documentado**: mi conteo de invariantes sumo los estados 011 y 001 como
si ambos violaran TR=1 con HS=0, y eso contaba las violaciones de AG como si fueran de TR. El
conteo correcto es **0 para TR y 22 para AG**, verificado con un guion independiente antes de tocar
el codigo. La asimetria que revelo es la que hace util la medicion: **la invariante de TR es dura y
la de AG no lo es**, y por eso P4 no puede imponerse a mano.

### La medicion nueva, y lo que decide la fase

El meta historico siempreuvo el score propio de HS entre sus features, porque las 12 probabilidades
OOF son 4 bases por 3 etiquetas. Eso hacia **inobservable** cuanto aportan TR y AG. Con el score
propio excluido, el meta alcanza **0.8433 de AUC para predecir HS**, frente a 0.8677 de V03 con el
suyo, luego el score propio solo aporta **+0.0244**.

**TR y AG SI llevan informacion sobre los errores de HS. Y no es utilizable**: ese mismo brazo
**pierde 0.007541** en el macro-F1, con la F1 de HS cayendo 0.022624, porque el umbral esta
congelado y un AUC alto no es una buena decision en el punto de corte. El brazo con el score propio
rescata 20 filas y dana 1, y solo mueve +0.000534 en el macro, por debajo del gate exploratorio.

### El reencuadre: el techo no esta en HS

**El techo de +0.046271 no proviene de mejorar la estimacion de HS.** El oraculo que lo produce no
cambia **ni una sola celda** de la columna HS: pone TR y AG a cero en las 835 filas donde el HS
verdadero es cero, borrando 196 falsos positivos, 78 de TR y 118 de AG. Es un efecto **aguas abajo**
de la condicion de HS, y ningun metodo que mejore la columna HS puede alcanzarlo.

El corolario medido no admite matiz: **corregir la etiqueta de HS deja TR y AG con delta exactamente
cero**. Y los 143 falsos negativos de HS, donde 107 filas tienen TR o AG verdadera a uno, **no los
toca el oraculo en ninguna celda**.

### Dos correcciones mas, antes de mirar resultados

`ajustar_base` abortaba ante cualquier warning, y con veinte ajustes por pliegue eso convertia un
aviso benigno en un fallo del experimento. Y el veredicto del informe decidia con un umbral de AUC de
0.52 **que me habia inventado**, en lugar del gate real del proyecto. Ademas mi primera verificacion
de donde viene el techo comprobaba la columna 0 del oraculo, que nunca modifica.

### Estado

**2 experimentos de 8 principales y 0 de 4 de reserva. 0 adoptables, 1 marginal, 1 rechazada.
0 submissions, 0 de 3 evaluaciones de holdout2, segunda particion no ejecutada, test no utilizado,
V03 congelada e intacta con identidad verificada en cada ejecucion.**

**La linea de inferencia estructurada de etiquetas queda CERRADA FORMALMENTE.** No se afirma que
TOXIC no pueda mejorar. Se afirma que la inferencia estructurada condicionada por HS no es una via
rentable con los mecanismos permitidos, porque el techo no esta en la estimacion de HS sino en la
aplicacion de la condicion de HS a las decisiones de TR y AG, via que ya estaba cerrada por
medicion desde v12.

**Sin push, sin merge adicional, sin reset, sin rebase, sin modificar otras ramas.**

## Fase v14: auditoria de entrenamiento partially-label para TR y AG (2026-09-26)

Worktree `/home/carlos/Git/PLN-v14`, rama `proyecto-kaggle-toxic-2026-v14`, desde `b07e410` que es
el estado validado de v13. v13, v12 y v11 no se modifican. **Un solo experimento, V14-00**, mas el
informe final V14-01, y el registro llega a **239 filas con 239 claves unicas y 23 campos, sin filas
malformadas**.

**Ninguna submission generada, ninguna enviada, ninguna recomendada. V03 se mantiene** con
0.7772114525165721 local y 0.78306 en el externo. **holdout2 intacto, con 0 de 3 evaluaciones
gastadas.** Segunda particion no ejecutada. Test no utilizado.

### La auditoria cerro la fase antes de gastar nada

El enunciado advertia que la perdida enmascarada de P1 **no debe confundirse con H2-M**, y por eso
proponia P1 para no repetirlo. La auditoria demuestra lo contrario, y con la mayor evidencia posible:

| comprobacion | valor |
| --- | --- |
| peor diferencia de coeficientes | **0.000e+00** |
| peor diferencia de probabilidades | **0.000e+00** |
| delta de la enmascarada | **-0.116581** |
| delta de H2-M registrado en v12 | -0.116581 |

**Una fila con peso cero en la perdida no aporta gradiente.** Luego el coeficiente lo determinan solo
las filas con peso uno, que son exactamente las que H2-M conserva. Enmascarar y eliminar dejan el
mismo subconjunto de filas aportando senal. Lo unico que puede distinguirlas es como se reescala el
termino de perdida, que es una constante de escala de la familia de calibracion y umbrales, ya
cerrada. Y la formulacion **reproduce un resultado negativo ya registrado, hasta el ultimo decimal**.

**Se cumple la condicion de parada numero 1. La fase se cierra con 0 de 4 principales y 0 de 2 de
reserva.**

### Mi propia verificacion fallo primero, y la segunda no

La verificacion inicial uso **una sola pliegue** y concluyo que la identidad era exacta con
0.000e+00. Al extenderla a los cinco pliegues aparecieron diferencias de 4,5e-01. **La causa era
mia**: en la reconstruccion de H2-M pasaba `sample_weight` nulo, es decir sin pesos de clase, cuando
el brazo M real usa `class_weight="balanced"` sobre la submuestra de HS=1. Estaba comparando H2-M
contra un ajuste sin ponderar, y mi H2-M reconstruido daba -0.132999 en lugar de -0.116581.

**Lo detecto** comprobar que el delta reconstruido coincidiera con el registrado en v12. Sin esa
comprobacion la conclusion erronea habria pasado, porque un numero pequeno no delata un modelo mal
montado.

### Los cuatro mecanismos "nuevos" tienen cero menciones historicas

La sonda textual **se marca a si misma**, porque el guion de auditoria nombra los doce mecanismos que
busca. Con el conteo desglosado:

| mecanismo | guiones historicos | conteo bruto |
| --- | --- | --- |
| aprendizaje con etiquetas parciales | **0** | 1 |
| positive-unlabeled learning | **0** | 1 |
| selective loss | **0** | 1 |
| masked BCE | **0** | 1 |

Es el patron que v12 documento, aqui con su cuantificacion. Y la sonda estructural da el lado
positivo: **13 llamadas con `sample_weight`**, ninguna definida por la indefinicion de la etiqueta.
La unica operacion del historial que condiciona la senal de TR y AG a HS=1 es H2-M.

### La conclusion comoda, evitada

Seria comodo concluir que las etiquetas de TR y AG en HS=0 deben seguir siendo negativos ordinarios
porque la alternativa fallo. **No puede afirmarse**: la alternativa no ha fallado, ha resultado ser el
mismo estimador que la formulacion que si fallo. No hay evidencia ni a favor ni en contra de la
semantica de observacion parcialmente definida, y las preguntas 1, 2 y 5 del enunciado se declaran sin
respuesta por evidencia.

Lo que si queda establecido es una restriccion de lo construible: **cualquier mecanismo que se
implemente enmascarando la perdida de clasificacion sobre las filas HS=0 va a reproducir H2-M**.

### Estado

**1 experimento, que es la auditoria. 0 ejecuciones. 0 adoptables, 0 marginales, 1 rechazada.
Fraccion del techo del oraculo recuperada -2,52 de signo negativo. 0 submissions, 0 de 3
evaluaciones de holdout2, segunda particion no ejecutada, test no utilizado, V03 congelada e
intacta.**

**Sin push, sin merge adicional, sin reset, sin rebase, sin modificar otras ramas.**

## Fase v15: cierre de la frontera final, inventario de senal (2026-09-26)

Worktree `/home/carlos/Git/PLN-v15`, rama `proyecto-kaggle-toxic-2026-v15`, desde `c5be6b9` que es
el estado validado de v14. v14, v13, v12 y v11 no se modifican. **Tres experimentos, V15-00 a
V15-02**, y el registro llega a **242 filas con 242 claves unicas y 23 campos, sin filas
malformadas**.

**Ninguna submission generada, ninguna enviada, ninguna recomendada. V03 se mantiene** con
0.7772114525165721 local y 0.78306 en el externo. **holdout2 intacto, con 0 de 3 evaluaciones
gastadas.** Segunda particion no ejecutada. Test no utilizado.

### La fase cambia el objetivo: de modelo a informacion

V15 deja de buscar modelos y pregunta **que informacion disponible en los datos no esta usando
V03**. El inventario es lo primero, y da tres respuestas.

**La categoria E no existe.** El fichero bruto tiene exactamente `id;text;HS;TR;AG` y el cargador no
descarta ninguna columna. No hay user ID, ni fecha, ni perfil. La unica columna no textual es `id`,
que es un contador de fila con valores 20002, 20003, 20005, y que ademas la seccion 3.E prohibe.

**V03 tiene dos propiedades demostrablemente incompletas**: no registra la posicion absoluta en el
documento, porque `TfidfVectorizer` extrae n-gramas sin la posicion, y no interacciona dimensiones,
porque word(1,2) solo ve una ventana de dos. De la primera se sigue una clase de senal no
expresable.

**De doce clases, tres no existen, cuatro ya estaban medidas y cinco estan en familias cerradas.**
Los rasgos estructurales de E2-18 costaron -0.0052 y -0.0051 en dos estimaciones independientes.

### La unica clase no cubierta, y por que se aisla

La **posicion en el documento**. Y hay que ser preciso: **no es nueva**. `pos_mencion_incial` fue
feature de modelo en `16_targeting_v2.py` linea 1204, y `mencion_inicial` aparece en 8 guiones mas.
Lo que no se ha probado es la posicion como **columna de V03**, que es donde V03 decide.

### Dos mediciones descartadas por invalidas, antes de escribir el experimento

**El AUC global de un rasgo no decide nada**: un AUC de 0.5598 es compatible con que V03 ya tenga
ese 0.5598 dentro de su prediccion.

**El R2 de predecir el rasgo desde los margenes de V03 tampoco**: tres numeros no pueden
reconstruir cuatro rasgos, luego sale bajo por construccion y parece una buena noticia que no lo es.

**La que si decide es el AUC estratificado por el margen de V03**, con null de 500 permutaciones
dentro de cada estrato y valor p empirico.

**Correcciones propias**: la primera implementacion hacia **una sola permutacion**, que es un punto
y no un null. Y solo se probo en P4 la primera de las tres candidatas que pasaron P3.

### El resultado, que tiene dos mitades

**La senal residual existe y es complementaria**: AUC de 0.5547 en los estratos de V03 con valor p
empirico por debajo de 1 entre 500. Es literalmente lo que la seccion 12 pregunta, y la respuesta es
que el rasgo **ve algo que V03 no ve**.

**Y no es utilizable**: las tres candidatas dan -0.002868, -0.001352 y -0.003037, y el dato que
decide es el **acierto CERO** entre las celdas modificadas, de 50, 14 y 57 celdas, **ninguna**
acierta, frente al umbral de detectabilidad de 0.5166. No es una mejora que no llega al gate, es
un cambio que dana sin ambiguedad.

### El dato que sostiene el cierre

El **mejor resultado local de toda la historia, 0.8465 en S08, no se adopto**: el externo fue
0.78163 frente a 0.78306 de V03. V03 sigue siendo la mejor solucion no por falta de intentos, sino
por evidencia en contra medida, y en un proyecto de 242 experimentos eso es una conclusion fuerte.

### Estado y cierre

**3 experimentos, 0 adoptables, 0 marginales, 3 rechazadas. 0 submissions, 0 de 3 evaluaciones de
holdout2, segunda particion no ejecutada, test no utilizado, V03 congelada e intacta.**

**CLOSED. V03 continua siendo la mejor solucion reproducible bajo las restricciones actuales. No
existe una fuente de senal permitida y genuinamente nueva identificada en el input disponible. No
se abre una V16 para buscar microganancias.**

**Sin push, sin merge adicional, sin reset, sin rebase, sin modificar otras ramas.**


---

## 2026-09-27 - Fase P45: auditoria de metrica y estructura conjunta

**Recuperacion forense previa (sin cambios en artefactos).** Se determino el estado real: rama
`v25`, HEAD `e1ad0bf`, `experiments.csv` con 311 filas, notebook con 181 celdas, V44 persistido
en el working tree pero no commiteado. Se determino que la cifra de 258 filas de mi memoria
conversacional no existia en ningun worktree ni commit: era una alucinacion, no un estado real.
Se determino tambien que la fase V46 citada en esa misma memoria **no existe**: 0 archivos.

**Reconciliacion documental de V44** (commit `2285423`). Fuentes de verdad segun jerarquia:
script ejecutado, JSON producido, artefactos, notebook, `experiments.csv`,
`plan/contexto.md`. Hallazgos y correcciones:

- `v44_01_vw_optimizers.py` **calcula** sus resultados; el JSON y el MD se generan de la misma
  ejecucion. La tabla del notebook (FTRL 0.597823) no la produce ningun script y sus
  rescates/danos no coinciden con el JSON. **El JSON gana.**
- La fila V44-01 de `experiments.csv` declaraba "VW 9.11.0, 10 hilos, 32 passes", config que
  el script nunca tuvo. La real es 9.11.8 con `passes=10`.
- La fila V44-02 decia "CANDIDATO UNICO / REPENDIENTE DE CORRER" cuando el informe ya existia
  con PLS en -0.040430.
- `v44_02_pls_correcto.py:290` tenia un fallback copy-pegado de V44-01 que devolvia
  `SPARSE_ONLINE_OPTIMIZATION = CLOSED`. Corregido a `PLS = CLOSED` y propagado al JSON y al
  MD **sin tocar ninguna cifra**.
- `plan/contexto.md` rotulaba como V44 resultados de V40 y V42. Rehecho; el prefijo historico
  quedo intacto.
- La seccion 37 del notebook la deberia escribir `v42_02_integrar_notebook.py`, pero su guard
  buscaba una cadena que ya estaba en la seccion 36 (escrita por `v40_02`), asi que aborto sin
  escribir. Por eso no hay seccion propia de V42 y la 37 contiene V44. No se renumero nada.
- Limitacion registrada: `v44_01` no fija los hilos de VW, asi que la corrida no es
  bit-reproducible. El JSON es el registro de la corrida efectuada. El veredicto no depende de
  ello.

**P45-A** (`experiments/v45_00_metrica_audit.py`, 10.9 s): `corrected_cv` reproducido con
diferencia `+0.000e+00`. Identidad con `f1_score(average="macro")` demostrada. Media de
pliegues frente a global reconciliada. `samples` excluido por escala. Ranking de umbrales
estable entre las tres hipotesis restantes. `METRIC_MATCH = TRUE`.

**P45-B** (`experiments/v45_01_joint_label_structure.py`, 0.4 s): distribucion de los ocho
estados, matriz 8x8, categorias de error, contraste de independencia (chi2 343.156, gl 20,
p < 1e-6) y correccion estructural dura negativa en las dos formas. Inventario de mecanismos
conjuntos leido del registro. `JOINT_LABEL_STRUCTURE = CLOSED`.

**Integracion** (`experiments/v45_02_integrar_notebook.py`): secciones 40 y 41 anadidas, 181 ->
185 celdas. Ambas celdas de codigo se ejecutaron con returncode 0. `reports/p45_results.csv`
con 49 filas.

**Errores propios durante la fase, corregidos antes de cerrar**: nombres de variable no ASCII
(CJK) en dos scripts; `average="macro_3"` pasado a sklearn; `e` indefinido en el generador de
MD; un bucle que **redefinia** `matriz` y hacia que la matriz 8x8 se guardara como un array de
1483x3; un `c` de bucle reutilizado como diccionario. Todos detectados por asercion o por
verificacion explicita del artefacto, no por el informe final.

**Sin push, sin merge, sin reset, sin rebase, sin PR. 0 submissions. V03 intacto.**


---

## 2026-09-28 - Fase P46: auditoria del campeon real

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, holdout y test sin usar. Tarea
exclusivamente documental.

**P46-00** (`experiments/v46_00_champion_audit.py`, exit code 0): 22 filas del leaderboard log
auditadas con artefacto y sha256; extraccion de las afirmaciones de CHAMPION.md por tabla;
reproduccion de la formula declarada; deteccion de la contradiccion interna; separacion de las
cuatro magnitudes. Artefactos: `reports/p46_champion_audit.{json,md,csv}`.

**P46-01** (`experiments/v46_01_integrar_notebook.py`, exit code 0): seccion 42, notebook 185 ->
187 celdas, celda de codigo ejecutada con returncode 0, 0 secciones duplicadas.

**Veredicto.** CASE B. `leaderboard_max_verified = 0.78306` (V03). `CHAMPION_STATUS =
INCONSISTENTE_AUTOCORREGIDO`. Decision registrada como D-117.

**Correcciones aplicadas a P45: splice de IDs y redaccion.**

- Las decisiones de P45 se escribieron como D-030 y D-031, IDs ya ocupados por la fase v5.
  Detectado al buscar el siguiente ID libre para P46. Renumeradas a D-115 y D-116, con nota de
  correccion. El contenido no cambio. El registro de decisiones queda con 116 encabezados y 0
  duplicados.
- `METRIC_MATCH = TRUE` era una formulacion demasiado fuerte. Desdoblado en
  `LOCAL_METRIC_REPRODUCED = TRUE` y `KAGGLE_SCORER_DIRECTLY_VERIFIED = FALSE`, porque el
  codigo del scorer del host no esta en el repositorio. Actualizado el script, el JSON, el MD,
  el CSV y la fila V45-00 del registro.

**Hallazgo no previsto: la columna `macro_f1` del registro mezcla bases de medida.** 4 filas
sobre holdout2 y 46 sobre el split D-004 supersedido. Las filas de V03 registran 0.8322 y
0.8356 en esa base. Un maximo ingenuo sobre la columna devuelve S08 con 0.8743, que es una
medida de holdout, no de corrected_cv. No se reescriben las filas historicas; se documenta el
defecto y se fija la magnitud utilizable desde `features/oof_v03_final.npz`.

**Errores propios durante la fase.** Una expresion rota en el generador de MD y dos
referencias a una clave renombrada, ambos detectados porque el script se ejecuto de verdad y se
comprobo el exit code, no solo `py_compile`. Un pipe a `tail` enmascaraba el exit code real en
la primera ejecucion; se corrigio capturando el codigo sin pipe.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P47: auditoria de disponibilidad de datos

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 Pseudo-labeling, 0 labels
externos, 0 embeddings nuevos. Holdout y test abiertos **solo** para contar filas y calcular
hashes.

**V47-00** (`experiments/v47_00_auditoria_datos.py`, exit code 0): inventario de 7 fuentes
oficiales/derivadas con filas, columnas, sha256 y clasificacion legal; 5 corpus externos con
licencia, documentos y tokens; reconciliacion de las tres cifras citadas; conteos de la
pregunta central; busqueda del precedente de augmentation. Artefactos:
`reports/p47_data_inventory.{json,md,csv}` y `reports/p47_augmentation_audit.md`.

**V47-01** (`experiments/v47_01_integrar_notebook.py`, exit code 0): seccion 43, notebook 187 ->
189 celdas, celda de codigo con returncode 0.

**Hallazgos.**

- `data/` es un directorio real; `raw` y `processed` son enlaces simbolicos a otro worktree y
  `external` es real. Se registro con `is_symlink()` y `resolve()`, no de memoria.
- `train 1731 = dev 1483 + holdout 248`, sin asignar 0.
- `holdout2` (247 filas) esta dibujado **de dev**, no del holdout: 0 de 247 caen dentro del
  holdout. Por tanto no anade etiquetas nuevas.
- 519 es el numero de positivos TR, no el holdout. El holdout tiene 248.
- No hay 44k filas sin etiquetar en ningun sitio. El "44" son las features del meta-modelo.
- V6-08 ya probo `espaciado`, `elongacion` y `repeticion_puntuacion` con la augmentation dentro
  del training fold, grupos preservados y control que reproduce V03. J1 = -0.0061.

**Correccion propia durante la fase.** Redacte primero que "`data/` es un enlace simbolico a X",
afirmacion **falsa** que no habia verificado. Se detecto al imprimir la estructura real, y se
sustituyo por una linea que se construye con `is_symlink()` y `resolve()` en tiempo de
ejecucion. Tambien un primer borrador de frase daba por simblicos `raw`, `processed` **y**
`external`, cuando `external` es un directorio real; la version final lista los destinos
realmente resueltos.

**Errores de tooling.** Una lambda dentro de un bucle capturaba la variable de iteracion
(B023, detectado por ruff); una insercion de helper rompio la indentacion; una excepcion ciega
`except Exception` quedo tras el lint. Los tres se corrigieron antes de cerrar. El `exit code` se
capturo **sin pipe**, porque en P46 un `| tail` habia enmascarado un fallo real.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P48: investigacion residual guiada por EDA

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test. Presupuesto
de 3 experimentos: **consumidos 0**.

**V48-00** (`experiments/v48_00_eda_traceability.py`, exit code 0): matriz de trazabilidad de 15
hallazgos del EDA contra las 318 filas del registro. Artefactos:
`reports/p48_eda_traceability.{json,md,csv}`.

**V48-01** (`experiments/v48_01_residual_atlas.py`, exit code 0): atlas de 1483 filas, 757 celdas
de error, clasificacion A/B/C contra los 20 canales OOF historicos, analisis de margen, errores
conjuntos, 24 slices y combinaciones, y veredicto de brecha. Artefactos:
`reports/p48_residual_atlas.{json,csv}`, `p48_rescue_analysis.csv`,
`p48_unrescued_analysis.csv`, `p48_pattern_candidates.csv`, `p48_research_gap.md`.

**V48-02** (`experiments/v48_02_integrar_notebook.py`, exit code 0): seccion 44, notebook 189 ->
192 celdas, dos celdas de codigo con returncode 0.

**Hallazgo central.** Ninguna slice se enriquece en rescate. Varias tienen lift de error de 1,21
a 1,60 con p < 0,01, pero su tasa de rescate es igual o inferior a la global de 0,9445.

**Tres correcciones de criterio durante la fase, todas detectadas por auditoria propia.**

1. El criterio de cierre por "ultima fila coincidente" daba 7 hallazgos abiertos y era erroneo:
   para "grupos de near-duplicate" la ultima fila era la auditoria del campeon de P46, que no
   tiene nada que ver. Redefinido sobre veredictos sustantivos, excluyendo `AUDIT_ONLY` y
   `DIAGNOSTIC_ONLY`. El resultado correcto es 1 hallazgo abierto, no 7.
2. El criterio de candidato a hipotesis exigia lift de la **tasa de error**, lo que selectaba
   patrones que solo describen donde V03 falla, un hecho ya conocido. Redefinido para exigir
   enriquecimiento en **rescate** mediante prueba binomial contra la tasa global. Los 6
   candidatos iniciales pasan a 0.
3. Dos patrones quedaron definidos por el propio error (`margen_bajo_error`,
   `margen_alto_error`), de modo que su lift era circular. Marcados como no candidatos. Y
   `urls` y `urls_2_o_mas` eran el mismo patron, porque `n_urls` solo toma 0, 2 y 4 en dev.

**Errores de codigo corregidos.** Columna `canales_acieren_min` mal escrita frente a su uso; el
insulto lexico no existe por fila fuera de `all_three_errors.csv`, que es precisamente la
poblacion "las tres mal" y habria sido circular, sustituido por `label_rarity` y `poblacion`;
`id` del parquet es `object` y hay que castearlo; un `diagnostico_de_margen` calculado en `_md` y
usado en `main`; y un texto que afirmaba "100 por ciento" cuando el valor calculado es 99,3.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P49: reconciliacion forense del rescate OOF

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.

**V49-00** (`experiments/v49_00_rescate_forense.py`, exit code 0): localizacion de los tensores
en dos worktrees, reconstruccion del pool de 20 canales validada contra `hardness_v4.csv`
(4449/4449 celdas, diferencia 0.00e+00), reproduccion exacta del 147 de V17-00, matriz de
rescate y damage por canal, prueba de azar frente a senal, margen sobre rescate observado y
slices error-dense frente a rescue-dense. Artefactos:
`reports/p49_rescue_reconciliation.{json,md,csv}`, `p49_rescue_matrix.csv`,
`p49_damage_matrix.csv`.

**V49-01** (`experiments/v49_01_integrar_notebook.py`, exit code 0): seccion 45, notebook 192 ->
194 celdas, celda de codigo con returncode 0.

**Hallazgos.**

- Los tensores `v2_oof_*.npz` y `oof_candidates_v4.npz` estaban en los worktrees `PLN` y
  `PLN-v8`. No se habian perdido.
- 757 son **label-cells**, no tweets. Los tweets con algun error son 441.
- 147 es el rescate de **una** variante; 715 es el de la **union de veinte**. Se reproducen
  ambos exactamente.
- 715 no es azar: se esperan 1,20 celdas irrecuperables bajo independencia y se observan 42.
- 19 de 20 canales tienen net negativo. El unico positivo es V03 con umbral 0.5.
- `p49_damage_matrix.csv` se escribe con la misma tabla que la de rescate, con la columna
  `rescate` renombrada a `damage`, para que ambos ficheros sean legibles por separado. Es una
  decision de comodidad de lectura, no dos medidas distintas: la matriz de damage por canal es
  la columna `damage` de `p49_rescue_matrix.csv`.

**Reclasificacion de P48.** Se anade una cabecera a `reports/p48_research_gap.md` que declara
superados dos elementos (la limitacion sobre los tensores y el criterio de enrichment) y
verificados el resto. No se borra ni se reescribe el informe original.

**Errores propios.** Identificador CJK introducido en el script; `f1_score` importado desde
`scipy.stats` en vez de `sklearn.metrics`; las claves del artefacto v4 no son `variantes` sino
los propios nombres de variante en primer nivel; y columnas de contexto que no se copiaban al
dataframe base. Los cuatro detectados por ejecucion real con `exit code` capturado sin pipe.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P50: seleccion contextual de canal y techo del oracle

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.

**V50-00** (`experiments/v50_00_brecha_y_oracle.py`, exit code 0): auditoria de brecha de las cinco
categorias contra los artefactos, reconstruccion del pool de 20 canales, techo del oracle, tabla de
fiabilidad condicional al desacuerdo, y construccion de la matriz de 53 variables observables por
label-cell. Artefactos: `reports/p50_selector_gap_audit.md`, `p50_oracle_ceiling.csv`,
`p50_channel_reliability.csv`, `p50_gap_and_oracle.json`, `features/p50_celda_features.npz`.

**V50-01** (`experiments/v50_01_selector.py`, exit code 0): baselines B0-B3, S1 con 21
`LogisticRegression` y cross-fitting de 5 pliegues canonicos, AUC out-of-fold por accion, metricas
por pliegue, interseccion de errores, analisis por margen y por confianza, y correlacion de
Spearman entre AUC y net real. Artefactos: `reports/p50_selector_results.{json,csv}`,
`p50_selector_analysis.md`.

**V50-02** (`experiments/v50_02_integrar_notebook.py`, exit code 0): seccion 46, notebook 194 -> 196
celdas, celda de codigo con returncode 0.

**Bug propio mas grave de la fase, y casi pasa.** La primera version de `f1_macro` aplastaba las
4449 celdas con `ravel` antes de llamar a `f1_score`. sklearn ve ahi un unico problema binario sobre
4449 muestras y devuelve el **F1 agregado, 0,8186415291**, no el macro-F1 multietiqueta. El valor
canonico `corrected_cv` de V03 es **0,7772114525165721**, que es la media de los tres F1 por etiqueta.
Con la version erronea, el delta del oracle salia **+0,1713** en vez de **+0,2098**: inflaba
exactamente la magnitud que la fase existe para medir. Corregido a `f1_score(y, p,
average="macro")` sobre la matriz 2D y verificado celda a celda contra el `corrected_cv` canonico.
Es exactamente el fallo que P45 audito en el scorer del proyecto, reincidido en codigo nuevo.

**Otros errores propios.** Desempaquetado invertido de `acciones.shape`; calculo del AUC sobre el
conjunto de entrenamiento en vez de out-of-fold, que habria inflado la cifra de identificabilidad;
un `np.repeat`innecesario en el analisis por margen; una f-string anidada rota en la tabla de AUC
del notebook; identificadores CJK en dos scripts. Los cinco detectados por ejecucion real con
`exit code` capturado sin pipe.

**S2 no se ejecuto**, conforme a la regla de la especificacion que lo prohibe cuando S1 ya falla de
forma clara. El gate de 0,003 no se pasa, luego tampoco se calcularon bootstrap ni segunda
particion.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P51: seleccion contextual consciente de la utilidad

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.
Presupuesto respetado: 2 experimentos, P51-01 (S1-U) y P51-02 (S2-U).

**V51-00** (`experiments/v51_00_utilidad_y_asimetria.py`, exit code 0): brecha de la
familia utilitaria contra los artefactos, reconstruccion de la utilidad exacta con los
conteos de cada pliegue de entrenamiento, y analisis de asimetria de direccion.
Artefactos: `reports/p51_utility_selector_audit.md`, `p51_direction_asymmetry.csv`,
`p51_utility_gap_and_utility.json`, `features/p51_utility_targets.npz`.

**V51-01** (`experiments/v51_01_selector_utilidad.py`, exit code 0): S1-U con 20
`LogisticRegression` entrenados solo sobre celdas en discrepancia, target `delta_F1 >
0`, cross-fitting de 5 pliegues canonicos, puntos de equilibrio analiticos, y tres
diagnosticos de calibracion. Artefactos: `reports/p51_utility_selector_results.{json,csv}`,
`p51_utility_selector_analysis.md`, `p51_utility_selector_confusion.csv`,
`p51_utility_selector_rescue.csv`.

**V51-02** (`experiments/v51_02_accion_utilidad_esperada.py`, exit code 0): S2-U con la
regla de utilidad esperada, permutacion del net de celdas y variante restringida a la
direccion 1->0. Artefactos: `reports/p51_s2u_results.{json,csv}`,
`p51_utility_selector_s2u.md`.

**V51-03** (`experiments/v51_03_integrar_notebook.py`, exit code 0): seccion 47,
notebook 196 -> 198 celdas, celda de codigo con returncode 0.

**Dos hallazgos que corrigen a P50.**

1. El target de utilidad es **mucho mas aprendible** que el de correccion: AUC
   out-of-fold de 0,5507 a 0,9249, media **0,8091**, frente al 0,55 a 0,77 de P50.
2. Una permutacion da una nula de net **-115** con IC95 [-136, -93] frente a un
   observado de **-62**, luego el selector es **53 celdas mejor que el azar** con p de
   una cola 0,0000. P50 afirmo que la informacion no era identificable, y eso era
   falso.

**Un hecho estructural.** En celdas binarias, si canal y V03 discrepan, la etiqueta
verdadera coincide con exactamente una de las dos, luego todo cambio es rescate o
dano y nunca ambos mal. Luego `delta_F1 > 0` equivale a `rescate` y el target binario
de P51 coincide por construccion con el de P50. Lo unico nuevo es la magnitud, y un
clasificador binario no puede usarla.

**El bootstrap se sustituyo por permutacion, y por que.** El gate no pasa, luego la
directiva no pide bootstrap. Ademas un bootstrap sobre las celdas cambiadas es
**degenerado**: al reasignar a cada celda la prediccion del selector de la celda
muestreada, el valor no cambia y el intervalo sale de ancho cero, que es lo que
ocurrio en la primera ejecucion. La permutacion si responde a la pregunta util.

**Errores propios.** Un triple-quote anidado en un heredoc rompio el script de parche
sin escribir nada; un bloque D1/D2/D3 insertado dos veces y despues eliminado de
ambas copias; `d_arriba` con la asignacion retirada por ruff; indexado de canales
aplanado contra mascaras 2D; y un `ravel` de masks con formas incompatibles. Todos
detectados por ejecucion real con `exit code` capturado sin pipe.

**Unicode.** Auditoria exhaustiva de los 30 ficheros de P50 y P51: **0 bytes > 127**
en todos, comprobados en forma literal y en forma escapada `\uXXXX`, contra los rangos
CJK, hiragana, katakana, hangul, CJK-ext y cirilico. Las secciones 45, 46 y 47 del
notebook dan 0 no-ASCII. El no-ASCII restante del repositorio es anterior a esta
sesion y el verificador del proyecto lo acepta. Dos entradas CJK que **si** introduje
yo en P50 (`??` y `??`) quedaron corregidas antes de ejecutarse.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P52: policy oracle, decision boundary y complementariedad

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.
**0 filas nuevas en `experiments.csv`**, porque toda la fase es auditoria y analisis
descriptivo, y la directiva lo prohibe expresamente.

**V52-00** (`experiments/v52_00_auditoria_y_techo.py`, exit code 0): auditoria del codigo
real de P51, re-derivacion cerrada e independiente de la utilidad, verificacion de la
asimetria, techo de estrato, rescate y dano exclusivos, complementariedad de los 190
pares, correlacion de error en el regimen V03-wrong, clustering descriptivo y curva de
valor. Artefactos: `reports/p52_policy_oracle.{json,csv}`,
`p52_channel_pair_analysis.csv`, `p52_channel_exclusivity.csv`,
`p52_error_correlation.csv`, `p52_policy_value_curve.csv`, `p52_policy_analysis.md`.

**V52-01** (`experiments/v52_01_integrar_notebook.py`, exit code 0): seccion 48, notebook
198 -> 200 celdas, celda de codigo con returncode 0.

**Correccion propia de una correccion previa.** P51 registro que la elevacion necesaria al
bajar de 1 a 0 era 1,03x. Es incorrecto: dividio `(1 - 0,6108) = 0,3892` por la tasa base
de 0,3761, pero el rescate al bajar es `P(y=0)` y su equilibrio es `0,6108`. El valor
correcto es **1,62x**, luego la direccion 1->0 era mas dificil de lo que P51 reporto.

**Una lectura que se descarta a si misma.** La curva de valor de la politica sale con net
+225 y **no es valor de politica**: la accion por estrato se elige con labels del training
fold, de modo que todo cambio es rescate por construccion. Se reporta marcada como
contaminada, y la conclusion se apoya solo en el `STRATUM_ORACLE`. Antes de arreglarlo, la
primera version de la curva daba `net = n_cambiadas` en los ocho umbrales, lo que delato el
problema.

**Errores propios.** `np.repeat` sobre un array 2D en vez de `ravel`, que daba 13347
elementos en vez de 4449; una clave de diccionario con un espacio, que rompia el informe;
`p51_eq` leido despues de su primer uso; un `and` sin sentido que producia un `base_up`
falso; y **tres entradas CJK** (`??`, `??` y `??`) introducidas al escribir el script y
el notebook, corregidas antes de ejecutar. Todos detectados por ejecucion real con
`exit code` capturado sin pipe.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P53: auditoria de politica observable de grano fino

**Alcance.** 0 modelos, 0 submissions, 0 busqueda de umbrales, 0 Optuna, 0 consultas a
Kaggle, 0 uso de holdout o test. **0 filas nuevas en `experiments.csv`**, porque toda la
fase es auditoria, y la directiva lo prohibe expresamente.

**V53-00** (`experiments/v53_00_politica_fina.py`, exit code 0, 305 s): auditoria del
`STRATUM_ORACLE` de P52, colapso estructural del espacio de acciones, distribucion de
mascaras, rescate y dano por patron, auditoria del rescate exclusivo, descomposicion A B C
D de los 190 pares, HHI estandar, y techos cross-fitted de P0 a P4 con soporte diagnostico.
Artefactos: `reports/p53_observable_policy.json`, `p53_observable_policy_audit.md`,
`p53_observable_policy.csv`, `p53_pairwise_channels.csv`, `p53_exclusive_rescue.csv`,
`p53_pattern_policy.csv`.

**V53-01** (`experiments/v53_01_integrar_notebook.py`, exit code 0): seccion 49, notebook
200 -> 202 celdas, celda de codigo con returncode 0.

**Correccion de conclusion propia.** P52 cerro la linea contextual y la directriz de P53
podia haberlo dado por definitivo. P53 lo abre porque la inferencia de P52 no se seguia: 18
estratos no acotan todas las particiones observables. Al abrirlo, aparece el colapso
estructural, que es mas fuerte que lo que P52 afirmaba y tambien mas limitado.

**Tres correcciones metricas a P52.** (1) `STRATUM_ORACLE` no esta contaminado: la pregunta
de la directriz estaba mal planteada, porque elegir la accion con `y_true` es justamente lo
que hace que un techo sea holgado, y un techo holgado por debajo de V03 acota todo lo mas
cenido. (2) La exclusividad bilateral estaba mal medida: P52 restringia `i_only` a las
celdas donde ambos canales discrepan, conjunto que por el colapso es degenerado, luego da 0
por construccion; P53 obtiene 189 de 190. (3) El HHI estaba mal definido con
proporciones que no suman 1.

**Errores propios.** El mas grave: use `canales_plano[0]` en vez de "cualquier canal
discrepa" al construir la accion de volteo, lo que restringio el colapso a un solo canal y
produjo un `TRUE_ORACLE` de 0,8088 en vez de 0,9870. Cero con dos canales contradiciendose
debio delatarlo, porque con 20 canales era imposible. Ademas: `+` sobre arrays de cadenas
de numpy, `p_max` y `p_mean` con NaN, una clave de JSON escrita despues del `write_text`,
y un analisis por patron que era **tautologico** porque se hacia sobre celdas de error de
V03, donde voltear siempre ayuda. Todos por ejecucion real con `exit code` sin pipe.

**Rendimiento.** 305 s, por recomputar el estado observable dentro del bucle de pliegues en
vez de precalcularlo. Es ineficiencia, no error, y no afecta a los resultados.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P54: atlas semantico de los errores no rescatables

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.
**0 filas nuevas en `experiments.csv`**: toda la fase es EDA y auditoria.

**V54-00** (`experiments/v54_00_atlas_no_rescatable.py`, exit code 0): poblaciones A, B y C
recalculadas desde cero y verificadas contra P49, tabla descriptiva de 33 atributos por
celda, distribucion A/B/C, comparacion B contra C con enrichment y p de un lado,
reconciliacion de los quince hallazgos del EDA original con **dos** enrichments, test del
mecanismo concedente, atlas completo de las 42 celdas y taxonomia retrospectiva.
Artefactos: `reports/p54_semantic_audit.json`, `p54_semantic_audit.md`,
`p54_nonrescuable_atlas.csv`, `p54_nonrescuable_taxonomy.csv`,
`p54_rescue_vs_nonrescue.csv`, `p54_eda_reconciliation.csv`.

**V54-01** (`experiments/v54_01_integrar_notebook.py`, exit code 0): seccion 50, notebook
202 -> 204 celdas, celda de codigo con returncode 0.

**La leccion metodologica de la fase.** Una lectura cualitativa de las 42 celdas llevo a
pensar que el residuo era `insulto lexico sin escalada`. Al medirlo, el lexico esta en el
66,67 por ciento de C **y en el 53,36 por ciento de A**, luego el enrichment es de 1,25x
y no discrimina. La lectura cualitativa era correcta sobre el subconjunto de AG y
equivocada sobre el conjunto. Es exactamente el riesgo que la directiva intenta evitar, y
solo se esquivo porque el calculo se hizo antes de escribir la conclusion.

**Un mecanismo propuesto y refutado por los datos.** La lectura sugeria que la negacion
actua de forma *concedente*, licenciando lo que sigue. Medido: una sola celda de 42
combina negacion y marcador de contraste, con enrichment de 0,76x. El mecanismo se
declara **sin explicar** en lugar de sustituirlo por otro.

**Errores propios.** El test binomial recibia **conteos** en vez de indicadores, y
`k > n` lo hacia fallar en la reconciliacion del EDA; un dict que ruff elimino como
"usado" en un fix previo y que hacia falta; precedencia de operadores en la mascara de
alta confianza, que mezclaba la precedencia de `&` con la de `|`; y un `RE_TODO_CAPS` sin
usar. Todos por ejecucion real con `exit code` capturado sin pipe.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P55: auditoria estrategica del pipeline final de V03

**Alcance.** 0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.
**0 filas nuevas en `experiments.csv`**: es una auditoria, no un experimento. V03 no se
modifica.

**V55-00** (`experiments/v55_00_auditoria_pipeline.py`, exit code 0): lectura del pipeline
real desde el codigo de `04_refit_submissions_v02_v03.py` y `14_oof_corregido_v2.py`,
integridad de artefactos con sha256, umbrales in-sample y **anidados**, estabilidad por
pliegue, tasa positiva, trayectoria con el tamano de entrenamiento, consistencia de
vectorizador y grupos, y balance de pliegues. Artefactos:
`reports/p55_pipeline_audit.{json,md}`, `p55_pipeline_audit.csv`,
`p55_threshold_transfer.csv`, `p55_artifact_integrity.csv`.

**V55-01** (`experiments/v55_01_integrar_notebook.py`, exit code 0): seccion 51, notebook
204 -> 206 celdas, celda de codigo con returncode 0.

**El hallazgo, y por que no es una mejora.** Los umbrales de V03 se adoptaron sobre un OOF
generado con un protocolo de agrupacion mas debil que el de la metrica reportada. Al
corregir el protocolo no se reoptimizaron. Los umbrales anidados muestran que V03 esta
0,03 por encima de lo que pide D-016, en los cinco pliegues, en HS y TR. Pero la estimacion
honesta del beneficio es **+0,000925**, por debajo del margen de adopcion de 0,003. Se
registra como `BUG_FOUND` con `ORIGINAL_RESULT` y `CORRECTED_RESULT` separados, y **no** se
llama mejora ni se gasta una submission.

**Por que importa la distincion entre in-sample y anidado.** El delta in-sample es
**+0,003629**, que habria superado el margen de adopcion. Es una trampa: el umbral se
eligio sobre el mismo OOF que se puntua, luego no es una mejora esperable. El delta
anidado, con el umbral elegido en cuatro pliegues y aplicado al quinto, es **+0,000925**.
Reportar el primero como.expected seria el error clasico de esta auditoria.

**Un metodo que se corrigio a si mismo.** La primera comprobacion de si el vectorizador se
ajustaba dentro del pliegue buscaba una cadena en el docstring y devolvio `False`. La
verificacion correcta mira el codigo: `construir_matrices` llama a
`vectorizador.fit_transform(textos_entrenamiento)`, luego **si** esta dentro y no hay
leakage de representacion. Un `False` erroneo habria reportado un defecto de leakage que no
existe.

**Errores propios.** Un triple-quote anidado en la celda del notebook rompio el literal; una
referencia a indice futuro al construir la trayectoria de tamanos; una clave de npz
inexistente; y `ruff --fix` elimino en una pasada anterior un diccionario que si se usaba.
Todos por ejecucion real con `exit code` capturado sin pipe.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P56: transferencia de umbrales corregida

**Alcance.** La unica variable es el procedimiento de estimacion de umbrales. Sin cambios
en representacion, clasificador, C, ngrams ni class weights. **1 submission: 0.** 0
consultas a Kaggle, 0 uso de holdout o test. **1 fila nueva en `experiments.csv`**
(`V56-01`), porque si hubo una ejecucion experimental reproducible.

**V56-00** (`experiments/v56_00_transferencia_umbrales.py`, exit code 0, 43 s):
reproduccion exacta de P55, anidamiento completo con 4 pliegues interiores agrupados y
refit de vectorizador y modelo dentro de cada outer train, distribucion de umbrales con
mediana y desviacion estandar, regla de agregacion unica (mediana), T0 contra T1 por
pliegue, por etiqueta y por celda, tasa positiva, bootstrap opcional y decision.
Artefactos: `reports/p56_threshold_transfer.{json,md}`,
`p56_threshold_transfer.csv`, `p56_threshold_stability.csv`.

**V56-01** (`experiments/v56_01_integrar_notebook.py`, exit code 0): seccion 52, notebook
206 -> 208 celdas, celda de codigo con returncode 0.

**La correccion de conclusion propia.** P56 se abrio  para no descartar el
mecanismo de P55 por el gate, y lo que encontro fue que **el mecanismo tampoco se
sostiene**. Los umbrales de V03 caen dentro del rango de las cinco estimaciones interiores
en las tres etiquetas, con desviaciones estandar de 0,019 a 0,043. El "fuera del rango en
los cinco pliegues" de P55 era un artefacto de su procedimiento mas simple.

**El numero que casi paso.** La mediana aplicada globalmente da **+0,00264**, con 4 de 5
pliegues positivos, y habria clasificado como CASE A por el margen de la directiva. Es
optimismo puro: definir el umbral con datos de los cinco pliegues y luego evaluarlo en los
cinco vale **+0,0053**, mas que la propia mejora. La cifra honesta es **-0,00266**, y por eso
la logica de decision se corrigio para que use el umbral por pliegue exterior y no la
mediana global.

**El contraste instructive.** El F1 sube +0,0026 mientras el recuento de celdas baja -11.
No es contradiccion: macro-F1 promedia tres etiquetas y un cambio de umbral reordaza
positivas. Pero el recuento de celdas es la medida honesta del efecto, y es negativo en las
tres etiquetas. Es el mismo patron que P54 encontro en el residuo, y por eso se reporta el
net y no solo el F1.

**Errores propios.** Concatenacion de arrays de cadenas de numpy con `+`, que no soporta
`<U21` mas `<U21`; e imports sin usar. El mas relevante: la logica de decision inicial
usaba la cifra global de T1 y habria reportado CASE A, que es exactamente el falso positivo
que P56 pretendia evitar.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P57: cierre estrategico y plan final

**Alcance.** Solo auditoria y sintesis. **0 modelos, 0 entrenamientos, 0 submissions, 0
consultas a Kaggle, 0 uso de holdout, 0 uso de test. 0 filas nuevas en `experiments.csv`**,
porque no hubo ninguna ejecucion experimental (?29).

**V57-00** (`experiments/v57_00_sintesis_estrategica.py`, exit code 0): cruza `experiments.csv`
(323 filas), `leaderboard_log.csv` (22), `submission_manifest.csv` (9), `CHAMPION.md` y los
informes `p45` a `p56`. Verifica el sha256 de V03 al arrancar y **aborta si no coincide**.
Clasifica cada experimento por metrica, protocolo, poblacion y anidamiento, y construye la
jerarquia de magnitudes. Artefactos: `p57_final_experiment_inventory.csv`,
`p57_family_status.csv`, `p57_candidate_matrix.csv`, `p57_open_questions.csv`,
`p57_strategic_closure.{md,json}`.

**V57-01** (`experiments/v57_01_integrar_notebook.py`, exit code 0): seccion 53, notebook
208 -> 210 celdas, celda de codigo con returncode 0 y cifras leidas de los artefactos.

**El hallazgo que cambia como hay que leer el registro.** Las columnas `f1` y `macro_f1` de
`experiments.csv` **no son una metrica cada una**. El maximo absoluto de `macro_f1` es 0.8743, y
es **macro-6 sobre holdout**. La diferencia entre macro-6 y la metrica del proyecto, medida
sobre el mismo OOF de V03, es **+0.04101**. Y `leaderboard_log.csv` registra para V03 un
`local_f1` de 0.7989, **+0.021689** por encima del `corrected_cv` canonico, que es el coste de
medir con D-004 en vez de D-016.

Esto no lo habia detectado ninguna fase, y es la clase de error que produce una conclusion
falsa con toda la aparencia de una conclusion bien construida. **La respuesta no es reescribir
el registro**, que es historico y correcto en su contexto, sino **anadirle la clasificacion**
para que la columna sea utilizable. Eso hace `p57_final_experiment_inventory.csv`.

**El hallazgo operativo.** `experiments.csv` V33-03 documenta un envio real y aceptado: ID
**56608788**, 2026-09-27, `toxic_submission_v32_05_url_length.csv`, LB 0.78306, delta
**exactamente 0** porque el fichero es identico a V03. **No esta en `leaderboard_log.csv`.** El
total real de envios es **23**, no 22, y tanto `PROJECT_STATE.md` como `CHAMPION.md` afirman
"0 de 22". La afirmacion sigue siendo cierta en su sustancia y **el recuento es erroneo**. El
log no se modifica: es un registro de mediciones y sus 22 filas son fiables.

Ese envio es, de paso, el control mas limpio del proyecto: +0.0045 de `corrected_cv` y **LB sin
moverse ni en la sexta cifra decimal**.

**El numero que resume la estrategia.** Diez envios de candidatos con mecanismo nuevo
(1 de kernels, 1 de especializacion, 3 de ensembles, 5 de meta) y **los diez transfirieron a la
baja**. Y el mejor CV comparable del proyecto, 0.7904 de svd300_rbf, **no es el campeon**: se
envio y perdio 0.00368. La cifra que un argumento de "hemos probado mas cosas y ninguna supera
a V03" tiene que citar es la **anidada**, 0.7885, porque el 0.7904 es el maximo de cuatro
candidatos comparados sobre el mismo OOF, luego lleva seleccion.

**El resultado incomodo de la seccion de candidatos.** Ningun artefacto cumple a la vez legal,
reproducible, sin leakage, no rechazado y con mecanismo claro. No es una omision: los tres con
mejor evidencia interna tienen veredicto externo medido y contrario, y el unico con delta
honesto positivo y sin veredicto externo (E2-13) se cerro por una razon de mecanismo, no de
metrica.

**La distincion que P57 insistio en separar, y que vale la pena repetir.** Hay dos sitios donde
"no funciono" y "nunca se probo bien" no son lo mismo:

- **PLS: cierre de tipo B.** Sus numeros no son evidencia porque el experimento se invalido. El
  resultado verdadero llego despues, como verificacion, y es negativo. La hipotesis no se
  rechaza; el experimento si.
- **Estructura de etiquetas en V13-02 y V14-00: nunca ejecutadas**, cerradas por condicion de
  parada con 0 de 4 experimentos. Ahi no cabe decir "no funciono". Se registran como pregunta
  abierta, no como cierre.

**Errores propios.** Colision de nombre: la funcion `log()` quedaba sombreada por la variable
`log = pd.read_csv(LB_LOG)`, y fallo en tiempo de ejecucion. Un `if` anidado redundante. Un
`int64` de pandas no serializable en JSON. Y un conteo que contaba claves de diccionario en vez
de envios, que imprimia "envio real no registrado: 7" cuando era 1. Los cuatro se detectaron
por ejecucion real con exit code capturado, no por revision visual.

**Nota sobre una trampa de lectura que se evito.** La primera version del clasificador no
reconocia `corrected_cv` como marcador del protocolo D-016, y por eso el maximo comparable
salia 0.7885 en vez de 0.7904 y la familia de svd300_rbf quedaba sin clasificar. El nombre de
la metrica **lleva dentro** el protocolo, y eso habia que codificarlo.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P58: auditoria final de candidatos existentes

**Alcance.** Auditoria pura. **0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a Kaggle,
0 uso de holdout, 0 uso de test labels. 0 filas nuevas en `experiments.csv`** (?21: P58 no es
experimento).

**V58-00** (`experiments/v58_00_auditoria_candidatos.py`, exit code 0, 0,2 s): verifica el
sha256 de V03 al arrancar y **aborta si no coincide**; verifica que el OOF canonico reproduce
`0.7772114525165721` al ultimo digito y **aborta si no**; verifica que los tres OOF comparten
ids y etiquetas; recomputa cinco descomposiciones celulares sin entrenar; comprueba que
reproducen `reports/v4_candidates.json` al sexto decimal y **aborta si no**; audita E2-13;
aplica los siete criterios; valida formalmente los 37 CSV sin modificar ninguno.
Artefactos: `reports/p58_final_candidate_audit.{md,csv,json}`.

**V58-01** (`experiments/v58_01_integrar_notebook.py`, exit code 0): seccion 54, notebook
210 -> 212 celdas, celda con returncode 0.

**El hallazgo que importa, y que corrige a P57.** P57 afirmo que E2-13-META-ENT-RIDGE era el
unico candidato con delta honesto anidado positivo y sin veredicto externo, y que solo le
faltaba una verificacion externa. **Era incorrecto, y el error venia de leer la columna
`decision` de `experiments.csv` en vez del `result` estructurado que hay en la misma fila.**

La columna dice `CERRADA por sparsity`. El `result` dice: cobertura informativa 0,52 por ciento
contra un umbral del 20 por ciento, y sobre su propio comparador el incremento es +0,0011 con
3 de 5 pliegues y complementariedad negativa. Ademas, `v2_meta_oof.npz` declara
`variante = META-BASES-RIDGE`, ridge alpha 3, 44 features, que **es el modelo que se envio como
`v08_meta_v2` con LB 0,78163**.

O sea: E2-13 no es un mecanismo nuevo, es v08_meta_v2 mas un canal que cubre el medio por ciento
de las instancias, y su delta de +0,0113 es el de la familia META, refutada 5 de 5 veces. **El
cierre por sparsity es metodologico y correcto**, y el caso limite real es `B_meta36`.

**La leccion sobre como se leen las fases anteriores.** `experiments.csv` resume el veredicto en
una columna de texto libre, y esa columna puede ser **una etiqueta de la causa en vez de la
conclusion del argumento**. El bloque `result` es el que tiene los numeros. Es la segunda vez
que un control automatico sobre texto en vez de sobre evidencia da un veredicto erroneo, y la
primera vez que el error no estaba en el control sino en como leimos el registro a mano.

**El caso limite, bien separado del veredicto.** `B_meta36` pasa el gate D-016 pre-registrado
(IC95 [+0,003147, +0,016633], 4 de 5 pliegues, acierto 0,6183, el mas alto de los cinco) y no
tiene **ningun** veredicto externo. No es finalista por tres razones independientes: no existe
artefacto de prediccion y la directiva 12 prohibe crearlo; el holdout esta agotado 5 de 5; y su
hermano `C` es estadisticamente el mismo modelo que `v08_meta_v2`
(`delta_vs_meta_v08 = -4,25e-05`). **Pasa cinco de siete y falla en el sexto por ausencia de
artefacto.** Nombrarlo `quasi-finalista` es mas util que esconderlo, porque es el unico punto
donde una decision futura tendria algo que decidir.

**El metodo que hizo posible auditar sin reentrenar.** `features/oof_candidates_v4.npz` contiene
los cuatro candidatos del gate de V4-X01, con ids y etiquetas identicos al OOF canonico, y
`v2_meta_oof.npz` contiene el OOF del meta. Con esos dos tensores y el OOF canonico de V03 se
recomputan las cinco descomposiciones, y **reproducen el artefacto del gate al sexto decimal**,
lo que prueba que son los mismos tensores. Eso es validacion formal de artefactos, que es lo
unico que la directiva 12 permite, y no es reentrenamiento.

**Validacion formal de los 37 CSV.** 1154 filas, columnas `id,HS,TR,AG`, ids y orden identicos
a `toxic_test.csv`, etiquetas binarias, sin nulos, sha256. **37 de 37 validos.** Ninguno se
modifica. Se confirma que `toxic_submission_v20_03_reproduccion_v03.csv` es bit a bit identico a
V03, que es la razon de que su envio diera delta exactamente 0.

**Errores propios.** Cinco fallos de ejecucion, todos detectados por la ejecucion real con exit
code capturado y no por revision visual: claves en espanol buscadas en ingles (`rescate` por
`rescue`), claves anidadas bajo `veredicto` buscadas en la raiz, dos caracteres acentuados que
se colaron en las cadenas del markdown y que no pasan por `a_ascii`, y el modulo
`unicodedata.decomposing` que no existe. El ultimo aparecio tambien en P57, y es el mismo error
dos fases seguidas: **conviene convertir los caracteres no-ASCII del diccionario de sustitucion
a escapes `\uXXXX` desde el principio, no a posteriori.**

**Sin push, sin merge, sin reset, sin rebase, sin PR.**


---

## 2026-09-28 - Fase P59: frontier research (tracks y gate)

**P59-00** (`experiments/p59_00_registro_hipotesis.py`, exit 0, 0,2 s): registro de
hipotesis con gate de novedad ejecutable. 16 precedentes verificados contra
`experiments.csv`, abortando si la cifra citada no esta en la fila real.

**El hallazgo que ahorra presupuesto: P59-G no se ejecuta.** V8-03 ya hizo
exactamente lo que propone el parrafo 33, a tres umbrales (1636 / 927 / 455
features; -0,003308, -0,004628, -0,018890; 1/5, 1/5, 0/5), con IC95 pareado, y se
cerro con "No se reintenta con mas agresividad ni con otra constante de
submuestreo". Un gate de novedad que no cierra pistas no esta haciendo su trabajo.

**P59-01** (`experiments/p59_01_voronoi_local.py`, exit 0, 90,0 s): diagnostico
Voronoi. 9 tests unitarios en verde, 5 teselaciones fold-internas, AUC del score
0,725194 contra error de V03 con IC95 [0,7061, 0,7446], y 0,471798 contra celda no
rescatable con IC95 [0,3954, 0,5469]. **Diagnostico, no experimento:** no entrena
clasificador ni evalua `corrected_cv`, luego no se registro en `experiments.csv`
(parrafo 50).

**P59-02** (`experiments/p59_02_voronoi_bloque.py`, exit 0, 80,8 s): bloque
complementario, 1 experimento registrado. Tres brazos para separar el coste del
envoltorio del aporte de la geometria. H1 y H2 fallan, `CLOSED`.

**P59-90** (`experiments/p59_90_integrar_notebook.py`, exit 0): integrador generico
e idempotente. Seccion 55 (P59 Frontier Research) y 56 (P59-A). Las dos celdas se
verifican **aisladas** con `returncode = 0`, no solo encadenadas.

**Errores propios de esta fase, todos cazados por `exit code` real.**

1. Dos precedentes citados **de memoria** en el registro de hipotesis. El gate los
   rechazo porque las cadenas no existian en las filas del registro. Es exactamente
   el fallo que el gate existe para impedir, y aparecio en el propio gate.
2. `celda_c[error].all()` como comprobacion de que toda celda no rescatable es un
   error: seleccionaba las 757 celdas con error, no las 42. Condicion trivialmente
   verdadera, luego inutil.
3. Comparar el `tweet_id` del atlas de P54 contra el **indice posicional** en vez
   del identificador real: 0 celdas localizadas donde deberian ser 42.
4. La matriz de distancias consulta-referencia construida como producto de
   distancias a prototipos, que no es una distancia entre puntos. El calculo
   estaba entero mal.
5. `centros[d[0].sum()]` como primer medoid: indexaba un array con un float.
6. Reemplazar globalmente `rescuable` por `rescatable` **corrompio la ruta** al
   atlas de P54. Los reemplazos globales de cadenas nunca son seguros.
7. Escaneo de bytes > 127 sobre un `.npz`: un ZIP es binario por naturaleza, luego
   el control no significaba nada. Sustituido por una asercion de dtype.
8. Editar la celda del notebook con cirugia de strings en vez de dejar que el
   integrador idempotente la regenerase: duplico la seccion 56. Se revirtio el
   notebook a HEAD y se reintegro.

**El punto de fondo, que es el mas caro de los ocho:** tres de los ocho fallos
eran de **suposicion** en vez de de medicion. La geometria se construyo entera
sobre una distancia mal definida, y el resultado habria sido publicado sin que
ningun control lo notara si no se hubiera escrito una celda de codigo que la
consume.

**Sin push, sin merge, sin reset, sin rebase, sin PR.**
<!-- P60 2026-09-28 -->
<!-- P60 2026-09-28 -->
## P60 - ejecucion

- 2026-09-28T18:06:45+00:00 P60 completa, exit 0.
- 0 submissions, 0 consultas al leaderboard, 0 uso de holdout, 0 uso de test.
- Baseline verificado al inicio y al cierre de cada pista: sha256 intacto, corrected_cv exacto, 1415 grupos con 0 partidos.
- Artefactos generados por script leyendo los JSON, sin cifras propias.

<!-- P61 2026-09-28 -->
## P61 (2026-09-28) - Capacidad, diversidad y distribucion de probabilidad

Tres experimentos: `P61-01` LightGBM con Optuna anidado, `P61-02` ensemble con pertenencia
y pesos anidados, `P61-03` segunda particion con control negativo. Ademas la
reconciliacion de los nueve problemas de instrumentacion de P60, la EDA de las
probabilidades OOF de V03 y el inventario con diversidad de los veinte canales del pool de
P50.

```text
P61-01  LightGBM + Optuna anidado    0.7513651502
         familia -0.0302397191
         busqueda +0.0043934168
         total -0.0258463023   2/5   CLOSED
P61-02  ensemble anidado             0.7835159596  +0.0063045070  gate superado
         control de atribucion: el modelo nuevo SUBTRAIA -0,000850
P61-03  segunda particion            test 1 -0.0011527969
         test 2 holdback medio -0.0015908404 con 7/10
         test 3 control negativo +0.0059889238   REJECTED
```

**El hallazgo que mas importa.** El control negativo NO colapsa. Con los tres peores
canales del pool, entre 0,04 y 0,24 por debajo de V03, la ganancia del ensemble sigue
siendo +0.0059889238. Un promediado ponderado que contiene a V03 mejora el OOF casi con
seguridad porque los pesos se eligen para exploitear su residuo sobre las mismas filas, y
el bootstrap no lo detecta porque mide ruido de muestreo y no sesgo de seleccion. Esto
explica de una sola causa el fracaso historico de ENS-01 a ENS-04.

**El diagnostico que queda.** Entre el 60 y el 68 por ciento de los errores de V03 son
errores de alta confianza, con la distancia al umbral POR ENCIMA de la de los aciertos.
El limite de V03 no es de decision ni de calibracion: es de representacion.

```text
no evidence of exploitable remaining model capacity/diversity
V03 remains the best verified leaderboard result.
```
<!-- /P61 2026-09-28 -->

<!-- P62 2026-09-29 -->
## P62 (2026-09-29) - Representacion, estructura linguistica y error de alta confianza

Cinco experimentos, tres pistas cerradas sin gastar nada, y una correccion de lectura.

```text
P62-01  L palabra__POS     -0,009572  1/5  CLOSED
P62-02  M triple lexico   -0,011236  0/5  CLOSED
P62-03  curva de presupuesto   ningun K mejora
P62-04  POS contra lemas   ambos pierden 8/8, el POS ayuda 6/8
P62-05  cobertura min_df=1  -0.0033508936  CLOSED
```

**La interaccion existe y es medible.** La synergy palabra-POS es +0.0657, el
17.41 por ciento de la senal de la palabra. El problema no
es que falte informacion, es que no cabe: repaga el
17.4 por ciento del coste de anadir el canal.

**La cobertura tampoco es la via.** `min_df=2` conserva solo el
28.3 por ciento de las formas superficiales
presentes, y `min_df=1` lo sube al 99.5 por
ciento y resta.

**La correccion.** Yo lei mal mi propia EDA: el perfil de longitud se llamaba
`estructura_corta` pero su cuartil superior es lo largo, luego la delta negativa decia que
V03 es MEJOR en los largos, no peor en los cortos. La comprobacion independiente lo
desmiente: V03 alcanza 0.9639 de AUC en los cortos y su F1
es 0,94 contra 0,72 en los largos. El limite esta en el ORDEN de los textos largos, donde
V28-01 ya habia localizado el techo estructural.

```text
no evidence of exploitable remaining model capacity in the linguistic representation
V03 remains the best verified leaderboard result.
```
<!-- /P62 2026-09-29 -->
