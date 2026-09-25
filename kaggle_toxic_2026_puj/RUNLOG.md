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
