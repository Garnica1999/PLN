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

<!-- P63 2026-09-29 -->
## P63 (2026-09-29) - Representacion a nivel de registro, estructura del dataset y composicion

Seis experimentos, dos auditorias y un dataset auxiliar. Ninguno supera a V03, y lo
interesante no son los negativos sino que cada uno lleva un mecanismo MEDIDO.

```text
P63-00  ID forensics      D. STRUCTURED_IDENTIFIER, confianza alta, 0 experimentos
P63-02  id predictivo     0,419136 contra 0,424038 de un control de igual dimension
P63-03  relacional        el bloque esta 92,92 por ciento vacio
P63-04  composicion       brecha especifica +0,0083528268, y +0,1055 en solitario
P63-05  texto largo       la brecha en MUY_LARGA es -0,009437
P63-06  integracion       brecha +0,0085891400, MARGINAL
P63-07  residuo           los bloques restan al margen, luego son redundantes con el
                         margen y no con los n-gramas
```

**El id no es un contador.** V15-00 lo afirmo sin medirlo. Train y test son dos muestras
disjuntas del mismo bloque contiguo, con interseccion 0 y union
2885 sobre 4999 posiciones. Es un indice de
coleccion, y su contenido es la posicion en un ensamblado, que no correlaciona con la
etiqueta.

**La composicion en ratios porta senal y no cabe.** Es un resultado distinto del de P62, no
uno contrario: un canal disperso de varios cientos de columnas compite por el presupuesto
L2 de un espacio de 21000 dimensiones, y 42 proporciones acotadas no compiten. Y aun asi
no mejora a V03, porque lo que aportan ya esta implicito en la confianza de V03.

**La hipotesis mas antigua del repositorio queda refutada.** V30-01 escribio que el efecto de
la URL y la mencion cambia de signo con la longitud, y nunca gasto un experimento. El bloque
de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente en la region
muy larga, con -0.009437. Refutada en su prediccion, no solo
no confirmada.

**El limite que queda es de orden, no de decision.** El margen hasta el techo moviendo solo
el corte va de +0.0000 en la region corta a +0.0890 en la
muy larga, y en la region CORTA es exactamente +0.0000: el umbral de V03 ya
es el optimo ahi sobre las tres etiquetas.

**Dos correcciones cambiaron la direccion de una conclusion** y tienen fila propia en el
registro. El confound de escala de P63-04 paso el delta de -0,1899987838 a
-0,0102687019 y cambio el signo de la brecha. El techo de P63-06 paso el margen de MUY_LARGA
de +0,1589 a +0,0890.

```text
no evidence of exploitable remaining model capacity in the record-level and
aggregate-composition dimensions of this representation

V03 remains the best verified leaderboard result.
```
<!-- /P63 2026-09-29 -->

<!-- P64 2026-09-29 -->
## P64 (2026-09-29) - Representacion del tweet largo y distribucion de la evidencia interna

Cuatro experimentos, dos auditorias, trece figuras. Ninguno supera a V03, y la respuesta
tiene dos mitades que no se pueden separar.

**La estructura interna es REAL.** P59-07 habia explicado su fracaso por cobertura, no por
concepto: con ventana de 32 tokens solo el
9.6 por ciento de las filas tiene mas
de una unidad, mientras que la segmentacion por puntuacion llega al
50.6 por ciento. Un factor de
5.29 en las celdas alcanzables, que no es una variacion
de parametro.

**Y el leave-one-out, que P59 nunca computo, muestra el mecanismo.** El 80,88 por ciento de
los documentos multi segmento de la region MUY_LARGA tienen segmentos que empujan la decision
en signos opuestos, la entropia interna sube de 0,1048 a 0,4565 con la longitud, y la
concentracion cae de 0,7554 a 0,6170. En los tweets largos la decision no descansa en un
segmento dominante sino en un reparto contradictorio. Eso es exactamente lo que una bolsa de
n-gramas aplana.

**Y no aporta.** El bloque de evidencia queda -0.0140959650 frente al
margen de V03, y el leave-one-out -0.0004650203, que es casi nula.
Tampoco gana en la region larga, que es donde P63 midio el margen, y el margen tampoco mejora
ahi, luego el problema de MUY_LARGA no es de representacion ni de confianza.

**El perfil segmental es longitud disfrazada.** El mejor rasgo da
0.589847 de AUC frente a
0.600506 de contar tokens, luego queda
cerrado como rasgo.

**La razon de fondo.** El margen de V03 solo ya explica sus propios errores con AUC 0,770875
contra el acierto, y el mejor rasgo de evidencia interna llega a 0,605302. La estructura no
falta: esta dentro de la confianza de V03.

```text
the internal evidence distribution of long tweets is real, measured and reproducible,
and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```
<!-- /P64 2026-09-29 -->

<!-- P65 2026-09-29 -->
## P65 (2026-09-29) - Representacion de largo alcance, alcance de operadores y jerarquia

Cuatro experimentos y una auditoria, trece figuras. Ninguno supera a V03, y la respuesta tiene
una forma que no estaba prevista.

**LA RESPUESTA: la senal existe y la adicion la rechaza.** Cinco candidatos de este proyecto
baten a los dos controles, el de capacidad y el de la confianza de V03, y ninguno mejora:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0.0596671770 | +0.0071281317 | -0,0039520688 |
| P65-03 cohesion documental | +0.0090835614 | +0.0083711943 | -0,0027090062 |

Lo que falta no es informacion: es sitio. Unas cientos de columnas compitiendo por el
presupuesto L2 de un espacio de veinte mil, y lo que la columna nueva necesita decir es mas
pequeno que lo que cuesta decirlo. Eso es una afirmacion sobre la ARQUITECTURA de la adicion,
y es falsable: predice que la misma evidencia fundida en el espacio de probabilidad, y no
concatenada en el de caracteristicas, sobrevive.

**EL GAP ANALYSIS CERRO MAS DE LO QUE ABRIO.** 6 abiertas y
10 cerradas, leidas en el codigo y no en el registro. Los skip-grams ya los
habia medido V6-05, con un vocabulario un 83 por ciento mayor y un resultado neutro. La
negacion y la cita las habian medido E2-07 y E2-08, y despues P64-06 y P64-07.

**EL CONTROL DE IMPLEMENTACION REPRODUCE A V6-05.** El rango de hueco 1 a 2, que es
exactamente el suyo, da
-0.0010515908 frente al -0,0001 que el
registro. Luego la implementacion de los pares con hueco es la misma, y los rangos que V6-05 no
alcanzo, de tres a cinco y de seis a diez tokens, tampoco aportan.

**Y LA LONGITUD DEL TWEET NO ES EL MECANISMO.** Un especialista ajustado en LARGA y MUY_LARGA
es BIT A BIT el mismo numero que un reentrenado sobre las mismas
714 filas sin ningun criterio de longitud:
+0.0000000000. Un negativo asi no es *faltan datos*,
que seria una excusa valida: dice que la especializacion no es el mecanismo, punto. Y el
especialista gana
+0.0212 en MUY_LARGA y
pierde -0.0257 en LARGA, que tiene
el triple de filas, luego hay una mejora local real que la metrica primaria rechaza.

```text
the information about long-range relations is present, measurable and not what V03 already
knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long rows with no
length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```
<!-- /P65 2026-09-29 -->

<!-- P66 2026-09-29 -->
## P66 (2026-09-29) - Late fusion: la evidencia de largo alcance en el espacio de scores

Dos experimentos y dos auditorias, trece figuras. Y es la primera vez que la senal de largo
alcance mueve el numero.

**LA RESPUESTA A LA PREGUNTA DE LA FASE: SI, Y ES ATRIBUIBLE AL CANDIDATO.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | +0.0026743167 | 4/5 | +0.0041133588 |
| MI_arbol + residual con score barajado | -0.0014390421 | | |

La brecha de atribucion es la cifra que manda: el mismo mecanismo con el score del candidato
**barajado** pierde casi todo.

**Y MI_arbol es el unico que funciona porque es NO LINEAL.** Las correlaciones entre z_V03 y
z_C de los siete candidatos van de +0,9294 a +0,9988: son casi la misma puntuacion. El AUC del
residuo deja a cinco de siete por debajo de 0,5, luego su parte unica destruye informacion en
vez de anadirla. El de MI_arbol es el unico por encima del azar, y su modelo es el unico no
lineal de la lista. Si el error de V03 fuera una funcion lineal del error de MI_arbol, la
regresion residual no dejaria nada.

**EL MATIZ, Y VA PEGADO AL RESULTADO.** De las
53 celdas que la fusion rescata, solo
6 son errores de
alta confianza y las demas estan pegadas al umbral, luego es sobre todo **recalibracion**. Y el
bootstrap por 1415 grupos da IC95
[-0.0044311634, +0.0102108745], que
**incluye el cero**. Veredicto PROMISING, no FINALIST.

**EL INVENTARIO FUE EL PROBLEMA REAL.** Ningun script de P64 y P65 escribio sus
probabilidades a disco, y `features/` esta en `.gitignore`. Los scores se regeneraron y se
verificaron contra el numero de cada fase: 5 de 7 reproducen con diferencia de 1e-11, uno es
no verificable y uno tiene una diferencia declarada. El regenerador **se paro** cuando dos no
cuadraron.

**Y UNA CORRECCION QUE CAMBIO EL VEREDICTO DEL MEJOR RESULTADO.** `blend_prob` usaba siempre
la probabilidad REAL del candidato y nunca la barajada, luego su control de barajado salia
identico al real POR CONSTRUCCION, la brecha daba cero falso, y el mejor resultado de la fase
queda INVALIDATED por un motivo que no era el suyo. Corregido, ese mismo resultado pasa a
+0,0033 con brecha +0,0093.

```text
adding columns to a twenty thousand dimension space does not fit, and correcting a three
column score does, for the SAME evidence and with the SAME base model

the only measured exception is MI_arbol, and its distinguishing feature is that the model
producing it is NON LINEAR

and the caveat: it is mostly recalibration, because 6 of 53 rescues are high confidence and
the IC95 includes zero

V03 remains the best verified leaderboard result.
```
<!-- /P66 2026-09-29 -->

<!-- P67 2026-09-29 -->
## 2026-09-29, P67

- `experiments/p67_00_reconciliacion.py`: reconcilia las dos cifras de P66, marca `FINAL_VALID_P66_RESULT = 0.0026743167` y audita el protocolo nested.
- `experiments/p67_01_controles_decisivos.py`: reentreno anidado del umbral, sustitucion cruzada de los 7 candidatos, enriquecimiento de los rescates y AUC del residuo contra la etiqueta estratificado por banda de margen.
- `experiments/p67_02_cerrar.py`: informe final, figuras, cumplimiento, registro y cierre idempotente.

Sin push, sin merge, sin rebase.
<!-- /P67 2026-09-29 -->

## 2026-09-29, P67 cierre documental

- `experiments/p67_03_cierre_documental.py`: congela 11 piezas de evidencia, separa las cuatro
  afirmaciones, documenta el dano de P66 con su marca de no reejecucion, y corre las dos
  verificaciones de ASCII por separado.
- Control nuevo: azar con beta fijo. La senal real es positiva en los seis betas y el azar
  negativo en los seis.
- Hallazgo algebraico: el mecanismo `residual` de P66 es un `blend_logit` con desplazamiento de
  umbral, con diferencia `8.882e-16`.
- 8 figuras nuevas. 0 bytes no ASCII en los artefactos generados.

Sin push, sin merge, sin rebase.

## 2026-09-29, post-P67: gap analysis y P68-H1

- `experiments/p67_04_gap_analysis.py`: interseccion de errores separada en A y B, matriz de
  atribucion entre las siete alternativas, clasificacion de once familias y el presupuesto.
  B = 215 celdas, 122 exclusivas. Propone `P68-H1`.
- `experiments/p68_01_enrutado_por_estructura.py`: enrutado de la decision por estructura del
  texto, 123 features, cero derivadas del score. Resultado **INVALIDATED**, delta
  -0.0018992302, 1 de 5 pliegues.
- El diagnostico: AUC a quien delegar 0,5491 frente a AUC donde V03 falla 0,6080. La
  estructura predice dificultad, no ganador.

Sin push, sin merge, sin rebase. Test y holdout sin tocar.

## 2026-09-29, P68-H2 y el cierre de la investigacion

- `experiments/p68_02_h2_auditoria.py`: audita si el rescue-reweighting es nuevo. **No lo es.**
  9 ponderados con indicador propio, 2 con indicador ajeno, y el propuesto es un subconjunto
  estricto de uno ya medido. Clasificado **DUPLICATE**. Coste: 0 experimentos.
- Barrido de los ocho ejes que el encargo senala: 0 sin cita historica.
- Condicion de cierre del encargo cumplida. Fase de investigacion **cerrada**.
- Notebook actualizado con P67 y P68.

Sin push, sin merge, sin rebase. Test y holdout sin tocar.

## 2026-09-29, auditoria de A_cost_sensitive_downweight

- `experiments/p69_01_auditoria_downweight.py`: recupera la procedencia completa, reconstruye
  la regla de pesos del codigo, clasifica la causa del criterio 6 como **A_falta_de_artefacto**
  y reproduce el candidato.
- `experiments/p69_02_segunda_particion.py`: repite el experimento congelado sobre la segunda
  particion agrupada (semilla 20260928). **CONFIRMED**, delta +0,0105253130 con 5 de 5.
- `experiments/p69_03_cierre_downweight.py`: consolida ORIGINAL, REPRODUCED y segunda
  particion, anade el bootstrap por grupos y fija el estado final **MARGINAL**.

Cadena regenerada: `27_label_consistency_v4.py` -> `28_hardness_v4.py` ->
`30_supervised_weighting_v4.py`. Los tres scripts sin drift desde sus commits originales.

Sin push, sin merge, sin rebase. Test y holdout sin tocar.

## 2026-09-29, confirmacion de A_cost_sensitive_downweight

- `docs/p69_especificacion_congelada.md` (`915991b`): semilla, protocolo, regla de decision y
  diseno de permutacion fijados **antes** de ejecutar.
- `experiments/p69_04_tercera_particion.py`: tercera particion, semilla 20261001.
  **CASO A**, tres de tres positivas.
- `experiments/p69_05_permutacion.py`: 1000 permutaciones, semilla 20261002.
  **p = 0,000999**, 0 de 1000 superan al observado.
- `experiments/p69_06_particiones_corregidas.py`: corrige el defecto de `ajustar_predecir`
  (class_weight 'balanced' recalculado con sample_weight) y repite las dos particiones.

Estado final: **PROMISING_UNCONFIRMED**, 5 de 6 componentes. No se adopta, no se envia nada.

Sin push, sin merge, sin rebase.

## 2026-09-29, cierre de A_cost_sensitive_downweight

- `experiments/p69_07_cierre_final.py`: consolida la evidencia final, escribe el apartado
  **LIMITACION DE PRECISION ESTADISTICA** y la nota de gobernanza. 0 experimentos.

Informes nuevos: `final_evidence.md`, `final_evidence.csv`, `governance_note.md`.

```text
research_status   CANDIDATE_PENDING_GOVERNANCE
V03               FROZEN
candidato         PROMISING_UNCONFIRMED
SUBMISSION        NO      TEST  NO      LEADERBOARD  NO
```

Sin push, sin merge, sin rebase.

## 2026-09-29, cierre experimental

- `experiments/p69_08_paquete_candidato.py`: empaqueta el candidato con manifest completo,
  tabla comparativa con V03, evidencia en CSV y nota de gobernanza. 0 experimentos.

```text
research_status     EXPERIMENTAL_RESEARCH_CLOSED
candidate_status    PROMISING_UNCONFIRMED
governance_status   PENDING
submission 0 | test_development_use 0 | leaderboard_tuning 0
```

La investigacion experimental termina aqui. El candidato queda congelado.

Sin push, sin merge, sin rebase.

## 2026-09-29, reconciliacion de estado

- `experiments/p69_09_reconciliar_estado.py`: separa los tres ejes de estado y marca
  `CANDIDATE_PENDING_GOVERNANCE` como `SUPERSEDED`. 0 experimentos, 0 ficheros historicos
  modificados.

```text
research_status     EXPERIMENTAL_RESEARCH_CLOSED
candidate_status    PROMISING_UNCONFIRMED
governance_status   PENDING
```

Sin push, sin merge, sin rebase.

## 2026-09-29, submission experimental preparada

- `experiments/p69_10_submission_candidato.py`: revalida V03 y el candidato, reajusta segun el
  protocolo pre-registrado, transforma el test con los vectorizadores del train, predice, y
  genera la submission con 38 checks.

```text
submission   submissions/toxic_submission_cost_sensitive_candidate.csv
sha256       6fc7c171a8af08b9ac80c744baae2aceb65f2c1488e5eda15382b5914930898f
filas        1154      columnas  id, HS, TR, AG
checks       38, fallos 0
```

No se sube a Kaggle. No se consulta el leaderboard. No se leen labels de test. No se usa el
holdout. No se generan variantes.

Sin push, sin merge, sin rebase.

## 2026-09-29, post-mortem de la transferencia CV -> LB

- `experiments/p70_01_sensibilidad_tamano.py`: 12 corridas de submuestreo por grupo.
  Desviacion 0,007214 = 0,88x el efecto. Control limpio: las 3 al 100% identicas.
- `experiments/p70_02_postmortem.py`: prevalencia, margenes, descomposicion e interseccion.
- `experiments/p70_03_correccion_postmortem.py`: **la causa dominante es la SELECCION del
  baseline**. V03 es el maximo de 21 mediciones; el candidato cae en la mediana.

```text
ventaja del maximo sobre la mediana   0,00643
el gap a explicar                     0,00585
```

Auditoria independiente del especialista de Kaggle: 3 objeciones, todas incorporadas.

Sin test, sin holdout, sin submission nueva, sin tuning. Sin push.


## 2026-09-29, P0 de domain-specific Word2Vec sobre Twitter/X

- `experiments/p71_00_auditoria_corpus.py`: auditoria de procedencia de cuatro corpus espanoles
  de Twitter/X. **Ninguno pasa.** 0 descargas, 0 entrenamientos.

```text
ALLOWED 0 | AMBIGUOUS 3 | PROHIBITED 1
CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY
```

El corpus principal (`pysentimiento/spanish-tweets`) falla por dos motivos independientes: no
declara licencia, y no se puede garantizar que el texto del test no este dentro sin compararlo
contra el test, que el encargo prohibe.

Sin push, sin merge, sin rebase.


## 2026-09-29, P0 ampliado: busqueda en tres plataformas

- `experiments/p71_01_auditoria_ampliada.py`: amplia la busqueda a HuggingFace (6 consultas),
  Kaggle (8 consultas, 42 candidatos unicos) y GitHub (4 consultas). **9 candidatos auditados,
  ALLOWED 0.**

```text
ALLOWED 0 | AMBIGUOUS 6 | PROHIBITED 2 | NOT_APPLICABLE 1
```

El hallazgo: el bloqueo ha cambiado de naturaleza. El candidato de Kaggle tiene licencia
explicita y periodo cerrado; lo que falta es que **el periodo de TOXIC no esta documentado**.

Sin push, sin merge, sin rebase.


## 2026-09-29, la linea Twitter se reabre: la licencia no era una regla

- `experiments/p71_02_correccion_criterio.py`: corrige el criterio de auditoria. La competencia
  no menciona licencias; era un campo a registrar que yo eleve a criterio eliminatorio.
- `experiments/p71_03_preparar_corpus.py`: descarga verificada, comprobacion de solapamiento y
  preparacion de los dos corpus con la misma tokenizacion.

```text
mmaguero/covid19-spanishtweets-...  AMBIGUOUS -> ALLOWED
solapamiento con train: 0 exactas, 0 normalizadas
Twitter 10.000.029 tokens | general 10.002.400 tokens
```

Pendiente para cerrar P1: compilar el binario FastText C++ (la spec de v5 prohibe gensim) y
entrenar los dos modelos.

Sin push, sin merge, sin rebase.


## 2026-09-29, el veredicto de la fase Twitter se invalida

Carlos miro las curvas de loss y pregunto si el modelo no estaria aprendiendo. Tenia razon.

- `experiments/p71_06_word2vec_optimizador.py`: mismo modelo con SGD y decaimiento lineal.
  **La loss se queda clavada en 1,386295**, que es el valor del estado de productos punto nulos.
- `experiments/p71_07_invalidar_veredicto.py`: marca el veredicto.

```text
CLOSED -> INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS
la hipotesis sigue SIN PONERSE A PRUEBA
```

Se probaron cuatro arreglos y ninguno funciono. Hay un bug en mi implementacion que no he
aislado, y queda declarado como tal.

Sin push, sin merge, sin rebase.


## 2026-09-29, el bug del Word2Vec queda aislado y corregido

Carlos pidio aislarlo con una prueba controlada. Tenia razon en pedirlo.

- Prueba de aislamiento: corpus diminuto de dos familias que nunca se mezclan, con pares REALES.
  La prueba anterior usaba pares aleatorios, que son inaprendibles, y por eso no movia la loss.
- El bug: la loss promediada sobre lotes de 8192 hace el gradiente 8192 veces mas pequeno.
  El learning rate efectivo por ejemplo era 0,025/8192 en vez de 0,025.
- La correccion: learning rate escalado por el lote (0,025 x lote), mas inicializacion aleatoria
  de la salida.
- experiments/p71_08_word2vec_corregido.py: la loss baja monotonicamente en los dos corpus.

  Adam            1,706 -> 1,884      SUBE
  SGD sin arreglo 1,386295 constante  CLAVADA
  SGD corregido   1,272 -> 1,095      BAJA

Pendiente: relanzar el downstream con los modelos _lrfix.

Sin push, sin merge, sin rebase.


## 2026-09-29, downstream con los modelos corregidos

- `experiments/p71_09_downstream_lrfix.py`: reejecuta el downstream con los modelos que
  entrenaron de verdad, y con la fusion corregida por normalizacion de filas.

```text
dominio embedding-only        -0,006025  GANA GENERAL
dominio fusion estandarizada  -0,007827  GANA GENERAL
dominio fusion normalizada    +0,001475  GANA TWITTER
```

**El signo cambia segun como se mida.** Ningun gate pasa.

**Segunda anomalia declarada**: la fusion normalizada deberia quedarse cerca de V03 y cae a
0,578, treinta puntos por debajo. Es un defecto de implementacion que no he aislado.

Sin push, sin merge, sin rebase.


## 2026-09-29, P0 y arranque de P1 de la fase de dimension y pooling

- `experiments/p72_00_auditoria_implementacion.py`: audita lo existente. **6 de 6
  confirmaciones OK.** Hallazgo: el 200d que existe en disco no es el que produjo el resultado
  historico; los dos artefactos no entrenados quedan marcados y P1 arranca del valido.
- `experiments/p72_01_dimensiones.py`: entrena 100d y 300d con todo lo demas congelado.
- `experiments/p72_02_dimension_downstream.py`: downstream de las tres dimensiones.

Sin push, sin merge, sin rebase.


## 2026-09-30, P75: complementariedad de Word2Vec + TextCNN frente a V03, y fusion minima

- `experiments/p75_01_complementariedad_textcnn.py`: OOF pareado de los dos modelos sobre
  exactamente las mismas filas, con la particion canonica. El TextCNN usa `E*` como mediana de tres
  splits internos por clusters y reentrena sobre el `outer_train` completo durante `E*` epocas
  fijas. **La comprobacion de alineamiento pasa**: V03 recalculado desde la tabla pareada de 4449
  unidades da `0.7772114525` exacto.
- Resultado: **rescue 268, damage 813, net -545** en unidades tweet-etiqueta. Las tres etiquetas en
  la misma direccion. El CNN corrige **268 de los 757 errores de V03 (35,4 por ciento)**, pero el
  dano lo triplica. AUC del score del CNN sobre el error de V03: **HS 0,4872, TR 0,6045, AG
  0,4973**.
- Fusion con los tres alphas pre-registrados, todos negativos: 0,05 -> -0,0026005064; 0,10 ->
  -0,0052438938; 0,15 -> -0,0080173237. Gate NO PASA. **VEREDICTO: CASO 5 `CNN_PERJUDICA`**.
  **Finalista: ninguno.**
- **Incidencia metodologica, conservada**: P75-01 perdio nueve de doce slices por unir la tabla de
  anotaciones sin comprobar el tipo del identificador (parquet con `id` de texto frente a OOF con
  `id` entero; interseccion cero; `reindex` a NaN; slices vacias saltadas por umbral).
  `experiments/p75_02_slices_recuperadas.py` las recalcula desde el OOF pareado guardado, sin
  reentrenar, leyendo solo la particion `dev`.
- **Advertencia sobre tiempos**: la corrida esta contaminada por un proceso ajeno consumiendo CPU
  (carga media 50,8 sobre 32 nucleos). El pliegue 1 tardo 624 s frente a 24 s de un intento previo
  de la misma corrida. **Esos tiempos no se usan para ninguna comparacion.**
- **La segunda particion NO se ejecuto**, porque no hubo finalista. La transferibilidad queda **sin
  medir, no refutada**.
- Validacion independiente del Kaggle Specialist: corrigio tres razonamientos propios (el techo
  oraculo es casi vacuo; el AUC no es ~0,5 en todo; la justificacion "no se midio el modelo suelto"
  era falsa) y cinco imprecisiones mas. Registrado en `DECISIONS.md` D-258. **Ningun numero cambia.**

Sin push, sin merge, sin rebase.


## 2026-09-30, P76: calibracion nested de V03 y del TextCNN, y fusion controlada

- `experiments/p76_01_calibracion_nested.py`: calibracion nested con las mismas 5 folds externas de
  P75. `outer_train -> inner_train + inner_calibration -> calibrador -> outer_validation`, con la
  outer_validation **completamente aislada**. Solo dos familias, `raw` y Platt, para no abrir grid.
  Las predicciones de outer_validation se reutilizan de P75.
- La calibracion **funciona como operacion estadistica**: Brier de V03 0.125771 -> 0.122455, del CNN
  0.192150 -> 0.188485; pendientes de V03 1.31 -> 1.00 y del CNN 0.59 -> 0.85.
- **Y no transfiere nada.** Las ocho configuraciones (cuatro condiciones x dos alphas) son
  negativas. La mejor es **C, que calibra solo al CNN**, con `-0.0017743921` a alpha 0.05 y 2/5
  folds. Pero la mejora es **inercia, no complementariedad**: rescata menos (11 frente a 17) y dana
  menos (14 frente a 21).
- **Razon estructural**: el AUC del CNN sobre el error de V03 no cambia al calibrar (0.4872 ->
  0.4910 en HS, 0.6045 -> 0.6032 en TR, 0.4973 -> 0.4995 en AG). Una calibracion es monotona: no
  reordena, luego no crea informacion discriminativa.
- **Hallazgo no esperado**: calibrar a V03 lo **empeora** 0.0086 (0.7772114525 -> 0.7685825353),
  porque los umbrales congelados se eligieron sobre sus probabilidades crudas. **Su sobreconfianza
  es portante**: es parte de la definicion de su punto de operacion. Por eso B y D estan condenadas
  de antemano y la unica comparacion que informa es A frente a C.
- **VEREDICTO: CASO B `CALIBRACION_MEJORA_COMPLEMENTARIEDAD_PERO_NO_F1`.** Gate superado por
  ninguna. **Finalista: ninguno.** El **bootstrap no se ejecuto** porque ninguna configuracion
  supero el gate preliminar (delta >= +0.003 y rescue > damage): es lo pre-registrado.
- Siete figuras en `figuras/p76/`.
- La segunda particion **sigue sin ejecutarse**: sigue sin haber finalista.

Sin push, sin merge, sin rebase.
- Verificacion de figuras de P76, registrada en `figuras/p76/VERIFICACION.md`: las dos reliability
  se **regeneraron** desde `reports/p75_paired_oof.csv` mas los parametros de Platt del JSON y se
  compararon con los PNG almacenados **pixel a pixel**, de modo que la comparacion incluye el texto
  del titulo. **Resultado: las dos son identicas, luego NO havia intercambio V03/CNN.** La duda
  era legitima porque el titulo no se puede leer de un PNG, pero era infundada. Ademas se
  recalcularon los cuatro miembros desde el JSON y reproducen su macro-F1 al decimal.

## 2026-09-30, P77: operador selectivo sobre el punto de operacion congelado de V03

- Pre-registro escrito y hasheado ANTES de ejecutar, sha256
  `024b0288af5746baaa1b4acfbdade3c94fb2fbcf29601f41447c893706e7b0ca`, en commit separado.
- Dos revisiones obligatorias (Kaggle Specialist y ML/DL) incorporadas antes de correr: las
  magnitudes de evidencia pasan a escala logit para no comparar margenes entre escalas distintas, y
  los cortes del operador determinista pasan a ser una regla de cuantil calculada solo dentro de
  `outer_train` y solo sobre las unidades discrepantes. Se anadieron dos condiciones mas estrictas
  al gate de finalista y el bootstrap paso a ser incondicional. Se **rechazo** proponer confirmar un
  finalista en el holdout, porque el encargo lo prohibe.
- Resultados: A V03 0.7772114525 | B CNN sola 0.5838755350 | C nivel 1 0.7579728990 | D nivel 2
  0.7259394272 | E gate 0.7215155047. **0 de 5 folds positivos en los tres operadores.**
- El resultado es **monotono en el numero de conmutaciones**: 154 -> -0.0192385535, 393 ->
  -0.0512720253, 543 -> -0.0556959478. Bootstrap incondicional con los tres IC95 excluyendo cero.
- **El gate si discrimina**: AUC en outer de 0.6528, 0.7378, 0.7160, 0.6104 y 0.7374; interno medio
  0.7016. Selecciona 191 de 268 rescates y descarta 461 de 813 danos. Precision del selector
  **0.3517** frente a una base de **0.2479**.
- **Y no alcanza**: el tipo de cambio sube de 0.3296 a 0.5426 y sigue muy por debajo de lo que el
  macro-F1 necesita. El techo no lo fija la calidad del selector sino las 1081 discrepancias de las
  que solo 268 son rescates.
- **La senal del gate es sobre todo de V03**: al conmutar, `m_v` cambia -0.6100 y `m_c` solo
  +0.1611. Es una buena regla de prudencia, no una explotacion de la complementariedad.
- **TR daña mas que HS y AG**, lo contrario de lo que sugeria el AUC 0.6045 de P75: predecir que
  V03 falla no es lo mismo que acertar donde V03 falla.
- **VEREDICTO: CASO 1 `NINGUN_OPERADOR_TRANSFIERE`. Finalista: ninguno. La segunda particion NO se
  ejecuta.** Ocho figuras en `figuras/p77/`, doce slices diagnosticas.
- Tres paradas de seguridad propias Saltaron y pararon la corrida: un target degenerado por comparar
  probabilidad con etiqueta en vez de decision, un `column_stack` que daba nueve features, y el
  signo del bootstrap invertido. Ninguna produjo un numero publicado.

Sin push, sin merge, sin rebase.

## 2026-09-30, auditoria de Word2Vec y TextCNN. Fases 0 a 3 del plan, y P78

Plan de ejecucion: `/home/carlos/.opencode/plan/toxic_word2vec_textcnn_auditoria_y_reparacion.md`.
Se audita la cadena P71 a P77 y se corrige. V03 congelado, W2V100 congelado, test holdout
leaderboard y submission en cero.

### Fase 0: entorno, congelado y metricas
- Hardware re-verificado en vivo: venv 3.14.7, torch 2.14.0+rocm7.2, RX 9070 XT, 32 hilos, 31 GB.
- `w2v_twitter_100d.npz` sha256 `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`,
  **coincide** con el congelado. Rama `proyecto-kaggle-toxic-2026-p67`.
- El grafo `kaggle-toxic-2026-p67` permitio recuperar las firmas de las 40 funciones implicadas sin
  releer 4995 lineas. Se uso solo para localizacion; la verdad es el codigo fuente.

### Fase 3, P79: la auditoria del entrenador Word2Vec. Commit 710a8a8
- **CONFIRMADO**: `SkipGram.forward` de p71_04:211-216 promedia el termino negativo sobre
  `batch x K` en vez de sumarlo. Ratio **1/K exacto** para K=1,5,10,20 con residuo cero frente a
  autograd. Sobre pares reales del artefacto congelado, el ratio de la norma del gradiente de entrada
  es **0.080338** y la loss 1.084121 frente a 9.785559 de la referencia. El codigo no es un objetivo
  distinto: es la referencia con la repulsion en 1/K.
- **NO CONFIRMADO**: la hipotesis del scheduler con semillas desalineadas **era falsa**. Ambas ramas
  usan `default_rng(SEMILLA + ep)`. El desvio real es de +0.215 por ciento en el total de lotes. Esto
  **corrige una afirmacion del plan de auditoria**.
- **Defecto real sin alcance en el artefacto**: la matriz de salida se inicializa a cero, luego el
  gradiente de la entrada es exactamente cero en el paso 1. Con Adam o SGD plano escapa en el paso 2:
  se pierde un paso de 1398.
- Tres bugs del propio script P79, corregidos antes de aceptar el numero. El grave: `gradiente_analitico`
  usaba `unsqueeze(1)` sobre un tensor de forma (B, K) y producia (B, B, K) en vez de (B, K, dim).
  Con K igual a dim pasaba el filtro de shape y **corrompia el gradiente en silencio**. Ademas se
  comparaba el gradiente analitico por ocurrencia con el de autograd ya acumulado, y se comparaba el
  gradiente completo de la salida contra su sola parte positiva.
- No se reentrena el Word2Vec y no se toca el artefacto congelado.

### Fases 1 y 2, P78: el control anidado con la normalizacion alineada
Commits dabf07d (pre-registro), e5045e7, db4e5d2, 47bc216. Revision de los cuatro especialistas.

- **P1 = C - D = +0.024893**, sd entre semillas 0.021588, 5/5 pliegues, IC95 de bootstrap
  (0.007902, 0.040645). Criterio pre-registado: **rama 1**.
- **Pero el gate es fragil**: el bootstrap propaga cero variabilidad de reentrenamiento, y el delta
  por semilla va de +0.0428 a **+0.0009**. El IC t con n=3 contiene el cero.
- **El efecto de la normalizacion es mas grande que el del embedding**: P4 = +0.056969 con
  desviacion de 0.001272, y es el 65 por ciento de la ganancia total C - B de +0.088397.
- La cobertura sube de **68.64 a 88.73 por ciento** y las filas totalmente muertas de **70 a 1**. El
  OOV era 68.3 por ciento ruido de caja, no falta de corpus.
- El termino de interaccion `P1 - P2 = P3 - P4 = +0.001586` admite aditividad aproximada en la
  media. **No** se afirma ortogonalidad: la sd de P3 es doce veces la de P4.
- El control pareado se verifico por fuera: las normas de fila coinciden con diferencia maxima
  4.77e-07, y el coseno intra-brazo es +0.1598 en Word2Vec frente a -0.0002 en la aleatoria.
- **Seccion de corpus CERRADA**: el techo de cobertura es de 88.73 a cerca de 91.3 por ciento, y la
  procedencia no lo permite. Lo que procede es `MIN_COUNT` de 3 a 2, que no necesita corpus nuevo.
- **P73 y P74 reetiquetados como exploratorios, protocolo contaminado.** Sus numeros se conservan.

### Las tres paradas, y por que importan
1. Tras 6 unidades: asercion 10 redactada y no observada; asercion 6 re-derivando los splits en vez
   de observarlos; `ic95_promediado` invalido que ademas **decidia una rama**; escritura no atomica.
2. Tras 15 unidades: **etiquetas de semilla falsas** en el JSON, por dividir por `N_INTERNOS` cuando
   `epoca_estrella` usa `7*j`; y el diagnostico de canales muertos sin contar las 70 filas con cero
   tokens cubiertos, que es el mecanismo real.
3. Tras 60 unidades y toda la metrica: `savefig` deducia el formato de la extension y
   `nombre.png.tmp_p78` no es un formato soportado. **La escritura atomica funciono**: no quedo ningun
   artefacto final a medias.

Y un bug que habria costado 7 horas: el smoke test encontro que la asercion 6 nueva exigia
`N_INTERNOS*j` y el protocolo real usa `7*j`, luego habria abortado despues de entrenar todo.

Las seis unidades comunes entre la primera y la ultima corrida son **bit a bit identicas**: las tres
rondas de correccion fueron de instrumentacion y de informe, no de calculo.

### Verificacion
Diez aserciones del pre-registro, todas en verde; la 6 sobre 180 llamadas reales capturadas
parcheando `p75_01.partir_interna`. ASCII byte a byte en dos pasadas. El manifiesto verifica con
`sha256sum -c` las diez piezas. El JSON se reproduce entero desde el CSV pareado y el OOF canonico
con calculo independiente, sin discrepancias.

```text
gap del mejor brazo frente a V03: 0.117450, en METRICA LOCAL
V03 remains the best verified leaderboard result.
```

Sin push, sin merge, sin rebase.

### Ingesta en Graphiti (2026-09-30)
- Grupo: `kaggle-toxic-2026-puj`. **7 episodios + 56 tripletas**, todos en ASCII puro.
- El contenido se construyo en un fichero maestro y se valido byte a byte ANTES de enviarlo,
  con el validador `validar_graphiti.py` que aborta si cualquier byte supera 127, si aparece
  comilla tipografica, guion largo, zero-width o caracter de otro alfabeto. El validador
  detecto dos veces que un nombre mio llevaba un caracter no ASCII y se corrigio antes de
  enviar.
- La ingesta se hizo **de forma secuencial** porque en paralelo Graphiti creaba nodos
  duplicados del mismo nombre: tres UUID distintos para el mismo concepto. Ademas los
  nombres conceptuales llevan prefijo `tema_` porque sin el, `compliance_p78` se resolvia
  dentro de `P78` y producia un bucle.
- Grafo final: **sin duplicados**. Un nodo por concepto, con las aristas tipadas.
- Tipos de arista: OBSERVED, METHODOLOGICAL_ISSUE, LIMITATION, DECISION, CORRECTS,
  REETIQUETED_AS, RESPECTS, VERIFIED, CONFIRMED, NOT_CONFIRMED, NOT_SUPERSEDING,
  EXPLOITED_AS, HAS, SUPPORTED_BY, OBSERVED.
- Cada `fact` lleva su provenance: fichero, commit y fecha. Eso hace que cualquier hecho
  del grafo sea rastreable hasta su linea de codigo o su hash, sin depender de la memoria
  de la sesion.

## 2026-09-30, P80: auditoria del entrenador Word2Vec. PARADA EN A12, sin informe completo

### Lo que se ejecuto

Comando: `/home/carlos/venv/bin/python -B experiments/p80_01_auditoria_w2v.py`. Codigo de salida **1**.
Duracion: **menos de 2 segundos**, de 21:37:40 a 21:37:41 UTC. Sin entrenamiento, sin GPU, sin red.

Pre-registro verificado antes de la primera lectura: `440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f`, commit `8726c9f`. Sin cambios.
Artefacto congelado verificado ANTES y DESPUES: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`, 29478868 bytes. Intacto.

El script **no se edito**. La parada la escribio el propio script, obediente al pre-registro.

### Las once afirmaciones que llegaron a ejecutarse

```text
A1  CONFIRMADO   H1  la loss promedia el termino negativo por K en vez de sumarlo
A2  CONFIRMADO   H1  el factor es exactamente 1/K
A3  CONFIRMADO   H1  p72_01 reutiliza la clase SkipGram de p71_04 sin tocarla
A4  CONFIRMADO   H1  gradiente, maximo residuo 3.469446951953614e-17
A5  CONFIRMADO   H1  la afirmacion negativa, con contraejemplos numericos
A6  CONFIRMADO   H2  los dos iteradores del scheduler, leidos por AST
A7  CONFIRMADO   H2  el desfase de semillas, uno por epoca
A8  CONFIRMADO   H2  la comparacion con P79
A9  CONFIRMADO   H3  identidad con el signo correcto, residuo 1.1102230246251565e-16
A10 CONFIRMADO   H3  el control, separacion minima 2.771907574551194
A11 CONFIRMADO   H4  el congelado tiene solo W y palabras
A12 STOP         H4  se dispara la parada de la seccion 7, clausula 5
```

Siete afirmaciones mas no se ejecutaron: A13, A14, A15, A16, A17, A18 y A19.

Resumen: **11 CONFIRMADO, 0 REFUTADO, 0 NO APLICABLE, 1 STOP, 7 sin ejecutar.**

### Donde paro y por que

La parada la manda el pre-registro, seccion 7, clausula 5: si A12 encuentra una fuente con la tabla de salida, se para antes de publicar porque H4 cambia de contenido. El script obedecio y escribio solo `reports/p80_auditoria_w2v.json` con el motivo y el detalle minimo. **No se escribio** `reports/p80_auditoria_w2v.md` **ni ninguna figura**: `figuras/p80/` no llega a existir.

Los siete candidatos de A12, y por que ninguno es una tabla de salida:

```text
4 npz   w2v_twitter.npz        W.npy  (77657, 200)
        w2v_twitter_300d.npz   W.npy  (77657, 300)
        w2v_twitter_lrfix.npz  W.npy  (77657, 200)
        w2v_twitter_sgd.npz    W.npy  (77657, 200)
3 json  p50_gap_and_oracle.json  matriz_observable/forma  (4449, 53)
        p59_H_nmf_gap.json       medicion_de_equivalencia/matriz/forma  (4449, 21)
        p74_null_replicado.json  auditoria_word2vec/shape  (77658, 100)
```

Los cuatro npz son las tablas de **entrada** `W` de **otras** corridas, y los cinco npz de `twitter_domain` tienen exactamente los mismos dos miembros, `W.npy` y `palabras.npy`. Ninguno guarda una tabla de salida. Los tres json son **claves de forma**: su valor es una lista de dos enteros que describe una forma, no un arreglo de pesos.

El problema esta en el detector de A12, que empareja cualquier `.npz` con primera dimension `77657` **sin mirar la dimension** y cuenta como candidata cualquier clave de json que se llame `shape` o `forma`. Con esos dos criterios, las tablas de entrada de las corridas 200d y 300d y tres metadatos de forma dan falso positivo. **No se corrigio nada**: vuelve a ml-coder.

### H2 en detalle, que es donde P79 fallo

P79 clasifico juntos los dos bucles del scheduler de `p72_01` y declaro que no habia desfase. Si lo hay, y es de una epoca. Con `SEMILLA = 20261016` y `EPOCAS = 5` en `p71_04:62-64`:

```text
p72_01:92   for ep in range(m.EPOCAS):       ep = 0,1,2,3,4
p72_01:93       r2 = np.random.default_rng(m.SEMILLA + ep)   semillas 20261016 .. 20261020
p72_01:95       planes.append(c)                              el plan se construye con esas

p72_01:99   for ep, _ in enumerate(planes, start=1):  ep = 1,2,3,4,5
p72_01:101      r2 = np.random.default_rng(m.SEMILLA + ep)   semillas 20261017 .. 20261021
p72_01:102      centros, contextos = cx["pares_de_epoca"](...)  el entrenamiento dibuja otros pares
p72_01:103      o = r2.permutation(len(centros))               y ademas permuta
```

El bucle del plan y el del entrenamiento usan los dos una variable llamada `ep`, pero son bucles distintos con base distinta: `range(5)` empieza en 0 y `enumerate(..., start=1)` empieza en 1. Resultado: la semilla `20261016` se planifica y nunca se entrena, y la `20261021` se entrena y nunca se planifica. El total de lotes del scheduler sale del plan, luego el curriculum de `lr` esta calculado sobre una epoca que no es la que se ejecuta.

A6, A7 y A8 quedaron **CONFIRMADO**, y el recorrido por AST con mapa de padres es lo que separa los dos bucles. P79 no los separo.

### Estado de los siete hallazgos

```text
H1  CONFIRMADO     A1 A2 A3 A4 A5
H2  CONFIRMADO     A6 A7 A8      el veredicto de P79 era incorrecto
H3  CONFIRMADO     A9 A10        P79 comparaba pos + neg contra -(pos + neg)
H4  SIN VEREDICTO  A11 CONFIRMADO, A12 STOP, A13 sin ejecutar
H5  SIN VEREDICTO  A14 sin ejecutar
H6  SIN VEREDICTO  A15 A16 sin ejecutar
H7  SIN VEREDICTO  A17 A18 A19 sin ejecutar
```

Ningun hallazgo queda REFUTADO: A5 era la unica afirmacion marcada REFUTAR y tambien salio CONFIRMADO.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P79 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar
modelos/ sin escrituras   el congelado se abrio en solo lectura, con huella comprobada
escrituras: reports/p80_auditoria_w2v.json, nada mas
```

El holdout se lee con una expresion regular sobre el texto y solo de tres claves: `n_tweets_dev`, `n_tweets_holdout` y `hash_ids_csv`. La lista de identificadores nunca llega a existir como objeto en memoria.

### Estado

P80 **queda inconclusa**: sin informe completo, sin markdown y sin figuras. Se devuelve a ml-coder el problema del detector de A12, que bloquea H4 a H7.

```text
V03 remains the best verified leaderboard result.
```

## 2026-09-30, P80 segunda corrida: las 19 afirmaciones pasan, y el script revienta al escribir el markdown

### Lo que se ejecuto

Comando: `/home/carlos/venv/bin/python -B experiments/p80_01_auditoria_w2v.py`, version
`d752b4b1342c8fba841d02bff3308e93834c94c2394f671ba48f13ab6f117e42`, 5983 lineas. Codigo de salida **1**.

Duracion: **2 s**, de 22:00:19 a 22:00:21 UTC. No fueron minutos: es una auditoria de enumeracion,
recorrido de AST y sondas diminutas, sin entrenamiento. El cronometro del propio script suma 1.803 s
y el reparto es test1 0.000, test2 0.018, test3 0.297, test4 0.550, test5 0.063, test6 0.171,
test7 0.056 y figuras 0.648.

Pre-registro `440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f` verificado, sin cambios.
Artefacto congelado verificado ANTES y DESPUES: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`,
29478868 bytes, intacto.

**El script no se edito.**

### Las 19 afirmaciones: todas ejecutadas, todas en verde

```text
A1 A2 A3 A4 A5 CONFIRMADO   H1  A6 A7 A8 CONFIRMADO   H2  A9 A10 CONFIRMADO   H3
A11 A12 CONFIRMADO          H4  A13 NO APLICABLE POR CONSTRUCCION
A14 CONFIRMADO              H5  A15 A16 CONFIRMADO    H6  A17 A18 A19 CONFIRMADO  H7

18 CONFIRMADO   0 REFUTADO   1 NO APLICABLE POR CONSTRUCCION   0 STOP
ninguna afirmacion sin ejecutar
```

Los siete hallazgos quedan **CONFIRMADO**: H1, H2, H3, H4, H5, H6 y H7.

### Donde fallo, y no lo arreglo

Fallo al escribir el **markdown**, con el JSON ya escrito en temporal y las figuras ya en disco.
Traceback literal:

```text
Traceback (most recent call last):
  File "/home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj/experiments/p80_01_auditoria_w2v.py", line 5976, in <module>
    main()
    ~~~~^^
  File "/home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj/experiments/p80_01_auditoria_w2v.py", line 5928, in main
    md_temporal.write_text("\n".join(escribir_markdown(afirmaciones, informe)) + "\n",
                                     ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj/experiments/p80_01_auditoria_w2v.py", line 4704, in escribir_markdown
    f"{a12['medida']['cobertura']['claves_json_recorridas']} claves recorridas.",
       ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^~~~~~~~~~~
KeyError: 'claves_json_recorridas'
```

El diagnostico es una sola clave vieja. La correccion de A12 **quito** `claves_json_recorridas` de
`medida.cobertura` y `escribir_markdown` la sigue leyendo. Las otras tres claves que esa funcion lee
de A12, `npz_inspeccionados`, `bytes_leidos_en_total` y `json_inspeccionados`, mas `n_candidatos`,
**si existen**: el fallo es exactamente una clave, no un cambio de esquema entero. Vuelve a
ml-coder. **No he tocado el script.**

### Lo que queda en disco

```text
reports/p80_auditoria_w2v.json            1936 B    el JSON de PARADA de la corrida ANTERIOR, sin tocar
reports/p80_auditoria_w2v.json.tmp_p80  174103 B    el informe COMPLETO, huerfano, con las 19 afirmaciones
reports/p80_auditoria_w2v.md             NO EXISTE
figuras/p80/p80_ratio_negativo_y_total.png   564ad933955351f8
figuras/p80/p80_lr_real_frente_a_ideal.png   1becec0d807e2723
figuras/p80/p80_sonda_de_relleno.png         61b8deaadecd071a
```

La escritura atomica **funciono**: el renombrado final no llego a ocurrir, asi que el informe
completo no se publico con apariencia de validado. El coste es que los resultados buenos quedan en un
temporal huerfano y el `.json` oficial sigue diciendo lo de la parada anterior. **No he renombrado
nada**: publicar un informe por la mano seria exactamente el atajo que el pre-registro prohibe.

El JSON completo tampoco lleva la asercion ASCII final ni las firmas de los PNG, porque esa parte se
escribe despues del markdown, que es justo donde revienta.

### H2, el desfase de semillas, con las lineas del fuente

`SEMILLA = 20261016` y `EPOCAS = 5` en `p71_04:62-64`. Leido por AST con mapa de padres, sin buscar
cadenas ni ventanas de lineas:

```text
p72_01:92   for ep in range(m.EPOCAS):          ep = 0..4
p72_01:93       r2 = np.random.default_rng(m.SEMILLA + ep)   rol=plan
                conjunto = {20261016, 20261017, 20261018, 20261019, 20261020}
p72_01:99   for ep, _ in enumerate(planes, start=1):  ep = 1..5
p72_01:101      r2 = np.random.default_rng(m.SEMILLA + ep)   rol=entrenamiento
                conjunto = {20261017, 20261018, 20261019, 20261020, 20261021}
```

`distintos = true`. Solo en el plan: `20261016`. Solo en el entrenamiento: `20261021`. El desfase es
de **una epoca**, y `total_lotes` sale del plan, luego el curriculum de `lr` se calibra sobre un numero
de lotes que no es el que se recorre. Con el recorte declarado de 2000 lineas: 1395 lotes planados
frente a 1398 recorridos, desviacion relativa 0.002150537634408602. El `lr` real termina en 0.004578
con 0 pasos en el piso, mientras que con el total real terminaria en 0.001 con 3 pasos en el piso.

Por que P79 se equivoco: su heuristica metio los dos bucles en el mismo grupo, y su booleano salia
False no porque los rangos coincidieran sino porque solo habia un grupo. Su propia lista tenia 6
elementos para `EPOCAS = 5`, y eso ya lo contradicia.

### H4, la matriz de salida no se puede reconstruir

A11: el congelado tiene **exactamente** dos claves, `W` y `palabras`, y una sola matriz, que es la de
entrada (`modelo.entrada.weight` segun `p72_01:124-127`). Vocabulario 77657, dim 100, float32, todos
los valores finitos, 77657 palabras unicas, 0 filas nulas. Nota: `p72_01:129` anuncia
`parametros = vocabulario * dim * 2` porque el modelo tiene dos tablas; que las dos se guarden es
otra cosa, y no se guardan.

A12: **0 candidatos**, con cobertura declarada. 19 `.npz` inspeccionados por cabecera sin
descomprimir, 412987 bytes leidos, 320 `.json` de `reports/` recorridos. El criterio ahora es forma
exacta `(77657, 100)`, porque en SkipGram las dos tablas son `nn.Embedding(n_vocab, dim)` con el
**mismo** dim. Todo lo que coincide en la primera dimension y no en la segunda se descarta con su
motivo, registrado. Ese era exactamente el defecto de la corrida anterior.

A13 queda **NO APLICABLE POR CONSTRUCCION**: la tabla de salida no existe en ninguna parte, no se
sustituye en silencio por `W` y **P80 no publica ninguna cifra sobre el artefacto**. Por tanto
`loss_codigo = 1.08412074`, `loss_referencia = 9.78555861` y las normas de gradiente 0.02042704 y
0.25426385 de P79 describen un par `(E, S)` con `E = S = W` que nunca existio durante el
entrenamiento: describen un modelo ficticio y **no son evidencia del estado final**. Se retiran sin
sustituto. Lo unico admisible sobre el artefacto es que lo produjo `p72_01` usando la clase `SkipGram`
de `p71_04` sin tocarla.

### H5, la fuga de early stopping en los BiGRU

A14 CONFIRMADO en los dos ficheros. En `v9_04:111` la funcion `entrenar_pliegue` recibe `ids_va` y
`y_va` como parametros de validacion e `ids_tr`, `y_tr` como de entrenamiento; el estado se restaura
por la perdida `perdida_va` (linea 126) y las probabilidades reportadas salen de `proba`, que depende
de `ids_va`. **No hay particion interna**: la metrica que decide cuando restaurar pesos es la del
pliegue externo que despues se puntua. Es la fuga que condemned P73 y P74.

### H6, el enmascarado tardio del padding

A15 CONFIRMADO en los dos. Cero `pack_padded_sequence`. En `v9_04:104-105` y en `v10_11:170-171` el
patron es identico: `salida, _ = self.gru(embebido)` seguido de
`suma = (salida * mascara).sum(dim=1)`. La GRU corre **antes** de enmascarar, luego los pasos de
relleno se consumen en la recurrencia. Con `padding_idx = 0` el vector de relleno es el cero, asi que
el efecto no es de explosion sino de **arrastre**: los pasos de relleno se acumulan en los estados y,
en la direccion inversa, se mezclan en las posiciones reales anteriores al relleno.

A16 CONFIRMADO con **sonda ejecutada**: matriz sintetica de 51 por 200, fila 0 a cero, longitudes
reales 12, 7 y 4 contra longitud de tanda 30, o sea 18 pasos de relleno anadidos. Distancia relativa
L2 **0.08399328135684547** con norma del vector real 0.9973462820053101. Reproducible: la segunda
pasada da el mismo valor. El defecto estructural pesa, y pesa en la direccion anunciada.

### H7, DL-03 uso las 1731 filas sin excluir el holdout

A17 CONFIRMADO. `n_validacion` por pliegue en DL-03: 347, 346, 346, 346, 346. Suma **1731**. Y
`1483 + 248 = 1731`, con `n_tweets_dev = 1483` y `n_tweets_holdout = 248`. El pool de validacion
cruzada de la fase DL son las 1731 filas etiquetadas completas, luego **el holdout v2 estaba dentro
del pool**. La via estructural lo apoya: `11_dl_desde_carlo.py:446` lee
`train = data_module.read_train()` y el bucle de CV de la linea 1362 itera sobre `pliegues`. Del
holdout solo se leyeron tres claves, con expresion regular sobre el texto y sin deserializar el
documento, para que la lista de identificadores no llegue a existir como objeto.

A18 CONFIRMADO: DL-01, DL-02 y DL-03 declaran los mismos `n_validacion` por pliegue. Y el OOF de
DL-01, 0.7948542132994776, frente al corrected CV de V03, 0.7772114525165721, **no es una mejora**: es
una diferencia aritmetica entre dos cifras calculadas sobre distinto conjunto de filas y con
distinto protocolo, contaminada por el holdout y por una seleccion de epoca no anidada. No se etiqueta
como delta, mejora, ventaja ni ganancia.

A19 CONFIRMADO: `11_dl_desde_cero.py:1140`, `entrenar_pliegue` restaura los pesos de la epoca con
mejor `f1_val`, es decir de la validacion del propio pliegue que despues se puntua, sin particion
interna. La fuga es la misma de H5, en la fase DL.

### Dos salvedades que el informe del script no lleva

Las dejo aqui porque un lector las necesita y el JSON no las publica.

**Primera, la regla 3 de A19 es una apuesta por nombre, no una medicion.** El desempate por nombre
puede clasificar como validacion un argumento que llega como parametro sin vinculacion local, por
ejemplo `x_val`. Es **latente**: no se activa en ninguno de los tres ficheros reales, y ahi manda el
vinculo local, que es lo correcto. Pero el mecanismo existe y depende del nombre, luego no debe
leerse como una prueba.

**Segunda, `asignaciones_del_llamante` usa `setdefault`.** Si un nombre se asigna dos veces, gana la
primera y **no se marca ambiguedad**. Se comprobo que eso falla hacia el lado conservador, es decir
devuelve REFUTADO y no sobre-detecta, que es el lado correcto para una afirmacion de auditoria. Aun
asi, un nombre asignado dos veces produce una lectura parcial sin que el informe lo advierta.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P79 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar
modelos/ sin escrituras   el congelado se abrio en solo lectura, con huella comprobada
```

### Estado

Los siete hallazgos estan **CONFIRMADO** y las diecinueve afirmaciones se midieron, pero **el
informe no se publico**: falta el markdown y el `.json` oficial sigue siendo el de la parada
anterior. Los resultados buenos estan en `reports/p80_auditoria_w2v.json.tmp_p80`, que se commitea
como evidencia huerfana. Vuelve a ml-coder por la clave `claves_json_recorridas`.

```text
V03 remains the best verified leaderboard result.
```

## 2026-09-30, P80 tercera corrida: informe publicado. 19 de 19 y 7 de 7

### Lo que se ejecuto

Comando: `/home/carlos/venv/bin/python -B experiments/p80_01_auditoria_w2v.py`, version
`201dcccbdd5e7169127b9d0a6772092d40d9edf74b002d063c5d2da4cde7482f`, 6361 lineas, 70 de 70 en
`--humo`. **Sin fallos, sin traceback, sin STOP.** Codigo de salida **0**.

Duracion: **2 s**, de 22:11:18 a 22:11:20 UTC. Cronometro interno 1.803 s: test1 0.000, test2 0.019,
test3 0.294, test4 0.555, test5 0.061, test6 0.171, test7 0.059, figuras 0.627 e informe 0.017.

Pre-registro `440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f` verificado y sin cambios.
Artefacto congelado `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`, 29478868 bytes,
verificado ANTES de la primera lectura y DESPUES de la ultima, intacto. **El script no se edito.**

### Publicacion

```text
reports/p80_auditoria_w2v.json  174656 B  sha256 27efa9a37a755464b121581110a333831bfdf5f3acf63f092f8086503f6ad8e6
reports/p80_auditoria_w2v.md     11657 B  sha256 f8bab0dd93dd6eb299a9b04d89083ef380c882d1d382bfd396d7e47c66ce4e8f
figuras/p80/  3 PNG, firmas verificadas
```

Las dos salidas se publicaron por **renombrado atomico**, y la asercion ASCII del propio script paso
byte a byte sobre el script, el json temporal y el markdown temporal. El JSON publicado lleva ya la
asercion ASCII final y la firma de los tres PNG, que en la segunda corrida no llegaban a escribirse.

**El `.tmp_p80` viejo ya no existe**, y no lo borre yo. El script reutiliza la misma ruta temporal, la
vacio con el JSON de esta corrida y la renombro sobre el `.json`. El contenido de la segunda corrida
sigue en la historia de git, en el commit `c317794`, con sha256
`890bbb1037b3034a893e88634b5db261a2d9f6acad278722d32c6a00007c834f`. No queda ningun temporal huerfano.

### Las diecinueve afirmaciones y los siete hallazgos

```text
A1  CONFIRMADO  A2  CONFIRMADO  A3  CONFIRMADO  A4  CONFIRMADO  A5  CONFIRMADO   H1
A6  CONFIRMADO  A7  CONFIRMADO  A8  CONFIRMADO                                 H2
A9  CONFIRMADO  A10 CONFIRMADO                                                 H3
A11 CONFIRMADO  A12 CONFIRMADO  A13 NO APLICABLE POR CONSTRUCCION              H4
A14 CONFIRMADO                                                         H5
A15 CONFIRMADO  A16 CONFIRMADO                                                 H6
A17 CONFIRMADO  A18 CONFIRMADO  A19 CONFIRMADO                                 H7

18 CONFIRMADO   0 REFUTADO   1 NO APLICABLE POR CONSTRUCCION   0 STOP   0 sin ejecutar
H1 H2 H3 H4 H5 H6 H7 = CONFIRMADO
```

### El markdown no sale vacio, y eso se ha comprobado

El bug anterior hacia tablas de afirmaciones **vacias y en silencio**, asi que no me basta con que el
fichero exista. Cuenta de filas de tabla en `reports/p80_auditoria_w2v.md`:

```text
filas de afirmaciones que empiezan por | A   = 19
filas de hallazgos que empiezan por | H       =  7
```

Y estan **pobladas**, no husk: cada fila de afirmacion lleva id, hallazgo, naturaleza, tolerancia y
veredicto, y cada fila de hallazgo lleva estado y lista de afirmaciones. Ejemplos literales:

```text
| A5  | H1 | REFUTAR | no aplica: es una correccion de una asercion indebida, no una medicion | **CONFIRMADO** |
| A13 | H4 | NO APLICABLE POR CONSTRUCCION | ... | **NO APLICABLE POR CONSTRUCCION** |
| H6  | CONFIRMADO | A15, A16 |
```

Aparece **una sola** vez el marcador `no disponible`, en la linea de A12, y es legitimo: el
arreglo de A12 quito `claves_json_recorridas` de `medida.cobertura` y la vista tolerante del
redactor devuelve `no disponible` ante una clave ausente en vez de reventar. La frase queda
`Ficheros JSON de reports/: 320, con no disponible claves recorridas`. Ninguna cifra real se vacio.

### Reproducibilidad: las diecinueve medidas son identicas a la segunda corrida

Comparacion del JSON de esta corrida contra el de la segunda, sacado de `c317794`, clave por clave:

```text
afirmaciones cuya medida difiere:  0 de 19
afirmaciones cuyo veredicto difiere: 0 de 19
veredictos_por_hallazgo identico: True
```

Las cifras que sostienen los hallazgos, las mismas:

```text
A7  solo_en_el_plan [20261016]   solo_en_el_entrenamiento [20261021]
A8  total_lotes_plan 1395   total_lotes_recorridos 1398
A12 n_candidatos 0
A16 distancia_relativa_l2 0.08399328135684547
A17 suma 1731   identidad_dev_mas_holdout se cumple True
```

La auditoria es determinista. Que dos corridas den medidas bit a bit identicas con dos versiones
distintas del script es la mejor evidencia de que el arreglo del redactor toco solo la instrumentacion
y no el calculo.

### Lo que queda establecido

**H2 CONFIRMADO.** `p72_01:92-95` planifica con `range(m.EPOCAS)` y `SEMILLA + ep`, o sea 20261016 a
20261020; `p72_01:99-103` entrena con `enumerate(planes, start=1)` y `SEMILLA + ep`, o sea 20261017 a
20261021. Los dos bucles se llaman `ep` y por eso P79 los clasifico juntos. Solo en el plan 20261016,
solo en el entrenamiento 20261021. 1395 lotes planados frente a 1398 recorridos. El veredicto de P79
era incorrecto.

**H4 CONFIRMADO.** El congelado solo guarda `W`, la tabla de entrada, y una sola matriz. A12 da 0
candidatos sobre 19 `.npz` por cabecera y 320 `.json`, con criterio de forma exacta `(77657, 100)`.
A13 NO APLICABLE POR CONSTRUCCION: la tabla de salida no existe en ninguna parte, **no se sustituye en
silencio por `W` y P80 no publica ninguna cifra sobre el artefacto**. Las cifras absolutas de P79
describen un par `(E, S)` con `E = S = W` que nunca existio y **no son evidencia del estado final**.

**H5 CONFIRMADO** en `v9_04` y `v10_11`: la metrica que decide cuando restaurar pesos es la del pliegue
externo que despues se puntua, sin particion interna. **H6 CONFIRMADO**: la GRU corre antes de
enmascarar (`v9_04:104-105`, `v10_11:170-171`), el efecto es de arrastre y no de explosion, y la sonda
da 0.08399328135684547. **H7 CONFIRMADO**: DL-03 entreno sobre 1731 filas etiquetadas completas, y
1483 + 248 = 1731, luego **el holdout v2 estaba dentro del pool** de la validacion cruzada.

### Tres salvedades que el informe del script no lleva

Se registran aqui porque un lector las necesita y el JSON no las publica. Las tres son **limitaciones
del alcance**, no fallos de medicion.

**Primera, la regla 3 de A19 es una apuesta por nombre, no una medicion.** El desempate por nombre
puede clasificar como validacion un argumento que llega como parametro sin vinculacion local, por
ejemplo `x_val`. Es **latente**: no se activa en ninguno de los tres ficheros reales, y ahi manda el
vinculo local, que es lo correcto. Pero el mecanismo existe y depende del nombre.

**Segunda, `asignaciones_del_llamante` usa `setdefault`.** Si un nombre se asigna dos veces, gana la
primera y **no se marca ambiguedad**. Se comprobo que eso falla hacia el lado conservador, es decir
devuelve REFUTADO y no sobre-detecta, que es el lado correcto para una afirmacion de auditoria. Aun
asi la lectura queda parcial sin que el informe lo advierta.

**Tercera, la sonda de A16 no mide los BiGRU reales.** La sonda corre sobre una red de prueba
sintetica de 51 filas por 200 columnas, con la fila 0 a cero y semilla fija, no sobre los BiGRU de
`v9_04` y `v10_11`. **La magnitud 0.08399328135684547 NO se extrapola** a los BiGRU reales: acredita
que el defecto estructural de A15 tiene peso medible y que el efecto va en la direccion anunciada, y
nada mas. El orden de las operaciones y el `padding_idx` de A15 si son medidas sobre el codigo real.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P79 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar
modelos/ sin escrituras   el congelado se abrio en solo lectura, con huella comprobada
```

### Estado

P80 queda **cerrado y publicado**. Los siete hallazgos CONFIRMADO, las diecinueve afirmaciones
medidas, informe en JSON y en markdown, tres figuras y ninguna escritura fuera de `reports/` y
`figuras/p80/`.

```text
V03 remains the best verified leaderboard result.
```

---

## P81 - 2026-09-30 - Auditoria de los cinco huecos de P80. FALLA EN A6, codigo de salida 1

Script `experiments/p81_01_cierre_huecos.py`, sha256
`212aa786998f682a4ccefca2d91aa77f0cbf510968a0e8d21f1c687083f8e913`, 4517 lineas, 67 de 67 en `--humo`.
APROBADO por `ml-reviewer`, que comprobo ademas que el humo discrimina: tres mutaciones distintas
bajaron el recuento cada una.

Documentos vinculantes comprobados por el propio script antes de leer nada mas, y comprobados
tambien a mano antes de ejecutarlo:

```text
docs/p81_preregistro_cierre_huecos.md  55e6900cce49bd7e2ee98b00e0c315081c04718edb18df4abd49f484755faabc  commit 86a9140
docs/p81_enmienda_1.md                 496223801147a414e74a53fb7cf407ae48ad4671c74e45daba4dce19f15801c6  commit c3dee26
```

### Duracion y codigo de salida

```text
corrida en segundo plano, 2026-09-30T23:39:30Z a 23:40:47Z   77 s   codigo de salida 1
corrida de comprobacion con subprocess, 62.3 s                 codigo de salida 1
mismo punto de fallo en las dos, bit a bit
```

### Punto exacto del fallo

```text
Traceback (most recent call last):
  File ".../experiments/p81_01_cierre_huecos.py", line 4510, in <module>
    main()
  File ".../experiments/p81_01_cierre_huecos.py", line 4356, in main
    resultado = funcion(*argumentos)
  File ".../experiments/p81_01_cierre_huecos.py", line 1846, in afirmacion_a6
    m1_suma = primario["primarios"]["SUMA"]["m1_media"]
              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
KeyError: 'm1_media'
```

**Causa, sin arreglar.** `elegir_primario` construye la clave por objetivo con
`clave = f"m1_{objetivo.lower()}"` en la linea 1265 y la escribe en la 1277, luego el brazo **SUMA**
guarda la clave `"m1_suma"`. `afirmacion_a6` en la linea 1846 busca para el brazo SUMA una clave
literal `"m1_media"`, que es la del brazo MEDIA, y revienta. `afirmacion_a5` en la linea 1711 lo
hace bien, `primario["primarios"]["SUMA"]["m1_suma"]`, luego el mismo patron literal aparece bien y
mal a dos lineas de distancia. Es un fallo de la misma familia que el que el revisor dejo sin
cobertura de humo directa: una clave literal escrita a mano contra un diccionario de claves
generadas.

**No se ha tocado el script.** Ni una linea, ni un nombre, ni un temporal. La correccion la decide
el orquestador.

### Estado de las veinte afirmaciones

```text
CONFIRMADO 5   REFUTADO 0   NO APLICABLE 0   STOP 0   sin ejecutar 15
A1 CONFIRMADO   A2 CONFIRMADO   A3 CONFIRMADO   A4 CONFIRMADO   A5 CONFIRMADO
A0 sin ejecutar, se calcula al final
A6 a A19 sin ejecutar, el fallo corta el recorrido
```

### Bloque 1: la curva completa de M1 por learning rate, los dos brazos

```text
lr 0.02  MEDIA 0.0000  SUMA 0.0000  diferencia +0.0000
lr 0.05  MEDIA 0.0000  SUMA 0.0000  diferencia +0.0000
lr 0.10  MEDIA 0.0056  SUMA 0.0389  diferencia +0.0333
lr 0.20  MEDIA 0.7389  SUMA 1.0000  diferencia +0.2611
lr 0.35  MEDIA 1.0000  SUMA 1.0000  diferencia +0.0000
lr 0.50  MEDIA 1.0000  SUMA 1.0000  diferencia +0.0000
```

Corpus plantado de 23004 pares, M3 del azar 0.000042 con 2000 permutaciones.

**Recuento de victorias por learning rate: la SUMA gana 2 de 6 puntos.** Los seis puntos, curva
completa, y `curva monotona: False`, como se anticipaba. El prototipo daba MEDIA 0.3611 contra SUMA
0.0000 en 0.20 y MEDIA 0.4667 contra SUMA 0.8278 en 0.35; aqui en 0.20 gana la SUMA 1.0000 contra
0.7389 de la MEDIA y en 0.35 y 0.50 empatan en el techo de 1.0000. **El prototipo no se reproduce**,
ni en el signo del punto 0.20 ni en la magnitud del punto 0.35.

**Primario declarado: lr 0.1.** La zona betonal de la enmienda 1, M1 mayor que 0.99, deja fuera
1.0000 de la SUMA en 0.20 y los dos 1.0000 de 0.35 y 0.50. El mejor utilizable de la SUMA queda en
0.0389 en 0.10 y el de la MEDIA en 0.7389 en 0.20.

**Lectura del titular tal como la declara el script:**

```text
A5: CONFIRMADO [EL_TEST_DISCRIMINA] | la diferencia entre los dos mejores de la rejilla es
-0.700000, por encima del minimo de 0.02, fuera de la banda muerta de 0.005, y los cuatro hash de
```

La linea se corta en 150 caracteres en `_registrar`, linea 4503, no en el dato. Y hay que leerla con
cuidado: la comprobacion es `abs(diferencia) >= TOL["discriminacion"]` en la linea 1726, luego
`-0.700000` **discrima por magnitud, no por direccion**. Los dos mejores son 0.0389 de la SUMA en
0.10 y 0.7389 de la MEDIA en 0.20, y **la SUMA pierde por 0.70**. El titular dice que el test
discrimina, y es cierto; lo que no dice es que discrimina **en contra** de la direccion que A6
espera. El texto "por encima del minimo" se refiere al valor absoluto. A6 es justo la afirmacion que
iba a medir la direccion, y es justo la que revienta.

### Bloque 2: no se ha ejecutado

El fallo ocurre en el bloque 1, en A6, antes de la linea 4360 que abre el bloque 2. **A8 no se ha
ejecutado.** La proporcion de OOV recalculada con el tokenizador real de `v9_04` **no se ha
obtenido**: no hay cifra, ni el 0.155020 del prototipo ni ninguna de las tres anteriores, ni el
reparto entre canal de relleno y canal de OOV. Los bloques 3 y 4 tampoco.

### Salidas

```text
reports/p81_cierre_huecos.json   NO EXISTE
reports/p81_cierre_huecos.md     NO EXISTE
ningun temporal .tmp_p81 en reports/
```

El script fallo antes de la escritura atomica, luego **nada se publico y nada se vacio**. La
guarda de la linea 476, que aborta si el destino ya existe, y la asercion ASCII en memoria de la
linea 471 a 475 guardaban todo antes de abrir nada, funcionaron como estaban escritas. El bug que trunco un
fichero a cero bytes en este proyecto no se ha reproducido.

La asercion ASCII byte a byte de `reports/p81_cierre_huecos.json` y `.md` y el verificador oficial
**no se pueden ejecutar**, porque los ficheros no existen. El recuento de `NO_DISPONIBLE` en el
markdown **no aplica**, porque no hay markdown.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar
modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
```

El artefacto congelado se abrio en solo lectura y salio con la misma huella,29 478 868 bytes, tras
las dos corridas. No se entreno el Word2Vec de diez millones de tokens. No se genero submission.

### Limite permanente

**El anclaje a una implementacion externa de SGNS queda SIN HACER** porque gensim no esta instalado
y con Python 3.14.7 no compila. Autoriza afirmar sobre el **OBJETIVO**, no sobre la calidad relativa
del artefacto frente a una libreria.

Dos salvedades mas. El texto ancla de `docs/p81_enmienda_1.md` se commiteo en `5f4faf7` con dos
tildes y se corrigio en `c3dee26` sin reescribir historia. Y el revisor dejo **sin cobertura de humo
directa** el fallo de la tupla pasada como clave literal, que esta corregido y verificado en vivo
pero sin caso de humo que lo fije; el fallo de `afirmacion_a6` que acaba de parar la corrida es de
esa misma familia y **tampoco lo cubre el humo**, porque las 67 comprobaciones pasan.

### Estado

**P81 queda INCONCLUSO.** 5 de 20 afirmaciones medidas, todas CONFIRMADO, ninguna REFUTADA, ninguna
STOP, y 15 sin ejecutar por un `KeyError` de clave literal en A6. Sin informe en JSON ni en
markdown. El bloque 1 aporta la curva completa y el recuento de victorias, que es lo que mas
interesa; el bloque 2 no aporta nada todavia.

```text
V03 remains the best verified leaderboard result.
```

---

## P81 - 2026-10-01 - SEGUNDA CORRIDA. Llega a A17 y se para en A18. 15 CONFIRMADO, 2 REFUTADO

Corrige y completa la entrada anterior del 2026-09-30, que quedo describiendo una corrida que
reventaba en A6. **Aquella corrida no publico informe**; esta tampoco, por un motivo distinto. Las
dos entradas se conservan.

Script `experiments/p81_01_cierre_huecos.py`, sha256
`00756ef3432ca2ee29fd2c99987a84f19b3c6f801dc1ea70a34cdd4bc78c966c`, 4930 lineas, 83 de 83 en `--humo`,
ejecutado por el orquestador. APROBADO por `ml-reviewer` en la version anterior; el delta posterior es
el arreglo de A6 mas los candados de coherencia del titular. Specs vinculantes sin cambio:
`55e6900c...faabc` en `86a9140` y `49622380...5801c6` en `c3dee26`.

### Duracion y codigo de salida

```text
2026-09-30T23:59:01Z a 2026-10-01T00:08:28Z   567 s   codigo de salida 1
sin traceback de bajo nivel, excepcion capturada por el manejador de la linea 4926
```

### Reproducibilidad, medida

La corrida se repitio con cronometro de subprocess: 615.5 s, codigo de salida 1, **el mismo punto de
fallo**. Los **17 veredictos son identicos uno a uno** entre las dos corridas, y la curva completa de
los seis learning rates con los dos brazos es **identica bit a bit**. El fallo de A18 es estructural y
no depende del azar de las semillas, como no dependia el de A6.

### Punto exacto del fallo

```text
Traceback (most recent call last):
  File ".../experiments/p81_01_cierre_huecos.py", line 4923, in <module>
    main()
  File ".../experiments/p81_01_cierre_huecos.py", line 4835, in main
    a18 = afirmacion_a18(contexto["modulos"]["p72_01"])
  File ".../experiments/p81_01_cierre_huecos.py", line 3331, in afirmacion_a18
    documentos = leer_primeras_n_lineas(CORPUS_TWITTER, RECRETE["lineas"], modulo_p72)
  File ".../experiments/p81_01_cierre_huecos.py", line 3407, in leer_primeras_n_lineas
    tokens = modulo_p71.RE_TOKEN.findall(linea)
               ^^^^^^^^^^^^^^^^^^^
AttributeError: module 'p72_01' has no attribute 'RE_TOKEN'
```

**Causa, sin arreglar.** `leer_primeras_n_lineas` recibe su tercer parametro con el nombre
`modulo_p71` y de ahi usa `modulo_p71.RE_TOKEN`. Quien llama, `afirmacion_a18` linea 3331, le pasa
`modulo_p72`, que es `p72_01`. `RE_TOKEN` esta definido en `p71_04`, que el script ya importa y que
ademas esta cargado en `contexto["modulos"]["p71_04"]`, y `afirmacion_a18` **no lo recibe**: su unico
argumento es `contexto["modulos"]["p72_01"]`, en la linea 4835. A18 necesita los dos modulos, uno
para las constantes del curriculum y otro para el tokenizador, y solo le dieron uno.

Es un fallo de **cableado de modulos**: el contrato del llamado se cumple para la mitad de lo que la
funcion necesita. Y el nombre del parametro, `modulo_p71`, era la senal de que se pedia el otro modulo
y se le paso el equivocado. Misma familia que los dos anteriores: **contrato asumido y no verificado
en la frontera**.

### Estado de las veinte afirmaciones

```text
CONFIRMADO 15   REFUTADO 2   NO APLICABLE 0   STOP 0   sin ejecutar 3
A1 CONFIRMADO   A2 CONFIRMADO   A3 CONFIRMADO   A4 CONFIRMADO   A5 CONFIRMADO
A6 REFUTADO     A7 REFUTADO
A8 CONFIRMADO   A9 CONFIRMADO   A10 CONFIRMADO  A11 CONFIRMADO  A12 CONFIRMADO
A13 CONFIRMADO  A14 CONFIRMADO  A15 CONFIRMADO  A16 CONFIRMADO  A17 CONFIRMADO
A0, A18 y A19 sin ejecutar; el fallo corta el recorrido
```

Bloques 2, 3 y 4 **completos**: A8 a A12 del bloque 2, A13 y A14 del bloque 3, A15, A16 y A17 del
bloque 4. Solo el bloque 5, el learning rate paso a paso, no llego a empezar.

### Bloque 1: las dos lecturas de la zona betonal

La curva completa de M1 por learning rate con los dos brazos, seis puntos, tres semillas, corpus
plantado de 23004 pares, M3 del azar 0.000042 en 2000 permutaciones:

```text
lr 0.02  MEDIA 0.0000  SUMA 0.0000   lr 0.20  MEDIA 0.7389  SUMA 1.0000
lr 0.05  MEDIA 0.0000  SUMA 0.0000   lr 0.35  MEDIA 1.0000  SUMA 1.0000
lr 0.10  MEDIA 0.0056  SUMA 0.0389   lr 0.50  MEDIA 1.0000  SUMA 1.0000
```

**Lectura por brazo**, que es la que se usa para el primario y la que declara `CRITERIO_ZONA_BETONAL`
en la linea 178: cada brazo excluye los puntos en los que ese brazo satura. La MEDIA conserva
0.02, 0.05, 0.10 y 0.20 y su mejor cifra es **0.7389 en lr 0.20**. La SUMA pierde el 0.20, que
satura a 1.0000, y su mejor cifra es **0.0389 en lr 0.10**.

```text
diferencia por BRAZO = 0.0389 - 0.7389 = -0.700000   gana la MEDIA por 0.700000
primario declarado: lr 0.1
```

**Lectura por punto**, que se calcula y se publica pero **no** se usa: se cae el punto entero si un
solo de los dos brazos lo declara betonal, luego quedan 0.02, 0.05 y 0.10. La mejor cifra de la
MEDIA es 0.0056 y la de la SUMA 0.0389.

```text
diferencia por PUNTO = 0.0389 - 0.0056 = +0.0333   gana la SUMA por 0.0333
```

**Las dos lecturas NO coinciden en signo**, que es exactamente lo que declara el criterio de la
linea 174: con estos numeros decidir solo con una seria ocultar la otra. El signo se da la vuelta
enteramente segun el criterio de exclusion.

**Recuento de victorias por learning rate: la SUMA gana 2 de 6 puntos, y la curva NO es monotona.**
El prototipo no se reproduce: daba MEDIA 0.3611 contra SUMA 0.0000 en 0.20 y MEDIA 0.4667 contra
SUMA 0.8278 en 0.35; aqui en 0.20 gana la SUMA 1.0000 contra 0.7389 y en 0.35 y 0.50 empatan en el
techo de 1.0000.

**El titular, tal como lo declara el script:**

```text
A5: CONFIRMADO [EL_TEST_DISCRIMINA_POR_MAGNITUD] | la diferencia entre los dos mejores de la rejilla
es -0.700000 en MAGNITUD, por encima del minimo de 0.02 y fuera de la banda muerta de 0.005
A6: REFUTADO | REFUTADO: la direccion es CONTRARIA a la hipotesis. La MEDIA gana por 0.700000 en
el primario, y el margen exigido es de de 0.02 a favor de la SUMA
A7: REFUTADO | el efecto no se sostiene: ['el conteo de victorias por semilla', 'el control negativo
tambien recupera estructura']
```

**Sale exactamente lo que se esperaba y es un resultado legitimo: el test NO apoya a la suma.** El
arreglo de A6 y los candados de coherencia hacen su trabajo. A5 etiqueta **solo** magnitud y ya no
puede leerse como apoyo a la SUMA; A6 mide la direccion y la da contraria, 0.700000 a favor de la
MEDIA contra los 0.02 exigidos a favor de la SUMA; y A7 se suma en REFUTADO porque el efecto no se
sostiene ni en el conteo de victorias por semilla ni en el control negativo, que tambien recupera
estructura.

Con diferencia `-0.700000`, la funcion `lectura_del_bloque_1` de la linea 1309 entra por la rama de
la linea 1339, `direccion == "favor_a_la_media"`, y devuelve **`EL_TEST_NO_APOYA_A_LA_SUMA`** con
`apoya_a_la_suma` falso. **Ese titular no se publico en ningun fichero**, porque el fallo cae antes
de `construir_json` y de `escribir_markdown`. Existe solo en memoria.

**Sin retoque.** No se movio la tolerancia de 0.02, ni la banda muerta de 0.005, ni la rejilla, ni el
criterio de zona betonal, ni el signo de la direccion. El REFUTADO de A6 y el REFUTADO de A7 se
publican tal cual salen.

### Bloque 2: ejecutado, con las cifras perdidas

A8 a A12 los ejecuto el bloque 2 completo. A8 **CONFIRMADO**: "las cuatro cifras estan medidas y
publicadas. La cifra de 11.2726 por ciento de la discusion de auditoria NO queda reproducida con
ninguna de las tres". A9 **CONFIRMADO**, las celdas quedan separadas de verdad, identificadores
distintos y fila de relleno identicamente cero. A10 **CONFIRMADO**, las seis magnitudes presentes con
al menos una cifra distinta de cero por condicion y por clase. A11 **CONFIRMADO**: "el canal de OOV
es al menos tan grande como el de relleno en las dos clases". A12 **CONFIRMADO**, dos pasadas con la
misma semilla con variacion exactamente cero.

**Las cuatro cifras de la proporcion de OOV NO son recuperables de esta corrida.** Se midieron y se
verificaron, pero vivieron en memoria y el proceso murio antes de `escribir_markdown`: no hay JSON ni
markdown donde leerlas, y la linea de registro de `_registrar` las corta a 150 caracteres, que es lo
que queda en el log. Del reparto entre canal de relleno y canal de OOV solo sobrevive el veredicto
cualitativo de A11, **el canal de OOV es al menos tan grande como el de relleno en las dos clases**,
y ninguna magnitud. El 0.155020 del prototipo y las tres cifras anteriores **siguen sin reproducirse
de forma verificable desde un fichero publicado**.

### Salidas

```text
reports/p81_cierre_huecos.json   NO EXISTE
reports/p81_cierre_huecos.md     NO EXISTE
ningun temporal .tmp_p81 en reports/
```

El fallo cae antes de la escritura atomica, luego **nada se publico y nada se vacio**. La asercion
ASCII byte a byte de los dos informes y el verificador oficial **no se pueden ejecutar**, porque los
ficheros no existen. El recuento de `NO_DISPONIBLE` en el markdown **no aplica**, porque no hay
markdown. El bug que llenaba el `.md` de huecos en silencio **no se ha reproducido**, porque el
script no llego a escribir. Que el fallo siga siendo de cableado y no de escritura es la mejor
noticia de las dos corridas: la defensa de escritura de P80 aguanta.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura, misma huella
```

### Limite permanente, y una conclusion nueva

**El anclaje a una implementacion externa de SGNS queda SIN HACER** porque gensim no esta instalado
y con Python 3.14.7 no compila. Autoriza afirmar sobre el **OBJETIVO**, no sobre la calidad relativa
del artefacto frente a una libreria. **Word2Vec no se afirma mejor que V03.**

**Conclusion nueva de esta corrida.** Con los numeros reales, **la pata EJECUTABLE del anclaje no
apoya a la suma**: A5 discrimina por magnitud, A6 mide la direccion y es contraria, y A7 no sostiene
el efecto. Luego **la pata que sostiene que el objetivo no era SGNS es la DOCUMENTAL**, la del codigo
C de word2vec, y **queda como unica base**. La reimplementacion propia no aporta apoyo
ejecutable, y eso hay que decirlo sin rodeos: la conclusion del proyecto sobre el objetivo se apoya
en la lectura del codigo de referencia y en nada medido aqui.

### Estado

**P81 sigue INCONCLUSO, y por segunda vez por un fallo de contrato en la frontera, no de calculo.**
17 de 20 afirmaciones medidas: 15 CONFIRMADO, 2 REFUTADO, 0 STOP, 3 sin ejecutar. Bloques 2, 3 y 4
completos. El bloque 1 **si** entrega la lectura que se buscaba: las dos lecturas de la zona betonal
con su signo opuesto, el recuento de victorias de 2 de 6, y el titular esperado. Sin informe
publicado, luego las cuatro cifras de OOV quedan sin publicar.

```text
V03 remains the best verified leaderboard result.
```

---

## P81 - 2026-10-01 - TERCERA CORRIDA. El propio escritura-incremental la para en A8. 6 CONFIRMADO, 2 REFUTADO, 0 ERROR

Corrige y completa las dos entradas anteriores, que quedaron describiendo corridas que no publicaron
informe. **Ninguna de las tres publico informe.** Las tres entradas se conservan.

Script `experiments/p81_01_cierre_huecos.py`, sha256
`5b5db4b0f2cd514fec1cd667d898c808dd5a6d090f27ae4e9f15c7b7967e9100`, 120 de 120 en `--humo` con salida 0,
APROBADO por `ml-reviewer`. Specs sin cambio: `55e6900c...faabc` en `86a9140` y `49622380...5801c6` en
`c3dee26`. **Nota de inventario:** el fichero tiene **5973 lineas**, no 5975; el sha256 coincide
completo con el declarado, luego es el fichero correcto y la cuenta venia de otro sitio.

### Duracion y codigo de salida

```text
64.4 s con cronometro de subprocess   codigo de salida 1
la corrida se paro a los 64 s, no a los diez minutos: el fallo es temprano y es de escritura
```

### Punto exacto del fallo

```text
[P81-01] 2026-10-01T00:48:57+00:00 | JSON parcial escrito en reports/p81_cierre_huecos.parcial.json: 7 afirmaciones, 0 aisladas con fallo
[P81-01] 2026-10-01T00:48:57+00:00 | bloque 2: tasas con el tokenizador real y sonda de celdas separadas
[P81-01] 2026-10-01T00:49:00+00:00 |    A8: CONFIRMADO | las cuatro cifras estan medidas y publicadas. La cifra de 11.2726 por ciento de la discusion de auditoria NO queda reproducida con ninguna de las tres
STOP escritura: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj/reports/p81_cierre_huecos.parcial.json ya existe y solo se admiten temporales hasta la publicacion atomica
```

**Causa, sin arreglar, y son dos defectos que se multiplican.**

1. `publicar_parcial`, linea 3882, escribe el parcial con `escribir_ascii_atomico` sobre
   `JSON_PARCIAL` y luego lo publica con `publicar_atomico` en la 3889, o sea **sobre su nombre
   definitivo**. La primera escritura, al terminar el bloque 1, deja el parcial en
   `reports/p81_cierre_huecos.parcial.json`. La segunda escritura, tras A8, vuelve a pedir ese mismo
   nombre y `escribir_ascii_atomico` lo **rechaza en la linea 517**, que es el guard de
   antitrunco de P80: si el destino existe, `STOP`. Un escritor incremental que publica sobre su
   nombre final choca de frente con un guard que prohibe sobrescribir el destino.
2. Ese `SystemExit` deberia haber sido un aviso. La linea 3895 tiene `except SystemExit: raise`, que
   **relevanta la excepcion antes de que la llegue a ver el `except Exception` de la 3897**, que es el
   queiba a registrar `aviso: no se pudo escribir el JSON parcial`. O sea que la red de seguridad que
   el propio escritor declara tener **no existe para este fallo**, y la corrida se cae entera.

**La ironia, y hay que decirla:** la caracteristica que se anadio para que un fallo final **no tirase
las mediciones anteriores es justo la que tira la corrida**, y se come precisamente la primera
medicion del bloque 2, la que pretendia salvar. Las cuatro cifras de A8 mueren por tercera vez
consecutiva, y esta por el mecanismo que deberia impedirlas.

### Estado de las veinte afirmaciones

```text
CONFIRMADO 6   REFUTADO 2   ERROR 0   STOP 0   sin ejecutar 12
A1 CONFIRMADO  A2 CONFIRMADO  A3 CONFIRMADO  A4 CONFIRMADO  A5 CONFIRMADO
A6 REFUTADO    A7 REFUTADO    A8 CONFIRMADO
A0 y A9 a A19 sin ejecutar
```

**Ninguna afirmacion quedo en ERROR.** El guard de aislamiento por afirmacion nunca llego a actuar:
`afirmaciones_aisladas_con_fallo` esta vacio. El fallo no ocurrio dentro de una afirmacion, ocurrio en
la capa de persistencia, entre A8 y la escritura que la iba a registrar. A8 **llego a CONFIRMADO** y
su veredicto esta en el log, pero **no quedo en disco**.

### Bloque 1: las dos lecturas de la zona betonal, ahora si en disco

Si estan publicadas, en `reports/p81_cierre_huecos.parcial.json`, con estos numeros exactos:

```text
umbral de zona betonal 0.99

por BRAZO, criterio declarado y el del primario
  MEDIA  excluidos 0.35 y 0.50   mejor no betonal 0.7388888888888889 en lr 0.20
  SUMA   excluidos 0.20, 0.35 y 0.50   mejor no betonal 0.03888888888888889 en lr 0.10
  diferencia_por_brazo = -0.7000000000000001   GANA LA MEDIA por 0.700000   primario en lr 0.1

por PUNTO, alternativa calculada y publicada, no usada para el primario
  sobran 0.02, 0.05 y 0.10   excluidos 0.20, 0.35 y 0.50
  MEDIA 0.005555555555555556   SUMA 0.03888888888888889
  diferencia_por_punto = +0.03333333333333333   GANA LA SUMA por 0.0333

las_dos_lecturas_coinciden_en_signo = False
```

**El signo se da la vuelta enteramente segun el criterio de exclusion.** Y la lectura por punto no es
una curiosidad: es la que explica por que el reparto de victorias sale 2 a 4.

**Recuento de victorias por learning rate: la SUMA gana 2 de 6 y la MEDIA 4 de 6. Reparto "2 a 4".**
Curva NO monotona, con **2 cambios de signo**. `recuento_en_banda_muerta: true`.

**Titular literal, y sale exactamente como se esperaba:**

```text
etiqueta: EL_TEST_NO_APOYA_A_LA_SUMA   apoya_a_la_suma: False   direccion: favor_a_la_media
explicacion: la diferencia entre los dos mejores de la rejilla es -0.700000, de magnitud suficiente
para discriminar pero con el SIGNO en contra de la SUMA. La MEDIA gana por 0.700000. El test
discrimina por magnitud, pero su direccion es contraria a la hipotesis de A6, y eso se publica: el
bloque 1 NO apoya al objetivo de la referencia
```

Y el recuento anade su propia lectura, que es la segunda capa de prudencia:

```text
ZONA DE NO CONCLUSION: el reparto de victorias por learning rate queda en 2 a 4 y eso es ruido con
tres semillas. El primario NO debe leerse como confirmacion
```

**A5 CONFIRMADO por magnitud, A6 REFUTADO por direccion, titular EL_TEST_NO_APOYA_A_LA_SUMA con la
MEDIA ganando por 0.700000, y recuento 2 a 4 en zona de no conclusion.** Las cuatro cosas que se
esperaban. Se publican tal cual. No se movio la tolerancia de 0.02, ni la banda muerta de 0.005, ni la
rejilla, ni el criterio de zona betonal, ni el signo de la direccion.

### Bloque 2: A8 corrio y CONFIRMO, y sus cifras murieron otra vez

A8 **CONFIRMADO**: las cuatro cifras estan medidas y publicadas, y el 11.2726 por ciento de la
discusion de auditoria **no** queda reproducido con ninguna de las tres. **Pero el parcial escrito al
terminar el bloque 1 solo contiene siete afirmaciones, A1 a A7.** La escritura que iba a meter a A8
en disco es la que revento, luego **las cuatro cifras de la proporcion de OOV no son recuperables por
tercera vez consecutiva**, y **tampoco el reparto entre canal de relleno y canal de OOV con
magnitud**. Del bloque 2 no queda ni A9, ni A10, ni A11, ni A12: A11 era el que daba el veredicto
cualitativo de que el canal de OOV es al menos tan grande como el de relleno en las dos clases, y esa
linea tampoco esta en disco.

El 0.155020 del prototipo y las tres cifras anteriores **siguen sin reproducirse de forma verificable
desde un fichero publicado**, y el motivo ahora ya no es un fallo de codigo: es que la persistencia
se cae antes de llegar a escribir el bloque 2.

### Bloque 5: A18 y A19 nunca se ejecutaron

El bloque 5 **no llego a empezar**. El recorrido se corta en la primera escritura del bloque 2, muy
antes. **A18 y A19 no tienen veredicto**: ni CONFIRMADO, ni REFUTADO, ni ERROR, ni sin ejecutar por
fallo propio, sencillamente no seanska. En consecuencia **la contingencia de que A19 salga REFUTADO
como consecuencia de un A18 en ERROR no se materializa**, y no hay ningun REFUTado quericheve a
descartar por fallo aguas arriba. El ultimo REFUTADO real es el de A6 y el de A7, y ambos son
mediciones de verdad sobre el bloque 1, ninguno depende de A18.

### Salidas

```text
reports/p81_cierre_huecos.parcial.json   24196 B   ESCRIBE, 7 afirmaciones A1 a A7
reports/p81_cierre_huecos.json            NO EXISTE
reports/p81_cierre_huecos.md              NO EXISTE
ningun temporal .tmp_p81
```

ASCII byte a byte del parcial: **24196 bytes, 0 bytes por encima de 127, ASCII puro**. El `.json` y el
`.md` completos no se pueden verificar porque no existen. El **verificador oficial**,
`tools/verificar_caracteres.py`, corrido sobre el proyecto entero: **codigo de salida 0, 0 caracteres
no permitidos**.

**NO_DISPONIBLE en el `.md`: no aplica, porque el `.md` no se llego a escribir.** En el unico informe
que si hay en disco, el parcial, las ocurrencias de `NO_DISPONIBLE` son **0**. El bug que llenaba el
informe de huecos en silencio **no se ha reproducido en esta corrida**; y en las dos anteriores
tampoco, porque ninguna llego a escribir el `.md`. La defensa de escritura de P80, la que trunca
antes de abrir, ha aguantado tres reventones seguidos.

### Avisos de contrato nuevos

Al arrancar salen **34 lineas** `aviso de contrato` sobre A0, A4, A5, A6, A7, A8, A9, A10, A11, A12,
A13, A15, A16, A17 y A19: el contexto no tiene el origen declarado `medidas.<parametro>` para la
mayoria de los parametros de las afirmaciones. **Son avisos, no fallos**, y la corrida sigue. Pero
conviene dejarlo escrito: el comprobador de contratos que se acaba de anadir todavia no tiene
declarada la procedencia de la mayor parte de lo que le pasa, luego **esa tabla de origenes esta a
medio construir**. Es la misma familia por tercera vez: contrato asumido, no verificado en la
frontera.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura, misma huella
```

### Limite permanente y conclusion

**El anclaje a una implementacion externa de SGNS queda SIN HACER** porque gensim no esta instalado
y con Python 3.14.7 no compila. Autoriza afirmar sobre el **OBJETIVO**, no sobre la calidad relativa
del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor que V03.**

**Conclusion, ya con el titular en disco y no solo en memoria.** La pata **EJECUTABLE** del anclaje
**no apoya a la suma**: A5 discrimina por magnitud, A6 mide la direccion y la encuentra contraria,
la MEDIA gana por 0.700000, y el reparto de victorias de 2 a 4 con la curva no monotona y dos cambios
de signo lo deja en zona de no conclusion con tres semillas. Luego **la base que sostiene que el
objetivo no era SGNS es la pata DOCUMENTAL, el codigo C de word2vec, y queda como base unica.** La
reimplementacion propia no aporta apoyo ejecutable, y eso se dice sin rodeos.

### Estado

**P81 sigue INCONCLUSO por tercera vez, y por tercera vez por un fallo de contrato y no de
calculo.** 8 de 20 afirmaciones medidas, 6 CONFIRMADO, 2 REFUTADO, 0 ERROR, 0 STOP, 12 sin ejecutar.
Lo que si queda en disco es el bloque 1 completo, con las dos lecturas, el recuento 2 a 4 y el
titular EL_TEST_NO_APOYA_A_LA_SUMA. Los bloques 2, 3, 4 y 5 no dejan nada. Ningun ERROR que aislar.

```text
V03 remains the best verified leaderboard result.
```

---

## P81 - 2026-10-01 - CUARTA CORRIDA. Las veinte medidas y diez parciales propios. Se para al serializar el informe final

Corrige y completa las tres entradas anteriores de P81, que quedaron describiendo corridas que no
publicaron informe completo. **Ninguna de las cuatro lo ha publicado.** Las cuatro entradas se
conservan.

Script `experiments/p81_01_cierre_huecos.py`, sha256
`4f7f8488ed9b27106248d8035e3bb749318b0032788616e745421d2df09adb27`, 6402 lineas, 126 de 126 en `--humo`
con salida 0, verificado por el orquestador. Specs sin cambio: `55e6900c...faabc` en `86a9140` y
`49622380...5801c6` en `c3dee26`.

### El arreglo funciona, y se nota

La corrida anterior se comia a si misma: `publicar_parcial` publicaba sobre su nombre definitivo y la
segunda escritura chocaba con el guard antitrunco. Ahora **cada escritura parcial va a un fichero
PROPIO, con la marca de corrida y el motivo**, de modo que la invariante "nunca sobrescribir un
destino existente" es cierta para todos los destinos sin excepciones. Y un fallo de escritura degrada
a AVISO con su causa en el log, no a parada. Solo una `Parada` con marca STOP aborta.

**Se escribieron los diez parciales propios**, uno al terminar cada bloque y uno tras cada
afirmacion del bloque 2, con nombres que llevan la marca de corrida:

```text
p81_cierre_huecos.parcial_20261001T010045_fin_del_bloque_1.json     24897 B
p81_cierre_huecos.parcial_20261001T010045_bloque_2_afirmacion_1_de_5_A8.json   27109 B
p81_cierre_huecos.parcial_20261001T010045_bloque_2_afirmacion_2_de_5_A9.json   35086 B
p81_cierre_huecos.parcial_20261001T010045_bloque_2_afirmacion_3_de_5_A10.json  38360 B
p81_cierre_huecos.parcial_20261001T010045_bloque_2_afirmacion_4_de_5_A11.json  39471 B
p81_cierre_huecos.parcial_20261001T010045_bloque_2_afirmacion_5_de_5_A12.json  41792 B
p81_cierre_huecos.parcial_20261001T010045_fin_del_bloque_2.json     41798 B
p81_cierre_huecos.parcial_20261001T010045_fin_del_bloque_3.json     45276 B
p81_cierre_huecos.parcial_20261001T010045_fin_del_bloque_4.json     52548 B
p81_cierre_huecos.parcial_20261001T010045_fin_del_bloque_5.json    148941 B
```

El `reports/p81_cierre_huecos.parcial.json` de la corrida rota **sigue ahi intacto**, sha256
`4670cad89407db551705cb18ae3f1c90e70fa707266034d554faa3d7b588a3ae`, 24196 B. **No se ha borrado**,
como se pidio: es evidencia de la corrida que fallo.

### Duracion y codigo de salida

```text
65.1 s con cronometro de subprocess   codigo de salida 1
no fueron los diez minutos porque el fallo es al final, no al principio: las veinte afirmaciones
se midieron enteras antes de caer
```

### Punto exacto del fallo

```text
  File ".../experiments/p81_01_cierre_huecos.py", line 6395, in <module>
    main()
  File ".../experiments/p81_01_cierre_huecos.py", line 6349, in main
    json.dumps(informe, ensure_ascii=True, indent=2)
  File "/usr/lib/python3.14/json/encoder.py", line 202, in encode
    raise TypeError(f'Object of type {o.__class__.__name__} '
                    f'is not JSON serializable')
TypeError: Object of type module is not JSON serializable
when serializing dict item 'p71_04'
when serializing dict item 'contexto_de_modulos'
```

**Causa, sin arreglar.** La linea 3835 mete en el cuerpo del informe
`"contexto_de_modulos": contexto.get("modulos", {})`, y ese diccionario contiene **objetos modulo
vivos**, `p71_04`, `p72_01`, `v10_11` y `v9_04`. Un modulo no es serializable a JSON, luego la
`json.dumps` de la linea 6349 revienta. Es la **ultima** sentencia efectiva de la corrida: todo lo
demas, las veinte afirmaciones y los diez parciales, ya estaba escrito en disco.

**La asimetria que lo explica:** los parciales serializan bien porque su cuerpo **no** incluye
`contexto_de_modulos`, y el informe final si. El campo se anadio para poder auditar que modulos se
cargaron, y para eso hace falta el modulo, y para eso el modulo no se puede volcar a JSON. Falta el
paso intermedio: publicar **los nombres** de los modulos, no los objetos.

Note que el guard de la linea 6344, que se niega a sobrescribir el informe si ya existe, se respeto
correctamente: los dos ficheros no existian, se paso de largo, y la muerte vino despues.

### Estado de las veinte afirmaciones

```text
CONFIRMADO 18   REFUTADO 2   ERROR 0   STOP 0   sin ejecutar 0
A0 CONFIRMADO   A1..A5 CONFIRMADO   A6 REFUTADO   A7 REFUTADO
A8..A19 CONFIRMADO
LAS VEINTE MEDIDAS. Ninguna en ERROR, ninguna sin ejecutar.
```

A0 **CONFIRMADO**: intacto en sha256, tamano y `mtime_ns` antes y despues de la primera y la ultima
lectura. Ninguna afirmacion quedo en ERROR, y el guard de aislamiento no necesito actuar porque no
hubo ningun fallo dentro de una afirmacion.

### Bloque 1: las dos lecturas, el reparto y el titular, tal cual se esperaba

```text
umbral de zona betonal 0.99

por BRAZO, criterio declarado y el del primario
  MEDIA  excluye 0.35 y 0.50          mejor no betonal 0.7388888888888889 en lr 0.20
  SUMA   excluye 0.20, 0.35 y 0.50   mejor no betonal 0.03888888888888889 en lr 0.10
  diferencia_por_brazo = -0.7000000000000001   GANA LA MEDIA por 0.700000   primario lr 0.1

por PUNTO, alternativa calculada y publicada, no usada
  sobran 0.02, 0.05 y 0.10   MEDIA 0.005555555555555556   SUMA 0.03888888888888889
  diferencia_por_punto = +0.03333333333333333   GANA LA SUMA por 0.0333

las_dos_lecturas_coinciden_en_signo = False
```

**Reparto de victorias por learning rate: la SUMA gana 2 de 6 y la MEDIA 4 de 6, "2 a 4".** Curva no
monotona, **2 cambios de signo**, `recuento_en_banda_muerta: True`.

**Titular literal:**

```text
EL_TEST_NO_APOYA_A_LA_SUMA   apoya_a_la_suma: False   direccion: favor_a_la_media
```

```text
la diferencia entre los dos mejores de la rejilla es -0.700000, de magnitud suficiente para
discriminar pero con el SIGNO en contra de la SUMA. La MEDIA gana por 0.700000. El test discrimina
por magnitud, pero su direccion es contraria a la hipotesis de A6, y eso se publica: el bloque 1
NO apoya al objetivo de la referencia
```

```text
ZONA DE NO CONCLUSION: el reparto de victorias por learning rate queda en 2 a 4 y eso es ruido con
tres semillas. El primario NO debe leerse como confirmacion
```

**Las cuatro cosas esperadas, y se publican sin ajustar:** A5 CONFIRMADO por magnitud, A6 REFUTADO por
direccion, titular `EL_TEST_NO_APOYA_A_LA_SUMA` con la MEDIA ganando por 0.700000, reparto 2 a 4 en
zona de no conclusion, y las dos lecturas con signo opuesto y `coinciden_en_signo` igual a False. **No
se movio la tolerancia de 0.02, ni la banda muerta de 0.005, ni la rejilla, ni el criterio de zona
betonal, ni el signo de la direccion.** El REFUTADO de A6 y el de A7 se publican tal cual salen.

### Bloque 2: LAS CUATRO CIFRAS, por fin en disco, y el reparto con magnitud

A8 **CONFIRMADO**. Estas son las cifras con el tokenizador real de `v9_04`, y **no habian sobrevivido
a ninguna corrida anterior**:

```text
proporcion_de_posiciones_oov        = 0.04124873567093729   4.124874 % de las posiciones
proporcion_de_posiciones_de_relleno = 0.7309191672285907    73.091917 % de las posiciones
oov_sobre_tokens                     = 15.329496 %           sobre los tokens que entran
numero_de_tipos_distintos_fuera_de_vocabulario = 2008
```

Y el resto del desglose, con los numeros que lo sostienen:

```text
n_filas_de_dev 1483   max_len_de_v9_04 64   posiciones_totales 94912
tokens_que_entran_en_la_matriz 25539   tokens_cubiertos 21624   tokens_oov 3915
posiciones_de_relleno 69373   vocabulario_usado 3439
numero_de_filas_con_cero_tokens_cubiertos 1   fraccion_de_filas_sin_cubiertos 0.0006743088334457181
codigo: v9_04.cargar_vec, v9_04.normalizar_emb_normal, v9_04.tokenizar y v9_04.secuencias
```

**Reparto entre canal de relleno y canal de OOV, con magnitud:** sobre las 94912 posiciones del
desarrollo, el relleno son 69373, el **73.091917 %**, y el OOV son 3915 tokens, el **4.124874 %**.
Sobre los 25539 tokens que llegan a la matriz, el OOV es el **15.329496 %**. El canal de OOV es una
orden de magnitud mas pequeno que el de relleno en posiciones, y algo mas grande que la mitad en
tokens.

**Lo que A8 deja escrito sobre las cifras anteriores, y es lo importante:**

```text
cifra_de_la_discusion_en_por_ciento = 11.2726
queda_reproducida = False
variante_que_la_reproduce = null
```

```text
las tres cifras de la discusion usan normalizaciones distintas y MAX_LEN 60, mientras que v9_04 usa
MAX_LEN 64 y normalizar_emb_normal. Recalcular con el codigo real es lo unico que hace comparables
los numeros
```

Las cifras que **no se arrastran** quedan listadas con su motivo: `p78_oov_normalizado 0.112658`,
`p78_oov_crudo 0.313639`, `p78_maxlen 60`, `p80_oov_de_la_discusion 11.2726`, contra
`v9_04_max_len 64`. **El 0.155020 del prototipo no reproduce ninguna de las cuatro**, y ahora eso esta
medido y publicado en vez de estimado. El **4.124874 %** es la cifra buena.

**Salvedad que hay que dejar escrita sobre A11.** A11 sale CONFIRMADO con el motivo "el canal de OOV es
al menos tan grande que el de relleno en las dos clases", y las magnitudes de la sonda son:

```text
v9_04_BiGRU               canal_de_relleno_c1 0.0   canal_de_oov_c2b_menos_c2 0.0   c2 0.1146087   c2b 0.1146087
v10_11_BiGRUAfinable      canal_de_relleno_c1 0.0   canal_de_oov_c2b_menos_c2 0.0   c2 0.1146087   c2b 0.1146087
distancia_entre_c2_y_c2b = 0.0 en las dos clases   banda_muerta_de_la_diferencia = true
```

**Los dos canales miden exactamente 0.0**, luego la comparacion "el OOV es al menos tan grande" es una
comparacion de **dos ceros**, y `banda_muerta_de_la_diferencia` lo declara. El veredicto es cierto y
no informa nada: **la magnitud que respalda el reparto de canales es la de A8, no la de A11.** A11
sigue siendo valido como prueba de que la sonda separa las celdas, no como medida del reparto.

### Bloque 5: A18 y A19, los dos CONFIRMADO

```text
A18: CONFIRMADO | la serie esta medida y publicada, y la clasificacion sale de aplicar el umbral:
     diferencia relativa de las medias 0.002143718728, umbral 0.005, luego DESPRECIABLE
A19: CONFIRMADO | la etiqueta existe y coincide con la regla de A18
```

Detalles que se publican: `total_lotes_plan 1395`, `total_lotes_recorridos 1398`, `pasos_totales 1398`,
`lr_inicial 6.4`, `lr_minimo 0.001`, y el recorte declarado, **primeras 2000 lineas del fichero, lote
256, 53243 tokens**, con la advertencia de que los recuentos absolutos no son los de la corrida real
y de que lo que se audita, la desviacion entre los dos totales, si escala de forma lineal. La serie
completa de `lr` paso a paso esta publicada.

A19 declara `coinciden: true`, `declaracion_de_impacto_del_scheduler: DESPRECIABLE` y
`diferencia_relativa_de_las_medias 0.0021437187276626846` contra `umbral 0.005`, y dice explicitly que
el desfase de una epoca **existe y es un defecto real de codigo**, porque el plan usa `range(EPOCAS)` y
el entrenamiento usa `enumerate(..., start=1)`, y que lo que se cualifica es su consecuencia numerica.

**Ninguno de los dos quedo en ERROR, y la contingencia de que A19 emita ERROR por A18 sin ejecutar no se
disparo**, porque A18 si se ejecuto. El arreglo de esa regla esta correcto y no llego a hacer falta.
**No hay ningun REFUTado que haya que descartar por fallo aguas arriba**: los dos REFUTADO, A6 y A7,
son mediciones del bloque 1 y no dependen de A18.

### Salidas

```text
reports/p81_cierre_huecos.json   NO EXISTE
reports/p81_cierre_huecos.md     NO EXISTE
10 parciales propios ESCRITOS, cada uno ASCII puro, 0 bytes por encima de 127
reports/p81_cierre_huecos.parcial.json  el de la corrida rota, INTACTO y conservado
ningun temporal .tmp_p81
```

El ASCII byte a byte de los **diez** parciales nuevos: **0 bytes por encima de 127 en los diez**. El
`.json` y el `.md` finales **no se pueden verificar porque no existen**. El **verificador oficial**
`tools/verificar_caracteres.py` sobre el proyecto entero: **salida 0, 0 caracteres no permitidos**.

**NO_DISPONIBLE en el `.md`: no aplica, porque el `.md` no se llego a escribir.** En los diez
parciales nuevos son **0**, y **0** tambien en el parcial viejo de la corrida rota. El bug que
llenaba el informe de huecos en silencio **no se ha reproducido en cuatro corridas seguidas**.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura. A0 lo confirma en sha256, tamano y mtime_ns.
```

### Conclusion y limite permanente

**La pata EJECUTABLE del anclaje de H1 no apoya la suma.** A5 discrimina por magnitud, A6 mide la
direccion y la encuentra contraria con la MEDIA ganando por 0.700000, y el reparto de 2 a 4 con dos
cambios de signo en la curva deja el primario en zona de no conclusion con tres semillas. Luego **la
base que sostiene que el objetivo no era SGNS es la pata DOCUMENTAL, el codigo C de word2vec, y queda
como base unica.**

**Limite permanente:** el anclaje a una implementacion externa de SGNS queda **SIN HACER** porque
gensim no esta instalado y no compila con Python 3.14.7. Autoriza afirmar sobre el **OBJETIVO**, no
sobre la calidad relativa del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor
que V03.**

### Estado

**P81 sigue INCONCLUSO, y por cuarta vez por un fallo de contrato y no de calculo.** Las veinte
afirmaciones estan medidas y **diez ficheros en disco las respaldan**, pero el informe completo en
JSON y en markdown no se publico. La diferencia con las tres corridas anteriores es que **esta vez no
se pierde ninguna medicion**: las cuatro cifras de OOV, el reparto con magnitud, las dos lecturas, el
recuento, el titular, A18 y A19 estan todos en `reports/p81_cierre_huecos.parcial_20261001T010045_fin_del_bloque_5.json`.
Lo unico que falta es el `.json` y el `.md` finales, y la causa es una sola: un objeto modulo dentro
del cuerpo del informe.

```text
V03 remains the best verified leaderboard result.
```

---

## P81 - 2026-10-01 - CORRIDA FINAL. PUBLICADO. 20 de 20, 18 CONFIRMADO, 2 REFUTADO, salida 0

Corrige y completa las cuatro entradas anteriores de P81. **Esta publica informe**, y las cuatro
anteriores quedan como las corridas que no lo lograron. Las cinco entradas se conservan.

Script `experiments/p81_01_cierre_huecos.py`, sha256
**`1e52280f64635637429e141440cfe5efce773e91419bf9b9010b8e7d0c6de39f`**, 6412 lineas. **El hash ha
cambiado respecto al `4f7f8488ed9b2710` de la corrida anterior, y es lo esperado**: ese cambio es el
arranque de la serializacion. Verificado antes de ejecutar. Specs sin cambio: `55e6900c...faabc` en
`86a9140` y `49622380...5801c6` en `c3dee26`.

### Duracion y codigo de salida

```text
65.4 s con cronometro de subprocess   CODIGO DE SALIDA 0
sin traceback, sin STOP, sin ERROR
```

### El arreglo

El informe metia `contexto_de_modulos` con los **modulos vivos**, y `json.dumps` no puede serializar
un modulo, luego el final reventaba aunque los diez parciales ya estuvieran escritos. Ahora se
publican **solo los NOMBRES** de los modulos, mas una tabla de que modulos exige cada afirmacion. En
el JSON publicado:

```json
"contexto_de_modulos": {"p71_04": "p71_04", "p72_01": "p72_01", "v10_11": "v10_11", "v9_04": "v9_04"}
```

### Estado de las veinte afirmaciones

```text
CONFIRMADO 18   REFUTADO 2   ERROR 0   STOP 0   sin ejecutar 0
A0 CONFIRMADO  A1..A5 CONFIRMADO  A6 REFUTADO  A7 REFUTADO  A8..A19 CONFIRMADO
LAS VEINTE MEDIDAS Y PUBLICADAS
```

### Que se publico en disco

```text
reports/p81_cierre_huecos.json  183426 B
  sha256 81a21033a83c8ffc560a5381cfed4d25f93e99c66ebc3976bc3777a65f381d4b
reports/p81_cierre_huecos.md     15752 B
  sha256 19e32b944db6d2927f4f58d1eb14505ccb2ab90fe9dc454dfe1062eabe7d30c4
21 parciales en reports/: 10 de la corrida 20261001T010045, 10 de esta 20261001T010636,
y el parcial.json de la corrida rota, conservado como evidencia
ningun temporal .tmp_p81
```

**ASCII byte a byte: los dos ficheros finales son ASCII PURO, 0 bytes por encima de 127.**
`reports/p81_cierre_huecos.json` 183426 B y 0, `reports/p81_cierre_huecos.md` 15752 B y 0. El
**verificador oficial** `tools/verificar_caracteres.py` sobre el proyecto entero: **salida 0, 0
caracteres no permitidos**.

**NO_DISPONIBLE en el `.md`: 0. Y 0 tambien en el `.json`.** El `.md` tiene 219 lineas y **20 filas de
tabla de afirmaciones, todas pobladas**. El bug que llenaba el informe de huecos en silencio **queda
descartado por quinta vez**, y ahora con el informe completo publicado y no con un fallo temprano.

### Los cinco resultados de P81

**1. El bloque 1 NO apoya a la suma.** A5 **CONFIRMADO** por magnitud, A6 **REFUTADO** por direccion,
titular **`EL_TEST_NO_APOYA_A_LA_SUMA`**, `apoya_a_la_suma` falso, `direccion` `favor_a_la_media`, y **la
MEDIA gana por 0.700000**. Reparto de victorias **2 a 4**, en zona de no conclusion, con curva **no
monotona y 2 cambios de signo**. Las dos lecturas de la zona betonal, con umbral 0.99:

```text
por BRAZO, criterio declarado y el del primario
  MEDIA  excluye 0.35 y 0.50          mejor no betonal 0.7388888888888889 en lr 0.20
  SUMA   excluye 0.20, 0.35 y 0.50   mejor no betonal 0.03888888888888889 en lr 0.10
  diferencia_por_brazo = -0.7000000000000001   GANA LA MEDIA por 0.700000   primario lr 0.1
por PUNTO, alternativa calculada y publicada, no usada para el primario
  sobran 0.02, 0.05 y 0.10   MEDIA 0.005555555555555556   SUMA 0.03888888888888889
  diferencia_por_punto = +0.03333333333333333   GANA LA SUMA por 0.0333
las_dos_lecturas_coinciden_en_signo = False
```

**No coinciden en signo**, y esa es la lectura honesta: el signo del resultado depende del criterio de
exclusion de la zona betonal. El texto publicado lo dice sin rodeos: "la diferencia entre los dos
mejores de la rejilla es -0.700000, de magnitud suficiente para discriminar pero con el SIGNO en contra
de la SUMA. La MEDIA gana por 0.700000. El test discrimina por magnitud, pero su direccion es
contraria a la hipotesis de A6, y eso se publica: el bloque 1 NO apoya al objetivo de la referencia".
Y el recuento anade: "ZONA DE NO CONCLUSION: el reparto de victorias por learning rate queda en 2 a 4 y
eso es ruido con tres semillas. El primario NO debe leerse como confirmacion".

**2. La pata EJECUTABLE del anclaje de H1 no apoya la suma.** Luego **la base unica que sostiene que el
objetivo no era SGNS es la pata DOCUMENTAL, el codigo C de word2vec**, sha256
`df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4`, 715 lineas, el mismo que anclan el
pre-registro y la enmienda 1.

**3. Limite permanente.** El anclaje a una implementacion externa queda **SIN HACER** porque gensim no
esta instalado y no compila con Python 3.14.7. Autoriza afirmar sobre el **OBJETIVO**, no sobre la
calidad relativa del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor que V03.**

**4. El bloque 2, con el tokenizador real de `v9_04`.** Proporcion de posiciones **OOV 0.041249**, de
**relleno 0.730919**, **OOV sobre los tokens que llegan a la matriz 0.153295**, y **2008 tipos fuera de
vocabulario**. Con el desglose: posiciones_totales 94912, tokens_que_entran_en_la_matriz 25539,
tokens_cubiertos 21624, tokens_oov 3915, posiciones_de_relleno 69373, vocabulario_usado 3439,
n_filas_de_dev 1483, max_len 64, filas_con_cero_tokens_cubiertos 1,
fraccion_de_filas_sin_cubiertos 0.0006743088334457181. **El 11.2726 por ciento de antes NO queda
reproducido**, `queda_reproducida: false` y `variante_que_la_reproduce: null`, porque las tres cifras
de la discusion usan normalizaciones distintas y MAX_LEN 60 frente a MAX_LEN 64 y
`normalizar_emb_normal` de `v9_04`. **Y el reparto entre canal de relleno y canal de OOV tiene
magnitud**: 69373 posiciones de relleno frente a 3915 tokens OOV, de 94912 posiciones. El 0.155020 del
prototipo no reproduce ninguna de las cuatro cifras.

**5. El bloque 5.** El desfase de una epoca del scheduler **es un defecto real de codigo**, porque el
plan usa `range(EPOCAS)` y el entrenamiento usa `enumerate(..., start=1)`, y **su consecuencia
numerica es DESPRECIABLE**: diferencia relativa de las medias **0.002144** contra un umbral de **0.005**.
A18 y A19 los dos **CONFIRMADO**, con `total_lotes_plan 1395` y `total_lotes_recorridos 1398`, y el
recorte declarado de las primeras 2000 lineas, 53243 tokens, con la advertencia de que los recuentos
absolutos no son los de la corrida real.

### El patron de proceso, que es informacion cara

**Cinco fallos en la serie de P81, todos de contrato en la frontera entre capas, dos de ellos de
calculo y el resto de cableado y de persistencia, y ninguno cubierto por los checks de humo hasta que
se reescribieron los casos de frontera.**

```text
1  KeyError de clave literal en A6, al buscar para SUMA la clave m1_media   calculo, corregido
2  AttributeError de RE_TOKEN en A18, por modulo equivocado al llamado      cableado, corregido
3  la tabla de origenes del comprobador de contratos, a medio construir     cableado, corregido
4  el escritor incremental publicando sobre su nombre definitivo             persistencia, corregido
5  contexto_de_modulos con modulos vivos dentro del cuerpo del informe       persistencia, corregido
```

**La leccion operativa, y es la que hay que llevarse:** un modo de humo que ejercita funciones con
datos sinteticos **no verifica el contrato entre capas**. Los 126 checks pasaban enteros mientras el
recorrido real reventaba, porque el humo comprobaba que cada funcion hiciera lo suyo con su entrada
propia y nunca que el llamador le pasara el modulo correcto, ni que lo que se fuera a publicar fuera
serializable. **Y hace falta comprobar por mutacion que cada comprobacion discrimina**: que una
mutacion deliberada baje el recuento, porque un caso que nunca puede fallar no es una prueba.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura, y A0 lo confirma en sha256, tamano y mtime_ns
```

### Estado

**P81 queda CERRADO Y PUBLICADO.** Las veinte afirmaciones medidas y publicadas: 18 CONFIRMADO,
2 REFUTADO (A6 y A7), 0 ERROR, 0 STOP, 0 sin ejecutar. Informe en JSON y en markdown, ambos ASCII
puros, con 0 ocurrencias de `NO_DISPONIBLE` y 20 filas de tabla pobladas. El bloque 1 **no apoya a la
suma**, y eso se publica tal cual.

```text
V03 remains the best verified leaderboard result.
```
## P82 - 2026-10-01 - CORRIDA FINAL. PUBLICADO. 23 de 23 CONFIRMADO, 0 REFUTADO, 0 ERROR, 0 STOP

Corrida buena, marca `20261001T172549Z_p2427889`, con los CINCO brazos. Informe final escrito y
verificado, y NO se ha vuelto a ejecutar ni editado nada para producing este registro.

```text
experiments/p82_01_rnn_atencion.py
  b19802dfa44b3b38ce6caf42829944e7320be82953f56ef60fb10a04c69328e6
reports/p82_rnn_atencion.json       94089 B
  3d0bc0d16a1ea8b427e0f2e0cfd1a765d962370477829ce9d11a2c81de7a9686
reports/p82_rnn_atencion.md          11887 B
  af808f8ecc4949bad2fbe81610b56e2300524f95267c19e368017d62e1cdcc7a
```

**Las veintitres afirmaciones, las veintitres CONFIRMADO.** Cero REFUTADO, cero ERROR, cero STOP,
cero sin ejecutar. A0 a A22.

### Los cinco brazos, media y desviacion entre las diez semillas

| brazo | nucleo | atencion | embedding | media | sd |
|---|---|---|---|---|---|
| B1 | GRU | NO | W2V_100d_congelado | 0.639659 | 0.011615 |
| B2 | GRU | SI | W2V_100d_congelado | 0.637554 | 0.014905 |
| B3 | LSTM | NO | W2V_100d_congelado | 0.615163 | 0.012924 |
| B4 | LSTM | SI | W2V_100d_congelado | 0.617194 | 0.020434 |
| B5 | GRU | SI | W2V_100d_permutado | 0.560747 | 0.019137 |

### Las cinco comparaciones pareadas, con t de Student sobre los deltas por semilla

| id | definicion | naturaleza | media | IC95 | signos + |
|---|---|---|---|---|---|
| P1 | B2 - B1, atencion en GRU | PRIMARIA | -0.002105 | [-0.012406, 0.008196] | 4 de 10 |
| P2 | B4 - B3, atencion en LSTM | SECUNDARIA | 0.002031 | [-0.016628, 0.020690] | 7 de 10 |
| P3 | B2 - B5, geometria del embedding | SECUNDARIA_DESCRIPTIVA | 0.076807 | [0.059823, 0.093791] | 10 de 10 |
| P4 | B1 - B5, lo mismo sin atencion | DIAGNOSTICA_DESCRIPTIVA | 0.078912 | [0.067144, 0.090680] | 10 de 10 |
| P5 | B1 - B3, GRU contra LSTM | DIAGNOSTICA_DESCRIPTIVA | 0.024496 | [0.013551, 0.035441] | 10 de 10 |

Metodo del intervalo que decide: `t_pareado_sobre_semillas`, t de Student sobre los deltas POR
SEMILLA. No es bootstrap.

**Limite de deteccion.** Desviacion planificada 0.021588, desviacion **medida** en esta corrida
0.01440007631814541, MDE con esa desviacion 0.014342, y gap **leido del informe de P78**
0.117450, no escrito a mano. Potencia objetivo declarada 0.8, potencia real en el ancla con la
desviacion medida 0.9979.

**Cronometro.** 270 ajustes de red, 5850.165344 segundos reales de un presupuesto de 5400. El
artefacto congelado se comprobo antes y despues y no cambio: sha256
`302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`, 29478868 bytes, mismo
`mtime_ns`, `coinciden_antes_y_despues` True. A0 lo confirma ademas en las tres cosas.

### La lectura, que es lo que importa

El efecto de la atencion es **indistinguible de cero en los dos nucleos y con signos opuestos**:
-0.002105 en GRU y +0.002031 en LSTM, los dos muy por debajo del minimo detectable de 0.014342. El
pre-registro decia que si la atencion tuviera el mismo signo en los dos nucleos seria la senal mas
fuerte posible, y que con signos opuestos seria la prueba mas fuerte de que era ruido. **Salio lo
segundo.**

**Y esto NO es que la atencion no aporte.** Es que este diseno, a diez semillas, no tiene potencia
para distinguir el efecto de cero. El limite de deteccion se calculo ANTES de correr, con potencia
0.8993 a diez semillas, y por eso el resultado se declara **EXPLORATORIO** y no confirmatorio. Un
resultado negativo NO se lee como que la atencion no aporta: se lee como que el efecto es inferior
a lo que este diseno resuelve. Cadena literal del informe:

```text
UN_RESULTADO_NEGATIVO_AQUI_NO_AFIRMA_QUE_LA_ATENCION_NO_APORTA_AFIRMA_QUE_EL_EFECTO_ESINFERIOR_AL_LIMITE_DE_DETECCION_DE_ESTE_DISENO
```

Lo que si es solido, con el signo coherente en diez de diez semillas y el intervalo entero por
encima de cero, es que **la geometria del embedding importa**: contra la matriz permutada,
0.076807 (P3) y 0.078912 (P4). Esa es la medicion que nlp-specialist pidio en la Fase 5 y que este
proyecto nunca habia hecho. Separa **existe geometria** de **esa geometria esta asignada a estos
tokens**, y la respuesta es que si.

**Y hay un hallazgo que no estaba en las hipotesis: GRU supera a LSTM por 0.024496, con diez de
diez.** No estaba en ninguna hipotesis del pre-registro y se dice como tal, no como si se hubiera
pedido. El mejor brazo se queda a **0.137552** de V03, que sigue siendo el mejor resultado
verificado.

### La cadena de incidencias, ocho intentos

Uno por intento, y ninguno se borro. **Ocho intentos = seis corridas en modo completo mas dos de
humo.** El detalle del recuento esta en la nota de recuento mas abajo, porque **no son ocho
directorios de parciales**: hay cinco subdirectorios con marca de corrida dentro de
`reports/p82_parciales/`, cinco ficheros sueltos en la raiz de ese mismo directorio, y cuatro
directorios de humo aparte.

1. **La primera corrida salio por CPU porque el script no tenia forma de usar la GPU.** Ni una sola
   mencion a GPU, ROCm o cuda, ni una llamada `to` con argumento de dispositivo. Consumio 66
   minutos sin cerrar el primer bloque, con 30 nucleos y la GPU al 7 por ciento. Se paro con su PID.
   **La causa era del script y no del encargo de ejecucion**, y se corrigio en el script.
2. **La medicion del coste por dispositivo** dio que la GPU gana por un factor dos, no mas
   (1.24 s por ajuste frente a 2.43 s con 16 hilos), y que **30 hilos es un 32 por ciento PEOR que
   16** (3.22 s frente a 2.43 s), por sobrecarga de planificacion con tensores pequenos. Ese dato
   viene de un micro-benchmark con la clase real del script y esta en la Enmienda 2 con sus numeros.
3. **La primera puerta se nego a publicar** porque A4 y A22 no se evaluaban en el modo completo sino
   solo en el humo, y las dos tienen marca de STOP. El script declaraba veintitres afirmaciones y el
   completo solo media veintiuna. La puerta lo detecto.
4. **Al autorizar el tramo 2 aparecio una contradiccion en A19:** la nota de tramo 2 se escribia
   SOLO cuando el tramo 2 no se ejecutaba, luego con la autorizacion la nota quedaba vacia y A19, que
   exige nota cuando se excede el presupuesto, fallaba. La primera corrida autorizada ejecuto los
   CINCO brazos y **se detuvo en A19 con STOP** tras 96 minutos contra 90 de presupuesto, con hito
   observado de 3666.928448 s contra 2400 de hito. Los 29 parciales estan a salvo en
   `reports/p82_parciales/20261001T141035Z_p2054042/`. **Se corrigio la nota, no A19**, porque A19
   es correcta y fue ella la que cazo el fallo.
5. **Un bug de cableado de una linea, encontrado aqui y no por el revisor:** el bucle del tramo 2 se
   controlaba con `if not nota_tramo2`, y con la autorizacion la nota es un texto largo y no vacio,
   luego **el tramo 2 NUNCA se ejecutaba aunque se autorizara**. Ahora se controla por
   `if decision_tramo2["ejecutar_tramo_2"]`, el BOLEANO DE DECISION, que no es lo mismo que la nota.
6. **El criterio de no negatividad de A17 se aplicaba tambien a `neto`, que es una resta.** Los tres
   netos reales son negativos, porque el brazo hace mas dano del que rescata frente a V03 (AG -126,
   HS -160, TR -94), y la comprobacion trataba como error de datos el hallazgo central del
   experimento. **Corregido el criterio, no relajada la afirmacion.**
7. **Dos afirmaciones comparaban una constante consigo misma** (LITERAL_TRAMO2 era una constante
   unica, y la tabla de limite de deteccion tambien), **y cinco con tupla anidada**, que hacian que
   TODO saliera CONFIRMADO, porque una tupla no vacia es truthy y el estado nunca veia el booleano.
   **Ninguno lo cazo el humo.**
8. **La leccion, que aplica a todo el proyecto: un modo de humo que ejercita funciones con datos
   sinteticos NO verifica el contrato entre capas, y un caso de humo que no puede fallar no prueba
   nada.** Ademas, un hallazgo scientifico relevante puede hacer fallar una comprobacion bien
   escrita si el criterio de la comprobacion estaba mal definido, y hay que **mirar el criterio
   antes de culpar a los datos**.

### Recuento de parciales, que NO se borran

```text
reports/p82_parciales/  5 subdirectorios con marca de corrida, 143 ficheros
  20261001T065649Z_p1233382    26 ficheros   intento 1
  20261001T080906Z_p1362206    28 ficheros   intento 2, el del informe del tramo 1
  20261001T141035Z_p2054042    29 ficheros   intento 3, el que se detuvo en A19 con STOP
  20261001T161225Z_p2293418    29 ficheros   intento 4
  20261001T172549Z_p2427889    31 ficheros   intento 5, el bueno, con los cinco brazos
  raiz: 5 ficheros sueltos, de la marca 20261001T051317Z_p1023604, el intento en CPU
  148 ficheros en total
reports/humo_p82_*  4 directorios, 44 ficheros cada uno, 176 en total, 2 marcas de humo
```

**El informe del tramo 1 se conserva y no se borra**, porque es evidencia de una corrida con menos
brazos, y no es el estado final de P82:

```text
reports/p82_rnn_atencion_tramo1.json
  dd87d42e9a5632347cb737240ec223563fd8b9ec123f9291353cf05bfbdf855e
reports/p82_rnn_atencion_tramo1.md
  bd75cfca1d07a0a86a7986de2d9a94c3e2a5753dfb46cea51f37036bfd9f118d
  marca 20261001T080906Z_p1362206, SOLO TRES BRAZOS, B1, B2 y B5
  21 CONFIRMADO y 2 REFUTADO de 23, tiempo real 3913.429214 s
  no publico los cinco brazos, luego no describe el resultado de P82
```

### Lo que P82 deja declarado para siempre

**P82 es la PRIMERA medicion con early stopping estrictamente anidado con particion interna.** Por
eso sus numeros **NO son comparables** con `v9_04` ni con `v10_11`, que elegian el checkpoint con el
mismo pliegue externo que luego puntuaban, que es la fuga que condeno a P73 y a P74. Queda escrito en
el pre-registro, seccion 4: los resultados de P82 no se comparan con `v9_04` (0.694343) ni con
`v10_11` (0.711639) ni con DL-03 (0.736367).

**Sigue vigente el limite permanente de P81:** no hay anclaje a una implementacion externa de SGNS
porque gensim no esta instalado y no compila con Python 3.14.7. P82 autoriza afirmar sobre el
**OBJETIVO** y no sobre la calidad relativa frente a una libreria.

**Enmienda 1** corrige el coste de la atencion de 8256 a 16448 parametros, porque 8256 asumia una
celda unidireccional de ancho 64 y los brazos son bidireccionales con salida de ancho 128. El valor
real medido es **16448 sin sesgos** y **16576 con sesgos**, y el informe publica las dos cifras y
declara la discrepancia en vez de retocarla. sha256 de la enmienda
`53c8555d1a5ef0fcebb06c4a6f061acf56e568d95242e907387f0af1b03420d8`.

**Enmienda 2** declara el dispositivo de calculo elegido **por medicion**, no por defecto: GPU AMD
Radeon RX 9070 XT, 16 hilos, sin precision mixta, con la consecuencia de que los valores absolutos y
los deltas son los de ese dispositivo y **no se comparan con los de otro**.

**Carlos autorizo el tramo 2**, que excede el presupuesto declarado, y eso quedo registrado como
**autorizacion explicita y no como unnoticed**. A19 lo atesta: situacion
`EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA`, margen -450.139675, presupuesto 5400,
hito declarado 2400, hito observado 3762.75905. El presupuesto y el hito **no se cambiaron**.

### Dos discrepancias de recuento y una de reloj, contra el encargo de registro

Se registran aqui porque son diferencias entre el encargo y lo que hay en disco y en el informe, y
un registro que las calla seria un registro falso.

1. **No hay ocho directorios de parciales.** Hay **cinco** subdirectorios con marca de corrida en
   `reports/p82_parciales/`, mas **cinco ficheros sueltos** en la raiz de ese directorio, de la marca
   `20261001T051317Z_p1023604`, que es el intento en CPU y que nunca llego a tener subdirectorio
   propio. Los **ocho intentos** si son ocho: seis corridas en modo completo mas dos de humo. Son
   cinco directorios, no ocho, y 148 ficheros.
2. **La Enmienda 2 dice "siete ficheros parciales de A0 a A3 y A8" y en disco hay cinco**, que son
   exactamente A0, A1, A2, A3 y A8. El recuento de la enmienda no cuadra con lo que quedo. No se
   corrige la enmienda, que esta commiteada en `9400b3c` y es parte del pre-registro; se deja
   publicado el numero real.
3. **El hito observado del informe es 3762.75905, no 3730.334861.** Ademas hay **dos** cifras de
   tiempo real en el mismo informe, cronometro `5850.165344` y A19 `5850.139675`, que difieren en
   0.025669 s, y el **resumen de cronometro dice `DENTRO_DE_PRESUPUESTO`** con
   `excedio_el_presupuesto` False y la nota publica vacia, cuando el tiempo real **supera** el
   presupuesto en 450.165344 s. A19, que recalcula desde el tiempo real, si lo detecta bien. Es un
   **defecto residual del bloque de resumen del informe, ya publicado y NO corregido aqui**, porque
   este registro no edita informes. Queda apuntado para una corrida posterior que lo arregle, y no
   cambia ninguno de los cinco brazos ni de las cinco comparaciones.

### Fronteras respetadas

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P81 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
```

### Estado

**P82 queda CERRADO Y PUBLICADO** con las veintitres afirmaciones CONFIRMADO, y su resultado es
**EXPLORATORIO** por el limite de deteccion, no confirmatorio. La geometria del embedding esta
medida y es solida; el efecto de la atencion es indistinguible de cero en este diseno.

```text
V03 remains the best verified leaderboard result.
```
## P82 - 2026-10-01 - TERCERA CORRIDA COMPLETA. Cierre. 23 de 23 CONFIRMADO y DETERMINISMO CONFIRMADO

Corrida `20261001T194000Z_p2731658`, de 99 minutos, que **corrige el defecto de redaccion del
cronometro** que la corrida anterior dejo publicado. El script cambio: sha256
`72c3c00fc4b235c4c8bd087abf9f4718a2409b1fdbae50049ccfe77d29fa941c`, 7278 lineas, y el hash anterior
era `b19802dfa44b3b38ce6caf42829944e7320be82953f56ef60fb10a04c69328e6`. El delta es el arreglo.

```text
reports/p82_rnn_atencion_20261001T194000Z_p2731658.json   98723 B
  8c1f613a8d8040815d2a6c3f7f95447861d435d9a75accb25c5a2df394d651db
reports/p82_rnn_atencion_20261001T194000Z_p2731658.md     13266 B
  4e3dc8223cbe92727dc237d391946e55291dc68ec89326137fc611fb9c6ff30e
```

**Las veintitres afirmaciones, las veintitres CONFIRMADO.** Cero REFUTADO, cero ERROR, cero STOP,
cero sin ejecutar. A0 a A22.

### El cronometro, coherente por fin

```text
tiempo_real_segundos                 5918.260188
budget_segundos                      5400
excedencia                           518.260188
situacion_del_presupuesto            EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA
nota_definitiva.excedio_el_presupuesto   True
A19.excedido                         True
A19.situacion                        EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA
A19.margen_segundos                  -518.240782
A19.comprobaciones.la_nota_corresponde_a_la_situacion_real   True
ajustes_de_red                       270
```

**El informe ya no dice a la vez que se supero y que no se supero.** Antes publicaba
`situacion_del_presupuesto` `DENTRO_DE_PRESUPUESTO` con `excedio_el_presupuesto` False y la nota
publicada vacia, con un tiempo real de 5850.165344 s que **supera** el presupuesto de 5400. Ahora las
tres piezas dicen lo mismo y en el mismo sentido.

### DETERMINISMO CONFIRMADO, y esto es lo importante

Los cinco brazos de esta corrida de 99 minutos dieron **EXACTAMENTE los mismos numeros** que la
corrida anterior de 98 minutos, comparados cifra a cifra:

| brazo | corrida 98 min | corrida 99 min | |
|---|---|---|---|
| B1 GRU sin atencion | 0.639659 sd 0.011615 | 0.639659 sd 0.011615 | identico |
| B2 GRU con atencion | 0.637554 sd 0.014905 | 0.637554 sd 0.014905 | identico |
| B3 LSTM sin atencion | 0.615163 sd 0.012924 | 0.615163 sd 0.012924 | identico |
| B4 LSTM con atencion | 0.617194 sd 0.020434 | 0.617194 sd 0.020434 | identico |
| B5 GRU con atencion, permutado | 0.560747 sd 0.019137 | 0.560747 sd 0.019137 | identico |

Y las cinco comparaciones identicas, **incluidos los dictados de delta por semilla**, que son la
prueba dura porque son diez numeros por comparacion y no una media:

| id | media | IC95 | signos + | deltas por semilla |
|---|---|---|---|---|
| P1 B2 - B1 | -0.002105 | [-0.012406, 0.008196] | 4 de 10 | identicos |
| P2 B4 - B3 | 0.002031 | [-0.016628, 0.020690] | 7 de 10 | identicos |
| P3 B2 - B5 | 0.076807 | [0.059823, 0.093791] | 10 de 10 | identicos |
| P4 B1 - B5 | 0.078912 | [0.067144, 0.090680] | 10 de 10 | identicos |
| P5 B1 - B3 | 0.024496 | [0.013551, 0.035441] | 10 de 10 | identicos |

**Que dos corridas independientes den lo mismo bit a bit es lo que hace que estas cifras sean
evidencia y no una medida que se le parezca.** El propio informe lleva escrito el criterio, en
`corrida_anterior.comparacion_obligatoria`: si al reejecutar los brazos no dan estas MISMAS medias,
hay no determinismo y hay que parar antes de publicar. Se cumplio.

El artefacto congelado sigue sin cambios: sha256
`302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`, `coinciden_antes_y_despues` True.

### El defecto, su causa exacta y el arreglo

**La causa era un argumento asumido.** La nota definitiva del tramo recibia un `False` escrito a mano
en vez de la comparacion real del reloj contra el presupuesto, y A19 se evaluaba **antes** de que el
reloj llegara al presupuesto. El efecto era que el informe publicaba que se habia superado el
presupuesto y que no se habia superado a la vez.

**Se arreglo de dos maneras, y las dos son la misma idea:**

1. **`construir_nota_de_tramos` ahora recibe el tiempo real y el presupuesto y calcula la excedencia
   dentro.** La firma pasa a ser `construir_nota_de_tramos(tiempo_real, presupuesto,
   ejecutar_tramo_2, hito, autorizado)`, y la excedencia sale de `tiempo_real - presupuesto`. **No
   existe forma de pasar un booleano por error**, porque el booleano ya no es un parametro: es una
   consecuencia. La tabla `PARAMETROS_DE_CONDICION` marca las posiciones 0 y 1 de esa funcion como
   posiciones que deben llegar calculadas.
2. **A19 es ahora la ULTIMA del pipeline del completo**, porque una afirmacion que vigila el
   presupuesto solo se puede medir al terminar. En el codigo, la llamada a `progreso.evaluar` de A19
   va despues de las de A20 y A21, que antes iban despues.

**El guard antitrunco NO se toco ni se relajo.** Como el destino canonico ya existia, la reejecucion
escribio con sufijo de marca y **declaro en el propio informe que conserva el anterior**:
`informe_anterior_conservado` dice `corrida_actual` `20261001T194000Z_p2731658` y
`motivo` "el destino canonico ya existia y el guard antitrunco se niega a sobrescribirlo, luego esta
corrida escribe con su marca y conserva el anterior intacto", con `ruta_del_anterior`
`reports/p82_rnn_atencion.json`.

**Ademas se anadio una auditoria AST que detecta literales booleanos en posiciones que deben llegar
calculadas**, `auditar_booleanos_literales`, que cubre **las dos** Niagara de la misma clase que ya
habian pasado en este fichero. Se invoca sobre el propio script y da cero infracciones, y se prueba
por mutacion: `construir_nota_de_tramos(True, 5400.0, False, 2400.0, False)` tiene que verse, y la
llamada con el tiempo real `5850.165344` no. En la puerta del humo el recuento baja de **164 a 162**
con esa mutacion, que es la prueba de que la deteccion **discrimina** y no solo existe.

### La leccion, y es la segunda vez que ocurre en este fichero

**Un argumento asumido, un `True` o un `False` escrito a mano donde deberia haber una comparacion.**
Ya paso con la tupla anidada que hacia que TODO saliera CONFIRMADO, y ahora con este `False` que hacia
que el informe se contradijera a si mismo. **El patron comun es el mismo y la defensa es la misma:**
una comprobacion que detecte el patron, mas un caso de humo que por mutacion demuestre que la
deteccion discrimina. **Un argumento que se asume no se verifica nunca, y lo que se verifica solo
cuando algo va mal.**

### Los tres pares de informes, que NO se borra

```text
reports/p82_rnn_atencion_20261001T194000Z_p2731658.json  8c1f613a8d804081  la vigente, unico con el cronometro coherente
reports/p82_rnn_atencion.json                            3d0bc0d16a1ea8b4  la de 98 min, con el defecto de redaccion
reports/p82_rnn_atencion_tramo1.json                     dd87d42e9a563234  tramo 1, tres brazos, 21 CONFIRMADO y 2 REFUTADO
```

Los parciales **NO se borran** y ahora son **once entradas**: seis subdirectorios con marca de corrida,
`20261001T065649Z_p1233382` con 26 ficheros, `20261001T080906Z_p1362206` con 28, `20261001T141035Z_p2054042`
con 29, `20261001T161225Z_p2293418` con 29, `20261001T172549Z_p2427889` con 31 y
`20261001T194000Z_p2731658` con 31, mas cinco ficheros sueltos en la raiz del directorio, para 179
ficheros en total. Los seis directorios de humo, con 44 ficheros cada uno y 264 en total.

### Cierre de fase

**P82 cierra la via de la RNN con atencion que Carlos autorizo abrir directamente, sobreescribiendo la
recomendacion propia y la del especialista de competencia.** El resultado **no la respalda y no la
refuta de forma concluyente**: el efecto es menor que lo que este diseno resuelve. El RNN completo se
queda a **0.137552 de V03**, luego **no es candidato**. La unica contribucion solida de P82 es la
medicion de la geometria del embedding contra la matriz permutada, que este proyecto no habia hecho
nunca y que separa **existe geometria** de **esa geometria esta asignada a estos tokens**. Y P82 es
la **primera medicion con early stopping estrictamente anidado**, luego sus numeros **no son
comparables** con `v9_04` ni con `v10_11`.

```text
V03 remains the best verified leaderboard result.
```

## 2026-10-02 (P84-01 - DL-01 corregido, corrida 2)
- 12:32 | P84 | Re-ejecucion completa tras corregir el informe MD (KeyError 'contadores'); evidencia previa preservada en `reports/p84_01_dl01_previo_fix.json` y `features/p84_01_oof_mlp_previo_fix.npz` | `/home/carlos/venv/bin/python experiments/p84_01_dl01_corregido.py` con GPU libre (D-008, 5/5 muestras) | 15 combinaciones (5 pliegues x 3 semillas) en 248.33 s, device cuda (AMD Radeon RX 9070 XT), 0 incidencias.
- Resultado primario (semilla 42, umbral 0.5): F1 0.758301 vs V03 0.777211 | delta -0.018910 | pliegues mejorados 1/5 | IC95 [-0.033910, -0.004150]; anidada 0.756968 y congelada 0.761659.
- Gate: base no cumplido (delta < +0.005, pliegues 1/5, IC95 no excluye cero) y D-057 no superado (delta < +0.010 con aumento de capacidad); estable True (mismo signo en 3 semillas) y brecha +0.004418 <= 0.05 | decision: CERRAR DL-01 COMO ARTEFACTO DE PROTOCOLO.
- Informes: `reports/p84_01_dl01.md` y `reports/p84_01_dl01.json`; OOF en `features/p84_01_oof_mlp.npz`; 15 checkpoints en `modelos/p84/`; 15 parciales en `reports/p84_01_parciales/`; fila P84-01 anadida una sola vez a `experiments.csv` (23 columnas).
- Reproducibilidad: max|delta| = 0.0 en deltas por semilla, F1 de las tres lecturas y probabilidades del npz nuevo contra el previo (tolerancia 1e-12); determinismo interno max|delta p| 0.0 <= 1e-6.

## 2026-10-02 (P84-02 - fastText supervisado, EXP-02)
- 13:01 | P84 | Corrida completa CPU (-thread 1) en 206.36 s, 5 pliegues x 2 brazos x 3 etiquetas, 0 incidencias | `/home/carlos/venv/bin/python experiments/p84_02_fasttext_supervisado.py` (log /tmp/opencode/p84_02_run.log) | Gate 0 OK: hashes, dev 1483 / holdout 0 / test 0, identidad V03 exacta (delta 0, tol 1e-12) | determinismo (principal, HS, pliegue 1) max|delta| 0.0, identicas True.
- Resultado primario (anidada): principal 0.583398 vs V03 0.777211 | delta -0.193813 | pliegues 0/5 | IC95 [-0.217172, -0.171220]; umbral 0.5: 0.584529; congelada: 0.387112. Control sin subwords: anidada 0.737771 | delta -0.039441 | 0/5 | IC95 [-0.054696, -0.024475]. principal vs control: -0.155385.
- ensemble_gain principal +0.000000 con IC95 [0,0] (1483/1483 celdas igualadas: proba principal clavada en 0.5, avg.loss ~ ln 2 reproducido en diagnostico aislado; el test de complementariedad queda vacio, no evidencia de no-complementariedad); control -0.028430 IC95 [-0.041775, -0.015810]. Correlaciones globales: principal r=0.519/rho=0.585; control r=0.919/rho=0.906.
- Gate: standalone NO y complementariedad NO -> CERRAR fastText supervisado sin rejilla nueva; elegible EXP-06 = False. Anomalia: brazo principal subentrenado (E*=5 en 5/5 pliegues, perdida en azar); brazo control sobreajustado (E*=20, brecha 0.269) pero tambien por debajo de V03.
- Informes: `reports/p84_02_fasttext.md` y `reports/p84_02_fasttext.json`; OOF en `features/p84_02_oof_fasttext.npz`; 10 parciales en `reports/p84_02_parciales/`; fila P84-02 anadida una sola vez a `experiments.csv` (452 filas de datos, 23 columnas en todas); temporales `experimentos/p84_02_tmp/` limpios (sin .bin/.vec/.txt).

## 2026-10-02 (P84-03 - ROS vs reweighting, EXP-03)
- 13:31 | P84 | Corrida completa CPU en 17.48 s (5 pliegues x 4 brazos x 3 etiquetas, 0 incidencias) | `/home/carlos/venv/bin/python experiments/p84_03_ros_equivalencia.py` (log /tmp/opencode/p84_03_run.log; script sha256 13e500f9... coincide con Enmienda 3) | Gate 0 OK: dev 1483 / holdout 0 / test 0, identidad A=V03 max|delta proba| 0.0, celdas distintas 0 (tol 1e-12).
- Equivalencia B1-A: max|delta proba| 7.188e-02 (tol 1e-3, NO PASA) y |delta F1| congelado 0.000266 (tol 0.001, pasa) -> REFUTACION PARCIAL con etiqueta CLOSED BY EQUIVALENCE; causa declarada: redondeo entero (recuentos q/q+1 no uniformes en 15/15 ajustes B1, desviacion max de peso relativa 0.556222; liblinear convergio en <=5 iteraciones, sin contribucion del solver).
- Deltas congelados vs A=V03: B1 +0.000266 IC95 [-0.004077,+0.004838] 3/5 pliegues; B2 -0.044992 IC95 [-0.062517,-0.028327] 0/5; B3 +0.000210 IC95 [-0.004257,+0.004751] 3/5; rescate/dano 0/0 en los tres brazos (predicciones binarias identicas en las lecturas).
- C_efectivo B3 (HS/TR/AG): p1 [2.2496,2.7960,2.4722], p2 [2.2530,2.8027,2.4722], p3 [2.2511,2.8003,2.4667], p4 [2.2544,2.8003,2.4701], p5 [2.2530,2.7993,2.4722]; recuentos B1 1-2 copias (HS/AG) y 2-3 (TR) sin positivos sin copia; B2 (r=0.5) deja 87-185 positivos sin copia por redondeo half-to-even.
- Decision confirmada: "ROS cerrado por equivalencia con reweighting y class_weight; no promovible (CLOSED BY EQUIVALENCE)"; ningun brazo a Gate 2 ni a EXP-06. Informes `reports/p84_03_ros.json/.md`, OOF `features/p84_03_oof_ros.npz` (A/B1/B2/B3 1483x3), 5 parciales nuevos en `reports/p84_03_parciales/`; fila P84-03 anadida una sola vez a `experiments.csv` (453 filas de datos, 23 columnas en todas).

## 2026-10-02 (P84-05 - EM con shrinkage, EXP-05)
- 16:38 | P84 | Corrida completa CPU (numpy puro) en 1.0 s | `/home/carlos/venv/bin/python experiments/p84_05_em_shrinkage_simulado.py` (log /tmp/opencode/p84_05_run.log; script sha256 9c6d62c7... coincide con Enmienda 5) | Gate 0 OK: dev 1483 / holdout 0 / test 0, identidad V03 exacta (corrected_cv 0.7772114525165721, V03@0.5 0.771778476256122, tol 1e-12).
- Delta medio (promedio de las 5 rho) -0.008394 vs umbral +0.002 | pliegues LOFO positivos 2/5 (p3 +0.000925, p4 +0.003307) | IC95 bootstrap con pesos [-0.015763, -0.001231].
- Descomposicion: componente EM +0.020019 (IC95 [+0.008198, +0.032761]), componente umbral -0.028414 | sanity rho=1 -0.005043 < 0 | guardarrail de tasa NO: max desviacion 0.014835 > 0.005 (AG 0.014835; HS -0.002023 y TR -0.004720 cumplen).
- Gate unico NO PASA (delta, 4/5 pliegues, IC95 positivo y guardarrail) -> decision: CERRAR LA FAMILIA EM/threshold (CLOSED); sin Gate 2 ni EXP-06; sin test, sin holdout, sin submissions.
- Informes `reports/p84_05_em.json/.md`, 5 parciales `reports/p84_05_parciales/p84_05_p1..p5.json`; fila P84-05 anadida una sola vez a `experiments.csv` (454 filas de datos, 23 columnas en todas, sin ids duplicados).

## 2026-10-02 (P84-04 - SGNS corregido y anclado, EXP-04)
- 18:34 | P84 | Corrida completa GPU en 15511.17 s (3 semillas x brazos S/M/R, R no entrena), 0 incidencias | `/home/carlos/venv/bin/python experiments/p84_04_sgns_corregido_anclado.py` (log /tmp/opencode/p84_04_run.log, EXIT=0) | Gate 0 OK: hashes congelados coinciden, dev 1483 / holdout 0 / test 0, identidad V03 max|delta proba| 0.0; ancla ANCLA LOGRADA (algebra err 1.28e-16/1.16e-16, gradiente 4.42e-08, pureza 1.0 oficial y nuestra en 16 palabras; fastText oficial: intento 1 NaN codigo -6, intento 2 OK).
- Resultado congelado (proba promediada de 3 semillas; por semilla S 0.613333/0.610643/0.613621, M 0.611190/0.602817/0.607611, R 0.616519/0.614104/0.607417): S 0.626091 | M 0.622888 | R 0.651825 | V03 0.777211 (@0.5 0.771778); anidada S 0.635549 | M 0.621945 | R 0.646108; flujo pareado S/M identico por semilla (hash_pares 1dca8e47, 8819fb86, 65ee53e8).
- Deltas pareados congelados: S vs M +0.003204 (3/5 pliegues, IC95 [-0.012606, +0.018844], rescate/dano 334/318) | S vs R -0.025734 (1/5, IC95 [-0.046887, -0.004500], 463/596) | S vs V03 -0.151120 (0/5, IC95 [-0.175340, -0.127571], 259/894) | ensemble_gain S+V03 -0.049004 (IC95 [-0.066411, -0.031799]).
- Gate: S vs M NO (3/5 e IC cruza cero), S vs R NO, S vs V03 standalone NO (menor que +0.005) y complementariedad NO -> CERRAR la linea SGNS corregido con el resultado verificado (sin rejilla nueva); elegible EXP-06 = False; brecha D-057 OK (S/M/R/V03 0.138356/0.137537/0.142208/0.204721; max +0.05). Anomalia: R (matriz aleatoria congelada) supera a S y M en congelada y anidada.
- Rutas: `reports/p84_04_sgns.json/.md`, `reports/p84_04_ancla.json`, `features/p84_04_oof_emb.npz` (proba por brazo/semilla 1483x3), 9 embeddings `modelos/p84/emb_2026101{6,7,8}_{S,M,R}.npz`, 9 npz + 9 json en `reports/p84_04_parciales/`; fila P84-04 una sola vez en `experiments.csv` (455 filas, 23 columnas en todas); temporales de entrenamiento ausentes (solo evidencia del ancla en `experimentos/p84_04_tmp/`, 609 KB) y humo conservado por diseno.

## 2026-10-02 (P84-03 - reejecucion por bug de conteo, Enmienda 3bis)
- 19:06 | P84 | Reejecucion completa CPU en 20.2 s tras Enmienda 3bis (sha256 2a837416...; log /tmp/opencode/p84_03_run2.log) | bug corregido: rescate/dano usan pred_ref; recuentos corregidos B1 61=32/29, B2 446=207/239, B3 63=34/29 sobre 4449; identidad A=V03 exacta (max|delta proba| 0.0); informe y fila CSV regenerados (P84-03 una sola vez, 455 filas de datos, 23 columnas en todas).
- Equivalencia B1-A: max|delta proba| 7.188e-02 (NO PASA, tol 1e-3) y |delta F1| congelado 0.000266 (pasa, tol 0.001) -> REFUTACION PARCIAL con etiqueta CLOSED BY EQUIVALENCE; deltas congelados vs A=V03: B1 +0.000266 IC95 [-0.004077,+0.004838] 3/5; B2 -0.044992 IC95 [-0.062517,-0.028327] 0/5; B3 +0.000210 IC95 [-0.004257,+0.004751] 3/5; C_efectivo y decision sin cambios; sin holdout/test/submissions.

## 2026-10-03 (Notebook consolidado: autocontenido, completo y limpio)
- 00:11 | Notebook | `experiments/p84_06_notebook_completo.py` regenera `notebooks/10_final_pipeline.ipynb`: 452 -> 457 celdas; cobertura 488/488 identificadores (455 del registro mas 33 ayudantes del inventario); `tools/verificar_caracteres.py` en 0 errores; todas las celdas de codigo compilan.
- Autocontenido: las funciones de `src/data.py`, `src/evaluation.py` y `src/embedding_utils.py` quedan embebidas en la celda de definiciones; sin imports de `src` ni inserciones en `sys.path` (verificado).
- Completo: seccion 127 (fase P84 con sus cinco experimentos y EXP-06), anexo A (seccion 128) con la lista completa y verificacion de completitud (seccion 129).
- Limpieza: simbolos no ASCII fuera de la ortografia espanola sustituidos (guion largo, signo menos, comillas angulares, simbolo de seccion, alpha, etc.).
- Smoke test: configuracion, definiciones embebidas, datos, metricas, reproducibilidad y las dos verificaciones ejecutadas sin errores; la ejecucion end-to-end completa sigue pendiente por coste.

## 2026-10-08 (Auditoria de oportunidades legales restantes: Caso B)
- Auditoria adversarial sin entrenamiento, sin GPU, sin test, sin holdout, sin leaderboard y sin submissions: 20 lineas auditadas; 0 ACTIONABLE, 5 BLOCKED_BY_RULES, 1 DUPLICATE, 14 CLOSED.
- Via A (TR): BLOCKED_BY_RULES. Las reglas son silenciosas sobre corregir etiquetas de entrenamiento; el material de revision humana sigue sin completar (80/80 columnas vacias) y el uso modelable cae en familias cerradas; V4-A mide 2,8 por ciento de contradiccion sin explicar el error.
- Via B (near-duplicados train->test): QUARANTINED. Exige autorizacion del organizador; techo medido nulo (V4-F04; sonda v18 con 0 celdas; <=48 celdas de 3462, bajo la banda de deteccion).
- Puerta 2 (H1..H5): las cinco ejecutadas y cerradas en v9 y posteriores (H5 delta exacto 0.0; H4 +0.0006 y despues negativo; H2 0 celdas; H1 V9-04/V10-11; H3 SDR/PLS). P57 (5 preguntas abiertas): ninguna cumple el umbral de reapertura.
- Entregables: `reports/toxic_remaining_legal_opportunities.md/.csv` y `reports/toxic_final_research_gap.md`; seccion 130 del notebook (459 celdas) con celda de verificacion (20 lineas, 0 ACTIONABLE); D-288 en DECISIONS.md.
- Sin push; el cierre documental se registra en un unico commit local. `tools/verificar_caracteres.py` en 0 errores y escaneo byte a byte de lo generado en 0 bytes > 127.
- Estado preservado: V03 FROZEN (0.7772114525165721 / LB 0.78306); candidato A_cost_sensitive_downweight intacto (PROMISING_UNCONFIRMED, external_transfer NEGATIVE con 0.77522, governance PENDING, research EXPERIMENTAL_RESEARCH_CLOSED).

## 2026-10-08 (Comprobacion documental: hipotesis Twitter/X explicitada)
- Comprobacion documental final de la hipotesis "Word2Vec desde cero sobre corpus no etiquetado de Twitter/X": aparece en el CSV (fila 18) y en el informe (seccion 3), pero la fila describia solo el primer P0 (CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY) y omitia la reapertura y la ejecucion.
- Reconstruccion verificada: P0 ampliado (9 candidatos) + correccion de criterio + autorizacion escrita (docs/p73_provenance_autorizacion_corpus.md, D-246) -> corpus mmaguero (CC BY-SA 4.0, periodo cerrado, 10.000.029 tokens, solapamiento cero con train); P71-09, P72, P73, P74, P75, P78, P79, P80, P81 y P84-04 ejecutados; P73/P74 invalidados por contaminacion; P78 sin transferencia util (brecha 0.117450 frente a V03); P84-04 negativo sobre corpus general.
- Clasificacion corregida: CLOSED por medicion (legal_status PERMITTED); el corpus de 597M del encargo original queda bloqueado (sin licencia, periodo abierto); la combinacion SGNS corregido x corpus Twitter no se ejecuto (cambio de corpus: no reabre por directiva).
- Actualizados: informe (seccion 6.6), CSV, gap, seccion 130 del notebook, nota de errata en la seccion 116, D-289 y este RUNLOG. Sin push.

## 2026-10-08 (P85-01: SGNS corregido sobre Twitter/X - Caso C, CLOSED)
- Un entrenamiento GPU (3248.8 s, 5 epocas, loss 3.146705 -> 2.594025, 467556 pasos) sobre `corpus_twitter_tokens.txt` (361994 docs, vocab 77657); cobertura dev 0.887342 (general 0.803623 con la misma normalizacion).
- Resultado: embedding-only twitter 0.629141 vs general 0.629044; fusion congelada twitter 0.708039 (delta -0.069172) vs general 0.709516 (delta -0.067695); 0/5 pliegues; bootstrap [-0.087269,-0.050543]; ensemble_gain 0.5 -0.075858; net fusion -285 (204/489).
- Decision preregistrada: C1..C5 falsas -> Caso C: CLOSED (no reabrir la familia de embeddings); sin ADOPTED; sin submission; sin push. Control general re-puntuado con EMB-NORMAL-TW v1 (el pooling publicado de P84-04 uso texto crudo); binario fastText fuera del alcance (Enmienda 1).
- Entregables: `reports/toxic_twitter_sgns_corrected.md/.csv`; `reports/p85_01_sgns_twitter.json`; `features/p85_01_oof_emb.npz`; `modelos/p85/emb_20261016_S_twitter.npz`; fila P85-01 (456 filas); seccion 131; D-290.
- Compliance: test 0, holdout 0, leaderboard 0, submissions 0, push 0; escaneo byte a byte de lo generado: 0 bytes > 127; `tools/verificar_caracteres.py` en 0 errores.
- Errata (2026-10-08): el apendice a `experiments.csv` y el CSV del informe usaban CRLF; corregido a LF y fila P85-01 re-anexada manualmente (registro: 456 filas, diff de una linea); Enmienda 2 del preregistro con el hash corregido.

## 2026-10-09 (P86-00: auditoria de novedad FM - DUPLICATE, sin ejecucion)
- Auditoria de solo lectura de la propuesta de FM dispersa sobre la representacion de V03: el precedente V25-01/V25-02 (2026-09-27) ya la ejecuto con el mismo protocolo y un control que reproduce V03 bit a bit; k agotado en 4, lambda_v optimo en 1, mejor +0.005487 con neto -40 y dLB esperado -0.0079; cierre CLOSE V25 / CERRAR EL PROYECTO y eje de geometria de representacion cerrado (D-190).
- Decision: DUPLICATE; no se ejecuta nada. 0 entrenamientos, 0 submissions, 0 push.
- Entregables: `reports/toxic_fm_novelty_audit.md/.csv`; fila P86-00 (457 filas); seccion 132; D-291; `experiments/p86_00_auditoria_novedad_fm.py` y `p86_01_integrar_notebook.py`.
- Controles: escaneo byte a byte 0 en lo generado; verificador oficial 0; bateria del notebook ejecutada.

## 2026-10-09 (P87-00: gobernanza de la anotacion TR - LOW_EXPECTED_VALUE, sin ejecucion)
- Auditoria de solo lectura de la via de revision humana de TR: material de 80 filas con columnas humanas vacias (10 en holdout: filtrar a dev, 140 prioritarias); "2,8 por ciento" de V4-A01 = inconsistencia textual demostrable, no tasa de error; error de TR menor dentro de los clusters contradictorios (0.023 vs 0.113); intervenciones +-0.0005; V28-03 sin potencia; P59-C MARGINAL; techo P57 ~+0,009 < banda 0,017.
- Protocolo disenado (no ejecutado) con criterios, doble revisor, adjudicacion, kappa, trazabilidad y originales inmutables; nota de autorizacion preparada y no recomendada.
- Decision: LOW_EXPECTED_VALUE (Caso B): no se solicita autorizacion; no se revisa ni corrige; TOXIC permanece cerrado; near-duplicados QUARANTINED.
- Entregables: `reports/toxic_tr_annotation_governance.md/.csv`; fila P87-00 (458 filas); seccion 133; D-292; `experiments/p87_00_gobernanza_anotacion_tr.py` y `p87_01_integrar_notebook.py`.
- Controles: escaneo byte a byte 0 en lo generado; verificador oficial 0; bateria del notebook ejecutada.
