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
  (A1 > A2 > A3). La lectura es que **un tuit repetido es señal, no ruido de recoleccion**,
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
