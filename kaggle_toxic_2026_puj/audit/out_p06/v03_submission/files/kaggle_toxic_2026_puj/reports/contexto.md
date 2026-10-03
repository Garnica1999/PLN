## Fase 1 - ml-architect (2026-09-22)

Estado del arte completado: 45 referencias verificadas hoy (DOI resuelto en Crossref, BibTeX oficial de ACL Anthology,
PDFs leidos en ACL/CEUR-WS, OpenAlex, arXiv, JMLR). Entregables: `reports/estado_arte.md`, `reports/estado_arte.csv`
(45 filas), `references.bib` (45 entradas `autorAnio`) y `reports/hipotesis.md` (H1-H10 con experimento E01-E10).

Hallazgo central: el sistema que GANO la Subtask A en espanol de HatEval (Argota Vega et al. 2019, MineriaUNAM,
DOI 10.18653/v1/S19-2079) uso cero redes neuronales y cero embeddings preentrenados: char n-gramas, n-gramas de palabras
funcionales 2-4, n-gramas de puntuacion 2-4, POS, patrones morfologicos (pron+verbo, verb+adj) y series de palabras
agresivas, con SVM lineal. Macro-F1 espanol 0.73 (baseline SVC 0.701), no el ~0.8 de los transformers
(DOI 10.1016/j.eswa.2020.114120). Compatibilidad verificada por el orquestador: 37/45 metodos compatibles de forma
directa y 8 con advertencia (parcial o prohibido por transformers/embeddings preentrenados).

Evidencia mas util y compatible: Nina-Alcocer 2019 (S19-2072), donde `TFIDF+EMOTIONS` dio EMR 0.711 vs 0.629 del
baseline y `TFIDF+STRUCT` el mejor F1 (0.780). Contraevidencia: combinar tres bloques de features DEGRADA el
rendimiento (Nina-Alcocer) y la votacion de ensambles NO supera al SVM lineal, 0.707 vs 0.701 (SINAI, S19-2084).

Siguiente fase: EDA y caracterizacion de TR (D-005). Prioridad por impacto/riesgo: E01 (char vs word), E04 (union),
E08 (post-proceso jerarquico) y E09 (umbral optimo) son de bajo riesgo y alto retorno; E03 (targeting TR) es la de
mayor incertidumbre y valor narrativo; E10 (stacking) va al final. Regla de decision pre-registrada: mejora >= +0.005
absoluto en la metrica primaria y en >= 3 de 5 folds.

## Fase 2 - nlp-specialist (2026-09-22)

EDA profundo, textometria y caracterizacion linguistica de TR. Script `experiments/01_eda_linguistico.py`
(idempotente); salidas en `reports/eda/` (21 tablas), `reports/figuras/`
(19 PNG) y `reports/eda_resumen.md` (formato OBSERVACION-HIPOTESIS-METODO-RESULTADO-INTERPRETACION-DECISION-SIGUIENTE PASO).

Hallazgos clave: (1) el corpus no tiene NINGUN emoji Unicode (normalizacion ASCII) y test tiene 6 bytes no ASCII;
(2) la mencion inicial y la segunda persona dominan la asociacion con TR (dentro de HS=1, n=756: TR=1 519,
TR=0 237); (3) el odio con objetivo colectivo explicito es casi siempre TR=0
(57.0 % de los TR=0 frente a 10.0 % de los TR=1),
y esas filas concentran las URLs; (4) prevalencias moderadas (desbalanceo <= 2.4:1), sin necesidad de remuestreo agresivo;
(5) la deduplicacion por near-duplicado no cambia el orden ni la magnitud de los efectos.

TR (D-005): la evidencia es compatible con la lectura HatEval de rango del objetivo (TR=1 = individuo, TR=0 = grupo/generico)
y deja abierta la lectura literal del enunciado. NO se decide: `reports/eda/tr_ejemplos_para_revision.csv` (40 TR=1 + 40 TR=0
con rasgos anotados) y `reports/eda/fase2_tr0_con_referencia_colectiva.csv` (casos limite) quedan para revision humana.

Hipotesis: H1, H3, H6, H8 y H9 reciben apoyo inicial (indirecto o estructural); H4, H5, H7 y H10 quedan sin evaluar.
Riesgo detectado: la senal de TR puede ser principalmente "es una respuesta a un usuario" (mencion inicial V de Cramer
0.512) y no "el objetivo es una persona"; E03 debe ablacionar la mencion para separarlo.

## Fase 4-5 - ml-coder (2026-09-22)

Barrido de 25 configuraciones de las Fases 4-5: PARTE A preprocesamiento (PRE-00..PRE-09), PARTE B representaciones (REP-01..REP-06) y PARTE C modelos (MOD-01..MOD-04) con 5-fold.
Validacion comun `StratifiedGroupKFold(shuffle=True, random_state=42)` agrupada por near-duplicado y estratificada por la combinacion HSxTRxAG; modelo fijo LR balanced C=1 liblinear en A y B; metrica primaria `f1_media_columnas`.
Referencia `PRE-00` (B1) reproducida en 0.7587 frente al B1 registrado 0.7587 (diferencia +0.000000).
Artefactos: `reports/variantes_representaciones.csv` (ranking), `reports/variantes_representaciones.json` (metricas por pliegue), `src/evaluation.py` (metricas reutilizables) y `experiments/03_variantes_representaciones.py`.

PARTE A: gana PRE-03 (0.7623, +0.0035 vs B1, pliegues 4/5, descartada); siguiente PRE-05 (0.7607); descartadas 9 de 10
PARTE B: gana REP-05 (0.7865, +0.0278 vs B1, pliegues 5/5, adoptada); siguiente REP-06 (0.7855); descartadas 5 de 6
PARTE C: gana MOD-03b_BEST (0.7927, +0.0339 vs B1, pliegues 5/5, adoptada); siguiente MOD-04 (0.7870); descartadas 8 de 9

Mejor representacion de la Parte B: REP-05 (0.7865).
Mejor configuracion global: MOD-03b_BEST (0.7927); variantes adoptadas por la regla pre-registrada (+0.005 absoluto y >= 3/5 pliegues): MOD-03b_BEST (0.7927), REP-05 (0.7865), REP-06 (0.7855), MOD-01_BEST (0.7849), MOD-03a_BEST (0.7723), MOD-01_W12 (0.7699), MOD-03b_W12 (0.7699).
MOD-04 (post-proceso jerarquico sobre el ganador): 0.7870 frente a 0.7927 de su origen MOD-03b_BEST -> -0.0057; resultado negativo de H8 con esta representacion, por lo que no se adopta como configuracion final.
Prueba de emojis: 0 emojis Unicode en train y test (train 100 % ASCII), por lo que la variante de emojis no aplica; evidencia en el JSON.
Todas las variantes no adoptadas quedan documentadas y descartadas con su evidencia en el CSV/JSON; ningun script previo se modifico (02 se conserva por trazabilidad).

## Fase 13a - ml-trainer (2026-09-22)

Refit de la configuracion ganadora `MOD-03b_BEST` (union TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf; LR balanced C=2.0 liblinear x3, umbral 0.5, sin post-proceso jerarquico) con `StratifiedGroupKFold(5, shuffle=True, random_state=42)` agrupado por near-duplicado y estratificado por HSxTRxAG.
F1 OOF reproducida: 0.792659 frente a 0.792659 de la Fase 4-5 (diferencia +0.000e+00, tolerancia 1e-06); redondeada 0.7927: replicacion exacta.
Probabilidades OOF guardadas en `features/oof_mod03b.npz` (ids, y_true, proba (n,3), etiquetas) para ensembles futuros.
Umbrales optimos por etiqueta (rejilla 0.20-0.80 paso 0.01 sobre OOF, nunca sobre test): HS 0.44 (F1 0.7799 -> 0.7897), TR 0.51 (F1 0.8287 -> 0.8296), AG 0.46 (F1 0.7694 -> 0.7774).
f1_media_columnas OOF con 0.5 = 0.7927 -> con umbrales optimos = 0.7989 (+0.0063); mejora por pliegue [0.0102, 0.0063, 0.003, -0.0023, 0.0137].
Umbrales optimos ADOPTADOS (regla +0.005 y >= 3/5 pliegues): +0.0063 absoluto y 4/5 pliegues mejorados; se genera v03.
Modelo final reajustado con todo el train en `models/mod03b_lr_union.joblib` (vectorizador + 3 modelos + umbrales + etiquetas).
Submissions generadas y validadas contra la plantilla oficial: `submissions/toxic_submission_v02_lr_union.csv`; `submissions/toxic_submission_v03_lr_union_thr.csv`.
Registro: `experiments.csv` (TH-01, S02, S03) y `submissions/leaderboard_log.csv` con `kaggle_f1` vacio, pendiente de que Carlos envie las submissions.
Sin fuga: el test solo se uso para predecir; los umbrales y la seleccion de modelo se decidieron con probabilidades out-of-fold. Detalle completo en `reports/fase13a_refit_umbral_metricas.json`.
ADVERTENCIA de honestidad: la ganancia de umbrales (+0.0063) es una cota OPTIMISTA (umbral y metrica comparten el mismo OOF); la verificacion anidada del revisor da +0.0014 con 2/5 pliegues positivos -> v03 es una apuesta y v02 es el control.
