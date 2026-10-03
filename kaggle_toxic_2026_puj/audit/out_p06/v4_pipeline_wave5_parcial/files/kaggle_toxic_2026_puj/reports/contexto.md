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

## Fase 13b - ml-trainer (2026-09-22)

Ensembles con evaluacion anidada sobre 4 modelos base OOF con el split agrupado de las Fases 4-5/13a (StratifiedGroupKFold(5, shuffle=True, random_state=42), near-duplicados, estratificado HSxTRxAG); sin fuga (vectorizador por pliegue).
Bases en `features/oof_bases.npz` (F1 OOF a 0.5 replicada con tolerancia 1e-6 frente a la Fase 4-5, dimensiones 4024/23407/27431 verificadas): M1 word+LR 0.7587 (B1), M2 union+LR 0.7927 (MOD-03b), M3 union+LinearSVC 0.7849 (sigmoid = decision>0) y M4 char_wb+LR 0.7567 (REP-01).
Diversidad por pares (`reports/ensembles_diversidad.csv`): correlacion de errores 0.63..0.84, desacuerdo a 0.5 0.042..0.106 y Q-statistic en la tabla.
Ensembles anidados (pesos/meta-modelo ajustados en 4 pliegues y evaluados en el restante): ENS-01 0.7906; ENS-02 0.7919 (umbrales 0.7949); ENS-03 0.7836; ENS-04 0.7916.
Ganador: ENS-02 (voto suave con pesos en rejilla simplex (paso 0.1)); F1 anidada 0.5 = 0.7919 (2/5 pliegues mejorados vs MOD-03b); mejor variante con umbrales = 0.7949 (delta vs MOD-03b +0.0023, delta vs el single con el mismo protocolo +0.0007).
Umbrales del ganador: 0.5 fijo = 0.7919; optimo OOF completo = 0.7986 (optimista); optimo anidado = 0.7937; mediana por pliegue = 0.7949 -> estrategia elegida mediana_por_pliegue (umbrales finales HS 0.43, TR 0.48, AG 0.44).
NO se genera submission v04: el ganador no supera a MOD-03b_BEST por +0.005 en la evaluacion anidada (delta +0.0023); se mantienen v02/v03 como control y apuesta.
Limitacion metodologica: las probabilidades base son OOF compartidas (la prediccion de un pliegue viene de modelos que vieron los otros, incluido el de evaluacion), por lo que la estimacion anidada sigue siendo algo optimista; el single se evaluo con el mismo protocolo y el test nunca se uso para decidir.
Artefactos: `features/oof_bases.npz`, `reports/ensembles_diversidad.csv`, `reports/ensembles_resultados.csv` y `.json`; registro en `experiments.csv` (ENS-01, ENS-02, ENS-03, ENS-04).

## Fase 12 - nlp-specialist (2026-09-22)
Analisis de error de `MOD-03b_BEST` sobre probabilidades OOF (sin reentrenar, sin test): `experiments/06_error_analysis.py` -> `reports/error_analysis.md` y 7 CSV en `reports/eda/error_*.csv` + 5 figuras.
F1 verificado: 0.7927 (0.5) y 0.7989 (umbrales 0.44/0.51/0.46); 425 FP y 375 FN; 251/123/101 ejemplos fallan 1/2/3 etiquetas.
Regla estructural `HS=0 => TR=0`: -0.0004 F1 (version completa con AG -0.0049): H8 cerrada como palanca.
Las 5 rebanadas mas debiles (n >= 40, sin artefactos in-sample): tiene_hashtag=1 (n=161, F1 0.586); tiene_pregunta=1 (n=158, F1 0.633); tiene_referencia_colectiva=1 (n=418, F1 0.639); cuartil_longitud=Q4 (n=432, F1 0.645); tiene_mencion=0 (n=909, F1 0.697).
Los FP de HS y AG se concentran en insulto interpersonal con mencion inicial sin colectivo: de 526 textos, 213 (40.49 %) estan etiquetados (0,0,0).
El error de TR vive fuera de las respuestas: F1_TR 0.875 con mencion inicial frente a 0.728 sin mencion ni colectiva y 0.597 sin mencion con colectiva.
AG: los FN contra grupos superan a los FP de broma (el 45.5 % de los FN tiene referencia colectiva frente al 13.9 % de los TP).
Hipotesis nuevas H11-H15 (evidencia, mecanismo, experimento y criterio en el reporte): H11 interacciones de registro bromista/afectivo; H12 bloque de objetivo individual vs colectivo para TR; H13 features de cita/atribucion para el regimen noticia; H14 lexico inducido de deshumanizacion para AG implicito; H15 (cierre) umbrales por rebanada, que se espera que no aporten.
Sin fuga: todo se calcula con OOF de train; los lexicos son in-sample y sus rebanadas se reportan aparte. Registro en `experiments.csv` pendiente para la Fase 14.

## Fase 13c - sonda v04 (2026-09-22)

Sonda de submission v04 decidida por Carlos (1 envio de Kaggle) con el ensemble `ENS-04` (par M2+M3: LR balanced C=2.0 y LinearSVC balanced C=1.0 sobre la union TF-IDF word (1,2)+char_wb (3,5)) y umbrales medianos por pliegue, para medir si los ensembles generalizan al leaderboard; el script `experiments/07_submission_v04_ens04.py` no envia nada a Kaggle.
ASSERT reproducido: F1 anidada con umbrales medianos = 0.798467 frente a 0.798467 de `reports/ensembles_resultados.json` (diferencia +0.0e+00, tolerancia 1e-03); redondea a 0.7985 y la F1 anidada a 0.5 (0.791609) coincide con la publicada.
Umbrales medianos por pliegue usados: HS 0.44 / TR 0.47 / AG 0.44 (mediana de los 5 optimos por pliegue de la rejilla 0.20-0.80 paso 0.01, identicos al registro de la Fase 13b).
ENS-04 eligio el par (M2, M3) con pesos (0.5, 0.5) en los 5 pliegues, por lo que su probabilidad anidada es el promedio simple de ambas bases; el OOF recalculado coincide con `features/oof_bases.npz` (M2 8.3e-12, M3 0.0e+00) y queda como evidencia en `features/oof_ens04_v04.npz`.
Submission `submissions/toxic_submission_v04_ens04_mediana.csv` validada contra la plantilla oficial (1154 filas, ids identicos y en orden, valores en {0,1}, sin nulos ni duplicados); sha256 c37cf1f5f9159b22...; modelo final en `models/fase13c_ens04_m2m3.joblib`.
Positivos predichos en test: HS 537 (46.53 %) / TR 348 (30.16 %) / AG 467 (40.47 %); prevalencias de train: HS 43.67 % / TR 29.98 % / AG 38.24 %.
Registro: `experiments.csv` (S04, f1 0.7985 = estimacion anidada) y `submissions/leaderboard_log.csv` (v04_ens04_mediana con `kaggle_f1` vacio, pendiente de que Carlos envie la submission).
ADVERTENCIA de honestidad: la F1 anidada (0.7985) es una cota algo optimista porque las probabilidades base son OOF compartidas; en v02 y v03 la familia union+char rindio -0.0128 y -0.0158 bajo su local, asi que el LB esperado ronda 0.783 +/- 0.003 y la referencia a batir es v03 (0.78306).
Sin fuga: el test solo se uso para predecir; ni la seleccion del par, ni los umbrales ni ninguna decision usaron el test. Idempotente: dos ejecuciones seguidas dejan el mismo contenido salvo los tiempos.

## Fase 6-8 - ml-coder (2026-09-22)

Features densas para H11 (registro bromista), H12 (objetivo individual frente a colectivo) y H14 (deshumanizacion) en `experiments/08_features_densas_h11_h12.py`, con el split de D-004 (StratifiedGroupKFold(5, shuffle=True, random_state=42), grupos de near-duplicado, estratificado por combo HSxTRxAG).
Bloques densos: H11 22 features; H12 15 features; H14 9 features; escalado con StandardScaler ajustado por pliegue. Los lexicos inducidos (log-odds AG, HS y referencia colectiva) y la lista de palabras frecuentes del detector de nombres propios se inducen DENTRO de cada pliegue de entrenamiento (hallazgo de la revision de la Fase C); el resto de features son deterministas (regex y listas cerradas).
Control MOD-03b_REF recalculado: 0.792659 frente a 0.792659 de `features/oof_mod03b.npz` (coincidencia binaria 1.0000); verificacion OK.
Resultados (f1_media_columnas OOF, delta vs MOD-03b 0.7927, pliegues mejorados): HIB-H12 0.7836 (-0.0091, 1/5); HIB-H11 0.7713 (-0.0213, 0/5); HIB-H14 0.7630 (-0.0297, 0/5); HIB-UNION 0.7610 (-0.0316, 0/5); B5-DENSE 0.6607 (-0.1319, 0/5).
Mejor configuracion: HIB-H12 (0.7836, descartada); mejor candidato de la fase HIB-H12 con -0.0091.
Evidencia anti-sobreajuste (F1 del pliegue de entrenamiento frente a OOF): HIB-H12 0.9724/0.7836 (brecha +0.1888); HIB-H11 0.9704/0.7713 (brecha +0.1990); HIB-H14 0.9612/0.7630 (brecha +0.1983); HIB-UNION 0.9585/0.7610 (brecha +0.1975); B5-DENSE 0.7349/0.6607 (brecha +0.0742).
Control de pipeline: HIB-UNION con las 46 columnas densas puestas a cero da 0.792659 (+0.000000 vs MOD-03b), por lo que la perdida del hibrido es atribuible a las features y no a la concatenacion.
Rebanadas debiles de la Fase 12 (hashtag, pregunta, referencia colectiva, Q4 y sin mencion) frente al control: 2 de 9 mejoran; deltas: {"tiene_hashtag|1": -0.0314, "tiene_pregunta|1": -0.0378, "tiene_referencia_colectiva|1": 0.0137, "cuartil_longitud|Q4": 0.0135, "tiene_mencion|0": -0.015, "region_regimen_noticia|1": -0.0027, "region_no_responde_con_colectiva|1": -0.0196, "tiene_mencion_inicial|1": -0.0057, "cuartil_longitud|Q1": -0.0105}.
Nota de lectura: las rebanadas usan umbral 0.5 (los valores del control difieren de la tabla de la Fase 12, que uso los umbrales por etiqueta 0.44/0.51/0.46) y sirven para comparar control y candidato con el mismo criterio.
Mejor configuracion de las Fases 6-8: HIB-H12 con f1_media_columnas 0.7836 (delta -0.0091 frente a MOD-03b_BEST 0.7927, 1/5 pliegues): descartada. Rebanadas mejoradas del mejor candidato frente al control: 2 de 9.
Artefactos: `reports/features_densas_resultados.csv` y `.json`, `reports/eda/features_densas_slices.csv` y filas B5-DENSE, HIB-UNION, HIB-H11, HIB-H12 y HIB-H14 en `experiments.csv`.
El test nunca se leyo; ninguna decision de la fase usa etiquetas de test. Ningun script previo se modifico.

## Fase 10/27 - ml-trainer (2026-09-22)

Optimizacion acotada de hiperparametros del pipeline ganador con Optuna (TPE, sampler seed=42), presupuesto 50 trials / 20 min y poda MedianPruner (arranque 10, warmup 1, minimo 3 comparables).
Espacio acotado: word n-gramas {(1,1),(1,2),(1,3)} x min_df {1,2,3,5} x max_df {0.9,1.0} x sublinear_tf {True,False}; char_wb {(3,5),(3,6),(2,5),(4,6)} x min_df {2,3,5} x sublinear_tf; LR con C log-uniforme [0.3,10] y class_weight {balanced, none}.
Cada trial = 5-fold del split de D-004 (StratifiedGroupKFold(5, shuffle=True, random_state=42), near-duplicados, estratificado HSxTRxAG) con vectorizador ajustado dentro de cada pliegue; metrica objetivo f1_media_columnas OOF (0.5).
Estudio: 50 trials en 295 s (28 completados, 22 podados); top-5: #24 0.8012; #17 0.7978; #44 0.7968; #16 0.7960; #41 0.7957.
Mejor trial #24 con f1_media_columnas OOF 0.8012 frente a 0.7927 de MOD-03b (delta +0.0086); pliegues por pliegue: mejor [0.805707, 0.800985, 0.780485, 0.80334, 0.815442] frente a referencia [0.801293, 0.78986, 0.777528, 0.793383, 0.801449].
Mejores parametros (JSON): {"word": {"ngram_range": [1, 3], "min_df": 5, "max_df": 0.9, "sublinear_tf": true}, "char": {"ngram_range": [3, 6], "min_df": 2, "sublinear_tf": true}, "modelo": {"C": 3.2575442857761407, "class_weight": "balanced"}}
Regla del proyecto (>= +0.005 y >= 3/5 pliegues): ADOPTADA (delta +0.0086, 5/5 pliegues mejorados).
Advertencia D-012: el delta NO alcanza +0.010; en la familia union word+char (gap local-LB -0.013 a -0.016) una mejora asi debe confirmarse con una sonda de leaderboard antes de darla por real.
Umbrales por etiqueta del mejor trial (rejilla 0.20-0.80 paso 0.01 sobre su OOF): HS 0.48, TR 0.51, AG 0.49; f1_media_columnas 0.8012 -> 0.8039 (+0.0027, cota optimista).
Submission `submissions/toxic_submission_v05_optuna.csv` generada con umbrales por etiqueta (HS 0.48, TR 0.51, AG 0.49) ajustados SOLO sobre el OOF del mejor trial y validada contra la plantilla oficial (1154 filas, ids en orden, valores en {0,1}, sin nulos ni duplicados).
Artefactos: `reports/optuna_estudio.json`, `reports/optuna_trials.csv`, `reports/figuras/fig_25_optuna_historia.png`, `models/fase10_optuna_lr_union.joblib`.
Registro: filas OPT-01, S05 en `experiments.csv` y fila v05_optuna en `submissions/leaderboard_log.csv` (kaggle_f1 vacio, pendiente de que Carlos envie la submission).
Sin fuga: el test solo se uso para predecir cuando se genero la submission; la seleccion del trial, la de los umbrales y la comparacion con MOD-03b usan exclusivamente probabilidades out-of-fold de train.
Idempotente: study_name, semilla, n_trials y timeout fijos; los registros se reescriben por id y no se duplican al re-ejecutar.

## Fase 13d - sondas de robustez (2026-09-23)

Sondas de robustez pre-registradas de D-013 (maximo 2 envios; script `experiments/10_sondas_robustez.py`): reducir la capacidad de la representacion de MOD-03b para probar si mejora la transferencia al test.
Control REF-MOD03b (C=2.0, min_df=2): F1 OOF con 0.5 = 0.792659 frente a 0.792659 de la Fase 4-5 (diferencia +0.000e+00); probabilidades identicas a `features/oof_mod03b.npz` (diferencia maxima 0.0e+00); dimension 27431; gap train-OOF +0.1818.
R1-C1 (`v06`): MOD-03b con C reducido de 2.0 a 1.0 (misma representacion min_df=2); dimension 27431; F1 OOF 0.5 = 0.7865; umbrales OOF HS 0.44 / TR 0.53 / AG 0.44; F1 OOF con umbrales = 0.7929 (+0.0064); std entre pliegues 0.0115; gap train-OOF +0.1635.
R2-MINDF5 (`v07`): MOD-03b con min_df=5 en ambos bloques y C=1.0 (vocabulario reducido); dimension 10515; F1 OOF 0.5 = 0.7953; umbrales OOF HS 0.44 / TR 0.53 / AG 0.49; F1 OOF con umbrales = 0.7993 (+0.0040); std entre pliegues 0.0083; gap train-OOF +0.1272.
Gap train-OOF (F1 in-sample de los pliegues de entrenamiento menos F1 OOF con 0.5): control +0.1818, R1-C1 +0.1635, R2-MINDF5 +0.1272 -> R1-C1 baja 0.0183 frente al control; R2-MINDF5 baja 0.0546 frente al control.
Positivos predichos en test con los umbrales OOF de v06: HS 514 (44.5 %) / TR 326 (28.2 %) / AG 446 (38.6 %).
Positivos predichos en test con los umbrales OOF de v07: HS 521 (45.1 %) / TR 337 (29.2 %) / AG 408 (35.4 %).
Referencia de calibracion: prevalencia de train HS 43.7 % / TR 30.0 % / AG 38.2 %; test estimado (0.93x train) HS 40.6 % / TR 27.9 % / AG 35.6 %.
Submissions validadas contra la plantilla oficial (1154 filas, ids identicos y en orden, valores en {0,1}, sin nulos ni duplicados): `submissions/toxic_submission_v06_robustez_c1.csv` y `submissions/toxic_submission_v07_robustez_mindf5.csv`.
Registro: `experiments.csv` (R1-C1, R2-MINDF5, S06, S07) y `submissions/leaderboard_log.csv` (v06_robustez_c1, v07_robustez_mindf5 con `kaggle_f1` vacio, pendientes de que Carlos envie las dos sondas).
Decision pre-registrada D-013: se adopta como modelo final solo la sonda que supere el LB de v03 (0.78306); prohibido sondear mas alla de estos dos envios sin una hipotesis nueva.
Honestidad: la F1 OOF con umbrales es una cota optimista (umbral y metrica comparten el mismo OOF) y en esta familia el gap local-LB medido es -0.013 a -0.016 (D-012), asi que el LB esperado de las sondas ronda 0.78 +/- 0.01.
Sin fuga: el test solo se uso para predecir; umbrales, seleccion y comparaciones usan exclusivamente probabilidades out-of-fold de train. Idempotente: dos ejecuciones dejan el mismo contenido salvo los tiempos.

## Fase 11 - dl-specialist (2026-09-23)
Requisito metodologico: tres redes desde cero (sin transformers ni embeddings preentrenados; vocabulario y preprocesamiento solo del pliegue de entrenamiento), cabezal multi-tarea de 3 logits con BCEWithLogitsLoss y pos_weight por etiqueta, AdamW lr 1e-3 / wd 1e-4, batch 32, maximo 30 epocas y early stopping estricto sobre la F1 de validacion (paciencia 5, mejores pesos restaurados).
DL-01 (mlp): F1 OOF 0.7949 +/- 0.0262; por etiqueta HS 0.7838 / TR 0.8217 / AG 0.7790; gap train-validacion +0.1854; 110339 parametros; vocab 0; dim 300; longitud 300; 19.0 s.
DL-02 (charcnn): F1 OOF 0.7410 +/- 0.0097; por etiqueta HS 0.7093 / TR 0.7763 / AG 0.7375; gap train-validacion +0.1807; 87059 parametros; vocab 91; dim 48; longitud 200; 10.3 s.
DL-03 (bilstm): F1 OOF 0.7364 +/- 0.0199; por etiqueta HS 0.7237 / TR 0.7862 / AG 0.6992; gap train-validacion +0.1896; 233411 parametros; vocab 1685; dim 64; longitud 40; 15.2 s.
Repeticion con semillas 42/43/44 sobre DL-01: F1 OOF 0.7947 +/- 0.0011 (por semilla {'42': 0.7949, '43': 0.7932, '44': 0.7959}).
Referencias: MOD-03b_BEST (TF-IDF union + LR) OOF 0.7927 / LB 0.78306; B1 OOF 0.7587.
Conclusion honesta: La mejor red desde cero (DL-01) alcanza 0.7949 de F1 OOF frente a 0.7927 de MOD-03b_BEST (TF-IDF union + LR) y 0.7587 de B1. La red supera a la referencia clasica en OOF, pero D-012/D-014 exigen una sonda de leaderboard antes de dar la mejora por real: el gap local-LB de esta familia es de -0.013 a -0.029.
AMP no usado: medido en esta GPU, con estos modelos el coste de autocast supera la ganancia de fp16 (BiLSTM 3.42 s vs 7.48 s en 3 epocas). torch.use_deterministic_algorithms(False) por coredump de GPU observado con True en ROCm 7.2; semillas 42/43/44.
Artefactos: /home/carlos/Git/PLN/kaggle_toxic_2026_puj/reports/dl_desde_cero_resultados.csv, /home/carlos/Git/PLN/kaggle_toxic_2026_puj/reports/dl_desde_cero_resultados.json, /home/carlos/Git/PLN/kaggle_toxic_2026_puj/reports/figuras/fig_26_dl_curvas.png, /home/carlos/Git/PLN/kaggle_toxic_2026_puj/models/dl_mlp.pt; filas DL-01/DL-02/DL-03/DL-01-SEEDS en experiments.csv; evidencia del pre-vuelo en experiments/gpu_preflight.log.
Sin submissions: la decision de envio es del orquestador. Sin fuga: el test no se cargo en este script.

## Fase v2 - P1 (2026-09-25)
BASE: corrected_cv 0.7718 (delta vs V03 -0.0054; pliegues mejorados 2/5); por etiqueta HS 0.753 / TR 0.818 / AG 0.745; ensemble_gain -.
HS_TR_AG_raw: corrected_cv 0.7737 (delta vs V03 -0.0035; pliegues mejorados 1/5); por etiqueta HS 0.753 / TR 0.817 / AG 0.751; ensemble_gain +0.0044.
HS_TR_AG_standard: corrected_cv 0.7711 (delta vs V03 -0.0061; pliegues mejorados 1/5); por etiqueta HS 0.753 / TR 0.810 / AG 0.750; ensemble_gain +0.0033.
HS_AG_TR_raw: corrected_cv 0.7744 (delta vs V03 -0.0028; pliegues mejorados 1/5); por etiqueta HS 0.753 / TR 0.819 / AG 0.751; ensemble_gain +0.0043.
HS_AG_TR_standard: corrected_cv 0.7716 (delta vs V03 -0.0056; pliegues mejorados 1/5); por etiqueta HS 0.753 / TR 0.810 / AG 0.752; ensemble_gain +0.0053.
AG_HS_TR_raw: corrected_cv 0.7726 (delta vs V03 -0.0046; pliegues mejorados 1/5); por etiqueta HS 0.754 / TR 0.819 / AG 0.745; ensemble_gain -0.0001.
AG_HS_TR_standard: corrected_cv 0.7672 (delta vs V03 -0.0100; pliegues mejorados 0/5); por etiqueta HS 0.752 / TR 0.805 / AG 0.745; ensemble_gain -0.0025.
META: corrected_cv 0.7850 (delta vs V03 +0.0078; pliegues mejorados 3/5); por etiqueta HS 0.776 / TR 0.818 / AG 0.762; ensemble_gain +0.0154.
META-ANIDADO: corrected_cv 0.7853 (delta vs V03 +0.0081; pliegues mejorados 4/5); por etiqueta HS 0.776 / TR 0.813 / AG 0.767; ensemble_gain +0.0152.
Referencias del protocolo v2 en el desarrollo: M2 a 0.5 = 0.7718; V03 con umbrales = 0.7772.
Meta estrictamente anidado (los 4 bases se reentrenan en cada pliegue): corrected_cv 0.7853 (C 0.1), por C {'0.1': 0.7853, '0.3': 0.7832, '1.0': 0.7847}, std entre pliegues 0.0140 (por debajo de la contingencia 0.025); la version con el OOF congelado daba 0.7850, luego el anidamiento NO inflaba el numero.
Gate P1: ADOPTADO (mejor variante META-ANIDADO, corrected_cv 0.7853); regla D-016 exige >= 0.7822 con >= 3/5 pliegues o ensemble_gain >= 0.0015 con IC bootstrap > 0.
Umbrales: variante META-ANIDADO con umbrales OOF {'HS': 0.48, 'TR': 0.56, 'AG': 0.44} -> 0.7888 (cota optimista).
Sin fuga: holdout congelado no utilizado; vectorizadores, escaladores, umbrales y seleccion solo sobre OOF del desarrollo; el meta-modelo que se reporta como verificacion reentrena los 4 bases dentro de cada pliegue exterior (CV interna de 4 pliegues).
Duracion: 244.5 s de pared. Script: experiments/15_cadenas_etiquetas_v2.py.

## Fase v2 - P2 (2026-09-25)
P2 (E2-05/E2-06) targeting estructural sobre el desarrollo (n=1483), pliegues corregidos v2, holdout congelado intacto.
BASE reproducido: corrected_cv 0.7718 a 0.5 (max |diff| vs OOF congelado 0.0e+00) y V03 con umbrales 0.7772 (HS 0.762 / TR 0.816 / AG 0.754).
Lexico de insulto inducido fold-local (log-odds con prior de Dirichlet, freq>=5, df>=3; media 36 terminos por pliegue), adjetivos negativos 13, inventario colectivo 43 (regla estructural + gentilicios por sufijo, sin etiquetas).
Anotacion: spacy; constantes estructurales: experiments/06_error_analysis.py.
CH-TGT@C=2.0: corrected_cv 0.6076 (delta vs V03 -0.1696; vs v2-A -0.1777; pliegues 0/5; HS 0.606 / TR 0.618 / AG 0.599; ensemble_gain -0.0324).
CH-TGT+DIST@C=2.0: corrected_cv 0.6484 (delta vs V03 -0.1288; vs v2-A -0.1369; pliegues 0/5; HS 0.650 / TR 0.673 / AG 0.622; ensemble_gain -0.0329).
TXT+CH@C=2.0: corrected_cv 0.7356 (delta vs V03 -0.0416; vs v2-A -0.0497; pliegues 1/5; HS 0.723 / TR 0.768 / AG 0.716; ensemble_gain -0.0122).
TXT+TGT-SPARSE@C=2.0: corrected_cv 0.7487 (delta vs V03 -0.0285; vs v2-A -0.0366; pliegues 1/5; HS 0.729 / TR 0.791 / AG 0.727; ensemble_gain -0.0098).
Rebanada de referencia colectiva: n=363 (AG positivos 115); recall AG V03 0.452 vs 0.470 de la mejor variante global; fuera de la rebanada V03 0.825 vs 0.781 (brecha dentro-fuera de V03 -0.373, que confirma que el FN de AG se concentra en la referencia colectiva).
Senal localizada para P6: el canal CH-TGT+DIST alcanza el mayor recall de AG en la rebanada (0.487 con CH-TGT+DIST@C=1.0) frente a 0.452 de V03, aunque su F1 global es muy inferior; el promedio simple de probabilidades con M2 degrada (gain negativo), asi que solo el stacking regularizado de P6 puede explotarlo. Sus probabilidades OOF quedan en features/v2_oof_targeting.npz.
Gate P2: CERRADO SIN MEJORA. No adoptada: no alcanza el nivel exigido con 3/5 pliegues ni la complementariedad exigida; la linea de targeting estructural se cierra documentando el resultado. Nivel exigido 0.7822 (cumple=False), pliegues 1/5 (cumple=False), complementariedad gain -0.0098 (cumple=False; IC bootstrap no aplica con gain <= 0).
Sin fuga: lexicos, escaladores, vectorizadores y modelos se ajustan solo con el pliegue de entrenamiento; umbral y seleccion de C sobre OOF del desarrollo.
Duracion: 25.7 s. Script: experiments/16_targeting_v2.py. Artefactos: reports/targeting_v2.csv y .json, reports/figuras/fig_28_targeting_v2.png, features/v2_oof_targeting.npz (canales para P6).
## Fase v2 - P6 (2026-09-25)
P6 (E2-18) stacking de segunda generacion sobre el desarrollo (n=1483), pliegues corregidos v2, holdout congelado intacto; insumos: OOF de 4 bases + 2 canales de targeting (CH-TGT+DIST, TXT+CH) y 11 rasgos estructurales (longitud, tokens, mencion, hashtag, URL, comillas, segunda persona, colectivo, risas, mayusculas sostenidas, elongacion).
Insumos verificados: etiquetas y pliegues de los dos OOF coinciden con el desarrollo (pliegues del targeting identicos a los corregidos); V03 reproducido en 0.7772 con el OOF congelado; meta anidado de P1 reproducido en 0.7853 (referencia 0.7853, dif 0.0e+00, mejor C 0.1 vs C 0.1 de P1).
Variantes (corrected_cv, mejor hiperparametro, delta vs v2-A 0.7853, pliegues mejorados vs v2-A): META-BASES 0.7850 (lr 1, -0.0003, 2/5); META+CH 0.7807 (lr 1, -0.0046, 1/5); META+CH+ESTR 0.7844 (lr 1, -0.0009, 2/5); META-BASES-RIDGE 0.7874 (ridge 3, +0.0020, 3/5); META+CH+ESTR-RIDGE 0.7822 (ridge 0.3, -0.0031, 2/5); META+CH+ESTR-SVC 0.7809 (svc 0.3, -0.0045, 2/5); META-BASES-ANID 0.7853 (lr 0.1, +0.0000, 0/5); META+CH+ESTR-ANID 0.7802 (lr 0.1, -0.0051, 1/5).
Canales de targeting en el meta: NO APORTAN. META+CH+ESTR 0.7844 vs META-BASES congelado 0.7850 y anidado 0.7853; IC bootstrap del delta vs bases congelado [-0.0077,+0.0062].
IC bootstrap a nivel de tweet (2000 remuestreos) del mejor meta (META-BASES-RIDGE, corrected_cv 0.7874, por etiqueta HS 0.775 / TR 0.821 / AG 0.766) frente al anidado de P1: [-0.0054,+0.0093] (fraccion positiva 0.696); frente a META-BASES congelado: [-0.0035,+0.0079].
Rebanada de referencia colectiva (n=363): recall de AG V03 0.452 -> META-BASES-ANID 0.548 -> mejor meta 0.574 (fuera de la rebanada 0.825 -> 0.830).
FP/FN del mejor meta frente a V03 (corregidos/introducidos): HS 49 corregidos vs 32 introducidos; TR 31 corregidos vs 37 introducidos; AG 52 corregidos vs 45 introducidos.
Gate P6: ADOPTADO - Adoptado por complementariedad (ensemble_gain +0.0089, IC [+0.0019,+0.0160]); el nivel 0.7874 no llega a 0.7903 y el IC del delta vs v2-A [-0.0054,+0.0093] incluye 0 (mejora de nivel dentro del ruido). Complementariedad con M2: ensemble_gain +0.0089 (objetivo >= 0.0015), IC [+0.0019,+0.0160].
Sin fuga: holdout congelado no utilizado; escaladores, meta-modelos y seleccion de hiperparametros solo con el entrenamiento de cada pliegue o con el OOF del desarrollo; las variantes -ANID reentrenan los 4 bases dentro del pliegue exterior. Advertencia: en las variantes con OOF congelado los scores de entrenamiento del meta provienen de bases que vieron el pliegue evaluado (stacking estandar); P1 midio ese sesgo en <= 0.0003.
Duracion: 71.3 s. Script: experiments/17_stacking_v2.py. Artefactos: reports/stacking_v2.csv y .json, reports/figuras/fig_29_stacking_v2.png, filas E2-18-* en experiments.csv, {'modelo': 'models/v2_meta_candidato.joblib', 'oof': 'features/v2_meta_oof.npz'}.

## Fase v2 - P7 (2026-09-25)
P7 (D-016): validacion del candidato en el holdout congelado (248 filas; particion verificada por sha256) y decision de submission v08. Evaluaciones al holdout: 0 previas + 2 de esta tarea = 2/5 (`reports/holdout_evaluaciones_v2.csv`).
Paso 1 - V03 de referencia en el holdout (M2 reajustado con todo el desarrollo; umbrales congelados 0.44/0.51/0.46): F1 0.8285 (HS 0.857 / TR 0.849 / AG 0.779; a 0.5 0.8260).
Paso 2 - meta Ridge alpha=3 (44 meta-features OOF del desarrollo; bases reajustados con el desarrollo; umbral 0.5) en el holdout: F1 0.8465 (HS 0.862 / TR 0.857 / AG 0.820); reproduce el candidato de P6 (dif max en desarrollo 0.0e+00).
Decision pre-registrada (meta >= V03 - 0.005): delta +0.0180 -> GO. El meta (0.8465) alcanza el criterio pre-registrado (V03 - 0.005 = 0.8235); delta +0.0180. Se genera la submission v08 (sonda v2 del meta).
Paso 3 - submission v08: bases reajustados con el train completo (1731), meta-features del test y umbral 0.5; validada contra la plantilla oficial (1154 filas, ids en orden, valores en {0,1}, sin nulos ni duplicados).
Positivos predichos en test: HS 517 (44.8 %) / TR 375 (32.5 %) / AG 445 (38.6 %); referencia train 43.7/30.0/38.2 % y test estimado 40.6/27.9/35.6 % (E00).
Registro: fila S08 en `experiments.csv` y `v08_meta_v2` en `submissions/leaderboard_log.csv` (kaggle_f1 vacio; la sonda la envia Carlos). Hipotesis de la sonda: el meta supera el LB de v03 (0.78306).
Sin fuga: el holdout solo se uso para medir los 2 finalistas y el test solo para predecir; el meta, su escalador y los umbrales se ajustaron antes con el OOF del desarrollo. Advertencia: 248 filas y 1 evaluacion puntual (error estandar ~0.03); el gap local-LB de la familia union es -0.013 a -0.016 (D-012).
Duracion: 3.0 s. Script: experiments/18_p7_holdout_y_submission_v08.py. Artefactos: reports/p7_holdout_resultados.json, reports/holdout_evaluaciones_v2.csv, models/v2_meta_final.joblib, submissions/toxic_submission_v08_meta_v2.csv.

## Fase v2 - P3+P4 (2026-09-25)
P3 (E2-03/E2-04) sintaxis sparse y P4 (E2-07/E2-08) cita/atribucion/negacion sobre el desarrollo (n=1483), pliegues corregidos v2, holdout congelado intacto.
Anotacion spaCy SIMBOLICA (POS/lema/morfologia/dep/cabeza) en cache propia (v2_syntax_annotations.json.gz); prohibido el uso de vectores y similitud.
BASE reproducido: corrected_cv 0.7718 a 0.5 (dif max vs OOF congelado 0.0e+00) y V03 con umbrales 0.7772.
P3 - mejor POS: SYN-POS@C=1.0 corrected_cv 0.5637 (delta -0.2135 vs V03, pliegues 0/5); TXT+POS 0.7656 (-0.0116).
P3 - DEPS: SYN-DEPS 0.5514 (-0.2258); TXT+DEPS 0.7653 (-0.0119).
P3 - MORPH: SYN-MORPH 0.5477 (-0.2295); TXT+MORPH 0.7567 (-0.0205).
P3 - union de los tres: SYN-3 0.5865 (-0.1907); TXT+SYN3 0.7376 (-0.0396).
P4 - CITA: CITA 0.7437 (-0.0335); TXT+CITA 0.7554 (-0.0218).
P4 - NEG: NEG 0.5267 (-0.2505); TXT+NEG 0.7657 (-0.0115).
P4 - union: TXT+CITA+NEG 0.7552 (-0.0220).
Complementariedad: mejor ensemble_gain -0.0028 sobre M2 a 0.5 (objetivo >= +0.0015 con IC > 0).
Gate P3: CERRADO SIN MEJORA (TXT+POS@C=2.0 0.7656); nivel exigido no cumple, pliegues 0/5.
Gate P4: CERRADO SIN MEJORA (TXT+NEG@C=2.0 0.7657); nivel exigido no cumple, pliegues 0/5.
Robustez: texto vacio, solo URL, solo mencion, RT, comillas desbalanceadas y sin insultos sin NaN/infinito; features independientes del orden del dataset.
Sin fuga: holdout congelado no utilizado; lexicos, vectorizadores, escaladores y modelos solo con el entrenamiento de cada pliegue.
Duracion: 25.6 s. Script: experiments/19_sintaxis_cita_negacion_v2.py. Artefactos: reports/sintaxis_cita_negacion_v2.csv y .json, reports/figuras/fig_30_sintaxis_cita_v2.png, features/v2_oof_syntax.npz y features/v2_oof_cita_negacion.npz.
## Fase v2 - P5 (2026-09-25)
P5 (E2-09 emocion/afecto, E2-15 NB-SVM y E2-10 multicanal ortografico) sobre el desarrollo (n=1483), pliegues corregidos v2, holdout congelado intacto.
Base reproducida: M2 a 0.5 0.7718 (dif max vs OOF congelado 0.0e+00); V03 con umbrales 0.7772; pliegues identicos al OOF congelado.
Lexicos externos: ninguno. `data/external/lexicons/` vacio y la sesion prohibe descargas; solo lexico de afecto inducido fold-local (Monroe 2008) documentado en el JSON (NRC EmoLex, ML-SentiCon, VADER-es y WordNet-Affect evaluados y descartados).
E2-09 marcadores estaticos: EMO-EST@C=2.0 0.5897 (-0.1875, 0/5)
E2-09 lexico inducido + ventanas +-3/5/10: EMO-LEX@C=1.0 0.6205 (-0.1567, 0/5)
E2-09 canal completo solo: EMO-ALL@C=1.0 0.6230 (-0.1542, 0/5)
E2-09 union con V03: TXT+EMO@C=2.0 0.6330 (-0.1442, 0/5); con el bloque denso reescalado (diagnostico de escala, fuera de la decision): TXT+EMO-NORM@C=2.0 0.6749 (-0.1023, 0/5)
E2-15 NB-SVM word: NB-WORD-LR@C=1.0 0.7538 (-0.0234, 0/5); char: NB-CHAR-LR@C=1.0 0.7380 (-0.0392, 0/5)
E2-15 NB-SVM union LR: NB-UNION-LR@C=1.0 0.7612 (-0.0160, 2/5) (correlacion de residuos vs M2 0.887); SVC: NB-UNION-SVC@C=1.0 0.7429 (-0.0343, 0/5)
E2-10 raw (A): ORT-A@C=1.0 0.7486 (-0.0286, 0/5)
E2-10 colapso de repeticiones: ORT-B@C=1.0 0.7488 (-0.0284, 0/5); sin puntuacion interna: ORT-C@C=2.0 0.7500 (-0.0272, 0/5)
E2-10 leetspeak: ORT-D@C=1.0 0.7486 (-0.0286, 0/5); union normalizada A+B+C+D: ORT-ABCD@C=2.0 0.7534 (-0.0238, 0/5)
E2-10 insulto enmascarado sobre raw: ORT-AE@C=2.0 0.7496 (-0.0276, 0/5)
Complementariedad: el ensemble_gain es la media de probabilidades con M2 menos 0.7718 (definicion de P1-P4); la concatenacion aprendida de scores pertenece a P6.
Gate E2-09: CERRADO sin mejora (TXT+EMO@C=2.0 0.6330, ensemble_gain -0.0782).
Gate E2-15: CERRADO sin mejora (NB-UNION-LR@C=1.0 0.7612, ensemble_gain +0.0038).
Gate E2-10: CERRADO sin mejora (ORT-AB@C=2.0 0.7534, ensemble_gain -0.0097).
Canales OOF guardados en `features/v2_oof_emocion_nbsvm_ortografia.npz` para el meta de P6 (4 canales de emocion, 4 de NB-SVM y 3 ortograficos).
Sin fuga: lexicos, inventario colectivo, vectorizadores, pesos NB, escaladores y modelos solo con el pliegue de entrenamiento; umbral y seleccion de C sobre OOF del desarrollo.
Duracion: 26.3 s. Script: experiments/20_emocion_nbsvm_ortografia_v2.py. Artefactos: reports/emocion_nbsvm_ortografia_v2.csv y .json, reports/figuras/fig_31_emocion_nbsvm_v2.png.
## Fase v2 - P8 (2026-09-25)
P8 cierra las 5 lineas restantes del backlog v2 (E2-13, E2-14, E2-16, E2-20, E2-21) sobre el desarrollo (n=1483), pliegues corregidos v2 y holdout congelado intacto.
Referencias reproducidas: M2 a 0.5 = 0.7718, V03 con umbrales = 0.7772, meta Ridge = 0.7874.
E2-14 calibracion: Platt e isotonic anidados; son transformaciones monotonas por etiqueta, asi que la F1 con umbrales reajustados es invariante salvo empates.
E2-14 mejor variante: META-ISOTONIC corrected_cv 0.7857 (delta vs su par sin calibrar -0.0017); Brier 0.1199 y log-loss 0.4035.
E2-13 encoder de entidades: cobertura informativa (mencion+hashtag) 0.005 < 0.20; mencion 969 unicas, hashtag 210, dominio 1 (t.co trivial).
E2-13 mejor variante: META-ENT-RIDGE corrected_cv 0.7885 (delta vs meta +0.0011): linea CERRADA por sparsity.
E2-16 umbrales por rebanada: mejor SHRINK-M100 corrected_cv 0.7750 (delta vs global anidado -0.0017).
E2-16 rebanadas usadas (n>=80): hashtag, mencion, url, pregunta, colectiva, corto40 y largo_q4; comillas queda fuera por n<80.
E2-16 evaluacion anidada: el umbral local se ajusta en los pliegues de ajuste (>= 8 positivos por celda) y se aplica al pliegue evaluado; la mezcla usa w = n/(n+m).
E2-20 especialistas: 21 pruebas de proporcion de errores (rebanada x etiqueta) con BH q=0.05; celdas significativas: 11.
E2-20 decision: los especialistas rinden menos que el global dentro de la rebanada (delta medio -0.0058 para la mezcla); la mezcla queda por debajo de la referencia -> CERRADA sin mejora.
E2-21 espacios por etiqueta: corrected_cv 0.7519 (delta vs espacio compartido -0.0199); coste de dimension del canal: 153 HS / 159 TR / 154 AG.
E2-21 usa proxies estructurales por regex y lexicos inducidos fold-local (Monroe 2008); sin spaCy en este paquete (la linea morfologica se cerro en P3).
Regla de adopcion (D-016) aplicada al mecanismo de cada linea, con el aporte medido contra su comparador directo: solo el meta calibrado supera 0.7822 (0.7857, 5/5 pliegues), pero no aporta sobre el meta sin calibrar (-0.0017); ninguna otra variante alcanza +0.005 con 3/5 pliegues ni complementariedad con IC > 0.
Cierre honesto: las 5 lineas se cierran sin mejora transferible; el backlog v2 queda al 100 % (24/24) y no se gasta ninguna evaluacion del holdout.
Duracion: 27.2 s. Script: experiments/21_backlog_restante_v2.py. Artefactos: reports/backlog_restante_v2.csv y .json, reports/figuras/fig_32_backlog_restante_v2.png.

## Artefactos de auditoria v2 (2026-09-25)
Artefactos de las FASES 43/44/47/50/51/55/88/89/90 generados desde los OOF congelados del desarrollo (n=1483); sin entrenar modelos, sin submissions y sin usar el holdout ni el test.
Interseccion (FASE 43/44): v03_base_M2_union_lr_c2 falla las 3 etiquetas en 98 tweets (6.61%, IC 95 % [5.39%, 7.89%]); meta_META-BASES-RIDGE en 103 (6.95%, IC 95 % [5.73%, 8.29%]).
Contraste historico de FASE 43: 101 tweets (5.83 %) sobre train completo n=1731 con el protocolo historico (reports/error_analysis.md); el desarrollo corregido no es comparable 1:1.
Tres etiquetas mal (FASE 43): 201 filas tweet x referencia (98 + 103); solo id y rasgos, sin texto crudo.
Error por combinacion (FASE 47): 14 filas (2 referencias x 7 combinaciones) con precision/recall/F1 por etiqueta y medias estructurales; en las etiquetas constantes del subconjunto precision = recall por construccion.
Complementariedad (FASE 50/51/91): 496 parejas de 32 canales OOF; el ensemble_gain solo se copia si ya estaba medido (media con M2 a 0.5) y no se recalculo ninguno.
Mejor pareja por correccion mutua simetrica: emocion_EMO-ALL@C=1.0 / sintaxis_SYN-3 (57.8% y 60.5% de los errores del otro).
Catalogo (FASE 55/88): cita_negacion=21, dependencia_etiquetas=71, emocion=71, entidades=12, nbsvm=3, ortografia=5, sintaxis=3, targeting=37, texto_base=2 (225 features leidas del codigo de experiments/14 a 21).
Dataset estructurado (FASE 55/89): features/v2_structured_features.parquet con 2885 filas (dev+holdout+test) x 123 features deterministas sin etiquetas; manifest en reports/v2_structured_features_manifest.md con las exclusiones documentadas.
Registro (FASE 90): filas AUD-01, AUD-02 y AUD-03 en experiments.csv (solo append) con decision 'artefacto de auditoria; sin modelo nuevo'.
Honestidad: ninguna cifra se ajusto para cuadrar con lo documentado; las discrepancias se registran en el manifest. El holdout congelado sigue intacto y no se genero ninguna submission.
Duracion: 2.8 s. Script: experiments/22_artefactos_auditoria_v2.py.
