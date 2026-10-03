# P84: auditoria adversarial de las premisas del plan (2026-10-02)

Alcance: verificacion documental y de artefactos en solo lectura. No se entreno
nada, no se generaron submissions y no se modifico codigo ni datos. La unica
escritura es este informe. Veredictos usados: CONFIRMADO, PARCIAL, REFUTADO,
NO VERIFICABLE. Las lineas citadas corresponden al estado del repositorio el
2026-10-02.

## Tabla de premisas

| # | claim | veredicto | evidencia (ruta:linea) | consecuencia para el plan |
|---|---|---|---|---|
| 1 | La metrica oficial es la media de los 3 F1 positivos binarios (HS, TR, AG); E00 (0.51494) y v01 (local 0.75778 vs LB 0.75677) la respaldan | CONFIRMADO | `src/evaluation.py:6-7` (definicion) y `:97-110` (`f1_media_columnas` = media de las 3 F1 de clase positiva); `submissions/leaderboard_log.csv:2` (E00 = 0.51494) y `:3` (v01 local 0.75778 vs LB 0.75677); `experiments.csv:2` (fila E00, decision ADOPTADO) | El protocolo local del plan (`f1_media_columnas`) es el correcto; no reabrir macro-6, micro ni samples como metrica de seleccion |
| 2 | V03: `corrected_cv` 0.7772114525165721 sobre 1483 filas con grupos fusionados; LB 0.78306; umbrales 0.44/0.51/0.46; mejor resultado verificado | CONFIRMADO | `features/oof_v03_final.npz` (artefacto: `corrected_cv` = 0.77721145; 1483 ids; umbrales [0.44 0.51 0.46]); `docs/p78_preregistro.md:73`; `docs/p84_preregistro.md:24-30` (recalculo exacto 0.7772114525165721; 1645 grupos); `submissions/leaderboard_log.csv:5` (LB 0.78306); maximo de `kaggle_f1` recalculado sobre `leaderboard_log.csv` = 0.78306 | Control congelado valido; todo candidato se compara con ese OOF y esos umbrales; V03 sigue siendo el techo externo verificado |
| 3 | DL-01 (0.7949) no es comparable con V03: pool de 1731 filas y seleccion de epoca sobre el mismo pliegue puntuado | CONFIRMADO | `experiments.csv:48` (DL-01, f1 0.7949); `reports/p80_auditoria_w2v.md:92-97` (pool 1731 = 1483 + 248; OOF 0.794854 vs 0.777211 no comparable; epoca elegida con el pliegue externo); `reports/p80_auditoria_w2v.json` (afirmaciones[16] y [17]: totales DL-01/02/03 = 1731); `auditoria/sugerencias_080.md:110-112` | EXP-01 (pool 1483 y epoca anidada) es la unica relectura valida; no citar 0.7949 como base de mejora |
| 4 | No existe un clasificador fastText supervisado en el proyecto; solo embeddings no supervisados (fase v5) y sus fusiones | CONFIRMADO con matiz | Busqueda repo-wide de `train_supervised` y de comandos supervisados: cero apariciones en `experiments/`, `experimentos/`, `src/` o `notebooks/`; las unicas apariciones estan en la libreria vendorizada `tools/fastText/python/fasttext_module/fasttext/FastText.py:505` (API disponible, no usada); `experiments/embedding_02_fasttext_baseline_v5.py:1-20` entrena vectores sin etiquetas con el binario C++; fusiones en `experiments/embedding_03_*` y `embedding_18_*`; `docs/p84_preregistro.md:13-16` (binario reconstruido desde `tools/fastText`) | EXP-02 es materialmente nuevo; el binario ya esta compilado (`tools/fastText/build/fasttext`), no requiere descargas; el matiz es que la API supervisada existe dentro de `tools/` pero nunca se ejecuto |
| 5 | No existe ejecucion de sobremuestreo aleatorio (ROS); SMOTE/ADASYN se descartaron en el EDA; focal loss se probo y cerro negativo. Fuentes: `experiments/`, filas V7-01 y V4-B06 de `experiments.csv`, `reports/estado_arte_v4.md` | PARCIAL | `experiments/01_eda_linguistico.py:2713` (SMOTE/ADASYN descartados en el EDA); `experiments.csv:186` (V7-01: FOCAL -0.0024, IC95 [-0.0086, +0.0039], cierre del track); busqueda repo-wide de `RandomOverSampler`, `oversampl`, `sobremuestr`, `random over`: 0 apariciones en codigo (solo prosa en `reports/estado_arte_v4.md:161-174,1023-1030`); V4-B06 NO existe como fila en `experiments.csv` (grep y barrido de ids sin resultados); V4-B06 figura solo como plan en `reports/estado_arte_v4.md:172,1023,1064` y `tasks/todo_v4.md:33`; el script ejecutado `experiments/29_reweighting_v4.py` usa `sample_weight` por dureza, no remuestreo | EXP-03 sigue justificado (ROS no se ejecuto), pero la cita "fila V4-B06" es erronea: corregirla a V7-01 (focal) y `estado_arte_v4` (plan V4-B06). La ausencia de ROS es evidencia por busqueda de texto, no prueba exhaustiva |
| 6 | El artefacto Word2Vec congelado no implementa el objetivo SGNS completo (termino negativo con peso 1/K); el control plantado de P81 no apoyo la direccion "suma"; el ancla externa quedo pendiente | CONFIRMADO | `experiments/p71_04_entrenar_word2vec.py:214-215` (`neg_l = logsigmoid(...).mean()` sobre batch x K: peso 1/K); `reports/p79_auditoria_sgns.json` (`ratio_del_termino_negativo` = 1.0/0.2/0.1/0.05 para K = 1/5/10/20); `reports/p80_auditoria_w2v.md:54-58` (H1 confirmado, linea citada 215); `reports/p81_cierre_huecos.md:51,63` (titular EL_TEST_NO_APOYA_A_LA_SUMA; MEDIA gana por 0.700000; A6 REFUTADO) y `:110,216` (anclaje a implementacion externa SIN HACER) | EXP-04 es el cierre correcto del hilo; el ancla contra una implementacion de terceros sigue pendiente y no autoriza afirmar que el artefacto sea peor que library; no prometer mejora |
| 7 | Ruido del leaderboard: 0.01219 (cota binomial) y 0.01524 para F1; 19.2 aciertos netos; sensibilidad por acierto neto | CONFIRMADO con salvedad | `experiments.csv:192` (V8-02); `reports/v8_resolucion_liderboard.md:15-18` (n = 1154, p = 0.78 -> 0.01219; banda 0.01524) y `:21-37` (7.93e-04 de metrica por acierto neto; 19.2 netos); `DECISIONS.md:2223-2224` (D-063) y `:2727` (D-070); la banda local >0.017 de D-070 es consistente | Las cifras describen fielmente lo registrado; salvedad importante: la cota binomial usa las 1154 filas del test completo, no la parte publica del LB (ver R1) |
| 8 | El objetivo +0.01694 esta en el orden del ruido y ninguna sonda previa lo alcanzo; la curva F1 vs tasa de positivos predicha tiene maximo en V03 | CONFIRMADO | `experiments.csv:199` (V8-09: curva unimodal, maximo en el cuartil de V03, 13 de 14 sondas dentro de la banda, segunda mejor a -0.00143); `DECISIONS.md:2266,2591,2671` (D-063: 1.1 veces el ruido, 21 aciertos netos); `auditoria/sugerencias_080.md:23,86-87`; maximo LB recalculado = 0.78306 (leaderboard completo) | No apostar por calibracion, umbrales ni pesos por etiqueta; el presupuesto debe ir a candidatos con evidencia nueva (EXP-01 a EXP-04) |
| 9 | Holdout v2 agotado: 6 usos registrados frente a 5 declarados; no debe usarse mas | CONFIRMADO | `reports/holdout_evaluaciones_v2.csv:1-7` (6 evaluaciones: P7-01, P7-02, FIN-02, FIN-04, P8, V33-01); `config/holdout_v2.json` (`regla_de_uso.maximo_evaluaciones` = 5; campo `correccion_auditoria_2026_10_02`: sexto uso, limite excedido); `auditoria/sugerencias_080.md:107-109` (menciona ademas usos no registrados) | No evaluar ni entrenar con las 248 filas; el plan ya lo asume; si algo, la premisa se queda corta porque hay usos adicionales documentados |
| 10 | El leaderboard publico usa ~70 % del test y el privado ~30 % (pagina de Kaggle) | CONFIRMADO | Pagina de leaderboard de Kaggle del concurso (https://www.kaggle.com/competitions/tweet-toxic-2026-puj/leaderboard), contenido renderizado verificado el 2026-10-02: "This leaderboard is calculated with approximately 70% of the test data. The final results will be based on the other 30%, so the final standings may be different." | La nota final y el leaderboard privado dependen del ~30 % restante; ademas obliga a rescalar el ruido del punto 7 (ver R1). `docs/competition_rules.md:3` declara como fuente la pagina Overview, que no contiene este reparto |

## Riesgos y errores detectados

- R1 (el hallazgo principal). Las premisas 7 y 10 son inconsistentes entre si en
  la interpretacion. `data/raw/toxic_test.csv` tiene 1154 filas; si el LB publico
  puntua el ~70 % (~808 filas), la cota binomial correcta para el publico es
  ~0.0146, no 0.01219 (recalculo con la misma formula del informe:
  0.01219 x sqrt(1154/808)), y para el privado (~346 filas) ~0.0223. Si se
  escala la banda de 0.01524 por la raiz del cociente de tamanos (estimacion,
  no medicion), el ruido equivalente pasa de 19.2 a ~23 aciertos netos en
  publico y ~35 en privado. El plan usa 19.2 y 21 como vara de
  detectabilidad: es optimista para el LB publico. El ruido empirico de 0.01524
  no depende de n de forma declarada, pero fue validado como "misma magnitud"
  contra una banda local; no se encontro en el repo una medicion de dispersion
  del LB publico con n = 808.
- R2. Cita erronea en la premisa 5: no existe fila V4-B06 en `experiments.csv`.
  V4-B06 (barrido de `class_weight` por etiqueta con sobremuestreo/subsampling)
  quedo como plan en `reports/estado_arte_v4.md` y `tasks/todo_v4.md`; no hay
  registro de ejecucion. La parte "focal cerro negativo" si esta verificada en
  V7-01.
- R3. Veredictos 4, 5 y 10 se apoyan en busquedas de texto y en contenido
  renderizado de una pagina dinamica, respectivamente. No son prueba exhaustiva;
  no se encontro nada que los refute. En particular, "0 apariciones de
  RandomOverSampler" no demuestra que ningun script haya replicado filas por
  otra via, aunque los barridos de `np.repeat`/`replace=True`/`duplicat` tampoco
  dieron resultados en `experiments/`.
- R4. La premisa 2 se verifico contra `submissions/leaderboard_log.csv`; si
  existiera un score externo no registrado, "mejor resultado verificado" no
  podria cerrarse. El envio mas reciente (`p82_sonda_rnn_b1`, 2026-10-01) dio
  0.6815, muy por debajo de V03.
- R5. P79 no tiene informe `.md` (solo JSON); sus cifras fueron corregidas en
  parte por P80 (A5, A13: el par E/S medido no existia). El defecto 1/K no se ve
  afectado: queda confirmado por P80 A3/A4 con la linea 215 de `p71_04`.
- R6. La verificacion de la premisa 10 no pudo hacerse por descarga directa
  (pagina dependiente de JavaScript sin navegador conectado); se hizo con el
  contenido renderizado de la pagina de leaderboard. Conviene reconfirmarla en
  la sesion autenticada de Carlos antes de fijar decisiones dependientes del
  30 % privado.
