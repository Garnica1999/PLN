# QA final P84: P84-04 (SGNS corregido) y cierre de fase

- Fecha: 2026-10-02
- Alcance: P84-04 y coherencia global de la fase P84 (P84-01..P84-05, EXP-06).
- Metodo: verificacion Prove-It sobre artefactos en disco. Recalculo independiente de
  F1 y deltas desde los npz con el venv del proyecto; reproduccion exacta de los
  bootstrap con src/evaluation.py. No se re-entreno ni se modifico codigo o datos.
- Entorno normativo: /home/carlos/venv/bin/python (Python 3.14.7, numpy 1.26.4).
- Evidencia base: reports/p84_04_sgns.json/.md, reports/p84_04_ancla.json,
  features/p84_04_oof_emb.npz, features/oof_v03_final.npz, modelos/p84/emb_*.npz,
  reports/p84_04_parciales/ (9 json + 9 npz + humo), experiments.csv, RUNLOG.md,
  DECISIONS.md, docs/p84_spec.md, docs/p84_preregistro.md,
  /home/carlos/.opencode/plan/kaggle_toxic_exploracion_modelos_f1_080.md y
  tools/verificar_caracteres.py. QA previo: reports/p84_qa_parcial.md (P84-01/02/03/05).

## 1. Verificacion P84-04

| Aspecto | Resultado | Evidencia |
|---|---|---|
| Ancla lograda | PASA | p84_04_ancla.json veredicto "ANCLA LOGRADA", ok true; algebraica err rel 1.2764e-16 / 1.1588e-16 (tol 1e-09); gradiente dif. finitas 4.4210e-08 (tol 1e-05); ejecutable pureza oficial 1.0 = nuestra 1.0 en 16 palabras (k=5, tol 0.05); embebida en sgns.json identica campo a campo |
| Evidencia del ancla en disco | PASA | experimentos/p84_04_tmp/ancla_plantado.txt sha256 e8cb4905...5879 == declarado; ancla_oficial.vec/.bin presentes; pureza oficial recomputada desde el .vec: 1.0 en las 16 palabras objetivo (se excluye </s>); fastText oficial intento 1 NaN codigo -6, intento 2 OK |
| Identidad V03 | PASA | gate0.identidad_v03: max delta proba 0.0 (tol 1e-12), f1 congelado 0.7772114525165721 == esperado, delta 0.0, f1@0.5 0.771778476256122 exacto, pred reconstruida igual; V03 recalculado por QA desde su npz: 0.7772114525165721 exacto |
| Gate 0 hashes congelados | PASA | 5/5 sha256 reales coinciden: tokens_10M 00b839da..., fasttext 59c51cc5..., oof_v03_final 55926581..., toxic_train 3d40e734..., grupos afabf9ae... |
| Contadores | PASA | gate0.evidencia.contadores: dev 1483 / holdout_filas_usadas 0 / test_filas_usadas 0 / toxic_test_abierto false; 248 de holdout apartadas por indice y no usadas; pliegues 297/297/296/296/297 sin grupos compartidos |
| Flujo pareado S/M | PASA | flujo_pareado_S_M: 3/3 semillas iguales true; hash_pares y hash_negativos coinciden entre S, M, los 9 parciales y el registro; secuencias por semilla 1dca8e47..., 8819fb86..., 65ee53e8... |
| F1 congelada | PASA | Recalculada por QA desde features/p84_04_oof_emb.npz con umbrales 0.44/0.51/0.46 y metrica_macro: S 0.6260910107, M 0.6228875099, R 0.6518248058; V03 0.7772114525165721; 9/9 valores por semilla exactos; coinciden JSON y CSV |
| Deltas pareados | PASA | Recalculados: S_vs_M +0.0032035008, S_vs_R -0.0257337951, S_vs_V03 -0.1511204418; coinciden JSON, MD, CSV, RUNLOG y plan |
| IC bootstrap 95% | PASA | Reproducidos EXACTAMENTE con src.evaluation.bootstrap_diferencia_pareada (10000 rep, semilla 20260926) sobre el npz: S_vs_M [-0.012606, +0.018844] mediana 0.003319; S_vs_R [-0.046887, -0.004500]; S_vs_V03 [-0.175340, -0.127571]; M_vs_V03 y R_vs_V03 tambien exactos (5/5) |
| ensemble_gain | PASA | S -0.0490044504 IC95 [-0.066411, -0.031799] < 0.0015; declarado "NO" |
| Decision de cierre | PASA | gate.decision "CERRAR la linea SGNS corregido con el resultado verificado (sin rejilla nueva)"; D-285; plan EXP-04 COMPLETADA; RUNLOG 18:34 |
| No elegibilidad EXP-06 | PASA | gate.elegible_exp06 false; D-286 "EXP-06 NO SE EJECUTA"; plan EXP-06 CERRADA SIN EJECUCION |
| Brecha D-057 | PASA | brecha_ok true; S 0.1383568371 / M 0.1375367204 / R 0.1422080099 / V03 0.204721418 <= +0.05 |
| Parciales (3 semillas x S/M/R) | PASA | 9 json + 9 npz completos sin humo; estado "completo"; firma sha256 script 26ca470f...e5975 == script real == Enmienda 4 del preregistro; proba de cada parcial identica bit a bit a la del npz global; umbrales anidados identicos; f1_congelada == main |
| Embeddings | PASA | 9 archivos modelos/p84/emb_2026101{6,7,8}_{S,M,R}.npz; W (100097, 200) + palabras + dim + semilla + brazo; 9/9 sha256 declarado == real; vocabulario 100097 == config.corpus.vocabulario |
| OOF global | PASA | features/p84_04_oof_emb.npz: 1483 filas; ids, y_true y pliegues identicos bit a bit a features/oof_v03_final.npz (1483 ids unicos, sin holdout); proba_promedio_* == media exacta de las 3 semillas; umbrales_congelados [0.44, 0.51, 0.46] |
| Fila experiments.csv | PASA | P84-04 en linea 456, una sola vez; 455 filas de datos, todas con 23 columnas (parser csv), sin experiment_id duplicado; f1 0.626091, macro_f1 0.674577, precision 0.565686, recall 0.695181; decision e interpretation == JSON |
| RUNLOG | PASA | Lineas 5086-5091: corrida GPU 15511.17 s, EXIT 0, Gate 0, ancla, F1, deltas, gate/decision, elegible EXP-06 False, rutas y "455 filas, 23 columnas" |
| Sin fuga en P84-04 | PASA | StandardScaler y LR ajustados en outer_train por pliegue; umbrales anidados elegidos dentro de outer_train; embeddings entrenados solo con corpus externo sin etiquetas; R usa matriz aleatoria congelada |
| Uso de test | PASA | Sin aperturas ni lecturas de toxic_test; solo menciones literales en comentarios y la clave toxic_test_abierto (mismo aviso menor que P84-01/02/03/05) |

## 2. Coherencia global de la fase

- DECISIONS.md: D-281 (linea 10044), D-282 (10064), D-283 (10080), D-284 (10097),
  D-285 (10110) y D-286 (10125) presentes y completas; D-285 registra el cierre de
  P84-04 y D-286 la no ejecucion de EXP-06 y la no superacion de V03.
- Plan maestro: bitacora EXP-01 (linea 226), EXP-02 (246), EXP-03 (261), EXP-04
  EN CURSO (275) y COMPLETADA (298), EXP-05 (289) y EXP-06 CERRADA SIN EJECUCION
  (314). La doble entrada de EXP-04 es la progresion natural de la bitacora.
- experiments.csv: exactamente una fila por P84-01..P84-05 (lineas 452-456); sin
  duplicados ni filas malformadas; todas las filas de datos con 23 columnas.
- RUNLOG.md: 5 entradas P84, una por experimento (12:32, 13:01, 13:31, 16:38, 18:34),
  con resultados, gate y cierre.
- tools/verificar_caracteres.py: exit 0, "OK: 0 caracteres no permitidos" (incluye
  docs y reports P84).
- Scripts: los 5 scripts p84 son nuevos (no rastreados) y son los unicos .py creados
  o modificados en la ventana P84 (12:25-18:34). Los previos modificados tienen mtime
  05:47-05:54 (auditoria previa) y el notebook 10:00, todos antes de la ventana.
- Submissions: ningun archivo nuevo ni modificado tras 12:25; ningun script P84
  escribe en submissions/ (solo una cadena informativa en p84_05).
- Script P84-04 mtime 14:13 < ancla 14:15 < corrida 14:15-18:34 < informe 18:34:
  cronologia coherente; JSON de corrida 18:34 con 0 incidencias.

## 3. Discrepancias

1. [Menor] La fila P84-04 deja vacia la columna weighted_f1 (el resto de metricas
   esta presente); consistente con el patron de las filas P84 ya auditadas.
2. [Menor] La cadena literal "toxic_test.csv" aparece en comentarios y en la clave
   toxic_test_abierto del script P84-04; no hay apertura ni lectura. Igual que en los
   otros cuatro scripts P84.
3. [Menor, entorno] Los IC de bootstrap se reproducen exactamente con el venv
   (numpy 1.26.4) y no con el python3 del sistema (numpy 2.5.3), que produce otros
   sorteos de default_rng. El entorno normativo del proyecto es el venv; no afecta a
   los artefactos.
4. [Menor, limpieza] Archivo parasito "=" de 0 bytes en la raiz del proyecto, creado
   12:56 durante la ventana P84. No es script, dato ni submission; recomiendo borrarlo
   al cierre documental.
5. [Informativo] La matriz aleatoria R supera a S y M en F1 congelada y anidada; esta
   declarado como anomalia en RUNLOG y D-285, sin ocultarse ni re-interpretarse.
6. [Informativo] El intento 1 del binario oficial de fastText fallo con NaN (codigo -6)
   y el intento 2 fue OK; queda registrado en p84_04_ancla.json y RUNLOG.

## 4. Definition of Done de la fase

| Criterio | Resultado |
|---|---|
| Resultado registrado por experimento | PASA: P84-01..05 en CSV, JSON/MD, RUNLOG y D-281..D-286 |
| Sin fuga | PASA: ajustes en outer_train, umbrales intra-train, embeddings externos sin etiquetas, identidad V03 exacta |
| Sin uso de test ni holdout | PASA: contadores 1483/0/0 en los 5 experimentos; solo filas de desarrollo en los npz |
| Sin submissions | PASA: ninguna escritura ni archivo nuevo |
| Script nuevo por experimento sin editar cerrados | PASA: 5 scripts nuevos; ningun script previo editado en la ventana |
| Decisiones y QA en disco | PASA: DECISIONS D-281..286, plan con EXP-01..06, p84_qa_parcial.md y este p84_qa_final.md |

## 5. Veredicto global

**GO para el cierre documental de la fase P84** (no para submission). P84-04 cierra con
ancla lograda, identidad V03 exacta, contadores 1483/0/0, flujo pareado S/M verificado
por hash, F1 congeladas y deltas reproducidos de forma independiente, IC de bootstrap
exactos en el venv, decision de cierre y no elegibilidad EXP-06 documentadas. Las
discrepancias son de forma o limpieza y no comprometen los resultados ni el no-uso de
test/holdout. V03 sigue siendo el mejor resultado verificado; no se recomienda ninguna
submission nueva.

## 6. Nota de correccion (Enmienda 3bis, 2026-10-02)

La refutacion adversarial final detecto que experiments/p84_03_ros_equivalencia.py
contaba rescate/dano con `pred` en lugar de `pred_ref` (lineas ~927-929), forzando
0/0 en todas las comparaciones. Corregido y reejecutado (Enmienda 3bis; sha256 del
script 2a83741691a4d7064b14bd74999481ad339cbb03739d735a49e0ad8865a31e19, sustituye
al de la Enmienda 3). Recuentos corregidos sobre 4449 celdas contra V03 con umbrales
congelados: B1 61 celdas (32 rescates / 29 danos, neto +3), B2 446 (207/239, neto -32)
y B3 63 (34/29, neto +5); A=V03 sigue con 0 celdas distintas. La decision CLOSED BY
EQUIVALENCE no cambia. Este informe no contenia el conteo 0/0; la nota documenta la
correccion de la fase. Referencias: docs/p84_preregistro.md seccion 11, RUNLOG.md
19:06 y reports/p84_refutacion_final.md seccion 2.4.
