# Inventario unificado de experimentos

Generado por `auditoria/generar_inventario.py` el 2026-10-02 a partir de `experiments.csv`, `experiments/*.py`, `notebooks/10_final_pipeline.ipynb`, `reports/`, `resultados/` y `figuras/`. Las cifras son reproducibles: basta reejecutar el script con el venv del proyecto.

## 1. Cobertura del registro y de los scripts

| Concepto | Total |
| --- | ---: |
| Filas de datos en `experiments.csv` | 411 |
| Filas bien formadas (23 columnas) | 368 |
| Filas malformadas (ancho distinto de 23, desde P67) | 43 |
| Scripts `.py` en `experiments/` | 376 |
| Scripts con fila o asociación de registro | 304 |
| Scripts sin fila de registro | 72 |
| Experimentos en el inventario | 483 |
| Experimentos con evidencia atribuida por id o script | 174 |
| Experimentos con evidencia solo a nivel de fase | 124 |
| Experimentos sin evidencia atribuible | 185 |

## 2. Scripts por tipo

| Tipo | Scripts |
| --- | ---: |
| experimento | 251 |
| auditoria | 43 |
| integracion_notebook | 39 |
| cierre | 26 |
| submission | 9 |
| registro | 5 |
| scaffold | 2 |
| ayudas | 1 |

## 3. Estado de los experimentos

| Estado | Experimentos |
| --- | ---: |
| con_evidencia_y_notebook | 74 |
| con_evidencia_sin_notebook | 100 |
| sin_evidencia | 177 |
| solo_notebook | 132 |

Criterio de evidencia: una ruta de `reports/`, `resultados/` o `figuras/` cuenta si su nombre contiene el id (variantes `-`/`_`) o alguna variante del basename del script sin `.py` (stem completo o stem sin el índice numérico). La evidencia que solo coincide con el token de fase (`v44`, `p60`, `embedding`) no se cuenta en `estado`: se informa aparte para no atribuir a un sub-experimento artefactos de toda la fase.

## 4. Experimentos sin documentar en el notebook (277)

Ids sin ninguna mención en el notebook (se listan como máximo 80, en orden de registro):

```
E2-14-MEJOR-M2	e2	experimento
E2-14-MEJOR-META	e2	experimento
E2-13-META-ENT-LR	e2	experimento
E2-13-ENT-SOLO-LR	e2	experimento
E2-16-LOCAL-SIN-SHRINKAGE	e2	experimento
E2-16-SHRINK-M25	e2	experimento
E2-16-SHRINK-M50	e2	experimento
E2-16-SHRINK-M100	e2	experimento
E2-16-MEJOR	e2	experimento
E2-21-HS	e2	experimento
E2-21-TR	e2	experimento
E2-21-AG	e2	experimento
AUD-01	otras	auditoria
AUD-02	otras	auditoria
AUD-03	otras	auditoria
FIN-02-C-EMO	otras	experimento
FIN-03-C-EMO-margenes	otras	experimento
FIN-04-C-EMO-SYN	otras	experimento
FIN-05-C-SYN	otras	experimento
FIN-06-C-REF-44-CH	otras	experimento
FIN-07-C-EMO-SYN-TGT-CYT	otras	experimento
FIN-08-C-EMO	otras	submission
FIN-09-C-SYN	otras	submission
FIN-10-C-EMO-SYN	otras	submission
V4-00-SCAFFOLD	v4	scaffold
V4-B04	v4	experimento
V4-G01-G03	v4	experimento
EBD-00-SCAFFOLD	embedding	scaffold
E-CORP-01	embedding	experimento
EBD-07	embedding	experimento
EBD-08	embedding	experimento
EBD-09	embedding	experimento
E-CORP-06	embedding	experimento
E-CORP-07	embedding	experimento
EBD-10	embedding	experimento
E-CORP-08	embedding	experimento
EBD-13	embedding	experimento
EBD-14	embedding	experimento
E-CORP-09	embedding	experimento
EBD-15	embedding	experimento
EBD-16	embedding	experimento
EBD-17	embedding	auditoria
EBD-18	embedding	experimento
EBD-19	embedding	experimento
EBD-20	embedding	experimento
EBD-21	embedding	experimento
EBD-22	embedding	experimento
EBD-23	embedding	experimento
EBD-24	embedding	experimento
EBD-25	embedding	integracion_notebook
EBD-26	embedding	integracion_notebook
EBD-27	embedding	integracion_notebook
V6-00	v6	experimento
V6-01	v6	experimento
V6-04	v6	experimento
V6-05	v6	experimento
V6-06	v6	experimento
V6-08	v6	experimento
V6-09	v6	experimento
V7-02	v7	experimento
V7-03	v7	experimento
V7-04	v7	experimento
V8-00	v8	auditoria
V8-01	v8	experimento
V8-05	v8	experimento
V8-10	v8	experimento
V8-11	v8	experimento
V8-12	v8	experimento
V8-13	v8	experimento
V10-01	v10	experimento
V10-08	v10	experimento
V10-09	v10	experimento
V11-01	v11	experimento
V11-02	v11	auditoria
V11-03	v11	auditoria
V11-06	v11	experimento
V11-07	v11	experimento
V11-05	v11	registro
V11-08	v11	cierre
V12-00	v12	auditoria
```
Lista truncada: se muestran 80 de 277 experimentos sin notebook.

## 5. Scripts sin fila de registro (72)

| Script | Familia | Tipo | Evidencia | Notebook |
| --- | --- | --- | --- | --- |
| `experiments/37_auditoria_v4.py` | otras | auditoria | NO | NO |
| `experiments/embedding_06_auditoria_v5.py` | embedding | auditoria | NO | NO |
| `experiments/embedding_22_auditoria_final_v5_1.py` | embedding | auditoria | NO | NO |
| `experiments/p59_00_registro_hipotesis.py` | p59 | registro | NO | NO |
| `experiments/p59_01_voronoi_local.py` | p59 | experimento | NO | NO |
| `experiments/p59_05_nmf_gap_analysis.py` | p59 | experimento | NO | NO |
| `experiments/p59_08_calibracion_local.py` | p59 | experimento | NO | NO |
| `experiments/p59_11_generativo_ngram.py` | p59 | experimento | NO | NO |
| `experiments/p60_05_ruido_por_etiqueta.py` | p60 | experimento | NO | NO |
| `experiments/p60_14_auditoria.py` | p60 | auditoria | NO | NO |
| `experiments/p60_registrar.py` | p60 | registro | NO | NO |
| `experiments/p61_00_reconciliacion.py` | p61 | auditoria | NO | NO |
| `experiments/p61_01_eda_probabilidades.py` | p61 | experimento | NO | NO |
| `experiments/p61_02_inventario.py` | p61 | experimento | NO | NO |
| `experiments/p62_06_notebook.py` | p62 | integracion_notebook | SI | NO |
| `experiments/p62_10_techo_cortos.py` | p62 | experimento | SI | NO |
| `experiments/p62_11_cerrar.py` | p62 | cierre | NO | NO |
| `experiments/p63_09_dataset_auxiliar.py` | p63 | experimento | NO | NO |
| `experiments/p64_05_multiple_instance.py` | p64 | experimento | SI | NO |
| `experiments/p64_06_cerrar.py` | p64 | cierre | NO | NO |
| `experiments/p64_06_quote_mention.py` | p64 | experimento | SI | NO |
| `experiments/p64_07_negacion.py` | p64 | experimento | SI | NO |
| `experiments/p64_08_lda.py` | p64 | experimento | NO | NO |
| `experiments/p65_05_cerrar.py` | p65 | cierre | NO | NO |
| `experiments/p66_04_cerrar.py` | p66 | cierre | NO | NO |
| `experiments/p72_02_dimension_downstream.py` | p72 | experimento | NO | NO |
| `experiments/p73_04_auditoria_cierre.py` | p73 | auditoria | SI | NO |
| `experiments/p78_01_control_anidado.py` | p78 | experimento | SI | NO |
| `experiments/p79_01_auditoria_sgns.py` | p79 | auditoria | SI | NO |
| `experiments/p80_01_auditoria_w2v.py` | p80 | auditoria | SI | NO |
| `experiments/p81_01_cierre_huecos.py` | p81 | cierre | SI | NO |
| `experiments/p82_01_rnn_atencion.py` | p82 | experimento | SI | NO |
| `experiments/p82_02_submission_rnn.py` | p82 | submission | SI | NO |
| `experiments/p83_01_bayesiano.py` | p83 | experimento | NO | NO |
| `experiments/v16_00_prevalencia_y_curva_lb.py` | v16 | experimento | SI | NO |
| `experiments/v16_01_generar_submissions.py` | v16 | submission | NO | NO |
| `experiments/v16_02_promedio_umbrales.py` | v16 | experimento | SI | NO |
| `experiments/v16_03_vecinos_de_capacidad.py` | v16 | experimento | SI | NO |
| `experiments/v17_00_inventario_y_complementariedad.py` | v17 | experimento | NO | NO |
| `experiments/v17_01_informe_final.py` | v17 | cierre | SI | NO |
| `experiments/v18_01_cambio_y_senal.py` | v18 | experimento | SI | NO |
| `experiments/v19_00_metrica_verificada.py` | v19 | experimento | SI | NO |
| `experiments/v19_01_valor_de_cada_correccion.py` | v19 | experimento | SI | NO |
| `experiments/v19_02_restriccion_tr_implica_hs.py` | v19 | experimento | SI | NO |
| `experiments/v20_01_informacion_incremental.py` | v20 | experimento | SI | NO |
| `experiments/v20_02_inferencia_y_candidata.py` | v20 | experimento | SI | NO |
| `experiments/v21_01_triaje_competitivo.py` | v21 | experimento | SI | NO |
| `experiments/v22_01_auditoria_preprocessing.py` | v22 | auditoria | SI | NO |
| `experiments/v23_01_auditoria_rbf_svc.py` | v23 | auditoria | SI | NO |
| `experiments/v23_02_submission_rbf_sigmoid.py` | v23 | submission | SI | NO |
| `experiments/v24_01_modelo_transferencia_complejidad.py` | v24 | experimento | NO | NO |
| `experiments/v26_04_integrar_notebook.py` | v26 | integracion_notebook | NO | NO |
| `experiments/v27_02_integrar_notebook.py` | v27 | integracion_notebook | NO | NO |
| `experiments/v38_03_integrar_v39.py` | v38 | integracion_notebook | NO | NO |
| `experiments/v42_02_integrar_notebook.py` | v42 | integracion_notebook | NO | NO |
| `experiments/v47_01_ajuste_pliegues.py` | v47 | experimento | NO | NO |
| `experiments/v47_01_especializacion.py` | v47 | experimento | NO | NO |
| `experiments/v47_01_especializacion_por_etiqueta.py` | v47 | experimento | NO | NO |
| `experiments/v48_02_integrar_notebook.py` | v48 | integracion_notebook | NO | NO |
| `experiments/v49_01_integrar_notebook.py` | v49 | integracion_notebook | NO | NO |
| `experiments/v50_00_brecha_y_oracle.py` | v50 | experimento | NO | NO |
| `experiments/v51_00_utilidad_y_asimetria.py` | v51 | experimento | NO | NO |
| `experiments/v51_03_integrar_notebook.py` | v51 | integracion_notebook | NO | NO |
| `experiments/v54_00_atlas_no_rescatable.py` | v54 | experimento | NO | NO |
| `experiments/v54_01_integrar_notebook.py` | v54 | integracion_notebook | NO | NO |
| `experiments/v55_00_auditoria_pipeline.py` | v55 | auditoria | NO | NO |
| `experiments/v55_01_integrar_notebook.py` | v55 | integracion_notebook | NO | NO |
| `experiments/v56_01_integrar_notebook.py` | v56 | integracion_notebook | NO | NO |
| `experiments/v57_00_sintesis_estrategica.py` | v57 | cierre | NO | NO |
| `experiments/v57_01_integrar_notebook.py` | v57 | integracion_notebook | NO | NO |
| `experiments/v58_00_auditoria_candidatos.py` | v58 | auditoria | NO | NO |
| `experiments/v58_01_integrar_notebook.py` | v58 | integracion_notebook | NO | NO |

Scripts sin ninguna cita en `reports/`, `resultados/` o `figuras/`: 215 de 376. La divergencia de nombres entre scripts (`p60_02_asl.py`) y artefactos (`p60_A_asl.json`) explica la mayor parte de las citas no encontradas.

## 6. Secciones del notebook sin registro

Secciones de nivel 2 cuyo token de fase (`P45`, `V03`, ...) no existe en `experiments.csv`. La columna de alternativas señala fases registradas con la otra letra de nomenclatura:

| Sección | Celda | Título | Tokens sin registro | Alternativas |
| ---: | ---: | --- | --- | --- |
| 10 | 25 | Fase v2 (Advanced Research): paquetes P0-P7, resultados exactos y decisiones | P0, P7 | P7->V7 |
| 18 | 44 | Fase v5 y v5.1: representación distribuida aprendida desde cero | V5 |  |
| 40 | 181 | P45 - Auditoria de la metrica (2026-09-27) | P45 | P45->V45 |
| 41 | 183 | P45 - Auditoria de la estructura conjunta HS x TR x AG (2026-09-27) | P45 | P45->V45 |
| 42 | 185 | P46 - Leaderboard and Champion Audit (2026-09-27) | P46 | P46->V46 |
| 43 | 187 | P47 - Data Availability Audit (2026-09-27) | P47 | P47->V47 |
| 44 | 189 | P48 - EDA-Driven Residual Research (2026-09-27) | P48 |  |
| 45 | 192 | P49 - OOF Rescue Reconciliation (2026-09-28) | P49 |  |
| 46 | 194 | P50 - Contextual Channel Selection (2026-09-28) | P50 | P50->V50 |
| 47 | 196 | P51 - Utility-Aware Contextual Gating (2026-09-28) | P51 | P51->V51 |
| 48 | 198 | P52 - Policy Oracle and Channel Complementarity (2026-09-28) | P52 |  |
| 49 | 200 | P53 - Fine-Grained Observable Policy Audit (2026-09-28) | P53 |  |
| 50 | 202 | P54 - Auditoria semantica de los errores no rescatables (2026-09-28) | P54 |  |
| 51 | 204 | P55 - Auditoria estrategica del pipeline final de V03 (2026-09-28) | P55, V03 |  |
| 52 | 206 | P56 - Transferencia de umbrales corregida (2026-09-28) | P56 | P56->V56 |
| 53 | 208 | P57 - Cierre estrategico y plan final (2026-09-28) | P57 |  |
| 54 | 210 | P58 - Auditoria final de candidatos existentes (2026-09-28) | P58 |  |
| 58 | 218 | P59-D - Influence functions sobre V03 | V03 |  |
| 64 | 230 | P59-E - Regularizacion por grafo sobre el objetivo de V03 | V03 |  |
| 71 | 244 | P60-D - Numero efectivo de muestras frente a la forma de V03 | V03 |  |
| 83 | 269 | La EDA de las probabilidades OOF de V03, que es la prioridad especial de la fase | V03 |  |
| 83 | 294 | La EDA de las probabilidades OOF de V03, que es la prioridad especial de la fase | V03 |  |
| 92 | 324 | Donde falla V03, por perfil linguistico | V03 |  |
| 99 | 337 | CORRECCION: V03 es MEJOR en los tweets cortos, y el problema son los largos | V03 |  |
| 121 | 437 | PHASE - SELECTIVE RESIDUAL FUSION V03 TEXTCNN | V03 |  |

La columna Celda usa \u00edndice base 0 del notebook. Los tokens `V03` corresponden al modelo campe\u00f3n y los `P0`/`P7` a paquetes internos de la fase v2, no a fases del registro: se listan por rigor, pero no son huecos de registro.

### Secciones numeradas repetidas

- Sección 81: P61 - Capacidad, diversidad y distribucion de probabilidad
- Sección 82: La reconciliacion de los nueve problemas de instrumentacion de P60
- Sección 83: La EDA de las probabilidades OOF de V03, que es la prioridad especial de la fase
- Sección 84: El inventario de modelos y la medicion de la diversidad
- Sección 85: La busqueda de capacidad LightGBM, con Optuna anidado
- Sección 86: El ensemble con pertenencia y pesos ANIDADOS, y el control de atribucion
- Sección 87: La segunda particion, y el control negativo que decide
- Sección 88: P61 - Cierre de la fase

### Números de sección ausentes en la numeración de nivel 2

- 32, 38

### Fases con scripts en `experiments/` sin ninguna mención en el notebook

| Fase | Scripts |
| --- | ---: |
| P78 | 1 |
| P79 | 1 |
| P81 | 1 |
| P82 | 2 |
| P83 | 1 |
| V13 | 3 |
| V21 | 1 |
| V22 | 3 |
| V36 | 2 |

## 7. Filas malformadas del registro

Las 43 filas malformadas usan un formato de barras (`id|fecha|...`) con comas sin comillas en el payload; `csv.reader` las parte en 1 a 12 campos. Su línea cruda se conserva intacta en la fuente y se usó para extraer id, fecha y scripts. Líneas afectadas:

```
370	P67-00	ancho=3
371	P67-01	ancho=5
372	P67-02	ancho=5
373	P67-03	ancho=5
374	P67-04	ancho=5
375	P68-01	ancho=4
376	P68-02	ancho=4
377	P69-01	ancho=4
378	P69-02	ancho=2
379	P69-03	ancho=5
380	P69-04	ancho=2
381	P69-05	ancho=5
382	P69-06	ancho=1
383	P69-07	ancho=4
384	P69-08	ancho=3
385	P69-09	ancho=2
386	P69-10	ancho=5
387	P70-01	ancho=3
388	P70-02	ancho=2
389	P70-03	ancho=1
390	P71-00	ancho=3
391	P71-01	ancho=3
392	P71-02	ancho=1
393	P71-03	ancho=2
394	P71-04	ancho=2
395	P71-05	ancho=3
396	P71-06	ancho=2
397	P71-07	ancho=1
398	P71-08	ancho=5
399	P71-09	ancho=1
400	P71-10	ancho=2
401	P72-00	ancho=1
402	P72-01	ancho=1
403	P72-03	ancho=2
404	P72-04	ancho=3
405	P73-01	ancho=2
406	P73-02	ancho=2
407	P73-03	ancho=7
408	P74-PRE	ancho=3
409	P74	ancho=12
410	P75	ancho=2
411	P76	ancho=8
412	P77	ancho=10
```

## 8. Huecos principales detectados

- El registro deja de ser CSV tabular en P67: 43 filas en formato de barras con comas sin comillas (anchos de 1 a 12), de modo que un lector estricto de CSV pierde 43 experimentos.
- Scripts sin fila de registro: 72 de 376; por familia destacan p59(5), p64(5), v16(4), p60(3), p61(3), p62(3).
- Fases con scripts en `experiments/` y ninguna mención en el notebook: P78, P79, P81, P82, P83, V13, V21, V22, V36.
- El notebook no tiene sección propia para P78, P79, P80, P81, P82 ni P83; el único `p80` del notebook es el percentil `umbral_de_dureza_p80`, no la fase P80.
- Números de sección ausentes en la numeración de nivel 2: 32, 38.
- Secciones numeradas repetidas: 81, 82, 83, 84, 85, 86, 87, 88 (el bloque P61, secciones 81-88, aparece dos veces: celdas 265-286 y 290-311).
- Experimentos solo en notebook, sin evidencia: 132; experimentos sin evidencia ni menci\u00f3n en el notebook: 177.
- Nomenclatura divergente: el registro usa ids V45-xx a V58-xx mientras el notebook titula esas fases P45 a P58 (ver sección 6).
- La evidencia se nombra por fase o por letra y no por el índice del script (por ejemplo `p60_02_asl.py` frente a `p60_A_asl.json`), lo que impide atribuirla automáticamente sin revisión manual.

