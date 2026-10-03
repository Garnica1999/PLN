# Inventario unificado de experimentos

Generado por `auditoria/generar_inventario.py` el 2026-10-02 a partir de `experiments.csv`, `experiments/*.py`, `notebooks/10_final_pipeline.ipynb`, `reports/`, `resultados/` y `figuras/`. Las cifras son reproducibles: basta reejecutar el script con el venv del proyecto.

## 1. Cobertura del registro y de los scripts

| Concepto | Total |
| --- | ---: |
| Filas de datos en `experiments.csv` | 450 |
| Filas bien formadas (23 columnas) | 450 |
| Filas malformadas (ancho distinto de 23, desde P67) | 0 |
| Scripts `.py` en `experiments/` | 376 |
| Scripts con fila o asociación de registro | 343 |
| Scripts sin fila de registro | 33 |
| Experimentos en el inventario | 483 |
| Experimentos con evidencia atribuida por id o script | 179 |
| Experimentos con evidencia solo a nivel de fase | 119 |
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
| con_evidencia_y_notebook | 83 |
| con_evidencia_sin_notebook | 96 |
| sin_evidencia | 170 |
| solo_notebook | 134 |

Criterio de evidencia: una ruta de `reports/`, `resultados/` o `figuras/` cuenta si su nombre contiene el id (variantes `-`/`_`) o alguna variante del basename del script sin `.py` (stem completo o stem sin el índice numérico). La evidencia que solo coincide con el token de fase (`v44`, `p60`, `embedding`) no se cuenta en `estado`: se informa aparte para no atribuir a un sub-experimento artefactos de toda la fase.

## 4. Experimentos sin documentar en el notebook (266)

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
Lista truncada: se muestran 80 de 266 experimentos sin notebook.

## 5. Scripts sin fila de registro (33)

| Script | Familia | Tipo | Evidencia | Notebook |
| --- | --- | --- | --- | --- |
| `experiments/37_auditoria_v4.py` | otras | auditoria | NO | NO |
| `experiments/embedding_06_auditoria_v5.py` | embedding | auditoria | NO | NO |
| `experiments/embedding_22_auditoria_final_v5_1.py` | embedding | auditoria | NO | NO |
| `experiments/p59_00_registro_hipotesis.py` | p59 | registro | NO | NO |
| `experiments/p60_14_auditoria.py` | p60 | auditoria | NO | NO |
| `experiments/p60_registrar.py` | p60 | registro | NO | NO |
| `experiments/p61_00_reconciliacion.py` | p61 | auditoria | NO | NO |
| `experiments/p62_06_notebook.py` | p62 | integracion_notebook | SI | NO |
| `experiments/p62_11_cerrar.py` | p62 | cierre | NO | NO |
| `experiments/p64_06_cerrar.py` | p64 | cierre | NO | NO |
| `experiments/p65_05_cerrar.py` | p65 | cierre | NO | NO |
| `experiments/p66_04_cerrar.py` | p66 | cierre | NO | NO |
| `experiments/p73_04_auditoria_cierre.py` | p73 | auditoria | SI | NO |
| `experiments/v16_01_generar_submissions.py` | v16 | submission | NO | NO |
| `experiments/v17_01_informe_final.py` | v17 | cierre | SI | NO |
| `experiments/v22_01_auditoria_preprocessing.py` | v22 | auditoria | SI | NO |
| `experiments/v23_01_auditoria_rbf_svc.py` | v23 | auditoria | SI | NO |
| `experiments/v23_02_submission_rbf_sigmoid.py` | v23 | submission | SI | NO |
| `experiments/v26_04_integrar_notebook.py` | v26 | integracion_notebook | NO | NO |
| `experiments/v27_02_integrar_notebook.py` | v27 | integracion_notebook | NO | NO |
| `experiments/v38_03_integrar_v39.py` | v38 | integracion_notebook | NO | NO |
| `experiments/v42_02_integrar_notebook.py` | v42 | integracion_notebook | NO | NO |
| `experiments/v48_02_integrar_notebook.py` | v48 | integracion_notebook | NO | NO |
| `experiments/v49_01_integrar_notebook.py` | v49 | integracion_notebook | NO | NO |
| `experiments/v51_03_integrar_notebook.py` | v51 | integracion_notebook | NO | NO |
| `experiments/v54_01_integrar_notebook.py` | v54 | integracion_notebook | NO | NO |
| `experiments/v55_00_auditoria_pipeline.py` | v55 | auditoria | NO | NO |
| `experiments/v55_01_integrar_notebook.py` | v55 | integracion_notebook | NO | NO |
| `experiments/v56_01_integrar_notebook.py` | v56 | integracion_notebook | NO | NO |
| `experiments/v57_00_sintesis_estrategica.py` | v57 | cierre | NO | NO |
| `experiments/v57_01_integrar_notebook.py` | v57 | integracion_notebook | NO | NO |
| `experiments/v58_00_auditoria_candidatos.py` | v58 | auditoria | NO | NO |
| `experiments/v58_01_integrar_notebook.py` | v58 | integracion_notebook | NO | NO |

Scripts sin ninguna cita en `reports/`, `resultados/` o `figuras/`: 214 de 376. La divergencia de nombres entre scripts (`p60_02_asl.py`) y artefactos (`p60_A_asl.json`) explica la mayor parte de las citas no encontradas.

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
| 50 | 202 | P54 - Auditoria semantica de los errores no rescatables (2026-09-28) | P54 | P54->V54 |
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
| V13 | 3 |
| V21 | 1 |
| V22 | 3 |
| V36 | 2 |

## 7. Filas malformadas del registro

Las 43 filas malformadas usan un formato de barras (`id|fecha|...`) con comas sin comillas en el payload; `csv.reader` las parte en 1 a 12 campos. Su línea cruda se conserva intacta en la fuente y se usó para extraer id, fecha y scripts. Líneas afectadas:

```
```

## 8. Huecos principales detectados

- El registro deja de ser CSV tabular en P67: 0 filas en formato de barras con comas sin comillas (anchos de 1 a 12), de modo que un lector estricto de CSV pierde 43 experimentos.
- Scripts sin fila de registro: 33 de 376; por familia destacan embedding(2), p60(2), p62(2), v23(2), v55(2), v57(2).
- Fases con scripts en `experiments/` y ninguna mención en el notebook: V13, V21, V22, V36.
- El notebook no tiene sección propia para P78, P79, P80, P81, P82 ni P83; el único `p80` del notebook es el percentil `umbral_de_dureza_p80`, no la fase P80.
- Números de sección ausentes en la numeración de nivel 2: 32, 38.
- Secciones numeradas repetidas: 81, 82, 83, 84, 85, 86, 87, 88 (el bloque P61, secciones 81-88, aparece dos veces: celdas 265-286 y 290-311).
- Experimentos solo en notebook, sin evidencia: 134; experimentos sin evidencia ni menci\u00f3n en el notebook: 170.
- Nomenclatura divergente: el registro usa ids V45-xx a V58-xx mientras el notebook titula esas fases P45 a P58 (ver sección 6).
- La evidencia se nombra por fase o por letra y no por el índice del script (por ejemplo `p60_02_asl.py` frente a `p60_A_asl.json`), lo que impide atribuirla automáticamente sin revisión manual.

