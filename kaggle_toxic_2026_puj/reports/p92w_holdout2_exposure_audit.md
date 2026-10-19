# P92-W - Auditoria de exposicion de holdout2 y reconstruccion segura

Fecha: 2026-10-13. Fase diagnostica: sin entrenamiento, sin predicciones, sin
submissions, sin leaderboard. No se releyeron labels, scores ni metricas de
holdout2: la auditoria usa exclusivamente metadatos de pertenencia ya
registrados y documentacion existente.

## 1. Objeto

La instruccion global del proyecto establece HOLDOUT2 = FROZEN y su no uso
durante la exploracion. El muestreo de P92 declaro excluir holdout2 en su
manifiesto, pero el marco usado (dev = 1483) lo contenia. Esta auditoria
documenta la exposicion real y reconstruye el paquete con exclusiones
verificadas.

## 2. Cronologia (reconstruida de artefactos existentes)

- 2026-09-26: se crea holdout2 (experiments/v10_01_holdout2.py) como pliegue de
  dev para deteccion de sobreajuste; el propio script advierte que NO es un
  conjunto virginal (sus filas participaron en decisiones de fases v1-v9).
- 2026-10-09: P92 construye el muestreo. El manifiesto declara excluidos test,
  holdout y holdout2; sin embargo el marco efectivo fue dev (1483), que incluye
  holdout2 (247). La auditoria de preparacion valido 14/14 contra la particion
  holdout de holdout_v2, sin comprobar holdout2.
- 2026-10-12: calibracion (15 tweets, 4 anotadores) y analisis P92-T; el
  analisis no consulto particiones reservadas.
- 2026-10-13: la auditoria de P92-V detecta 35 filas principales y 3 de
  calibracion con pertenencia a holdout2 (por interseccion de ids) y lo
  registra como aviso; P92-W escala a auditoria formal y reconstruccion.

## 3. Inventario de acceso

- Consultado (solo metadatos): data/processed/holdout2_ids.csv (lista de 247
  ids; el archivo no contiene labels ni scores); cabecera de v10_01_holdout2.py
  (procedencia); declaraciones de auditorias P90 ('congelado, no tocado').
- NO consultado: textos, labels, scores, predicciones o metricas de holdout2
  como conjunto; no existe ningun artefacto especifico de contenido de holdout2.
- En P92-W la cola P91 (candidatos) se filtro por id reservado ANTES de usar
  cualquier otro campo; el OOF canonico no se cargo en esta fase.

## 4. Casos expuestos (metadatos de pertenencia)

- Filas principales afectadas (35/250): P92R-0001, P92R-0004, P92R-0015, P92R-0018, P92R-0024, P92R-0029, P92R-0030, P92R-0042, P92R-0045, P92R-0055, P92R-0066, P92R-0072, P92R-0086, P92R-0100, P92R-0107, P92R-0116, P92R-0117, P92R-0126, P92R-0146, P92R-0160, P92R-0172, P92R-0173, P92R-0180, P92R-0183, P92R-0199, P92R-0210, P92R-0212, P92R-0221, P92R-0232, P92R-0233, P92R-0235, P92R-0237, P92R-0243, P92R-0244, P92R-0246.
- Calibracion afectada (3/15): P92C-10, P92C-14, P92C-15.
- Deteccion: interseccion de ids en la clave interna contra la lista de
  holdout2; no requirio leer ningun contenido de holdout2.

## 5. Clasificacion de la exposicion

```text
HOLDOUT2_TEXT_EXPOSURE:              DEMOSTRADO (38 filas: textos dentro de
                                    paquetes de anotacion; 3 revisados por humanos)
HOLDOUT2_HUMAN_ANNOTATION_EXPOSURE: DEMOSTRADO para 3 filas de calibracion
                                    (4 respuestas humanas + notas + analisis);
                                    NO OCURRIO para las 35 principales (sin anotaciones)
HOLDOUT2_LABEL_ACCESS:              En P92: no se leyeron/analizaron labels por fila
                                    (diseno ciego; analisis de calibracion sin labels).
                                    Historico: las filas eran dev antes de V10; su
                                    participacion en analisis agregados de dev no es
                                    verificable por fila -> NO VERIFICABLE por fila
HOLDOUT2_SCORE_ACCESS:              El OOF canonico cubre dev (incluye esas filas);
                                    no se uso para seleccion ni analisis en P92/P92-W;
                                    uso historico agregado -> NO VERIFICABLE por fila
HOLDOUT2_METRIC_ACCESS:             NO OCURRIO en P92/P92-V/P92-W (ninguna metrica
                                    se computo sobre holdout2 como conjunto)
```

No se afirma que holdout2 haya permanecido completamente intacto: la exposicion
real queda documentada arriba, sin inferir acceso a labels o scores a partir de
la revision de textos.

## 6. Influencia en la guia v2 (evaluacion)

- De los 3 casos expuestos de calibracion, 2 (P92C-14, P92C-15) figuran entre las seis filas HS citadas
  como evidencia central de la guia v2 (patron odio vs. ofensa); P92C-10 se
  describio como caso sin duda. Por tanto NO puede demostrarse que las
  decisiones operativas de v2 sean independientes de la evidencia expuesta.
- Se registra: GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE.
- Evaluacion limitada con los 12 casos elegibles de la
  calibracion (sin reabrir los 3 excluidos; usando el analisis ya existente):
  UTD coincidente C/D en HS: 4/12 (P92C-05, P92C-09, P92C-11, P92C-13); con alguna senal de duda: 10/12.
  El patron de ambiguedad normativa que fundamenta Q1 se conserva en
  evidencia elegible; esto sostiene la direccion operativa, pero no
  sustituye la independencia plena (los conteos originales 6/15 incluian 2
  filas expuestas).
- v2 se mantiene como POLITICA OPERATIVA interna (nunca fue definicion
  oficial). Se registra la necesidad de una revision humana de la guia
  apoyada solo en evidencia elegible si se desea independencia plena; la
  nueva calibracion del paquete holdout-safe sirve a ese fin.

## 7. Artefactos derivados que contienen informacion de los casos expuestos

- artifacts/human_annotation_v1/ (38 textos; sin anotaciones) [historico]
- artifacts/human_annotation_v2/ (38 textos) [QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT]
- calibration_response_A/B/C/D.csv (3 filas con respuestas humanas)
- reports/p92_calibration_analysis.md, p92t_audit_results.json, p92s2_*, p92u_*,
  p92_guideline_decision_record.md, p92v_* y notebook (secciones 135-143)
- reports/p92_internal_sampling_key.csv (pertenencia y labels de los 265)
- Esta auditoria y el manifiesto P92-W (metadatos de pertenencia)

## 8. Tratamiento

- Las respuestas humanas y los analisis derivados de los 3 casos de
  calibracion quedan EXCLUIDOS de futuras evaluaciones que requieran holdout2
  sin exposicion; no se les pedira a los anotadores volver a etiquetarlos.
- Las 35 filas principales expuestas quedan excluidas de la muestra nueva;
  la reconstruccion usa la poblacion dev2 (1236) con semillas nuevas.
- No se modifico ningun label original; ninguna discrepancia futura sera
  correccion automatica (vocabulario de estados de P92-V vigente).

## 9. Hashes relevantes

```text
data/raw/toxic_train.csv: 3d40e7345b588860...
features/oof_v03_final.npz: 559265816ce4f631...
data/processed/holdout_v2_ids.csv: e5b62e5f22c4c63a...
submissions/toxic_submission_v03_lr_union_thr.csv: 7d7478f271e6edfd...
artifacts/human_annotation_v1/annotator_A.csv: 7dfd5b898d3905f2...
artifacts/human_annotation_v1/annotator_B.csv: 956646fa479f0c65...
```

Hashes completos de los paquetes en cuarentena (v2, sin cambios):

- annotator_A/README_annotator.md: 9c38f06586e80712638e75c55e443c373d8fe2f2955a44dcf123cf7d8e4f991b
- annotator_A/annotation_guidelines_v2_frozen.md: 77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4
- annotator_A/annotator_A.csv: 1238591610216bf3b8378bfdf5e9bc2fa034644da4f19e2dc01b5e104bc456ac
- annotator_B/README_annotator.md: 470295b7bf4bd0e341384824109f593d8500cb3adabfaea128fd1e9322c0c994
- annotator_B/annotation_guidelines_v2_frozen.md: 77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4
- annotator_B/annotator_B.csv: 8085697fe408d7a83636de495673627af800dbc9e0542fe7323f05c7b7501e22
- manifest.json: 2f9dddd7804c1fbeef392d25de6a343c93d6d12e216f19d0232b8d452aecb514
