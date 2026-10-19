# P92-Y - Informe de entrega: calibracion holdout-safe con guia v2.1

Fecha: 2026-10-13. Estado: **CALIBRATION_V2_1_PACKAGE_READY** + HUMAN_CALIBRATION_PENDING.
Sin entrenamiento, sin predicciones, sin submissions, sin leaderboard;
sin distribucion automatica; el paquete principal no se modifica.

## 1. Preparacion

- La calibracion comun ya existia (`calibration/calibration_sample.csv`,
  15 tweets, vacia) y se reutilizo sin regenerar; no se re-selecciono nada.
- Se crearon las dos copias ciegas en `calibration_handoff/annotator_A/` y
  `annotator_B/` con los MISMOS 15 review_id en ORDEN DISTINTO
  (semillas de orden: A=20261024, B=20261025).
- Cada carpeta incluye: calibration_sample.csv, README_annotator.md y
  annotation_guidelines_v2_1.md (copia byte a byte de la guia v2.1).

## 2. Controles ejecutados

- cal_15_filas: OK
- cal_ids_coinciden_con_clave: OK
- cal_anotaciones_vacias: OK
- cal_version_v2: OK
- cal_sin_reservados: OK
- cal_en_dev2: OK
- cal_sin_solape_principal: OK
- cal_textos_identicos_fuente: OK
- guia_hash_interno: OK
- guia_hash_completo: OK
- manifiesto_paquete_intacto: OK
- gate_x_registra_manifiesto: OK
- ordenes_distintos: OK
- handoff_15_y_15: OK
- handoff_misma_membresia: OK
- handoff_orden_distinto: OK
- handoff_anotaciones_vacias: OK
- handoff_textos_30: OK
- handoff_guia_v21_identica: OK
- handoff_blinding_sin_fugas: OK
- clave_interna_fuera: OK
- verificador_ascii: OK
- textos de la calibracion contra la fuente: 15/15
- textos en los handoffs: 30/30

## 3. Elegibilidad y cegamiento

- 15/15 ids en dev2 (poblacion elegible); 0 solapes con test, holdout
  historico y holdout2 (verificado con metadatos de particion).
- 0 solapes con los componentes principales A/B.
- Archivos visibles sin etiquetas originales, ids de origen, OOF,
  predicciones, thresholds, folds, grupos, motivos ni clave interna.

## 4. Guia y hashes

- Guia v2.1 en ambos handoffs: sha256 completo `fd8d4c65ccd194fbce866321c7dbbbdc6870653f606e3111f147e41bc7fc9204`;
  sha256 interno (sin linea de hash) `ff699f54a48da7797f7305cf71a2718cfc12b3d99eab6f0eb80e0560cd3ed38a`.

| archivo | sha256 |
|---|---|
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/README.md | `24415943c6bf58df583f6ce194b0c063e34bfe1065c3501833e4b35227326a46` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_A/README_annotator.md | `3f71a0a8f7a07d489e42e1dbcce1712b729843e61d3534cb521d6e4e3859cad7` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_A/annotation_guidelines_v2_1.md | `fd8d4c65ccd194fbce866321c7dbbbdc6870653f606e3111f147e41bc7fc9204` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_A/calibration_sample.csv | `b9db5a97912e5cc742ef8d2c2e77bc9a9320948f72caa20f3c74f8d762c73af5` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_B/README_annotator.md | `1f03754760c49f0e623d4f101280d031ab94852077b90e2346cfae7f52fd8e98` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_B/annotation_guidelines_v2_1.md | `fd8d4c65ccd194fbce866321c7dbbbdc6870653f606e3111f147e41bc7fc9204` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_B/calibration_sample.csv | `d6885144da7a2efa125eeef0ad50e75789378155671adad15c26788768adc799` |

## 5. Instrucciones exactas para la entrega

1. Entregar a la persona A la carpeta COMPLETA
   `artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_A/`
   (3 archivos) y a la persona B la carpeta `annotator_B/` (3 archivos).
2. Por separado, sin comunicacion entre ambas hasta que las DOS hayan
   entregado sus archivos.
3. Cada persona completa los 15 tweets y devuelve SOLO su CSV renombrado
   como `calibration_response_A.csv` / `calibration_response_B.csv`.
4. No completar ni simular respuestas; las columnas de anotacion se llenan
   unicamente por las personas.
5. Al recibir ambas respuestas: validar esquema (15 filas, tokens
   permitidos), almacenar verbatim en `calibration_responses/` de esta
   ronda y solo entonces abrir la discusion.
6. La guia v2.1 es politica operativa interna: no presentarla como
   resolucion normativa oficial; la ronda evalua claridad, no prevalencia.

## 6. Estado

```text
CALIBRATION_V2_1_PACKAGE_READY
HUMAN_CALIBRATION_PENDING
MAIN_ANNOTATION_NOT_DISTRIBUTED
V03 = FROZEN
NO NEW MODELING PHASE
```

- Verificador ASCII oficial: OK: 0 caracteres no permitidos en los ficheros propios del proyecto.
- La muestra principal de 250 tweets seguira sin distribuirse hasta recibir
  ambas calibraciones y decidir si la guia puede mantenerse congelada.
