# P92-U - Gate de distribucion de la muestra principal

Fecha: 2026-10-12. Estado: **NOT_READY (por diseno)**: la muestra principal de
250 tweets NO se distribuye en esta fase. Este documento define las condiciones
exactas que deben cumplirse antes de distribuirla y su estado actual.

## 1. Las seis condiciones del gate

| # | condicion | estado actual | detalle |
|---|---|---|---|
| 1 | Version aplicable de la guia identificada y congelada | PENDIENTE | v1 es historica; v2 esta propuesta (no congelada); la version aplicable se decide en el registro |
| 2 | Criterios procedimentales resueltos | PENDIENTE | Q4 a Q8 sin decision registrada (formato, UTD, citas, tono, duda de fila) |
| 3 | Ambiguedades normativas restantes documentadas | CUMPLIDA (mantener al dia) | documentadas en `p92_calibration_analysis.md`, propuesta v2 seccion 4 y registro Q1/Q3 |
| 4 | Ambos paquetes con la misma version de instrucciones | PENDIENTE | hoy ambos usan v1 (manifest); al congelar la guia nueva se actualizan ambos paquetes de forma trazable |
| 5 | Cegamiento e integridad de paquetes verificados | PARCIAL | verificado contra v1 en P92 (14/14); requiere re-verificacion contra la version congelada en el momento de distribuir |
| 6 | Decision de iniciar la anotacion principal registrada | PENDIENTE | plantilla en `p92_guideline_decision_record.md` seccion 3 |

## 2. Estado actual de los paquetes (evidencia)

- `annotator_A.csv` y `annotator_B.csv`: 250 tweets cada uno, anotaciones vacias,
  review_id opacos; hashes verificados (seccion 4). NO distribuidos.
- `manifest.json` del paquete: guideline_version v1, hashes de los cinco archivos,
  listado de cegamiento (ocultos: etiquetas, probabilidades, predicciones, folds,
  componente, motivos; visibles: review_id, text, columnas vacias, notas, version).
- La clave interna (`reports/p92_internal_sampling_key.csv`) permanece separada.

## 3. Que falta exactamente para poder distribuir

1. Completar el registro de decisiones (Q1 a Q8) y el gate (seccion 3 del registro).
2. Congelar la version aplicable de la guia con changelog y SHA256.
3. Actualizar los dos paquetes a la version congelada (misma version para ambos),
   con manifest nuevo y trazable; re-verificar cegamiento e integridad.
4. Registrar la decision de inicio y distribuir.

## 4. Si persiste una ambiguedad semantica material

Dos rutas posibles (a decidir y registrar por los responsables; no excluyentes):

- Ruta A - piloto exploratorio con UTD: distribuir con la version congelada que
  conserve UNABLE_TO_DETERMINE explicito para las categorias cuya definicion no
  alcance. No se fuerzan etiquetas binarias para cumplir el formato original.
- Ruta B - esperar aclaracion normativa (Q2): mantener la muestra sin distribuir
  hasta recibir respuesta del organizador o decidir formalmente continuar sin ella.

## 5. Auditoria de integridad (re-verificada en esta fase)

| archivo | sha256 (16) | esperado | ok |
|---|---|---|---|
| artifacts/human_annotation_v1/calibration_responses/calibration_response_A.csv | a20521989aa3fea9 | a20521989aa3fea9 | OK |
| artifacts/human_annotation_v1/calibration_responses/calibration_response_B.csv | 41e50c1f1d48760e | 41e50c1f1d48760e | OK |
| artifacts/human_annotation_v1/calibration_responses/calibration_response_C.csv | 8de76e21046d5091 | 8de76e21046d5091 | OK |
| artifacts/human_annotation_v1/calibration_responses/calibration_response_D.csv | 763229f9a9caa176 | 763229f9a9caa176 | OK |
| artifacts/human_annotation_v1/calibration_responses/calibration_response_A_normalized.csv | 7e7b3c805620e613 | 7e7b3c805620e613 | OK |
| artifacts/human_annotation_v1/calibration_responses/calibration_response_B_normalized.csv | 7cca95e20a1243c0 | 7cca95e20a1243c0 | OK |
| artifacts/human_annotation_v1/annotator_A.csv | 7dfd5b898d3905f2 | 7dfd5b898d3905f2 | OK |
| artifacts/human_annotation_v1/annotator_B.csv | 956646fa479f0c65 | 956646fa479f0c65 | OK |
| submissions/toxic_submission_v03_lr_union_thr.csv | 7d7478f271e6edfd | 7d7478f271e6edfd | OK |
| features/oof_v03_final.npz | 559265816ce4f631 | 559265816ce4f631 | OK |
| data/raw/toxic_train.csv | 3d40e7345b588860 | 3d40e7345b588860 | OK |
| data/processed/holdout_v2_ids.csv | e5b62e5f22c4c63a | e5b62e5f22c4c63a | OK |
| experiments/baseline_v03_frozen/model_v03.joblib | 263ed5bd0fc0de7c | 263ed5bd0fc0de7c | OK |

- Respuestas humanas originales: intactas (sin cambios).
- Normalizados A/B: presentes y trazables (hash registrado en P92-T y PROVENANCE).
- Muestra principal: no distribuida; anotaciones vacias.
- V03, OOF canonico, etiquetas originales, particion y submissions: intactos.
- Test y holdouts: no utilizados en esta fase (ni siquiera hasheados para test).
- Sin entrenamiento, sin predicciones, sin pseudo-labeling, sin clasificadores
  externos.

## 6. Recordatorio de estado congelado

```text
CALIBRATION = ANALYZED
GUIDELINE_V1 = HISTORICAL
GUIDELINE_V2 = PROPOSED_PENDING_DISCUSSION
MAIN_ANNOTATION = NOT_DISTRIBUTED
V03 = FROZEN
MODEL_RETRAINING = NOT_AUTHORIZED
SOURCE_LABEL_MODIFICATION = NOT_AUTHORIZED
```

## 7. Registro de la decision de inicio (referencia)

La decision de iniciar la anotacion principal se registra en
`reports/p92_guideline_decision_record.md` (seccion 3). Hasta entonces, el gate
permanece cerrado y ningun paquete se entrega.
