# P92-S - Auditoria de la entrega de calibracion humana

Fecha: 2026-10-09 (fecha del proyecto). Preparacion y verificacion de los dos paquetes de
calibracion (ronda de familiarizacion). **No se simularon anotaciones**: las columnas de
respuesta estan vacias y el acuerdo no se calculara hasta recibir las dos devoluciones
reales. Sin entrenamiento, sin etiquetas, sin test/holdout, sin submissions.

## 1. Hashes de los artefactos de origen

| artefacto | sha256[:32] |
|---|---|
| artifacts/human_annotation_v1/README_annotators.md | 24f3f7b733e0593b8733c76cb344cdba... |
| artifacts/human_annotation_v1/calibration_sample.csv | 42b13e52ec11d20838d90525ed34e65e... |
| artifacts/human_annotation_v1/manifest.json | 57e44b0535dbd3893adeea2f966270dd... |
| reports/p92_annotation_guidelines_v1.md | 731a244fc1361d856e0939aa823d9ad2... |
| reports/p92_internal_sampling_key.csv | ae358ad827285e8129754be61e2bd1d8... |
| reports/p92_sampling_manifest.json | e84bb372aed67932f55fcecc8ea6249d... |

## 2. Tamano de la muestra

15 tweets (los 15 originales de calibracion, semilla 20261012); **45 anotaciones de
etiqueta esperadas por anotador** (15 x 3), con 3 respuestas posibles por anotacion
(POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE). La muestra NO se volvio a seleccionar.

## 3. Elegibilidad

- Los 15 `review_id` de la calibracion coinciden EXACTAMENTE con los 15 de la clave
  interna con componente CALIBRACION.
- La calibracion es DISJUNTA de los componentes A y B de la muestra principal (verificado
  por id original).
- Todos los casos pertenecen a la particion dev (elegible); sin test ni holdouts.

## 4. Integridad de textos

- Los textos de los paquetes coinciden EXACTAMENTE con la fuente elegible
  (`data/raw/toxic_train.csv`, filas dev) para los 15 casos (verificacion por id via la
  clave interna).
- Las copias de `calibration_sample.csv` en A y B son byte-identicas al artefacto original
  (mismo sha256).

## 5. Esquema

- Columnas exactas: `review_id`, `text`, `HS_annotation`, `TR_annotation`, `AG_annotation`,
  `uncertainty_notes`, `guideline_version`; 15 filas; `guideline_version` = v1.
- CSV parseables; sin columnas extra; sin respuestas.

## 6. Cegamiento

- `review_id` opacos (P92C-##); sin etiquetas originales, sin probabilidades, sin
  predicciones, sin umbrales, sin folds, sin motivos de seleccion y sin grupos.
- Escaneo de tokens sensibles en los 6 archivos de los paquetes: 0 coincidencias.
- La clave interna NO esta en ningun paquete; cada paquete contiene solo los 3 archivos
  esperados.

## 7. Aislamiento de la clave interna

`reports/p92_internal_sampling_key.csv` permanece en `reports/`, fuera de
`artifacts/human_annotation_v1/calibration_handoff/`. Los paquetes no permiten inferir la
etiqueta original.

## 8. Hashes de los archivos generados

| archivo | sha256[:32] |
|---|---|
| artifacts/human_annotation_v1/calibration_handoff/annotator_A/README_annotator.md | 4c97a25ccbeef6085467f666ce87b74e... |
| artifacts/human_annotation_v1/calibration_handoff/annotator_A/annotation_guidelines_v1.md | 731a244fc1361d856e0939aa823d9ad2... |
| artifacts/human_annotation_v1/calibration_handoff/annotator_A/calibration_sample.csv | 42b13e52ec11d20838d90525ed34e65e... |
| artifacts/human_annotation_v1/calibration_handoff/annotator_B/README_annotator.md | 10983a64d62f7c0daec6aa1357daa417... |
| artifacts/human_annotation_v1/calibration_handoff/annotator_B/annotation_guidelines_v1.md | 731a244fc1361d856e0939aa823d9ad2... |
| artifacts/human_annotation_v1/calibration_handoff/annotator_B/calibration_sample.csv | 42b13e52ec11d20838d90525ed34e65e... |
| reports/p92s_calibration_handoff_manifest.json | (este manifiesto) |

## 9. Verificaciones

| check | estado |
|---|---|
| cada_paquete_con_exactamente_3_archivos | OK |
| clave_interna_ausente_de_los_paquetes | OK |
| columnas_de_anotacion_vacias | OK |
| csv_parseables | OK |
| guia_version_v1_declarada | OK |
| mismos_15_ejemplos_en_A_y_B | OK |
| readmes_distintos_por_anotador | OK |
| review_ids_opacos | OK |
| sin_respuestas_simuladas | OK |
| sin_tokens_sensibles_en_los_paquetes | OK |
| textos_identicos_a_la_fuente | OK |

## 10. Estado de Git y verificador

- Git: sin commits ni push; HEAD `e43c594` sin cambios; cambios locales conservados
  (el arbol NO esta limpio).
- Verificador oficial `tools/verificar_caracteres.py`: OK: 0 caracteres no permitidos en los ficheros propios del proyecto.

## 11. Limitaciones conocidas

- Guia provisional v1: las definiciones oficiales no cubren todos los limites semanticos;
  la calibracion puede motivar una v2 con changelog (se congelara antes de la tarea
  principal).
- Sin respuestas humanas aun: ningun calculo de acuerdo; ninguna inferencia de etiquetas.
- Los paquetes de calibracion no deben mezclarse con la muestra principal.

## 12. Decision

```text
CALIBRATION_PACKAGE_READY
HUMAN_CALIBRATION_PENDING
MAIN_ANNOTATION_NOT_DISTRIBUTED
V03 = FROZEN
NO NEW MODELING PHASE
```
