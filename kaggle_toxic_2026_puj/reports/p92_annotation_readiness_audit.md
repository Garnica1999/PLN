# P92 - Auditoria de preparacion del paquete de anotacion (readiness)

Fecha: 2026-10-09 (fecha del proyecto). Resultado real de las verificaciones ejecutadas en
esta fase. La anotacion humana NO se ha ejecutado: comienza cuando los archivos se
entreguen a dos personas reales.

## 1. Decision

```text
ANNOTATION_PACKAGE_READY_WITH_LIMITATIONS
HUMAN_ANNOTATION_PENDING
```

Situacion B: el paquete puede utilizarse como piloto exploratorio bajo la guia provisional
v1; existen ambiguedades documentadas (definiciones oficiales sin criterios operativos;
relacion HS/TR sin regla formal) y por tanto NO puede afirmarse que represente fielmente
la semantica pretendida por los organizadores.

## 2. Verificaciones de integridad (P16)

| item | resultado | evidencia |
|---|---|---|
| V03 model hash unchanged | OK | `263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca` |
| canonical OOF hash unchanged | OK | `559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab` |
| historical submission hash unchanged | OK | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` |
| original label arrays unchanged | OK | y_true del OOF == etiquetas del train para los 1483 dev (gate del script) |
| original train texts unchanged | OK | hash `3d40e734...` verificado; los 250 textos del paquete identicos a la fuente |
| test not accessed | OK | archivo no abierto; sin tokens de acceso en el AST del script |
| holdout / holdout2 not accessed | OK | solo metadatos de particion; 0 ids del holdout en la clave interna |
| leaderboard not accessed | OK | sin uso |
| no models trained / no new predictions | OK | fase de preparacion; 0 entrenamientos |
| no labels changed / no pseudo-labeling | OK | ninguna etiqueta modificada; sin pseudo-labels |
| no external labeled data / classifier / embeddings | OK | no utilizados |
| no submissions created or modified | OK | sin submissions |

## 3. Verificaciones del paquete (14/14)

| check | estado |
|---|---|
| anotaciones_vacias | OK |
| calibracion_disjunta | OK |
| columnas_visibles_exactas | OK |
| mismo_conjunto_A_B | OK |
| orden_diferente | OK |
| plantilla_adjudicacion_750 | OK |
| plantilla_sin_decisiones | OK |
| review_ids_opacos | OK |
| sin_columnas_sensibles | OK |
| sin_holdout_en_clave | OK |
| sin_repetidos_en_paquete | OK |
| tamano_principal | OK |
| textos_identicos_a_la_fuente | OK |
| todos_en_dev | OK |

## 4. Recuentos y composicion

Principal: **250 tweets** (A 100 + B 150) = **750 anotaciones de etiqueta** por anotador
(3 respuestas posibles por anotacion: POSITIVE/NEGATIVE/UNABLE_TO_DETERMINE).
Calibracion: 15 tweets (separada). Plantilla de adjudicacion: 750 filas sin decisiones.
Clave interna: 265 filas (250 principales + 15 calibracion). Componente B seleccionado de
329 disponibles (PILI 26 / DRC 22 / HCD 118 / LSM 11), cap 2 por grupo. El codigo DRC es un
identificador heredado del criterio de seleccion (EMPIRICAL_STRUCTURE_ANOMALY, ver P91-R):
NO es una violacion normativa confirmada; los 22 casos historicos estan los 22 en B.

## 5. Auditoria ASCII y de formatos

- `tools/verificar_caracteres.py` (verificador oficial): **0 caracteres no permitidos**.
- Parseabilidad: adjudication_template.csv   750     annotator_A.csv   250     annotator_B.csv   250     artifacts/human_annotation_v1/manifest.json   JSON OK     calibration_sample.csv   15     reports/p92_audit_results.json   JSON OK     reports/p92_sampling_manifest.json   JSON OK
- Hashes de los artefactos:

| artefacto | sha256[:32] |
|---|---|
| artifacts/human_annotation_v1/README_annotators.md | ccd4c51a9163d9578f3586a184a10d7f... |
| artifacts/human_annotation_v1/adjudication_template.csv | 4237fa3f4e607ce3730aca65f12f3d9e... |
| artifacts/human_annotation_v1/annotator_A.csv | 7dfd5b898d3905f21e431d4a076655d0... |
| artifacts/human_annotation_v1/annotator_B.csv | 956646fa479f0c6515af7c76ec323d5d... |
| artifacts/human_annotation_v1/calibration_sample.csv | 42b13e52ec11d20838d90525ed34e65e... |
| artifacts/human_annotation_v1/manifest.json | d7a0d596803e7858578fa4227e4dcdfb... |
| reports/p92_annotation_guidelines_v1.md | 5f626bda5116bbb52df58e6fce0002b6... |
| reports/p92_human_annotation_plan.md | e1aa4844a2f659e98f67c6c804ae31a3... |
| reports/p92_internal_sampling_key.csv | ae358ad827285e8129754be61e2bd1d8... |
| reports/p92_sampling_manifest.json | 1800079faee94b3dbc115b9c97981314... |

## 6. Ambiguedades declaradas (los anotadores pueden registrarlas, no resolverlas)

1. HS: "odio" vs "ofensa" sin criterio operativo oficial.
2. AG: "agresivo" vs "vulgar" sin criterio operativo oficial.
3. Relacion HS/TR (y HS/AG) sin regla formal documentada; 0 excepciones TR en dev y 22 excepciones AG.
4. Casos limite sin instruccion: citas, sarcasmo, discurso reproducido.
5. La guia es v1 provisional; la calibracion puede generar v2 (con changelog) antes de congelar.

## 7. Requisitos para completar la anotacion

Entregar `calibration_sample.csv` y el archivo personal a DOS personas anotadoras; completar
la calibracion y discutirla; congelar la version de la guia; completar la ronda principal
sin discutir casos; devolver los CSV; entonces (y solo entonces) calcular acuerdo y
adjudicar con un tercero cualificado. La clave interna permanece fuera del paquete.

## 8. Cierre

El paquete esta listo con limitaciones metodologicas documentadas; ninguna etiqueta fue
modificada; V03 permanece congelado; no se autoriza ninguna fase de modelado.
