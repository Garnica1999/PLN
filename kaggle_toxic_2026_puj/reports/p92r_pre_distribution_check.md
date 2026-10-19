# P92-R - Verificacion final antes de distribuir el paquete humano

Fecha: 2026-10-09 (fecha del proyecto). Comprobaciones exclusivamente documentales; no se
entreno nada, no se cambiaron las muestras ni los paquetes CSV, y no se simularon
anotaciones humanas.

## 1. Terminologia de las anotaciones (corregida)

El termino "750 decisiones binarias por anotador" era impreciso: cada anotacion de etiqueta
admite **3 respuestas posibles** (`POSITIVE`, `NEGATIVE`, `UNABLE_TO_DETERMINE`), por lo que
las 750 tareas no son necesariamente decisiones binarias. Terminologia aplicada en todos los
documentos: **"750 anotaciones de etiqueta por anotador; 3 respuestas posibles por
anotacion"**. `UNABLE_TO_DETERMINE` se conserva (no se elimino ninguna respuesta permitida).

Archivos corregidos: `reports/p92_human_annotation_plan.md`,
`reports/p92_annotation_readiness_audit.md`, `reports/p92_annotation_guidelines_v1.md`
(frase "etiqueta binaria" -> "POSITIVE o NEGATIVE"), README del paquete y la seccion 139 del
notebook (regenerada via su integrador). **Las muestras y los paquetes CSV no cambiaron.**

## 2. Aclaracion del codigo DRC

El codigo `DOCUMENTED_RULE_CONFLICT` / `DRC` aparece en: la clave interna (22 filas: 12
puras + 8 con HCD + 2 con LSM), el sampling manifest (tiers), el plan y la auditoria de
readiness. **NO aparece en ningun archivo visible del paquete** (verificado). La
correspondencia historica se comprobo: **los 22 casos DRC de P91 estan los 22 en el
componente B** (22/22). Documentacion explicita anadida en el plan, la auditoria de
readiness, el sampling manifest (`reason_code_notes`, aditivo) y el notebook: el codigo es
un **identificador heredado del criterio de seleccion** (tension estructural empirica
AG=1/HS=0; P91-R la clasifico `EMPIRICAL_STRUCTURE_ANOMALY`) y **NO constituye una
violacion normativa confirmada**. Se conserva el codigo de procedencia; no se cambiaron
etiquetas, no se seleccionaron candidatos nuevos y los motivos siguen ocultos a los
anotadores.

## 3. Protocolo de calibracion (orden confirmado y explicitado)

El README del paquete incluye ahora un bloque "Orden de la calibracion" que exige:
1. las DOS personas anotan `calibration_sample.csv` de forma independiente y completa;
2. se conservan las respuestas originales tal como se entregaron;
3. solo tras cerrar esa ronda se comentan las interpretaciones;
4. si cambian las instrucciones, se publica una nueva version de la guia (con changelog);
5. la version aplicable a la muestra principal se congela ANTES de iniciar esa anotacion.
La calibracion no se incorpora a las estimaciones poblacionales ni se mezcla con la muestra
principal (plan P92, seccion 3). Ninguna actualizacion de guia puede reinterpretar en
silencio anotaciones ya realizadas.

## 4. Integridad final

| comprobacion | resultado |
|---|---|
| A/B contienen exactamente los mismos casos principales | OK (250/250; mismos ids y mismos textos) |
| orden de filas diferente entre paquetes | OK |
| tres columnas de anotacion siguen vacias | OK (A, B, calibracion y plantilla) |
| clave interna fuera del paquete compartible | OK (ninguna clave en `artifacts/human_annotation_v1/`) |
| motivos de seleccion no visibles | OK (ningun archivo del paquete contiene los codigos) |
| textos, etiquetas y muestras sin cambios | OK (CSV del paquete con los hashes de P92: A 7dfd5b898d3905f2..., B 956646fa479f0c65..., calibracion 42b13e52ec11d208..., adjudicacion 4237fa3f4e607ce3...) |
| manifiesto del paquete coherente con disco | OK (hashes recomputados tras la correccion del README) |
| sampling manifest con nota DRC | OK (`reason_code_notes` presente) |
| V03 / OOF / submission / train intactos | OK (263ed5bd... / 55926581... / 7d7478f2... / 3d40e734...) |
| holdouts / test no accedidos; leaderboard no usado | OK |
| verificador oficial `tools/verificar_caracteres.py` | **0 caracteres no permitidos** (resultado real) |

## 5. Archivos modificados (hashes antes -> despues)

| archivo | antes | despues |
|---|---|---|
| reports/p92_human_annotation_plan.md | e1aa4844a2f659e9... | b8c20c1fe74d85c5... |
| reports/p92_annotation_readiness_audit.md | de1d6a4ba5962ba5... | 67bc4da31732c1fc... |
| reports/p92_annotation_guidelines_v1.md | 5f626bda5116bbb5... | 731a244fc1361d85... |
| artifacts/human_annotation_v1/README_annotators.md | ccd4c51a9163d957... | 24f3f7b733e0593b... |
| reports/p92_sampling_manifest.json | 1800079faee94b3d... | e84bb372aed67932... |
| artifacts/human_annotation_v1/manifest.json | d7a0d596803e7858... | 57e44b0535dbd389... |
| experiments/p92_03_integrar_notebook.py | (texto de la seccion) | editado (terminologia + nota DRC) |
| notebooks/10_final_pipeline.ipynb | seccion 139 previa | seccion 139 regenerada (455 celdas) |
| reports/p92r_pre_distribution_check.md | - | nuevo (este informe) |

Los paquetes CSV (`annotator_A.csv`, `annotator_B.csv`, `calibration_sample.csv`,
`adjudication_template.csv`) **no se modificaron**.

## 6. Estado de Git

Sin commits ni push; cambios locales conservados (el arbol NO esta limpio). Al cierre de
esta revision: rama `main`, HEAD `e43c594` sin cambios; 74 archivos sin seguimiento (73
previos + este informe) + 1 modificado (notebook). Ruta nueva de esta revision:
`reports/p92r_pre_distribution_check.md`.

## 7. Decision

```text
ANNOTATION_PACKAGE_READY_WITH_LIMITATIONS
HUMAN_ANNOTATION_PENDING
V03 = FROZEN
NO NEW MODELING PHASE
```

El paquete puede distribuirse a dos personas anotadoras bajo la guia provisional v1. No se
simularon anotaciones; las columnas de respuesta permanecen vacias a proposito.
