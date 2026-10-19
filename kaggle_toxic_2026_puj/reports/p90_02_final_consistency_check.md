# P90-02 - Comprobacion documental final (FINAL CONSISTENCY CHECK)

Fecha: 2026-10-10. Alcance: verificacion documental de los entregables de la auditoria P90.02.
**No repite la auditoria; no entrena modelos; no usa test, holdout ni holdout2; no genera
submissions; no modifica resultados historicos; no hace commits ni operaciones remotas; no borra
ni sobrescribe artefactos.** Fuentes de verdad: `git status --short`, el arbol de archivos, el
manifiesto y el inventario de P90-02, y las tablas ya validadas.

## 1. Comprobaciones efectuadas

### 1.1 Inventario de artefactos

Estado Git verificado hoy (`git status --short` en `kaggle_toxic_2026_puj/`): **1 modificado**
(`notebooks/10_final_pipeline.ipynb`) y **22 sin seguimiento** (19 en `reports/` + 3 en
`experiments/`). No hay archivos borrados ni renombrados.

Los **19 archivos del entregable P90.02** en `reports/` (todos sin seguimiento (`??`), todos
presentes en el manifiesto (`output_artifacts`) y todos enumerados en la respuesta final de
P90.02). Esta comprobacion agrega despues 2 documentos nuevos con prefijo `p90_02_`
(`p90_02_final_consistency_check.md` y `p90_02_group_audit_addendum.md`), que no forman parte
del entregable original ni de su manifiesto:

| # | archivo | estado Git | en manifiesto | en respuesta P90.02 |
|---|---|---|---|---|
| 1 | p90_02_artifact_inventory.csv | ?? | SI | SI |
| 2 | p90_02_audit_manifest.json | ?? | SI | SI |
| 3 | p90_02_audit_results.json | ?? | SI | SI |
| 4 | p90_02_casebook_validation.csv | ?? | SI | SI |
| 5 | p90_02_chatgpt_handoff.md | ?? | SI | SI |
| 6 | p90_02_corpus_profile.csv | ?? | SI | SI |
| 7 | p90_02_error_diagnosis.csv | ?? | SI | SI |
| 8 | p90_02_error_intersections.csv | ?? | SI | SI |
| 9 | p90_02_historical_eda_matrix.csv | ?? | SI | SI |
| 10 | p90_02_hypothesis_status.csv | ?? | SI | SI |
| 11 | p90_02_integrity_and_corpus_audit.md | ?? | SI | SI |
| 12 | p90_02_label_distribution.csv | ?? | SI | SI |
| 13 | p90_02_metric_reconciliation.csv | ?? | SI | SI |
| 14 | p90_02_representative_cases.csv | ?? | SI | SI |
| 15 | p90_02_review_sample_audit.csv | ?? | SI | SI |
| 16 | p90_02_slice_metrics.csv | ?? | SI | SI |
| 17 | p90_02_slice_overlap.csv | ?? | SI | SI |
| 18 | p90_02_slice_registry.csv | ?? | SI | SI |
| 19 | p90_02_text_provenance.csv | ?? | SI | SI |

Scripts nuevos (3): `experiments/p90_02_auditoria_integral.py`,
`experiments/p90_02b_informe_y_handoff.py`, `experiments/p90_03_integrar_notebook.py` (todos
`??`; presentes en el manifiesto).

**Resolucion de la diferencia 22 vs 19**: la respuesta final declaro "22 artefactos nuevos en
`reports/`" (resumen ejecutivo) y enumero 19 archivos en `reports/` (seccion de archivos
generados). La verificacion establece que **`reports/` contiene exactamente 19 archivos** y que
**22 es el total de archivos nuevos** de la tarea (19 en `reports/` + 3 scripts en
`experiments/`). La atribucion de los 22 a `reports/` fue una imprecision de redaccion de la
respuesta de chat; no falta ningun archivo y la enumeracion de 19 es correcta y completa. No
existe ningun archivo `p90_02_*` fuera de `reports/` y `experiments/` (verificado con busqueda
exhaustiva; solo aparecen ademas dos `.pyc` de `__pycache__` de `py_compile`, ignorados por git
y ajenos a los entregables).

**Universos de conteo (para evitar confusiones futuras)**:

| universo | cantidad | detalle |
|---|---|---|
| archivos generados en `reports/` | 19 | tablas + informes + JSON de P90-02 |
| scripts nuevos en `experiments/` | 3 | generadores de la auditoria |
| archivos nuevos en total | 22 | 19 + 3 (cifra citada como "22" en la respuesta) |
| salidas en el manifiesto | 23 | 19 + 3 + notebook modificado |
| artefactos FUENTE en el inventario | 22 | universo distinto: oof, train, particion, grupos, pipeline, hashes, submission, modelo, casebook (8), provenance, lexico, p57, oportunidades, experiments.csv, DECISIONS |

Correspondencia manifiesto <-> disco: 19 rutas `reports/` en el manifiesto = 19 en disco (0
faltantes, 0 sobrantes); las 23 rutas del manifiesto existen todas. Correspondencia
inventario <-> manifiesto: 22 fuentes en ambos; hashes identicos en las 22 (0 discrepancias;
OOF `559265816ce4f631...`).

### 1.2 Auditoria de grupos

La documentacion de grupos del informe P90-02 (seccion 15) era parcial: cubria grupos
near-duplicate en dev (1479; 6 filas no unitarias; grupos 112 y 338) pero no la distribucion de
tamanos completa, los grupos fusionados (union de validacion), los pseudo-autores ni las
divergencias de labels. Se completo con el addendum **`reports/p90_02_group_audit_addendum.md`**
(solo desarrollo autorizado; metadatos historicos reutilizados; sin grupos ni particiones
nuevas). Cifras clave verificadas:

- Near-duplicate solo (dev): 1479 grupos; tamanos reales {1: 1477, 3: 2}; 6 filas no unitarias;
  1 de 2 grupos no unitarios con labels divergentes (grupo 112).
- Pseudo-autor (historico): 17 autores >= 3 tweets (77 tweets, 4.4 %); PA-01: 740 autores
  distintos, 39 con >1 tweet; 1645 grupos fusionados (train); F1 0.7838 vs 0.7927 (D-015).
- Union de validacion (`grupo_near_dup_autor`, dev): **1415 grupos**; tamanos
  {1: 1378, 2: 20, 3: 10, 4: 3, 5: 2, 6: 1, 7: 1}; 105 filas no unitarias; 13 grupos con labels
  divergentes (43 filas). La cifra 1415 coincide con la unidad de remuestreo de P66 y con el
  subconjunto "textos unicos" de V4-F02.
- Cluster de consistencia (contexto V4-F02): `label_consistency_v4.npz`; 84 filas en train,
  **68 en dev**; F1 unicos 0.7568 vs cluster 0.9769; techo de propagacion 0.0.

### 1.3 Estado de lineas historicas (reconciliacion cierre/bloqueo)

Las tres lineas citadas aparecen a la vez como "cerradas" (por resultados) y "bloqueadas" (por
reglas) porque describen **facetas distintas**: el mecanismo ya fue ejecutado y cerrado con
evidencia, y la extension no ejecutada queda bloqueada o prohibida. Clasificacion final:

| linea | ejecucion historica | estado de familia | extension futura | clasificacion final |
|---|---|---|---|---|
| Propagacion por near-duplicate | V4-F01-F04 (techo nulo medido; 0 coincidencias literales test-train; vecindario delta +0.018 con IC que incluye 0; sonda v18 con 0 celdas). Artefactos: experiments/35_discovery_v4.py; reports/discovery_v4.json, distribution_shift_v4.csv, error_taxonomy_v4.md | retrieval CLOSED (closure A) | train->test transductiva: QUARANTINED; audit: BLOCKED_BY_RULES | **EXPERIMENT_ALREADY_EXECUTED_AND_CLOSED + FUTURE_EXTENSION_BLOCKED_BY_RULES** |
| Expansion de datos | P47: inventario 1731 oficiales / 1483 dev / 248 holdout / 0 filas etiquetadas sin usar. Artefactos: reports/p47_data_inventory.{json,csv,md}; corpus_manifest_v5.json | data_expansion CLOSED (closure E) | sin fuente legal de etiquetas: PROHIBITED; audit: BLOCKED_BY_RULES | **EXPERIMENT_ALREADY_EXECUTED_AND_CLOSED + FUTURE_EXTENSION_BLOCKED_BY_RULES** |
| Embeddings desde cero | E2-19 + V38-01 (supervisado 0.595466; factorizacion de rango 3); DL-01 from-scratch 0.794900 bajo D-004 (sin transferencia bajo D-016); v5/v5.1; P73-P75; P78-P80; P84-02; P84-04; SGNS corregido -0.151120; 19 informes embedding_* | embeddings CLOSED (closure E) | "mas corpus/dimensiones/hiperparametros": PERMITTED pero DUPLICATE; variante preentrenada: PROHIBITED (D-030; audit: BLOCKED_BY_RULES) | **EXPERIMENT_ALREADY_EXECUTED_AND_CLOSED + DUPLICATE_OF_EXISTING_HYPOTHESIS** (variante preentrenada: BLOCKED_BY_RULES) |

Nota: "Embeddings desde cero" no es FUTURE_EXTENSION_BLOCKED_BY_RULES porque su repeticion es
legal pero redundante (DUPLICATE); lo bloqueado por reglas es la variante preentrenada/LLM.

### 1.4 Consistencia entre documentos

- **Nombres de archivos**: las 24 referencias a rutas del informe integral existen; la
  referencia del handoff existe. Las tablas citadas por el informe (inventario, metricas,
  slices, historico) son las generadas.
- **Cantidades**: 48 lineas historicas = 48 filas de `p90_02_historical_eda_matrix.csv` y de
  `p90_02_hypothesis_status.csv`; 19 slices = 19 filas del registro = 11 + 3 + 5 reportadas;
  13 casos representativos; 441 tweets / 757 celdas; muestra 100 + 185. Todo consistente entre
  informe, handoff y tablas.
- **Estados experimentales**: 37 CLOSED / 6 BLOCKED_BY_RULES / 3 DUPLICATE / 1 INVALIDATED /
  1 COMPLEMENTARY identico en informe (secciones 20/22), handoff (seccion F) y CSV.
- **Cifras del baseline**: `corrected_cv` 0.7772114525165721 (historico) / ...722 (recomputado,
  diferencia 1.1e-16) presente en informe (3 menciones), handoff (2) y manifiesto (1); LB
  0.78306 en informe, handoff e inventario; umbrales 0.44/0.51/0.46 en manifiesto y textos;
  hashes V03 intactos (submission `7d7478f271e6edfd...`, modelo `263ed5bd0fc0de7c...`).
- **Decisiones finales**: V03 FROZEN y recomendacion A. CLOSE_FAILURE_ANALYSIS presentes y
  coherentes en informe (seccion 25) y handoff (seccion I).

## 2. Discrepancias encontradas

1. **Conteo "22 vs 19"** (imprecision de redaccion de la respuesta final de P90.02, no de los
   artefactos): declaraba "22 artefactos nuevos en `reports/`"; el numero correcto de archivos
   en `reports/` es 19 y 22 es el total de archivos nuevos (19 + 3 scripts). Resuelta: no falta
   ningun archivo; la enumeracion de 19 es completa. Cualquier otra descomposicion de conteo de
   la respuesta (p. ej. en el bloque de estado Git) queda igualmente corregida por el
   `git status --short` de P90.02 (22 sin seguimiento en total, 1 modificado; ver la seccion 5
   para el estado final de esta comprobacion).
2. **Trazabilidad de la cifra 68** (cluster near-duplicate): no estaba documentada su
   definicion en el informe P90-02. Resuelta: proviene de `features/label_consistency_v4.npz`
   (`en_cluster`, generado por `experiments/27_label_consistency_v4.py`, 4 criterios) y de
   `reports/discovery_v4.json` (robustez_duplicados); documentada en el addendum.
3. **Cobertura de grupos incompleta** en la seccion 15 del informe (faltaban distribuciones,
   grupos fusionados, pseudo-autores y divergencias de labels). Completada en el addendum.
4. Sin discrepancias de hashes (22/22 inventario-manifiesto), de cifras del baseline, de
   estados experimentales ni de decisiones finales.

## 3. Correcciones documentales realizadas

- **No se modifico ningun artefacto existente** (regla de conservacion): el informe, el
  handoff, las tablas, el manifiesto y el inventario permanecen intactos.
- Nuevos documentos de esta comprobacion: `reports/p90_02_final_consistency_check.md` (este
  informe) y `reports/p90_02_group_audit_addendum.md` (addendum de grupos).
- **El handoff NO requiere actualizacion**: no contiene conteos de artefactos generados ni las
  cifras afectadas por las imprecisiones; sus cifras y decisiones coinciden con los artefactos
  verificados.
- La correccion de la cifra "22" queda registrada aqui; su origen era la respuesta de chat, no
  un archivo del repositorio (los archivos no contenian esa cifra).

## 4. Cuestiones que siguen abiertas

Heredadas del informe P90-02 (documentadas, no bloqueantes): (1) mecanismo de los 466 errores
lejanos (>= 0.10) sin causa demostrada; (2) impacto cuantificable de la normalizacion ASCII, no
medido; (3) regla de anotacion formal detras de TR=1 => HS=1 (identidad empirica documentada);
(4) calibracion como mejora legal, no medida. Limitacion del addendum: el detalle por
pseudo-autor no esta serializado en los artefactos historicos.

## 5. Estado final de Git

```text
Rama: main | HEAD: e43c594 (P90-00) | sin commits en esta tarea | sin push
git status --short (al cierre de esta comprobacion):
 M notebooks/10_final_pipeline.ipynb
 ?? 21 archivos reports/p90_02_* (19 del entregable P90.02 + este informe + el addendum)
 ?? 3 scripts experiments/p90_0[23]*
Total: 1 modificado + 24 sin seguimiento. Sin borrados ni renombrados.
origin/main = e0fcff8; main local +8
```

No se ejecutaron commits, push, merge ni operaciones destructivas. DECISIONS.md, RUNLOG.md y
todos los artefactos historicos permanecen sin cambios.

## 6. Confirmacion de no entrenamiento, submissions ni acceso prohibido

- **Entrenamientos: 0.** No se ejecuto ningun modelo ni optimizacion.
- **Submissions: 0.** No se genero ni modifico ninguna submission.
- **Test/holdout/holdout2: no utilizados.** Se leyeron exclusivamente metadatos de particion y
  grupos (`holdout_v2_ids.csv` filtrado a `particion == dev` antes de usar cualquier campo;
  `grupos_near_duplicados.csv` es solo id/split/grupo/tamano). No se inspecciono contenido,
  etiquetas ni textos de particiones reservadas.
- **V03 sin modificar**: hashes de submission y modelo verificados hoy (intactos).
- **Sin operaciones remotas**; sin borrado de artefactos; sin sobrescritura.

## 7. Recomendacion

**AUDIT_CLOSED_WITH_DOCUMENTED_LIMITATIONS**

Justificacion: la evidencia del repositorio es internamente consistente (inventario, manifiesto,
informe, handoff, tablas y decisiones coinciden; 0 discrepancias de hashes y cifras); la unica
discrepancia de conteo estaba en la redaccion de la respuesta de chat y queda documentada y
corregida aqui; la cobertura de grupos se completo con el addendum; y las cuestiones abiertas
heredadas estan declaradas y acotadas (sin impacto sobre la integridad de la auditoria). No se
requiere una comprobacion de integridad adicional.
