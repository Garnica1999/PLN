# P92-V - Auditoria del paquete principal v2

Fecha: 2026-10-13. Estado: **MAIN_ANNOTATION_PACKAGES_READY**.
Fase diagnostica: sin entrenamiento, sin predicciones, sin modificacion de
etiquetas, sin acceso a test ni holdouts, sin leaderboard, sin submissions.

## 1. Membresia y procedencia

- Mismos 250 review_id que human_annotation_v1 (por anotador); orden por
  anotador preservado de v1; ordenes de A y B distintos entre si.
- Componentes: A=100, B=150.
- Todos los original_example_id en toxic_train.csv: True; en pool dev: 250/250.
- Excluidos de la particion holdout de holdout_v2: 0 coincidencias;
  lista de test (plantilla submission.csv, solo columna id): 0 coincidencias.
- holdout2 (pliegue de dev creado en V10 para deteccion de sobreajuste;
  explicitamente no virginal; disjunto del holdout de holdout_v2): 35/250
  filas principales y 3 filas de calibracion coinciden. La definicion de
  elegibilidad del plan P92 (aprobada con 14/14) excluye la particion holdout de
  holdout_v2 y el test; no excluye este pliegue de dev. Limitacion heredada y
  documentada; esta fase reutiliza los mismos 250 por mandato. La anotacion de
  calidad no consume holdout2 como set de evaluacion (sin reentrenamiento;
  cualquier correccion futura requeriria autorizacion separada y consideraria
  la integridad de evaluacion de estas filas).
- toxic_test.csv NO accedido (ni texto ni etiquetas ni ids del archivo).

## 2. Verificacion de textos

- Comparaciones texto contra la fuente elegible (toxic_train.csv): 500/500 OK.
- Los CSV v2 replican byte a byte los textos de v1 (mismos review_id).

## 3. Estado de las anotaciones y esquema

- Columnas: review_id, text, HS_annotation, TR_annotation, AG_annotation,
  uncertainty_notes, tweet_note, guideline_version (8 columnas; CH-09).
- Anotaciones y notas VACIAS en ambos archivos; guideline_version = v2.

## 4. Cegamiento

- Escaneo de fuga (tokens identificadores de material interno: ids originales,
  componentes, motivos de seleccion, folds, OOF, V03, ids P92R/P92C dentro de
  las instrucciones): hallazgos: ninguno.
- Nota de metodo: una primera pasada con tokens genericos detecto dos falsos
  positivos (un nombre de usuario que contiene 'oof' dentro de un texto,
  verificado contra la fuente; y la palabra 'umbrales' en la lista de
  prohibiciones de la guia). El escaneo se refino a tokens identificadores;
  los textos se verifican ademas byte a byte contra la fuente elegible.
- Sin etiquetas originales, ids originales, OOF, predicciones, umbrales, folds,
  motivos de seleccion ni grupos; sin clave interna.

## 5. Chequeos

- v2_A_250_filas: OK
- v2_A_columnas: OK
- v2_A_anotaciones_vacias: OK
- v2_A_version: OK
- v2_A_mismos_textos_v1: OK
- v2_B_250_filas: OK
- v2_B_columnas: OK
- v2_B_anotaciones_vacias: OK
- v2_B_version: OK
- v2_B_mismos_textos_v1: OK
- v2_mismos_casos_A_B: OK
- v2_orden_distinto_A_B: OK
- v2_membresia_igual_v1: OK
- blinding_sin_tokens_fuga: OK
- ascii_puro: OK
- guia_copias_identicas: OK
- manifest_espejo_identico: OK

## 6. Hashes

| archivo | sha256 |
|---|---|
| artifacts/human_annotation_v2/annotator_A/README_annotator.md | `9c38f06586e80712638e75c55e443c373d8fe2f2955a44dcf123cf7d8e4f991b` |
| artifacts/human_annotation_v2/annotator_A/annotation_guidelines_v2_frozen.md | `77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4` |
| artifacts/human_annotation_v2/annotator_A/annotator_A.csv | `1238591610216bf3b8378bfdf5e9bc2fa034644da4f19e2dc01b5e104bc456ac` |
| artifacts/human_annotation_v2/annotator_B/README_annotator.md | `470295b7bf4bd0e341384824109f593d8500cb3adabfaea128fd1e9322c0c994` |
| artifacts/human_annotation_v2/annotator_B/annotation_guidelines_v2_frozen.md | `77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4` |
| artifacts/human_annotation_v2/annotator_B/annotator_B.csv | `8085697fe408d7a83636de495673627af800dbc9e0542fe7323f05c7b7501e22` |
| reports/p92_annotation_guidelines_v2_frozen.md | `77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4` |
| v1/annotator_A.csv | `7dfd5b898d3905f21e431d4a076655d034b7f791730e6ccac3b9d9612ca9546b` |
| v1/annotator_B.csv | `956646fa479f0c6515af7c76ec323d5d9d9ca5a9c47004249490c1a507ba8351` |
| reports/p92_annotation_guidelines_v2_frozen.md (hash interno) | `46be258ad47074405abce99e79ef831970d7b877e1987705de375432413117f8` |
| reports/p92v_manifest.json | `2f9dddd7804c1fbeef392d25de6a343c93d6d12e216f19d0232b8d452aecb514` |
| artifacts/human_annotation_v2/manifest.json (espejo identico) | `2f9dddd7804c1fbeef392d25de6a343c93d6d12e216f19d0232b8d452aecb514` |

## 7. Resultado

- Sin errores: MAIN_ANNOTATION_PACKAGES_READY (NO distribuidos).

Avisos:
- holdout2: 35/250 filas principales pertenecen al pliegue holdout2 (subconjunto de dev; creado en V10 como pliegue de deteccion de sobreajuste, explicitamente no virginal; la definicion de exclusion del plan P92 no lo incluye; ver auditoria, seccion de limitaciones)
