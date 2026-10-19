# P92-X - Auditoria de procedencia de la guia (v2 -> v2.1)

Fecha: 2026-10-13. Sin consultar los textos ni respuestas de los 3 casos
excluidos: la auditoria usa informes de procedencia, decisiones documentadas,
definiciones oficiales y artefactos de la guia.

## 1. Tabla por clausula

| clause_or_example_id | content_type | source | official_or_operational | supporting_eligible_evidence | possible_holdout2_influence | distribution_status |
|---|---|---|---|---|---|---|
| C-01 | Definicion oficial de HS | OFFICIAL_DEFINITION | docs/competition_rules.md (verbatim) | official | n/a | ninguna | vigente en v2.1 |
| C-02 | Definicion oficial de TR | OFFICIAL_DEFINITION | docs/competition_rules.md (verbatim) | official | n/a | ninguna | vigente en v2.1 |
| C-03 | Definicion oficial de AG | OFFICIAL_DEFINITION | docs/competition_rules.md (verbatim) | official | n/a | ninguna | vigente en v2.1 |
| C-04 | Q1: definicion literal + UTD (sin nueva definicion de odio) | GENERAL_OPERATIONAL_RULE | decision del propietario 2026-10-13 (P92-V) | operational | 4/12 filas elegibles con UTD coincidente en HS; 10/12 con duda | indirecta en el conteo original; base elegible verificada; conteo citado corregido en v2.1 | vigente en v2.1 |
| C-05 | Q2: registro de gobernanza | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | no aplica (proceso) | ninguna | vigente en v2.1 |
| C-06 | Q3: no equiparar vulgaridad, insulto y agresividad | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | casos elegibles 06 y 13 (AG) | ninguna | vigente en v2.1 |
| C-07 | Q4: contenido observable; sin inferir intencion oculta | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | casos elegibles 07 y 13 (tono) | ninguna | vigente en v2.1 |
| C-08 | Q5: texto completo; la cita no implica adhesion | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | casos elegibles 01 y 07 (contexto/citas) | ninguna | vigente en v2.1 |
| C-09 | Q6: alcance de UTD (por etiqueta, con nota) | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | casos elegibles 01, 04 y 07; patron HS elegible | ninguna material | vigente en v2.1 |
| C-10 | Q7: dudas separadas (etiqueta vs tweet; columna tweet_note) | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | marcas de fila de A/B en filas elegibles (01, 02, 04, 05, 13) | ninguna | vigente en v2.1 |
| C-11 | Q8: formato exclusivo de tokens | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V); README y guia v1 | operational | desviacion de formato documentada (nivel documento) | ninguna | vigente en v2.1 |
| C-12 | Esquema de 8 columnas (incluye tweet_note) | GENERAL_OPERATIONAL_RULE | P92-U/Q7; adoptado en P92-V | operational | n/a (diseno de captura) | ninguna | vigente en v2.1 |
| C-13 | Instrucciones exactas de UNABLE_TO_DETERMINE | GENERAL_OPERATIONAL_RULE | P92-T/P92-U; decision Q6 | operational | casos elegibles 01, 04 y 07 | ninguna material | vigente en v2.1 |
| C-14 | CH-01 formato (0/1 invalido) | GENERAL_OPERATIONAL_RULE | documentacion de la calibracion (nivel formato) | operational | n/a | ninguna | vigente en v2.1 |
| C-15 | CH-02 nota obligatoria con UTD | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | notas de casos elegibles | ninguna | vigente en v2.1 |
| C-16 | CH-03 criterios operativos | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | ver C-04 y C-06 | indirecta (cubierta por C-04) | vigente en v2.1 |
| C-17 | CH-04 duda de fila (tweet_note) | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | marcas de fila elegibles | ninguna | vigente en v2.1 |
| C-18 | CH-05 tono, citas y contexto | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | casos elegibles 01, 07 y 13 | ninguna | vigente en v2.1 |
| C-19 | CH-06 sin textos en la guia | GENERAL_OPERATIONAL_RULE | decision de diseno (P92-T/P92-V) | operational | n/a | ninguna | vigente en v2.1 |
| C-20 | CH-07 preguntas formalizadas | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | n/a | ninguna | vigente en v2.1 |
| C-21 | CH-08 independencia HS/TR/AG | GENERAL_OPERATIONAL_RULE | decision del propietario (P92-V) | operational | n/a | ninguna | vigente en v2.1 |
| C-22 | CH-09 esquema de columnas | GENERAL_OPERATIONAL_RULE | P92-U/Q7 | operational | n/a | ninguna | vigente en v2.1 |
| C-23 | Evidencia '6 de 15 filas' (seccion 6 de v2) | POTENTIAL_HOLDOUT2_INFLUENCE | analisis P92-T (agregado original de la calibracion) | operational (evidencia) | sustituto elegible: 4/12 (y 10/12 con duda) | SI: 2 de las 6 filas citadas pertenecen a particiones reservadas | REEMPLAZADA en v2.1 |
| C-23R | Evidencia elegible '4 de 12 / 10 de 12' (v2.1) | EXAMPLE_DERIVED_FROM_ELIGIBLE_DATA | recalculo P92-W sobre artefactos existentes (solo casos elegibles) | operational (evidencia) | 4/12 con UTD coincidente en HS; 10/12 con duda | ninguna (solo casos elegibles) | vigente en v2.1 |
| C-24 | Ejemplos textuales en la guia | EXAMPLE_DERIVED_FROM_ELIGIBLE_DATA | n/a | n/a | n/a | n/a | ninguno presente (0 ejemplos en v2 y v2.1) |

Nota: la columna de contenido resume cada clausula; el texto completo esta
en `reports/p92_annotation_guidelines_v2_frozen.md` (v2, historico) y
`reports/p92_annotation_guidelines_v2_1_holdout_safe.md` (v2.1, vigente).

## 2. Resultado

- Tipos encontrados: OFFICIAL_DEFINITION (3), GENERAL_OPERATIONAL_RULE (19),
  POTENTIAL_HOLDOUT2_INFLUENCE (1: el conteo '6 de 15', reemplazado en v2.1),
  EXAMPLE_DERIVED_FROM_ELIGIBLE_DATA (1: la evidencia elegible 4/12; 0 ejemplos
  textuales en la guia), UNRESOLVED_PROVENANCE (0).
- Las reglas operativas se sostienen con fundamento documentado y evidencia
  elegible; se conservan sin cambios semanticos.
- Se creo la revision v2.1: reemplaza el conteo original por evidencia
  elegible, agrega la seccion de procedencia y el changelog R-01..R-03.
- La revision NO elimina retroactivamente la exposicion historica: el flag
  GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE de P92-W sigue vigente.

## 3. Actualizacion de los paquetes

- Guia v2.1: `reports/p92_annotation_guidelines_v2_1_holdout_safe.md`
  (sha256 sin linea de hash: ff699f54a48da7797f7305cf71a2718cfc12b3d99eab6f0eb80e0560cd3ed38a).
- En ambos paquetes se reemplazo `annotation_guidelines_v2_frozen.md` por
  `annotation_guidelines_v2_1_holdout_safe.md` (hash retirado de la copia v2: 77de86cab947c8cd...;
  el archivo v2 historico permanece en reports/ y en el paquete en cuarentena).
- READMEs de anotador y manifiesto actualizados; la muestra y las respuestas
  NO cambiaron (CSVs byte a byte).

## 4. Hashes del paquete (post-revision)

| archivo | sha256 |
|---|---|
| artifacts/human_annotation_v2_1_holdout_safe/README.md | `5a2951741d64852f713aa52678cfa3b6d690f051b0b9872b3011d9484a1331b8` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_A/README_annotator.md | `8b0043180ff7857eb65f9148dc067e9a71f88064b4b64103188ef0aa0a30a334` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_A/annotation_guidelines_v2_1_holdout_safe.md | `fd8d4c65ccd194fbce866321c7dbbbdc6870653f606e3111f147e41bc7fc9204` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_A/annotator_A.csv | `6525a308dc0926ea622822b125c32412d85caa8afb98a413d368984c9cb5464c` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_B/README_annotator.md | `daec8552197603f4d9c49b0226886aaaf7408413a72a97f9b68baf38aa1b285c` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_B/annotation_guidelines_v2_1_holdout_safe.md | `fd8d4c65ccd194fbce866321c7dbbbdc6870653f606e3111f147e41bc7fc9204` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_B/annotator_B.csv | `5c699b372d632ab3d71bc8586b3967b94f323b9abec433e392b6d9dc30731577` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration/README_calibration.md | `c60d9e9054da233a75fdff7c07d0511747271949316c31881d078c6fe5325d6d` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration/calibration_sample.csv | `eed1ffffdd4f6a9f5ad6a1888595cebae3aba8e4eceaf1953c1dc0e3b5fe6d1e` |
| artifacts/human_annotation_v2_1_holdout_safe/manifest.json | `00998e6006035c9fc69bb2db7c74a5e1f2837ab85687e766d202415b969a6e13` |
