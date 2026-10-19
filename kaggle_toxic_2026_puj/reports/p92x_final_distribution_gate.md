# P92-X - Gate final de distribucion (paquete holdout-safe, revision v2.1)

Fecha: 2026-10-13. Clasificacion: **READY_WITH_DOCUMENTED_GUIDELINE_LIMITATION**.

## 1. Gates

| gate | descripcion | estado | evidencia |
|---|---|---|---|
| G1 | exclusiones de membresia y particion verificadas | OK | 265/265 ids en dev2; 0 solapes con test/holdout/holdout2; cola: 69/69 exclusiones con causa reservada |
| G2 | A/B misma membresia y orden distinto | OK | CSVs del paquete v2.1 |
| G3 | columnas de anotacion vacias | OK | 250+250+15 filas sin respuestas |
| G4 | clave interna aislada | OK | reports/p92w_internal_sampling_key.csv |
| G5 | version/hash de guia consistentes | OK | v2.1 en ambos paquetes; sha256 ff699f54a48da779... |
| G6 | procedencia de la guia documentada | OK (con limitacion) | p92x_guideline_provenance_audit.md; 1 clausula reemplazada; flag P92-W vigente |
| G7 | composicion del componente B reconciliada | OK | tiers exclusivos PILI 19 + DRC 19 + HCD 112 + LSM 0 = 150; simulacion reproduce la seleccion |
| G8 | integridad de textos | OK | 515/515 contra la fuente |
| G9 | auditoria ASCII oficial | OK | OK: 0 caracteres no permitidos en los ficheros propios del proyecto. |

## 2. Decision

- **READY_WITH_DOCUMENTED_GUIDELINE_LIMITATION**: todos los gates de datos pasan (G1-G5, G7-G9) y la
  procedencia de la guia queda documentada (G6) con una limitacion no
  eliminable: la influencia indirecta de casos excluidos en la evidencia
  historica de la calibracion.
- NO se afirma que el uso de holdout2 haya sido completamente deshecho: la
  exposicion historica permanece documentada (P92-W) y el flag
  GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE sigue vigente.
- La revision v2.1 reduce nuevas dependencias documentales: cita solo
  evidencia elegible (4/12; 10/12) y elimina el conteo original (6/15).
- La distribucion la decide el propietario; los paquetes NO se entregan
  automaticamente.

## 3. Hashes

| artefacto | sha256 |
|---|---|
| reports/p92_annotation_guidelines_v2_1_holdout_safe.md (hash interno) | `ff699f54a48da7797f7305cf71a2718cfc12b3d99eab6f0eb80e0560cd3ed38a` |
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
| reports/p92w_internal_sampling_key.csv | `13191b56b7b6b61d347a6fbb023dc13e32c0472f9cb22e2504ad2ff062b40129` |

