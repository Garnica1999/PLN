# P92-W - Auditoria del paquete holdout-safe (human_annotation_v2_1)

Fecha: 2026-10-13. Estado: **NEW_HUMAN_ANNOTATION_PACKAGE = READY_WITH_VERIFIED_EXCLUSIONS**.

## 1. Poblacion y exclusiones

- Poblacion elegible dev2 = 1236 ids (dev 1483 menos holdout2 247).
- Reglas: elegible si pertenece a dev en la particion congelada y no a
  holdout2, holdout historico ni test; ninguna fila quedo sin particion
  demostrable (1731 = 1236 + 248 + 247).
- Test: ids de la plantilla submission.csv (solo columna id); el archivo
  toxic_test.csv no se accedio.

## 2. Muestreo

- A: SRS de 100 sobre 1236 (semilla 20261017); B: hasta
  150 por tiers con cap 2 por grupo (semilla 20261018); calibracion:
  SRS de 15 fuera de A y B (semilla 20261019).
- Probabilidad de inclusion A: 100/1236; margenes recalculados
  (+-4.49 pp (N=1236, n=100); +-10.30 pp (peor caso);
  DEFF 1.2, FPC; margenes recalculados, no reutilizados; ver manifiesto).

## 3. Verificaciones

- filas_250_250_15: OK
- mismos_casos_A_B: OK
- orden_distinto_A_B: OK
- sin_duplicados: OK
- anotaciones_vacias: OK
- version_v2: OK
- textos_identicos_fuente: OK
- todos_elegibles: OK
- sin_reservados: OK
- componentes_disjuntos: OK
- guia_identica: OK
- blinding_sin_tokens_fuga: OK
- clave_interna_fuera: OK
- v2_preservado: OK
- hashes_canonicos_ok: OK
- ascii_puro: OK
- textos contra la fuente: 515/515

## 4. Hashes de los archivos nuevos

| archivo | sha256 |
|---|---|
| artifacts/human_annotation_v2_1_holdout_safe/README.md | `9b6fb386d7649ed98ab3e9cc2be0a265681d784efbd5d862f8cbc486c2f25361` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_A/README_annotator.md | `685780dd1ec52bb9334c39ef65f00f8fd62f62e949633022a3468392b3db7025` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_A/annotation_guidelines_v2_frozen.md | `77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_A/annotator_A.csv | `6525a308dc0926ea622822b125c32412d85caa8afb98a413d368984c9cb5464c` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_B/README_annotator.md | `7b4488d5d49a4d30c6c7b3351e31c66d7af167c80d265ed87730ab52877c5172` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_B/annotation_guidelines_v2_frozen.md | `77de86cab947c8cd8452bd65ebc76bb371e4715d6da2ec184534dafa568f9db4` |
| artifacts/human_annotation_v2_1_holdout_safe/annotator_B/annotator_B.csv | `5c699b372d632ab3d71bc8586b3967b94f323b9abec433e392b6d9dc30731577` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration/README_calibration.md | `c60d9e9054da233a75fdff7c07d0511747271949316c31881d078c6fe5325d6d` |
| artifacts/human_annotation_v2_1_holdout_safe/calibration/calibration_sample.csv | `eed1ffffdd4f6a9f5ad6a1888595cebae3aba8e4eceaf1953c1dc0e3b5fe6d1e` |
| artifacts/human_annotation_v2_1_holdout_safe/manifest.json | `c3f10d604061b0e2d2aec56965f945937c9f7184ab929a3bbfe2fd770e61c60b` |

## 5. Relacion con paquetes anteriores

- human_annotation_v1: historico, nunca distribuido, contiene las 38 filas
  expuestas; no usar para el piloto.
- human_annotation_v2: QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT
  (aviso aditivo en artifacts/human_annotation_v2/QUARANTINE_NOTICE.md).
- human_annotation_v2_1_holdout_safe: este paquete (NO distribuido).

## 6. Resultado

- Sin errores: paquete listo con exclusiones verificadas (NO distribuido).
