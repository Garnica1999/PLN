# Paquete human_annotation_v2_1_holdout_safe (interno)

Fecha: 2026-10-13. Estado: preparado, NO distribuido.
Relacion con versiones previas:
- `human_annotation_v2` queda QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT
  (35 filas principales y 3 de calibracion pertenecian a holdout2; ver
  reports/p92w_holdout2_exposure_audit.md).
- `human_annotation_v1` es historico y tampoco debe distribuirse (contiene las mismas
  filas expuestas, sin anotaciones).
- Este paquete usa la revision v2.1 del manual (PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2; revision documental de procedencia, reglas sin cambios)
  y una poblacion reconstruida (dev2 = dev sin holdout2) con semillas nuevas.

Estructura: annotator_A/, annotator_B/ (CSV + guia + README), calibration/ (muestra
nueva + README), manifest.json. La clave interna esta en
reports/p92w_internal_sampling_key.csv (fuera de este directorio).

No distribuir automaticamente: la entrega la decide el propietario del proyecto.
