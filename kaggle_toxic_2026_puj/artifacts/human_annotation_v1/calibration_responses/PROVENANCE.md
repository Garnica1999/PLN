# Procedencia de las respuestas de calibracion (P92-S2, version 2)

Fecha: 2026-10-12. Dato humano; se conserva SIN CAMBIOS (las notas pueden contener
caracteres no ASCII del texto original de los anotadores; no se convierten).

- calibration_response_A.csv: copia byte a byte del archivo completado EN EL SITIO por el
  anotador A en artifacts/human_annotation_v1/calibration_handoff/annotator_A/
  (modificado 2026-10-12 21:49). Formato recibido: respuestas 0/1; notas 0/1; BOM de Excel.
- calibration_response_B.csv: idem para el anotador B.
- calibration_response_C.csv: recibido por mensaje de Carlos (2026-10-12); anotaciones y
  notas transcritas VERBATIM; texto materializado desde la fuente autorizada por review_id
  (C declaro el texto sin cambios).
- calibration_response_D.csv: recibido por mensaje de Carlos (2026-10-12); anotaciones y
  notas transcritas VERBATIM; los textos pegados por Carlos coinciden EXACTAMENTE con la
  fuente elegible (verificado, 0 diferencias) y se almacenan tal cual.
- Los textos de las cuatro respuestas coinciden con la fuente elegible (60 filas).
- Las plantillas originales del paquete de calibracion permanecen intactas.

## Addendum P92-T (2026-10-12): codificacion de A y B confirmada

- Regla confirmada por el responsable del proyecto: en las columnas de clase (HS, TR, AG),
  0 = NEGATIVE y 1 = POSITIVE; en uncertainty_notes, 1 = "dificil de determinar para
  cualquier clase" (flag de fila; NO equivale a UNABLE_TO_DETERMINE por etiqueta).
- Fuente: aclaracion explicita de Carlos (2026-10-12), registrada en
  reports/p92_calibration_analysis.md (seccion 2) y reports/p92t_audit_results.json.
- Derivados que aplican la regla (no modifican los originales):
  calibration_response_A_normalized.csv (sha256[:16] 7e7b3c805620e613) y
  calibration_response_B_normalized.csv (7cca95e20a1243c0).
- Los cuatro archivos de respuesta de esta carpeta permanecen SIN CAMBIOS; hashes
  completos verificados en reports/p92t_audit_results.json.

## NOTA ADITIVA P92-T-R (2026-10-12): estado condicional de la regla

La regla del addendum anterior queda CONDICIONADA. La auditoria de procedencia
(reports/p92tr_encoding_provenance_audit.md) encontro que la aclaracion existe como
instruccion registrada en la sesion, pero no esta persistida en el proyecto, no enumera
literalmente la direccion del mapeo y no cuenta con confirmacion directa de A y B.
Estado: ENCODING_UNRESOLVED al estandar estricto. Los archivos normalizados
(calibration_response_A_normalized.csv, calibration_response_B_normalized.csv) son
derivados CONDICIONALES a esa codificacion y no deben usarse como acuerdo definitivo.
Solicitud de confirmacion directa preparada (reports/p92tr_ab_confirmation_request.md;
no enviada). Los cuatro archivos originales de esta carpeta siguen SIN CAMBIOS.
