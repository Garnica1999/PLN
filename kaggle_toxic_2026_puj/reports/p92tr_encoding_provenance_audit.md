# P92-T-R - Auditoria de procedencia de la codificacion humana de A/B

Addendum independiente. Fecha: 2026-10-12. Fase diagnostica: sin
entrenamiento, sin modificacion de etiquetas, sin distribucion, sin
test/holdout, sin submissions. No sobrescribe respuestas humanas ni
resultados historicos: corrige la interpretacion de forma aditiva.

## 1. Objeto

Auditar la afirmacion de P92-T: "ENCODING_CONFIRMED; 0 = NEGATIVE; 1 =
POSITIVE; uncertainty_notes = 1 significa dificultad para determinar
cualquier clase", atribuida a "una aclaracion explicita del usuario del
2026-10-12". Estandar aplicado: si las dos convenciones no pueden demostrarse
con evidencia localizable, no se presentan como aclaraciones confirmadas.
La consistencia con las respuestas de C/D no se usa como evidencia.

## 2. Evidencia localizada

### 2.1 Busqueda en disco (2026-10-12)

- Archivos con la evidencia CRUDA del mensaje del usuario: NINGUNO. Las
  busquedas de "clase correspondiente" y "En las 2 anotaciones" no
  devuelven coincidencias en el proyecto.
- Archivos con la afirmacion (registros DERIVADOS, sin evidencia cruda):
  reports/p92_calibration_analysis.md (seccion 2), reports/p92t_audit_results.json
  (codificacion_AB), artifacts/human_annotation_v1/calibration_responses/
  PROVENANCE.md (addendum), reports/p92s2_calibration_results.json
  (campo "confirmada_por_carlos": "2026-10-12"), reports/
  p92u_human_discussion_pack.md, reports/p92_guideline_decision_record.md,
  notebooks/10_final_pipeline.ipynb (secciones 140 y 141) y
  experiments/p92_05_analisis_calibracion.py.
- Estado documentado ANTES de la supuesta aclaracion (informe de recepcion v2,
  reports/p92s2_calibration_reception_report.md, en disco):
  - "A/B usaron 0/1 en lugar de los tokens (analisis bajo el supuesto
    0=NEGATIVE, 1=POSITIVE, PENDIENTE DE CONFIRMACION)".
  - Decisiones pendientes: "1. Confirmar la codificacion de A/B: 0=NEGATIVE,
    1=POSITIVE (y ausencia de UTD). 2. Aclarar el significado del 0/1 en
    uncertainty_notes de A/B."

### 2.2 Instruccion del usuario (registrada en la sesion de trabajo del 2026-10-12)

Cita verbatim tal como consta en el registro de la sesion (no persistida en
disco hasta este addendum):

```text
En las 2 anotaciones, la primera y la segunda, 0 y 1 significan que son de la clase correspondiente, las que son 1 fuera de las 3 clases es que son dificiles de determinar para cualquier clase.
```

Cadena de procedencia: (1) se detecto que A y B devolvieron 0/1 (desviacion
de formato); (2) el agente solicito la aclaracion en la sesion; (3) el
responsable del proyecto respondio con la instruccion citada; (4) P92-T la uso
como base de "ENCODING_CONFIRMED"; (5) este audit verifica su alcance y
persistencia.

Alcance de la cita:

- Notas (uncertainty_notes = 1): "las que son 1 fuera de las 3 clases...
  dificiles de determinar para cualquier clase" -> contenido LITERAL.
- Clases (0/1): "0 y 1 significan que son de la clase correspondiente" ->
  establece que 0/1 codifican pertenencia a la clase; la direccion (1 =
  POSITIVE, 0 = NEGATIVE) es la lectura natural (convencion binaria y
  coherencia con la clausula de notas), pero NO queda enumerada literalmente.
- Nivel: aclaracion del responsable/coordinador del proyecto; NO es una
  confirmacion directa de los anotadores A ni B.
- Persistencia: inexistente en el proyecto hasta este addendum, que la deja
  registrada.

### 2.3 Formato solicitado a A y B (verificacion documental)

- README del handoff (annotator_A/README_annotator.md): "Usa exclusivamente
  POSITIVE, NEGATIVE o UNABLE_TO_DETERMINE en las tres columnas
  correspondientes"; "Deja uncertainty_notes vacio si no hay ninguna
  aclaracion necesaria".
- Guia v1 (seccion 8): "Columnas HS_annotation, TR_annotation, AG_annotation:
  una de POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE por celda".
- Resultado: A y B usaron un formato DISTINTO del solicitado (0/1 en las
  cuatro columnas, con BOM de Excel). Desviacion confirmada documentalmente;
  las respuestas originales se conservan intactas.

## 3. Estado de la codificacion (estandar estricto)

| convencion | contenido | evidencia localizada | confirmacion directa A/B | estado |
|---|---|---|---|---|
| uncertainty_notes=1 | "dificil de determinar para cualquier clase" | instruccion registrada (cita 2.2); contenido literal | no | ATRIBUIDA (contenido literal); pendiente de confirmacion directa |
| 0/1 -> NEGATIVE/POSITIVE | 0 = no pertenece; 1 = pertenece (POSITIVE) | instruccion registrada; direccion = lectura natural, NO enumerada literalmente | no | NO DEMOSTRADA al estandar estricto |

Resultado global: **ENCODING_UNRESOLVED** (estandar estricto de este audit).

- La atribucion de P92-T no fue una invencion: la instruccion existe y queda
  citada y persistida en este addendum.
- Sin embargo: (1) no estaba persistida en ningun artefacto del proyecto;
  (2) no enumera literalmente la direccion del mapeo de clases; (3) es de nivel
  coordinador, sin confirmacion directa de A ni de B.
- Consecuencia: ninguna de las dos convenciones se presenta como "aclaracion
  confirmada"; la de notas queda como contenido literal pendiente de
  confirmacion directa; la de clases queda como no demostrada.
- Justicia interpretativa: la instruccion hace de 0=NEGATIVE/1=POSITIVE la
  lectura natural y coherente; este audit no afirma que sea incorrecta, solo
  que no cumple el estandar de demostracion exigido. La confirmacion directa
  de A y B es la accion que cierra la incertidumbre (seccion 5).

## 4. Impacto sobre los resultados de P92-T (por niveles de evidencia)

### 4.1 Validos como observaciones crudas (sin supuesto de codificacion)

- Respuestas de C y D, incluidos sus UNABLE_TO_DETERMINE; acuerdo C-D
  (12/15 HS, 14/15 TR, 11/15 AG); conteos UTD; catalogo de ambiguedades
  (seis casos HS + secundarios), basado en notas de C/D y documentos.
- Valores crudos 0/1 de A y B; acuerdo A-B por igualdad de valores crudos:
  10/15 (HS), 11/15 (TR), 12/15 (AG)
  (supone que A y B usaron la misma convencion, sostenido por el formato
  comun y la instruccion registrada; no requiere la direccion del mapeo).
- Presencia cruda de las marcas de fila de A y B (A: P92C-02, 04; B: P92C-01,
  05, 13), con su significado atribuido segun la seccion 3.

### 4.2 Condicionales a la codificacion documentada (0=NEGATIVE, 1=POSITIVE)

- Todas las estadisticas que combinan A/B con C/D: pares AC, AD, BC, BD
  (acuerdo, dice, kappas).
- Combinacion exacta de los 4 por etiqueta y global (1/15); alfas de
  Krippendorff con A/B (0.243/0.466, 0.104/0.093, 0.354/0.491).
- Los archivos normalizados calibration_response_A/B_normalized.csv
  (derivados condicionales; no deben usarse como acuerdo definitivo).

### 4.3 Confirmados

- Ninguna estadistica queda en este nivel al estandar estricto (la
  interpretacion de A/B depende de la codificacion no demostrada).

### 4.4 No afectado

- La utilidad de la calibracion para revisar la guia se mantiene: el
  catalogo de ambiguedades, los seis casos HS, la propuesta v2, el registro de
  decisiones y el gate descansan en C/D y en los valores crudos, no en la
  codificacion de A/B.
- El desacuerdo entre anotadores no se mezcla con errores confirmados del
  dataset: nada de lo observado demuestra errores en las etiquetas originales.

## 5. Accion minima para resolver la incertidumbre

Solicitud directa a A y B (borrador listo en
reports/p92tr_ab_confirmation_request.md; NO enviada): dos preguntas de
seleccion sobre el significado de 0 y 1 en las columnas de clase y sobre el
significado de 0/1 en uncertainty_notes. Una respuesta escrita breve de cada
anotador, recogida por separado, cierra la incertidumbre y permitiria promover
las estadisticas de 4.2 a confirmadas (con un addendum nuevo). Mientras tanto:
A/B condicional; C/D valido.

## 6. Correcciones aditivas aplicadas

- Notas aditivas (marcador "NOTA ADITIVA P92-T-R") al final de:
  reports/p92_calibration_analysis.md,
  artifacts/human_annotation_v1/calibration_responses/PROVENANCE.md,
  reports/p92u_human_discussion_pack.md y
  reports/p92_guideline_decision_record.md.
- Este addendum queda como registro autoritativo de la procedencia; los
  documentos anteriores no se modifican en su contenido historico.
- No se modificaron: respuestas humanas (hashes en la seccion 7),
  normalizados (bytes intactos; marcados como condicionales),
  p92t_audit_results.json (historico), paquetes principales ni notebook
  historico (secciones 140 y 141 se mantienen; su calificacion queda
  corregida por este addendum).

## 7. Hashes de los archivos originales (re-verificados hoy)

| archivo | sha256 |
|---|---|
| calibration_response_A.csv | `a20521989aa3fea94e059f0666713b437a333d2cead31f9e4198d24dde9c9596` |
| calibration_response_B.csv | `41e50c1f1d48760eaf293a8e5f51b9dd0d036b19e8d48f92a2e1ce34c024a49f` |
| calibration_response_C.csv | `8de76e21046d5091bfa360cce6fb5299d2ef9951685b46ae85178ce45e477d0f` |
| calibration_response_D.csv | `763229f9a9caa176df7b9ee1ff247269aa6ea51259a9ce3f8ca658dc3432f8a4` |
| calibration_response_A_normalized.csv | `7e7b3c805620e6139bfa94ce695864f6a8badde495a63a08b6ad44f07206e808` |
| calibration_response_B_normalized.csv | `7cca95e20a1243c01dcfb23578b82f181261dd8210bd0a919465acd2a88b6dae` |

Los cuatro archivos originales coinciden con los hashes registrados al
momento de la recepcion: siguen intactos.

## 8. Integridad y estado

- Respuestas humanas originales: sin cambios. Normalizados: bytes sin
  cambios (marcados condicionales). Muestra principal: no distribuida.
- V03, OOF canonico, etiquetas originales, particion y submissions: intactos.
- Test y holdouts: no utilizados. Sin entrenamiento ni predicciones.
- Verificador ASCII oficial: ejecutado sin hallazgos (ver informe de fase).
- Git: sin commit ni push; cambios locales conservados.
