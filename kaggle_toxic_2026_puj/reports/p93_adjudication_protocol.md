# P93 - Protocolo de adjudicacion (tercera persona cualificada)

Estado: preparado; NO iniciado; la distribucion la decide el propietario.

## 1. Alcance

- Cola inicial: 176 celdas donde A y B coinciden en una
  etiqueta binaria distinta del gold original (candidatos, NO errores).
- Los desacuerdos humanos (P3) podran adjudicarse en un lote separado si se
  autoriza; este protocolo no los resuelve automaticamente.

## 2. Materiales

- Para el adjudicador: SOLO el paquete ciego
  (p93_adjudication_blind_candidates.csv: texto, etiqueta, respuestas A y B,
  notas), sin gold, sin predicciones, sin motivos de seleccion.
- El archivo interno de candidatos (con gold y motivos) NO se comparte.

## 3. Decisiones posibles por caso

1. Confirmar el gold (la diferencia se explica por ambiguedad de politica).
2. Proponer revision del gold (con justificacion escrita y cita de la regla).
3. UNABLE_TO_DETERMINE (falta contexto o norma).

## 4. Reglas

- Ninguna etiqueta se modifica durante P93 ni con este protocolo.
- Una correccion futura requerira adjudicacion documentada y autorizacion
  separada (vocabulario de estados de P92-V: POSSIBLE_LABEL_ISSUE ->
  ADJUDICATED_DIFFERENCE -> CONFIRMED_ANNOTATION_ERROR ->
  APPROVED_DATASET_CORRECTION).
- El adjudicador no vera V03 ni sus scores; su juicio se basa en el texto,
  la guia aplicable y las respuestas humanas.
- Registrar por caso: decision, fundamento, fecha y firma; las decisiones se
  conservan separadas de las respuestas originales.
