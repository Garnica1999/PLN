# P92-V - Decisiones de politica operativa del proyecto (Q1-Q8)

POLICY_ID: PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2
Fecha de decision: 2026-10-13. Decision adoptada por: propietario del proyecto.
Naturaleza: POLITICA OPERATIVA INTERNA para el piloto de anotacion. NO son
definiciones oficiales de la competencia. No consta una discusion de consenso
entre los cuatro anotadores de calibracion; estas decisiones no se les
atribuyen.

## 0. Registro de gobernanza (Q2)

```text
governance_clarification = NOT_REQUESTED
official_semantics = UNRESOLVED
internal_pilot = AUTHORIZED
```

## 1. Decisiones Q1-Q8 (operativas)

### Q1 - HS: odio frente a ofensa

- decision: Enfoque conservador: definicion oficial literal mas incertidumbre explicita. No se inventa una nueva definicion normativa de odio. Si el caso no puede resolverse responsablemente con la definicion oficial y estas instrucciones: UNABLE_TO_DETERMINE. Esa respuesta no se convierte automaticamente en negativa.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-03, CH-07

### Q2 - Consulta al organizador (registro de gobernanza)

- decision: El piloto continua sin esperar una aclaracion externa. Registro: governance_clarification = NOT_REQUESTED; official_semantics = UNRESOLVED; internal_pilot = AUTHORIZED. No se afirma que las cuestiones normativas esten resueltas oficialmente.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-07

### Q3 - AG: agresividad frente a vulgaridad

- decision: No equiparar automaticamente vulgaridad, insulto y agresividad. Aplicar la definicion oficial disponible. Usar UNABLE_TO_DETERMINE cuando no exista evidencia suficiente para decidir.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-03

### Q4 - Broma, ironia y sarcasmo

- decision: Evaluar el contenido observable del texto. No inferir una intencion oculta sin evidencia textual suficiente. Si el tono altera materialmente la interpretacion y no hay contexto suficiente: UNABLE_TO_DETERMINE, explicando la duda.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-05, CH-03

### Q5 - Citas y discurso reproducido

- decision: Evaluar el texto completo como aparece en el ejemplo. No atribuir automaticamente al autor una opinion citada ni asumir que reproducir un insulto equivale a respaldarlo. Usar UNABLE_TO_DETERMINE cuando no pueda determinarse la funcion de la cita con el contexto disponible.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-05

### Q6 - Alcance de UNABLE_TO_DETERMINE

- decision: Permitido ante falta de contexto o ambiguedad normativa/semantica material. Se asigna por etiqueta. Registrar una nota que explique el motivo cuando se utilice. No forzar una respuesta binaria solo porque el dataset original sea binario.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-02, CH-04

### Q7 - Duda por etiqueta y duda por tweet

- decision: Mantener ambas separadas. Las tres anotaciones HS/TR/AG se registran independientemente. Cuando exista una dificultad que afecte a todo el tweet, registrarla en una nota general adicional (columna tweet_note) sin sustituir las respuestas individuales.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-04, CH-09

### Q8 - Formato

- decision: Usar exclusivamente POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE. No se aceptan valores 0/1 en la anotacion principal.
- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)
- cambios de guia asociados: CH-01, CH-02

## 2. Regla de correccion futura (vocabulario de estados)

Ninguna discrepancia entre la anotacion humana y el label original es un error
confirmado. Estados diferenciados:

- MODEL_DISAGREEMENT: un modelo difiere del label original; no dice nada sobre el label.
- ANNOTATOR_DISAGREEMENT: anotadores humanos difieren entre si sobre un caso.
- POSSIBLE_LABEL_ISSUE: evidencia acumulada sugiere revisar el label; sin adjudicacion.
- ADJUDICATED_DIFFERENCE: diferencia revisada y decidida por adjudicacion documentada.
- CONFIRMED_ANNOTATION_ERROR: error demostrado y registrado tras adjudicacion.
- APPROVED_DATASET_CORRECTION: correccion aprobada formalmente (autorizacion separada; no ocurre en P92-V).

Una correccion futura necesitara anotaciones humanas independientes,
adjudicacion documentada y autorizacion separada. Ninguna etiqueta original se
modifica en P92-V y no se entrenaran modelos con las nuevas anotaciones.

## 3. Consecuencias y artefactos

- Guia congelada: reports/p92_annotation_guidelines_v2_frozen.md
  (sha256 sin linea de hash: 46be258ad47074405abce99e79ef831970d7b877e1987705de375432413117f8).
- Paquetes: artifacts/human_annotation_v2/annotator_A/ y annotator_B/
  (CSV vacio + guia congelada + README). NO distribuidos.
- Las respuestas de calibracion de los cuatro anotadores permanecen bajo la
  guia v1 (historico); no se reinterpretan retroactivamente.
