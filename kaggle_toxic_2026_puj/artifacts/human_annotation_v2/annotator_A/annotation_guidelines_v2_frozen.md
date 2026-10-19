# P92 - Guia de anotacion operativa v2 CONGELADA (piloto interno)

POLICY_ID: PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2
Version de guia: v2 (valor de columna guideline_version: "v2")
Fecha de congelacion: 2026-10-13
SHA256 (contenido sin la linea que contiene este hash): 46be258ad47074405abce99e79ef831970d7b877e1987705de375432413117f8

## 0. Naturaleza de este documento (leer primero)

- Esta guia es una POLITICA OPERATIVA INTERNA del proyecto para el piloto de
  anotacion principal. NO es una definicion oficial de la competencia y NO
  resuelve las ambiguedades normativas oficiales (seccion 6).
- La palabra "frozen" (congelada) se refiere exclusivamente a que esta version
  queda fijada para este piloto. No significa que las ambiguedades oficiales
  esten resueltas.
- Separacion de etiquetas de naturaleza usada en todo el documento:

```text
OFFICIAL_DEFINITION        -> texto oficial verbatim (docs/competition_rules.md)
PROJECT_OPERATIONAL_RULE   -> regla operativa adoptada por el propietario del
                              proyecto (2026-10-13); no oficial
UNRESOLVED_POLICY_QUESTION -> pregunta normativa que sigue abierta
```

## 1. Definiciones oficiales (OFFICIAL_DEFINITION, verbatim)

- **HS**: "indica si hay discurso de odio (1 = si, 0 = no)".
- **TR**: "indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)".
- **AG**: "indica si el texto es agresivo (1 = si, 0 = no)".

Estas definiciones son la UNICA fuente normativa disponible. No existen criterios
operativos oficiales, guia de anotadores original ni ejemplos resueltos por el
organizador (verificado en fases previas).

## 2. Reglas operativas del proyecto (PROJECT_OPERATIONAL_RULE, adoptadas 2026-10-13)

### Q1 - HS: odio frente a ofensa

Enfoque conservador: definicion oficial literal mas incertidumbre explicita. No se inventa una nueva definicion normativa de odio. Si el caso no puede resolverse responsablemente con la definicion oficial y estas instrucciones: UNABLE_TO_DETERMINE. Esa respuesta no se convierte automaticamente en negativa.

### Q2 - Consulta al organizador (registro de gobernanza)

El piloto continua sin esperar una aclaracion externa. Registro: governance_clarification = NOT_REQUESTED; official_semantics = UNRESOLVED; internal_pilot = AUTHORIZED. No se afirma que las cuestiones normativas esten resueltas oficialmente.

### Q3 - AG: agresividad frente a vulgaridad

No equiparar automaticamente vulgaridad, insulto y agresividad. Aplicar la definicion oficial disponible. Usar UNABLE_TO_DETERMINE cuando no exista evidencia suficiente para decidir.

### Q4 - Broma, ironia y sarcasmo

Evaluar el contenido observable del texto. No inferir una intencion oculta sin evidencia textual suficiente. Si el tono altera materialmente la interpretacion y no hay contexto suficiente: UNABLE_TO_DETERMINE, explicando la duda.

### Q5 - Citas y discurso reproducido

Evaluar el texto completo como aparece en el ejemplo. No atribuir automaticamente al autor una opinion citada ni asumir que reproducir un insulto equivale a respaldarlo. Usar UNABLE_TO_DETERMINE cuando no pueda determinarse la funcion de la cita con el contexto disponible.

### Q6 - Alcance de UNABLE_TO_DETERMINE

Permitido ante falta de contexto o ambiguedad normativa/semantica material. Se asigna por etiqueta. Registrar una nota que explique el motivo cuando se utilice. No forzar una respuesta binaria solo porque el dataset original sea binario.

### Q7 - Duda por etiqueta y duda por tweet

Mantener ambas separadas. Las tres anotaciones HS/TR/AG se registran independientemente. Cuando exista una dificultad que afecte a todo el tweet, registrarla en una nota general adicional (columna tweet_note) sin sustituir las respuestas individuales.

### Q8 - Formato

Usar exclusivamente POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE. No se aceptan valores 0/1 en la anotacion principal.

## 3. Unidad de anotacion y esquema de columnas

- Unidad: el tweet completo, tal como aparece; no normalices el texto.
- Las tres etiquetas (HS, TR, AG) se deciden de forma independiente entre si.
- Columnas del CSV:

```text
review_id            identificador opaco; NO modificar
text                 texto del tweet; NO modificar
HS_annotation        POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE
TR_annotation        POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE
AG_annotation        POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE
uncertainty_notes    motivo cuando uses UNABLE_TO_DETERMINE; aclaraciones por etiqueta
tweet_note           nota general SOLO si la dificultad afecta a todo el tweet
guideline_version    no modificar ("v2")
```

## 4. Procedimiento por etiqueta e instrucciones exactas de UNABLE_TO_DETERMINE

Para cada etiqueta, por separado:

1. Lee el texto completo.
2. Aplica la definicion oficial y las reglas operativas de la seccion 2.
3. Si puedes decidir con responsabilidad: POSITIVE o NEGATIVE.
4. Si no puedes: UNABLE_TO_DETERMINE y motivo en uncertainty_notes.

UNABLE_TO_DETERMINE (UTD):

- No es una tercera clase semantica: declara que no puedes asignar POSITIVE ni
  NEGATIVE con responsabilidad usando la definicion oficial y estas reglas.
- Se decide POR ETIQUETA: puedes usar UTD en HS y no en TR, etc.
- Casos tipicos: falta el tweet al que responde; referencia o apodo no
  identificable sin contexto; ambiguedad normativa material (p. ej., ofensa
  sexista individual frente a discurso de odio, mientras la seccion 6 siga
  abierta); funcion de una cita no determinable.
- Al usar UTD, escribe el motivo en uncertainty_notes (obligatorio).
- NUNCA conviertas UTD en NEGATIVE ni POSITIVE automaticamente, ni durante la
  anotacion ni en el analisis posterior.
- No fuerces una respuesta binaria solo porque el dataset original sea binario.
- Si la dificultad afecta a TODO el tweet, escribe ademas una nota general en
  tweet_note (sin sustituir las respuestas por etiqueta).

## 5. Changelog frente a la guia v1 (adoptado)

| # | Cambio | v1 | v2 congelada | Base |
|---|---|---|---|---|
| CH-01 | Formato | tokens por celda (0/1 no previsto) | tokens exclusivos; 0/1 explicitamente invalido | desviacion de formato documentada en la ronda de calibracion |
| CH-02 | Notas con UTD | nota opcional | nota de motivo obligatoria cuando se usa UTD | calibracion (notas de C y D) |
| CH-03 | Criterios operativos | no existian | Q1/Q3: definicion literal + UTD; sin nuevas definiciones | decision del propietario, 2026-10-13 |
| CH-04 | Duda de fila | no previsto | Q7: UTD por etiqueta + columna tweet_note | el formato 0/1 de la calibracion resulto ambiguo de interpretar |
| CH-05 | Tono, citas y contexto | sin orientacion | Q4/Q5: contenido observable; sin inferir intencion; UTD si el contexto no alcanza | observado en la calibracion (contexto insuficiente y tono) |
| CH-06 | Ejemplos | no habia | esta guia no incluye textos; los ejemplos estructurales quedan en el material interno del proyecto | material interno del proyecto |
| CH-07 | Preguntas normativas | listadas sin gestion | formalizadas (seccion 6); consulta NO solicitada (Q2) | Q2 |
| CH-08 | Independencia HS/TR/AG | recomendada | reforzada; UTD por etiqueta | Q6/Q7 |
| CH-09 | Esquema de columnas | 7 columnas | 8 columnas: se agrega tweet_note | Q7 |

## 6. Preguntas normativas que siguen abiertas (UNRESOLVED_POLICY_QUESTION)

1. Frontera odio/ofensa para una ofensa sexista individual dirigida a una
   persona. Evidencia: la ronda de calibracion registro ambiguedad normativa
   material en este punto (6 de 15 filas, con UNABLE_TO_DETERMINE registrado
   por mas de un anotador).
2. Frontera agresivo/vulgar y tratamiento de generalizaciones hostiles sin
   insulto explicito.
3. Relacion formal (si existe) entre HS y TR, y entre HS y AG.
4. Tratamiento normativo de citas, sarcasmo y discurso reproducido.

Registro de gobernanza (Q2): governance_clarification = NOT_REQUESTED;
official_semantics = UNRESOLVED; internal_pilot = AUTHORIZED. Esta guia permite
registrar incertidumbre mientras estas preguntas sigan abiertas; no las resuelve.

## 7. Cegamiento y reglas de independencia

- No consultes ni intentes deducir etiquetas originales, predicciones de
  modelos, umbrales ni material interno del proyecto.
- No busques los tweets en internet.
- No uses herramientas automaticas (LLM, clasificadores) para asignar etiquetas.
- No comentes casos con la otra persona anotadora antes de terminar tu ronda.

## 8. Registro de desacuerdos y regla de correccion futura

Los desacuerdos no se resuelven en esta ronda: se conservan y se remiten a una
adjudicacion posterior, separada y documentada. Vocabulario de estados del
proyecto (nada es error automatico):

- MODEL_DISAGREEMENT: un modelo difiere del label original; no dice nada sobre el label.
- ANNOTATOR_DISAGREEMENT: anotadores humanos difieren entre si sobre un caso.
- POSSIBLE_LABEL_ISSUE: evidencia acumulada sugiere revisar el label; sin adjudicacion.
- ADJUDICATED_DIFFERENCE: diferencia revisada y decidida por adjudicacion documentada.
- CONFIRMED_ANNOTATION_ERROR: error demostrado y registrado tras adjudicacion.
- APPROVED_DATASET_CORRECTION: correccion aprobada formalmente (autorizacion separada; no ocurre en P92-V).

Una correccion futura de etiquetas requerira anotaciones humanas independientes,
adjudicacion documentada y autorizacion separada.

## 9. Entrega

- Devuelve el CSV completado; no borres filas ni columnas ni cambies el orden.
- Guarda el archivo con el nombre exacto indicado en el README de tu paquete.

