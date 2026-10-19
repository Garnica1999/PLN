# P92 - Guia de anotacion v2 (PROPUESTA para discusion humana)

Fecha: 2026-10-12. Estado: **V2_PROPOSED_PENDING_DISCUSSION**. Esta guia NO esta
congelada y NO es aplicable todavia: es una propuesta para la discusion de los
cuatro participantes de la calibracion. La ronda principal no debe iniciarse hasta
cerrar la discusion, registrar las decisiones humanas y congelar la version
aplicable (seccion 5).

Base: `reports/p92_annotation_guidelines_v1.md` + evidencia de la calibracion
(`reports/p92_calibration_analysis.md`).

Regla de construccion de esta propuesta: no se inventa ninguna definicion
normativa. Cada enunciado se clasifica como:

```text
OFFICIAL_DEFINITION          -> texto oficial verbatim (competition_rules.md)
PROPOSED_OPERATIONAL_RULE    -> criterio operativo propuesto, pendiente de decision
UNRESOLVED_POLICY_QUESTION   -> pregunta normativa sin respuesta documentada
```

## 1. Definiciones oficiales (verbatim)

OFFICIAL_DEFINITION (docs/competition_rules.md; transliteracion ASCII de acentos):

- **HS**: "indica si hay discurso de odio (1 = si, 0 = no)".
- **TR**: "indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)".
- **AG**: "indica si el texto es agresivo (1 = si, 0 = no)".

Estas tres definiciones siguen siendo la UNICA fuente normativa disponible. No
existen criterios operativos oficiales, guia de anotadores original ni ejemplos
resueltos por el organizador (verificado en fases previas).

## 2. Instrucciones operativas

### 2.1 Distinguir una posible ofensa sexista de discurso de odio

UNRESOLVED_POLICY_QUESTION: no existe hoy ningun criterio documental (oficial o
provisto por el curso) que permita decidir cuando una ofensa sexista individual
dirigida a una persona constituye discurso de odio. La calibracion lo confirmo de
forma empirica: dos anotadores independientes (C y D) registraron la misma duda en
las mismas 6 de 15 filas de la muestra.

PROPOSED_OPERATIONAL_RULE (para discusion; NO normativa): se proponen tres lecturas
alternativas para que la discusion humana elija o descarte:

- O1 (estricta): HS = POSITIVE solo cuando la hostilidad se dirige a alguien por su
  identidad o pertenencia a un grupo (mujeres, migrantes, personas trans, etc.).
  El insulto por motivos personales sin componente identitario no seria HS.
- O2 (amplia): HS = POSITIVE cuando el texto degrada, deshumaniza o expresa
  hostilidad hacia una persona o grupo, aunque el motivo sea individual.
- O3 (literal): mantener la definicion oficial sin criterio operativo y usar
  UNABLE_TO_DETERMINE cuando la definicion no permita decidir.

### 2.2 Que hacer cuando el criterio no existe o no resuelve el caso

PROPOSED_OPERATIONAL_RULE: registrar UNABLE_TO_DETERMINE en la etiqueta afectada y
explicar el motivo en `uncertainty_notes`. No forzar una respuesta binaria. La duda
no invalida la anotacion: es un dato valido y esperado mientras la pregunta
normativa siga abierta. La incertidumbre registrada alimenta la consulta al
organizador (seccion 4).

### 2.3 Insulto, vulgaridad, agresividad y discurso de odio

OFFICIAL_DEFINITION: solo la de AG ("indica si el texto es agresivo").

No se asume que estos conceptos sean equivalentes. Orientacion propuesta:

- Vulgaridad: uso de lenguaje soez sin agresion dirigida. No implica AG ni HS.
- Insulto: agresion verbal dirigida a alguien. Puede implicar AG; su relacion con
  HS depende de la pregunta abierta de la seccion 2.1.
- Agresividad (AG): PROPOSED_OPERATIONAL_RULE: AG = POSITIVE si hay agresion
  dirigida (insulto, amenaza, orden hostil) hacia una persona o grupo. Queda como
  UNRESOLVED_POLICY_QUESTION si una generalizacion hostil sin insulto ni amenaza
  explicita cuenta como AG (caso real de la calibracion: generalizacion contra
  inmigrantes presentada como amenaza laboral).
- Discurso de odio (HS): segun la definicion oficial; sin criterio operativo
  documentado (seccion 2.1).

### 2.4 Citas, discurso reproducido, sarcasmo, ironia y falta de contexto

OFFICIAL_DEFINITION: no existe ninguna regla oficial para estos casos.

PROPOSED_OPERATIONAL_RULE (para discusion):
- Evaluar el contenido hostil expresado en el texto, no la forma retorica.
- Sarcasmo o ironia: si el texto expresa hostilidad hacia alguien, evaluar la
  etiqueta correspondiente con el mismo criterio que una expresion literal.
- Citas o discurso reproducido: considerar el contexto disponible en el propio
  texto; si el texto no permite saber si hay adhesion, denuncia o burla, registrar
  UNABLE_TO_DETERMINE y anotarlo.
- Expresiones idiomaticas (por ejemplo, formulas coloquiales de enfasis) no
  cuentan por si solas como agresion.
- Contexto insuficiente: si falta el tweet original o la referencia y no se puede
  decidir responsablemente, usar UNABLE_TO_DETERMINE + nota. No adivinar.

### 2.5 Criterios para elegir UNABLE_TO_DETERMINE

Formulacion heredada de la v1 (se mantiene): UTD no es una clase semantica.
Usarlo cuando:

1. La definicion oficial no permite decidir responsablemente.
2. Falta el contexto necesario (blanco no identificable, referencia ininteligible).
3. Existen dos lecturas competidoras y ninguna es claramente aplicable (por ejemplo,
   odio vs. ofensa mientras la seccion 2.1 siga abierta).

Reglas: se decide POR ETIQUETA (una etiqueta puede ser UTD y otra no); la nota de
motivo es obligatoria cuando se use UTD; UTD nunca se convierte automaticamente en
POSITIVE ni NEGATIVE; si la fila entera es dificil, marcar UTD en las etiquetas
afectadas y explicarlo en la nota (sin flags binarios aparte: el incidente A/B
mostro que un 0/1 adicional fuera de las columnas se vuelve ambiguo).

### 2.6 Independencia de HS, TR y AG

OFFICIAL_DEFINITION: tres definiciones independientes. PROPOSED_OPERATIONAL_RULE:
anotar cada etiqueta por separado, sin jerarquias. En particular, no asumir que
TR = 1 exige HS = 1 ni que AG = 1 exige HS = 1: la relacion observada en los datos
no esta documentada como regla formal. Cada etiqueta admite su propio UTD.

### 2.7 Ejemplos estructurales anonimizados (derivados de la calibracion)

Ejemplos de TIPO de ambiguedad, sin textos verbatim y sin respuestas oficiales:

- E1: insulto sexista directo a una persona, sin referencia a un grupo.
  Pregunta abierta: odio vs. ofensa (seccion 2.1).
- E2: ataque personal con insultos sexistas o vulgares y ordenes hostiles a una
  persona. Pregunta abierta: odio vs. ofensa; AG probablemente POSITIVE.
- E3: insulto leve con posible tono de broma entre personas conocidas.
  Pregunta abierta: agresivo vs. vulgar; el tono no decide por si solo.
- E4: generalizacion hostil contra un grupo (por ejemplo, presentarlo como amenaza
  laboral) sin insulto ni amenaza explicita. Pregunta abierta: cuenta como AG.
- E5: respuesta con apodo o referencia no identificable sin el tweet original.
  Tratamiento propuesto: UTD + nota (contexto insuficiente).
- E6: sarcasmo o cita con expresiones idiomaticas. Tratamiento propuesto: evaluar
  el contenido expresado; UTD si el contexto no alcanza.

## 3. Changelog v1 -> v2 (cada diferencia)

| # | Cambio | v1 | v2 propuesta | Evidencia |
|---|---|---|---|---|
| CH-01 | Formato de respuesta | "una de POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE por celda" | igual, mas: los valores 0/1 quedan explicitamente invalidos; ejemplo documentado del incidente A/B | A y B devolvieron 0/1 (P92-S2, P92-T) |
| CH-02 | Nota de motivo con UTD | nota opcional | nota OBLIGATORIA cuando se usa UTD | 6 filas con UTD de C y D sustentadas en notas; las notas hicieron interpretable cada duda |
| CH-03 | Criterios operativos | no existian | nueva seccion 2 con criterios marcados PROPOSED_OPERATIONAL_RULE, separados de OFFICIAL_DEFINITION y UNRESOLVED_POLICY_QUESTION | calibracion: la duda se concentro en odio vs. ofensa (6 filas HS) y agresivo vs. vulgar |
| CH-04 | Duda a nivel de fila | no previsto | 2.5: marcar UTD en las etiquetas afectadas + nota (sin flag binario aparte) | flag 0/1 de A/B resulto ambiguo de interpretar |
| CH-05 | Citas, sarcasmo, contexto | listados como preguntas abiertas sin orientacion | 2.4 con orientacion operativa propuesta + UTD cuando el contexto no alcanza | casos 01 y 07 (contexto insuficiente) y 06/07 (sarcasmo/idiomatismo) |
| CH-06 | Ejemplos | no habia | 2.7 con ejemplos estructurales anonimizados | tipos de ambiguedad observados en la calibracion |
| CH-07 | Preguntas normativas | 4 limitaciones listadas sin formulacion para terceros | seccion 4: formuladas como preguntas para decision humana y eventual consulta al organizador | evidencia empirica de la calibracion |
| CH-08 | Independencia HS/TR/AG | recomendada | reforzada en 2.6, con UTD por etiqueta | ningun anotador uso jerarquias; se mantiene como recordatorio |

Se conservan sin cambios: unidad de anotacion (tweet completo), cegamiento,
registro de desacuerdos, prohibicion de herramientas automaticas, esquema de
columnas y el principio de no inventar reglas.

## 4. Limitaciones semanticas que continuan sin resolverse

1. UNRESOLVED_POLICY_QUESTION (principal): cuando una ofensa sexista individual
   constituye discurso de odio. Evidencia: 6/15 filas de calibracion con UTD
   coincidente de C y D; 2 mas con UTD de D en la misma duda (03, 04).
2. UNRESOLVED_POLICY_QUESTION: agresivo vs. vulgar, y si la generalizacion hostil
   sin insulto cuenta como AG (casos 06 y 13 de la calibracion).
3. UNRESOLVED_POLICY_QUESTION: relacion formal entre HS y TR (y HS y AG).
4. UNRESOLVED_POLICY_QUESTION: tratamiento de citas, sarcasmo y discurso
   reproducido; la seccion 2.4 solo propone orientacion operativa.

Pregunta propuesta para el organizador (redaccion para discusion): "Cuando un
insulto o agresion sexista se dirige a una persona concreta por su identidad, sin
referirse a un grupo, debe anotarse HS = 1, HS = 0, o no existe criterio?".

## 5. Decision de version y condiciones de congelamiento

Estado actual: V2_PROPOSED_PENDING_DISCUSSION. La guia solo puede considerarse
congelada para la muestra principal despues de:

1. Resolver las cuestiones procedimentales (formato, UTD, notas: secciones 2.2,
   2.5 y CH-01/CH-02).
2. Registrar las decisiones humanas sobre los criterios operativos (2.1, 2.3, 2.4).
3. Documentar las cuestiones normativas que siguen abiertas (seccion 4).
4. Aprobar la version aplicable a la anotacion principal.
5. Registrar version y hash de la guia congelada.

Si la definicion oficial no permite resolver una ambiguedad material, no se simula
certeza normativa: se documenta la limitacion. La ronda principal puede continuar
como referencia exploratoria con UTD registrado, o esperar una aclaracion adicional
del organizador; esa eleccion es parte de la discusion humana.
