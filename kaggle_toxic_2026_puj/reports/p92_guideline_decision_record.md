# P92-U - Registro de decisiones de la guia de anotacion (plantilla editable)

Estado: **PLANTILLA**. Los campos de decision humana estan VACIOS a proposito.
No inventar consenso ni simular una reunion. Este registro lo completan las
personas responsables (anotadores y responsables del proyecto) durante la
discusion, usando `reports/p92u_human_discussion_pack.md` como material.

Instrucciones de uso:

1. Para cada pregunta (Q1 a Q8), completar `human_decision`, `rationale`,
   `decision_owner` y `decision_date`; actualizar `remaining_uncertainty` y
   `guideline_change` segun lo acordado.
2. Los campos `issue`, `official_rule`, `observed_evidence` y `options_considered`
   ya estan prellenados desde la evidencia de P92-T/P92-U; pueden ampliarse.
3. Ninguna decision se aplica a los paquetes principales hasta que la version de
   la guia resultante este congelada, con changelog y SHA256 (seccion final).

## 1. Registro por pregunta

### Q1 - Distincion entre odio y ofensa sexista individual

- question_id: Q1
- issue: Distincion entre odio y ofensa sexista individual
- official_rule: HS: "indica si hay discurso de odio (1 = si, 0 = no)" (docs/competition_rules.md). No existe criterio operativo, lista de grupos protegidos, ejemplos ni FAQ (verificado en fases previas).
- observed_evidence:
  - 6/15 filas de la muestra con UTD coincidente de C y D en HS (P92C-05, 09, 11, 13, 14, 15); todas son ofensas o insultos sexistas dirigidos a personas concretas.
  - D uso UTD con la misma duda normativa en P92C-03 y P92C-04; en 04 la nota de C registra la misma duda (odio vs. ofensa) aunque respondio POSITIVE.
  - Ninguna de las notas menciona pertenencia a un grupo: la duda es individuo (persona concreta) frente a la categoria oficial 'discurso de odio'.
  - La ambiguedad es sistematica y localizada, no ruido: dos anotadores independientes convergen en las mismas filas y el mismo motivo.
- options_considered:
  - O1 (estricta): HS = POSITIVE solo cuando la hostilidad se dirige a alguien por su identidad o pertenencia a un grupo.
  - O2 (amplia): HS = POSITIVE cuando el texto degrada o expresa hostilidad hacia una persona o grupo, aunque el motivo sea individual.
  - O3 (literal + UTD): Mantener la definicion oficial sin criterio operativo y usar UNABLE_TO_DETERMINE cuando no permita decidir.
- status: REQUIRES_GOVERNANCE_CLARIFICATION
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: O3 (literal + UTD) como respaldo si la discusion no adopta una regla.
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-03, CH-07 (confirmar o ajustar)

### Q2 - Solicitar aclaracion normativa al responsable de la competicion

- question_id: Q2
- issue: Solicitar aclaracion normativa al responsable de la competicion
- official_rule: No aplica: no existe canal documentado de aclaraciones para anotadores; la pagina de la competencia no publica criterios operativos.
- observed_evidence:
  - P91 dejo la gobernanza en revision (GOVERNANCE_REVIEW_REQUIRED) por la misma ausencia de criterios operativos.
  - La calibracion confirmo que la duda es material (6/15 filas HS + 2 mas) y sistematica entre anotadores independientes.
- options_considered:
  - S1 enviar consulta: Enviar una consulta escrita breve al organizador (redaccion propuesta en la seccion 5 de este paquete).
  - S2 no enviar; adoptar O3: No elevar la consulta y operar con la definicion literal + UTD.
  - S3 enviar y continuar en paralelo: Enviar la consulta y comenzar la ronda principal con UTD, registrando que se anoto sin la aclaracion.
- status: UNRESOLVED
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: Ninguna especifica: la no-respuesta no cambia por si sola la guia; aplica la opcion conservadora de Q1/Q3.
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-07 (confirmar o ajustar)

### Q3 - Distinguir agresividad de vulgaridad

- question_id: Q3
- issue: Distinguir agresividad de vulgaridad
- official_rule: AG: "indica si el texto es agresivo (1 = si, 0 = no)". Sin criterio operativo ni ejemplos.
- observed_evidence:
  - P92C-06: D UTD en AG - generalizacion hostil contra un grupo sin insulto ni amenaza explicita ('la agresividad es discutible'); A, B y C respondieron NEGATIVE.
  - P92C-13: C UTD en AG - insulto leve con posible tono de broma; duda agresivo vs. vulgar; D respondio POSITIVE.
  - P92C-07: D UTD en AG - sarcasmo o cita sin contexto; agresividad no inequivoca.
  - AG fue la etiqueta con menor coincidencia exacta de los 4 anotadores (6/15).
- options_considered:
  - A1 (agresion dirigida): AG = POSITIVE si hay agresion dirigida (insulto, amenaza, orden hostil) hacia una persona o grupo; la vulgaridad no dirigida no basta.
  - A2 (amplia): Incluir tambien generalizaciones hostiles hacia grupos aunque no haya insulto dirigido.
  - A3 (literal + UTD): Sin criterio operativo; UTD cuando 'agresivo' no sea claro.
- status: REQUIRES_GOVERNANCE_CLARIFICATION
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: A3 (literal + UTD) como respaldo.
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-03 (confirmar o ajustar)

### Q4 - Tono de broma, ironia y sarcasmo

- question_id: Q4
- issue: Tono de broma, ironia y sarcasmo
- official_rule: Ninguna regla oficial (no hay instrucciones de casos limite).
- observed_evidence:
  - P92C-13: 'podria ser broma entre conocidos' (nota de C) como razon de duda en HS y AG.
  - P92C-07: sarcasmo o critica; 'De puta madre' interpretada como expresion idiomatica (nota de D).
  - Ningun caso de la muestra produjo consenso de que la broma neutralice la hostilidad.
- options_considered:
  - T1 (contenido expresado): Evaluar el contenido hostil expresado; el tono retorico (broma, ironia, sarcasmo) no decide por si solo.
  - T2 (tono neutraliza): Una broma entre conocidos neutraliza agresion u odio.
  - T3 (UTD si el tono es determinante): Usar UTD cuando el tono sea lo que decide el caso y no sea verificable.
- status: UNRESOLVED
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: T1 + T3 (contenido expresado; UTD cuando el tono sea determinante e inverificable).
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-05, CH-03 (confirmar o ajustar)

### Q5 - Citas y discurso reproducido

- question_id: Q5
- issue: Citas y discurso reproducido
- official_rule: Ninguna regla oficial.
- observed_evidence:
  - P92C-07: critica o sarcasmo sobre descendientes de una persona descrita como 'genocida'; sin el contexto enlazado la intencion no es verificable (notas de C y D).
  - P92C-01: apodo referenciado ('nio polla') que no puede interpretarse sin el contexto original.
  - No hubo en la muestra ningun caso con cita textual entre comillas; la ambiguedad observada es de referencias y reproduccion indirecta.
- options_considered:
  - C1 (contenido tal como se expresa): Evaluar el contenido del texto tal como aparece, incluida la cita, sin inferir la intencion del autor.
  - C2 (inferir intencion): Intentar distinguir denuncia de adhesion.
  - C3 (C1 + UTD): Aplicar C1 y usar UTD cuando el texto no permita saber si hay adhesion, denuncia o burla.
- status: UNRESOLVED
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: C3 (C1 + UTD).
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-05 (confirmar o ajustar)

### Q6 - Cuando corresponde UNABLE_TO_DETERMINE

- question_id: Q6
- issue: Cuando corresponde UNABLE_TO_DETERMINE
- official_rule: No aplica: UTD no existe en el formato oficial de la competencia; es un mecanismo del protocolo de anotacion, ya presente en la guia v1.
- observed_evidence:
  - Contexto insuficiente: P92C-01 (TR y AG), P92C-04 (HS), P92C-07 (HS y AG) - blanco o referencia no identificables.
  - Ambiguedad normativa: P92C-05, 09, 11, 13, 14, 15 (HS) - la definicion oficial no alcanza.
  - Practica observada: C uso UTD en 9 celdas y D en 11 (columnas de clase); A y B no usaron UTD en clases (registraron duda como flag de fila).
- options_considered:
  - U1 (solo contexto): UTD solo cuando falta contexto (lectura restrictiva de v1).
  - U2 (contexto + ambiguedad normativa): UTD cuando falte contexto o cuando la definicion oficial no permita decidir (practica de C y D en la calibracion).
  - U3 (forzar binario): No usar UTD; toda celda POSITIVE o NEGATIVE.
- status: UNRESOLVED
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: U2 (U1 como subconjunto), documentada como respaldo.
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-02, CH-04 (confirmar o ajustar)

### Q7 - Duda general del tweet frente a incertidumbre por etiqueta

- question_id: Q7
- issue: Duda general del tweet frente a incertidumbre por etiqueta
- official_rule: No aplica (mecanismo interno del protocolo).
- observed_evidence:
  - A y B registraron la duda con un flag binario de fila en notas (A: P92C-02, 04; B: P92C-01, 05, 13); su significado no era derivable de los documentos y requirio una aclaracion externa; ademas, el flag no indica que etiqueta se duda.
  - C y D usaron UTD por etiqueta + nota de motivo.
  - 4 de las 5 filas con flag coinciden con UTD de C o D; P92C-02 quedo con flag sin motivo y sin UTD asociado.
- options_considered:
  - R1 (UTD por etiqueta + nota): Practica de C y D: UTD en la etiqueta afectada con nota de motivo.
  - R2 (R1 + nota de fila): UTD por etiqueta + nota libre adicional cuando la fila entera sea dificil.
  - R3 (mantener flag binario): Conservar el mecanismo 0/1 de fila.
- status: UNRESOLVED
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: R2 (R1 + nota de fila) como respaldo.
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-04 (confirmar o ajustar)

### Q8 - Formato obligatorio de respuestas para la ronda principal

- question_id: Q8
- issue: Formato obligatorio de respuestas para la ronda principal
- official_rule: No aplica (formato del protocolo: tokens en las tres columnas; la guia v1 y el README ya lo exigian).
- observed_evidence:
  - A y B devolvieron 0/1 en las cuatro columnas (con BOM de Excel), pese a que el README y la guia exigian POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE; la interpretacion requirio una aclaracion explicita del responsable del proyecto (P92-T, ENCODING_CONFIRMED).
  - C y D devolvieron tokens + notas de texto libre.
  - La desviacion no invalido las respuestas (fueron interpretables), pero costo una ronda completa de verificacion.
- options_considered:
  - F1 (tokens exclusivos): POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE en las tres columnas; sin 0/1; guardar como CSV plano (sin reabrir en Excel si altera el formato).
  - F2 (tokens + aceptar 0/1): Aceptar ambos formatos.
- status: UNRESOLVED
- human_decision: (vacio - completar en la discusion)
- rationale: (vacio - completar en la discusion)
- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora documentada: F1 (ya vigente en v1); la confirmacion humana queda pendiente.
- decision_owner: (vacio - completar en la discusion)
- decision_date: (vacio - completar en la discusion)
- guideline_change: CH-01, CH-02 (confirmar o ajustar)

## 2. Correspondencia decision -> changelog CH-01 a CH-08 de la propuesta v2

| cambio | descripcion | preguntas relacionadas | decision humana asociada |
|---|---|---|---|
| CH-01 | Formato de respuesta (tokens obligatorios; 0/1 invalido) | Q8 | (vacio) |
| CH-02 | Nota de motivo obligatoria cuando se usa UTD | Q6, Q8 | (vacio) |
| CH-03 | Criterios operativos propuestos (odio/ofensa; agresivo/vulgar; tono) | Q1, Q3, Q4 | (vacio) |
| CH-04 | Duda de fila: UTD por etiqueta + nota (sin flag binario aparte) | Q6, Q7 | (vacio) |
| CH-05 | Citas, sarcasmo, ironia y contexto | Q4, Q5 | (vacio) |
| CH-06 | Ejemplos estructurales anonimizados | (derivado de Q1 y Q3) | (vacio) |
| CH-07 | Preguntas normativas formalizadas (consulta al organizador) | Q1, Q2, Q3 | (vacio) |
| CH-08 | Independencia HS / TR / AG reforzada | (sin pregunta directa; se mantiene) | (vacio) |

## 3. Decision de inicio de la anotacion principal (gate)

| campo | valor |
|---|---|
| decision_iniciar_anotacion_principal | (vacio: SI / NO / DIFERIDA) |
| version_de_guia_aplicable | (vacio: identificar version congelada) |
| sha256_guia_congelada | (vacio) |
| responsable_que_aprueba | (vacio) |
| fecha | (vacio) |

## 4. Procedimiento posterior a las decisiones (resumen)

1. Mantener v1 como artefacto historico (no se modifica).
2. Aplicar a una version nueva de la guia EXCLUSIVAMENTE las decisiones
   registradas aqui (la propuesta v2 no se congela por si sola).
3. Generar el changelog explicito de la version resultante y separar: reglas
   oficiales, reglas operativas aprobadas y preguntas todavia abiertas.
4. Calcular el SHA256 de la version resultante y registrarlo.
5. Si persiste una ambiguedad semantica material, mantenerla documentada y
   determinar si el piloto continua de forma exploratoria con UTD explicito.
6. Actualizar los paquetes principales a la version congelada (trazable: nuevo
   manifest con hashes) y re-verificar el gate antes de distribuir.

---

## NOTA ADITIVA P92-T-R (2026-10-12)

Donde este registro menciona "ENCODING_CONFIRMED" (evidencia de Q7 y Q8), leer el estado
corregido: ENCODING_UNRESOLVED al estandar estricto, pendiente de confirmacion directa
de A y B (ver reports/p92tr_encoding_provenance_audit.md y
reports/p92tr_ab_confirmation_request.md). La estructura de las preguntas no cambia;
solo se corrige la calificacion de esa evidencia.

---

## DECISION DEL PROPIETARIO DEL PROYECTO (P92-V, 2026-10-13): politica operativa v2 congelada

No consta una discusion de consenso entre los cuatro anotadores; estas decisiones fueron
adoptadas por el propietario del proyecto como POLITICA OPERATIVA INTERNA para el piloto,
no como definiciones oficiales. Registro por pregunta (detalle en
reports/p92v_operational_policy_decisions.md):

| pregunta | decision adoptada (operativa) | decision_owner | decision_date | status | cambios |
|---|---|---|---|---|---|
| Q1 | definicion oficial literal + UTD explicito; sin nueva definicion de odio | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-03, CH-07 |
| Q2 | no esperar aclaracion externa; NOT_REQUESTED / UNRESOLVED / AUTHORIZED | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-07 |
| Q3 | no equiparar vulgaridad, insulto y agresividad; UTD si no hay evidencia | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-03 |
| Q4 | contenido observable; sin inferir intencion; UTD si el tono decide y falta contexto | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-05, CH-03 |
| Q5 | texto completo; no atribuir la cita al autor; UTD si la funcion no es determinable | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-05 |
| Q6 | UTD por etiqueta (contexto o ambiguedad material); con nota; sin forzar binario | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-02, CH-04 |
| Q7 | dudas separadas: por etiqueta + nota general adicional (tweet_note) | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-04, CH-09 |
| Q8 | formato exclusivo POSITIVE/NEGATIVE/UNABLE_TO_DETERMINE; 0/1 no valido | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-01, CH-02 |

Consecuencias: guia congelada reports/p92_annotation_guidelines_v2_frozen.md
(PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2; sha256 sin linea de hash: 46be258ad47074405abce99e79ef831970d7b877e1987705de375432413117f8); paquetes v2
preparados en artifacts/human_annotation_v2/ (NO distribuidos). Las respuestas de
calibracion permanecen bajo la guia v1 y no se reinterpretan retroactivamente.
