# P92-U - Paquete final de discusion humana (autocontenido)

Para: los cuatro participantes de la calibracion (A, B, C y D) y los
responsables del proyecto. Fecha: 2026-10-12. Estado:
**HUMAN_DISCUSSION_PREPARED** (las decisiones se registran en
`reports/p92_guideline_decision_record.md`; esta fase no decide nada).

Este documento es la version final y autocontenida del paquete de discusion;
no reemplaza ni sobrescribe el paquete de P92-T
(`reports/p92_calibration_discussion_pack.md`, historico).

Reglas de la discusion:

- No se muestran etiquetas originales, probabilidades OOF, predicciones de V03,
  umbrales ni motivos de seleccion.
- No se decide aqui quien tiene razon; no hay adjudicacion automatica.
- La mayoria simple no resuelve por si sola una cuestion normativa.
- Resultado esperado: un registro de decisiones humanas y cuestiones
  pendientes, con la correspondencia frente a los cambios CH-01 a CH-08.
- Las notas citadas se transliteran a ASCII; los originales con acentos
  permanecen en los archivos de respuesta de cada persona.

## 1. Contexto y evidencia disponible (resumen)

- Calibracion: 15 tweets anotados de forma independiente por 4 personas con la
  guia v1; la codificacion 0/1 de A y B quedo confirmada (0 = NEGATIVE,
  1 = POSITIVE; en notas, 1 = dificil de determinar para cualquier clase).
- Acuerdo exacto (de 15 filas): HS 12/15, TR 14/15, AG 11/15 en el par C-D;
  kappa binario C-D = 1.000 en las tres etiquetas; coincidencia exacta de los 4
  en las tres etiquetas: 1/15.
- UTD por etiqueta (columnas de clase): HS C=7 D=8; TR C=1 D=0; AG C=1 D=3.
  A y B no usaron UTD en clases; registraron su duda como marca de fila
  (A: P92C-02, 04; B: P92C-01, 05, 13).
- La ambiguedad normativa dominante (odio vs. ofensa sexista individual) se
  concentra en 6 filas HS con UTD coincidente de C y D.
- Limitacion muestral: todos los numeros son de esta muestra de 15 tweets, con
  denominadores visibles; no se extrapolan a la muestra principal ni al dataset.

## 2. Agenda de decision (Q1 a Q8)

Estados usados: `UNRESOLVED` = decision humana pendiente;
`REQUIRES_GOVERNANCE_CLARIFICATION` = la definicion oficial no alcanza y se
requiere aclaracion del responsable de la competicion (o una decision humana
explicita de operar con la opcion conservadora); `DECIDED` = no usado en esta
fase (se actualizara en el registro). Ninguna pregunta esta decidida: el agente
no toma decisiones en nombre de los participantes.

### Q1. Distincion entre odio y ofensa sexista individual

**Definicion oficial disponible**: HS: "indica si hay discurso de odio (1 = si, 0 = no)" (docs/competition_rules.md). No existe criterio operativo, lista de grupos protegidos, ejemplos ni FAQ (verificado en fases previas).

**Evidencia observada en la calibracion**:
- 6/15 filas de la muestra con UTD coincidente de C y D en HS (P92C-05, 09, 11, 13, 14, 15); todas son ofensas o insultos sexistas dirigidos a personas concretas.
- D uso UTD con la misma duda normativa en P92C-03 y P92C-04; en 04 la nota de C registra la misma duda (odio vs. ofensa) aunque respondio POSITIVE.
- Ninguna de las notas menciona pertenencia a un grupo: la duda es individuo (persona concreta) frente a la categoria oficial 'discurso de odio'.
- La ambiguedad es sistematica y localizada, no ruido: dos anotadores independientes convergen en las mismas filas y el mismo motivo.

**Alternativas propuestas** (ventajas / riesgos):
- O1 (estricta): HS = POSITIVE solo cuando la hostilidad se dirige a alguien por su identidad o pertenencia a un grupo.
  - Ventajas: Lectura restrictiva y comun de 'discurso de odio'; decisiones reproducibles cuando el objetivo es un grupo; reduce el riesgo de llamar 'odio' a cualquier ofensa.
  - Riesgos: Puede sub-anotar agresiones identitarias a individuos (p. ej., una mujer insultada por ser mujer sin mencion de grupo); no hay respaldo documental explicito para restringir a grupos.
- O2 (amplia): HS = POSITIVE cuando el texto degrada o expresa hostilidad hacia una persona o grupo, aunque el motivo sea individual.
  - Ventajas: Trato uniforme de la hostilidad dirigida; menos UTD; decisiones mas simples.
  - Riesgos: Equipara 'ofensa' y 'odio', que el texto oficial trata como conceptos distintos ('discurso de odio' frente a 'agresivo'); puede sobre-extender la etiqueta.
- O3 (literal + UTD): Mantener la definicion oficial sin criterio operativo y usar UNABLE_TO_DETERMINE cuando no permita decidir.
  - Ventajas: No inventa norma; fiel al texto oficial; preserva la incertidumbre de forma trazable; es la unica opcion que no requiere una interpretacion no documentada.
  - Riesgos: Mas celdas UTD en la ronda principal; no produce etiquetas binarias para estas filas; traslada la decision al analisis posterior.

**Opcion conservadora documentada (NO aprobada)**: O3 (literal + UTD) como respaldo si la discusion no adopta una regla.

**Decision requerida**: Los responsables humanos (anotadores y responsables del proyecto) eligen O1, O2 u O3, o deciden elevar la pregunta (ver Q2).

**Estado**: `REQUIRES_GOVERNANCE_CLARIFICATION` | Cambios relacionados: CH-03, CH-07

### Q2. Solicitar aclaracion normativa al responsable de la competicion

**Definicion oficial disponible**: No aplica: no existe canal documentado de aclaraciones para anotadores; la pagina de la competencia no publica criterios operativos.

**Evidencia observada en la calibracion**:
- P91 dejo la gobernanza en revision (GOVERNANCE_REVIEW_REQUIRED) por la misma ausencia de criterios operativos.
- La calibracion confirmo que la duda es material (6/15 filas HS + 2 mas) y sistematica entre anotadores independientes.

**Alternativas propuestas** (ventajas / riesgos):
- S1 enviar consulta: Enviar una consulta escrita breve al organizador (redaccion propuesta en la seccion 5 de este paquete).
  - Ventajas: Unica via de resolver normativamente; deja constancia institucional; puede beneficiar a todos los equipos.
  - Riesgos: Depende de los tiempos del organizador; la respuesta puede no llegar antes de la ronda principal; el envio lo hacen los responsables (el agente no contacta a nadie).
- S2 no enviar; adoptar O3: No elevar la consulta y operar con la definicion literal + UTD.
  - Ventajas: Autonomia total; sin dependencia externa.
  - Riesgos: El vacio normativo queda como riesgo permanente del piloto; las decisiones no quedan respaldadas por el organizador.
- S3 enviar y continuar en paralelo: Enviar la consulta y comenzar la ronda principal con UTD, registrando que se anoto sin la aclaracion.
  - Ventajas: No bloquea la ronda principal; si llega la aclaracion, se aplica hacia adelante.
  - Riesgos: Convivencia de criterios si la respuesta llega tarde; requiere documentar el punto de corte (sin re-anotar lo ya hecho).

**Opcion conservadora documentada (NO aprobada)**: Ninguna especifica: la no-respuesta no cambia por si sola la guia; aplica la opcion conservadora de Q1/Q3.

**Decision requerida**: Los responsables deciden si se envia la consulta, quien la firma y cuando; la redaccion propuesta esta lista para revision.

**Estado**: `UNRESOLVED` | Cambios relacionados: CH-07

### Q3. Distinguir agresividad de vulgaridad

**Definicion oficial disponible**: AG: "indica si el texto es agresivo (1 = si, 0 = no)". Sin criterio operativo ni ejemplos.

**Evidencia observada en la calibracion**:
- P92C-06: D UTD en AG - generalizacion hostil contra un grupo sin insulto ni amenaza explicita ('la agresividad es discutible'); A, B y C respondieron NEGATIVE.
- P92C-13: C UTD en AG - insulto leve con posible tono de broma; duda agresivo vs. vulgar; D respondio POSITIVE.
- P92C-07: D UTD en AG - sarcasmo o cita sin contexto; agresividad no inequivoca.
- AG fue la etiqueta con menor coincidencia exacta de los 4 anotadores (6/15).

**Alternativas propuestas** (ventajas / riesgos):
- A1 (agresion dirigida): AG = POSITIVE si hay agresion dirigida (insulto, amenaza, orden hostil) hacia una persona o grupo; la vulgaridad no dirigida no basta.
  - Ventajas: Regla operativa clara para la mayoria de casos; coincide con el uso observado (los casos AG POSITIVE implican destinatario); reduce UTD por vulgaridad.
  - Riesgos: No cubre la generalizacion hostil sin insulto (caso 06); es una interpretacion, no una definicion oficial.
- A2 (amplia): Incluir tambien generalizaciones hostiles hacia grupos aunque no haya insulto dirigido.
  - Ventajas: Captura hostilidad grupal; menos UTD.
  - Riesgos: Se aleja del sentido comun de 'agresivo'; riesgo de solapamiento con HS y TR.
- A3 (literal + UTD): Sin criterio operativo; UTD cuando 'agresivo' no sea claro.
  - Ventajas: Sin interpretacion no documentada; incertidumbre trazable.
  - Riesgos: Mas UTD; decisiones menos consistentes entre anotadores.

**Opcion conservadora documentada (NO aprobada)**: A3 (literal + UTD) como respaldo.

**Decision requerida**: Los responsables humanos eligen A1, A2 u A3 (o elevan la parte no resoluble via Q2).

**Estado**: `REQUIRES_GOVERNANCE_CLARIFICATION` | Cambios relacionados: CH-03

### Q4. Tono de broma, ironia y sarcasmo

**Definicion oficial disponible**: Ninguna regla oficial (no hay instrucciones de casos limite).

**Evidencia observada en la calibracion**:
- P92C-13: 'podria ser broma entre conocidos' (nota de C) como razon de duda en HS y AG.
- P92C-07: sarcasmo o critica; 'De puta madre' interpretada como expresion idiomatica (nota de D).
- Ningun caso de la muestra produjo consenso de que la broma neutralice la hostilidad.

**Alternativas propuestas** (ventajas / riesgos):
- T1 (contenido expresado): Evaluar el contenido hostil expresado; el tono retorico (broma, ironia, sarcasmo) no decide por si solo.
  - Ventajas: Regla uniforme; evita que 'era broma' excuse cualquier hostilidad; coincide con el uso observado (los cuatro anotadores evaluaron contenido).
  - Riesgos: El sarcasmo puede invertir el significado; requiere lectura cuidadosa.
- T2 (tono neutraliza): Una broma entre conocidos neutraliza agresion u odio.
  - Ventajas: Refleja intuiciones sobre contexto social.
  - Riesgos: Imposible de verificar sin conocer la relacion entre las personas; abre la puerta a excusar hostilidad; sin respaldo documental.
- T3 (UTD si el tono es determinante): Usar UTD cuando el tono sea lo que decide el caso y no sea verificable.
  - Ventajas: Honesta con la incertidumbre; conservadora.
  - Riesgos: Mas UTD; conviene combinarla con T1 para no trasladar toda la carga a UTD.

**Opcion conservadora documentada (NO aprobada)**: T1 + T3 (contenido expresado; UTD cuando el tono sea determinante e inverificable).

**Decision requerida**: Los responsables humanos eligen la combinacion operativa y la registran.

**Estado**: `UNRESOLVED` | Cambios relacionados: CH-05, CH-03

### Q5. Citas y discurso reproducido

**Definicion oficial disponible**: Ninguna regla oficial.

**Evidencia observada en la calibracion**:
- P92C-07: critica o sarcasmo sobre descendientes de una persona descrita como 'genocida'; sin el contexto enlazado la intencion no es verificable (notas de C y D).
- P92C-01: apodo referenciado ('nio polla') que no puede interpretarse sin el contexto original.
- No hubo en la muestra ningun caso con cita textual entre comillas; la ambiguedad observada es de referencias y reproduccion indirecta.

**Alternativas propuestas** (ventajas / riesgos):
- C1 (contenido tal como se expresa): Evaluar el contenido del texto tal como aparece, incluida la cita, sin inferir la intencion del autor.
  - Ventajas: Reproducible; no requiere informacion externa.
  - Riesgos: Puede penalizar textos que citan para denunciar.
- C2 (inferir intencion): Intentar distinguir denuncia de adhesion.
  - Ventajas: Mas justo con textos de denuncia.
  - Riesgos: Requiere adivinar; sin contexto es especulacion; no verificable.
- C3 (C1 + UTD): Aplicar C1 y usar UTD cuando el texto no permita saber si hay adhesion, denuncia o burla.
  - Ventajas: Combina reproducibilidad con honestidad; coincide con la practica observada.
  - Riesgos: Mas UTD.

**Opcion conservadora documentada (NO aprobada)**: C3 (C1 + UTD).

**Decision requerida**: Los responsables humanos eligen la regla operativa y la registran.

**Estado**: `UNRESOLVED` | Cambios relacionados: CH-05

### Q6. Cuando corresponde UNABLE_TO_DETERMINE

**Definicion oficial disponible**: No aplica: UTD no existe en el formato oficial de la competencia; es un mecanismo del protocolo de anotacion, ya presente en la guia v1.

**Evidencia observada en la calibracion**:
- Contexto insuficiente: P92C-01 (TR y AG), P92C-04 (HS), P92C-07 (HS y AG) - blanco o referencia no identificables.
- Ambiguedad normativa: P92C-05, 09, 11, 13, 14, 15 (HS) - la definicion oficial no alcanza.
- Practica observada: C uso UTD en 9 celdas y D en 11 (columnas de clase); A y B no usaron UTD en clases (registraron duda como flag de fila).

**Alternativas propuestas** (ventajas / riesgos):
- U1 (solo contexto): UTD solo cuando falta contexto (lectura restrictiva de v1).
  - Ventajas: Acota el uso; mantiene el foco en el contexto.
  - Riesgos: Obliga a decidir casos normativamente indecidibles (contradice 'no inventar reglas'); ya resulto insuficiente en la practica.
- U2 (contexto + ambiguedad normativa): UTD cuando falte contexto o cuando la definicion oficial no permita decidir (practica de C y D en la calibracion).
  - Ventajas: Coherente con la evidencia; evita certezas simuladas; preserva trazabilidad.
  - Riesgos: Mas UTD; requiere que el analisis posterior trate la incertidumbre explicitamente.
- U3 (forzar binario): No usar UTD; toda celda POSITIVE o NEGATIVE.
  - Ventajas: Formato binario uniforme.
  - Riesgos: Simula certeza inexistente; contradice el protocolo; indefendible ante auditoria; descartada por P91 y P92 (no inventar reglas).

**Opcion conservadora documentada (NO aprobada)**: U2 (U1 como subconjunto), documentada como respaldo.

**Decision requerida**: Los responsables humanos confirman el alcance de UTD y lo registran.

**Estado**: `UNRESOLVED` | Cambios relacionados: CH-02, CH-04

### Q7. Duda general del tweet frente a incertidumbre por etiqueta

**Definicion oficial disponible**: No aplica (mecanismo interno del protocolo).

**Evidencia observada en la calibracion**:
- A y B registraron la duda con un flag binario de fila en notas (A: P92C-02, 04; B: P92C-01, 05, 13); su significado no era derivable de los documentos y requirio una aclaracion externa; ademas, el flag no indica que etiqueta se duda.
- C y D usaron UTD por etiqueta + nota de motivo.
- 4 de las 5 filas con flag coinciden con UTD de C o D; P92C-02 quedo con flag sin motivo y sin UTD asociado.

**Alternativas propuestas** (ventajas / riesgos):
- R1 (UTD por etiqueta + nota): Practica de C y D: UTD en la etiqueta afectada con nota de motivo.
  - Ventajas: Preciso (indica que etiqueta); consistente con la guia v1; trazable.
  - Riesgos: Si la fila entera es dificil, requiere marcar varias celdas.
- R2 (R1 + nota de fila): UTD por etiqueta + nota libre adicional cuando la fila entera sea dificil.
  - Ventajas: Captura ambos niveles; elimina la ambiguedad del flag binario.
  - Riesgos: Dos lugares para la misma informacion; requiere disciplina de uso.
- R3 (mantener flag binario): Conservar el mecanismo 0/1 de fila.
  - Ventajas: Formato compacto.
  - Riesgos: Demostrado ambiguo (su significado no era derivable de los documentos); no indica etiqueta; descartado por la evidencia.

**Opcion conservadora documentada (NO aprobada)**: R2 (R1 + nota de fila) como respaldo.

**Decision requerida**: Los responsables humanos confirman el mecanismo y lo registran.

**Estado**: `UNRESOLVED` | Cambios relacionados: CH-04

### Q8. Formato obligatorio de respuestas para la ronda principal

**Definicion oficial disponible**: No aplica (formato del protocolo: tokens en las tres columnas; la guia v1 y el README ya lo exigian).

**Evidencia observada en la calibracion**:
- A y B devolvieron 0/1 en las cuatro columnas (con BOM de Excel), pese a que el README y la guia exigian POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE; la interpretacion requirio una aclaracion explicita del responsable del proyecto (P92-T, ENCODING_CONFIRMED).
- C y D devolvieron tokens + notas de texto libre.
- La desviacion no invalido las respuestas (fueron interpretables), pero costo una ronda completa de verificacion.

**Alternativas propuestas** (ventajas / riesgos):
- F1 (tokens exclusivos): POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE en las tres columnas; sin 0/1; guardar como CSV plano (sin reabrir en Excel si altera el formato).
  - Ventajas: Esquema sin ambiguedad; ya exigido por v1; elimina el costo de verificacion.
  - Riesgos: Requiere cuidado al editar el CSV; se mitiga con instrucciones de guardado.
- F2 (tokens + aceptar 0/1): Aceptar ambos formatos.
  - Ventajas: Tolerante con herramientas de edicion.
  - Riesgos: La correspondencia 0/1 no es universalmente intuitiva (0 = NEGATIVE); ya exigio una aclaracion; descartada por la evidencia.

**Opcion conservadora documentada (NO aprobada)**: F1 (ya vigente en v1); la confirmacion humana queda pendiente.

**Decision requerida**: Los responsables humanos confirman F1 para ambos paquetes y lo registran.

**Estado**: `UNRESOLVED` | Cambios relacionados: CH-01, CH-02

## 3. Los seis casos HS con UTD coincidente de C y D (evidencia de Q1)

En las seis filas, C y D registraron UNABLE_TO_DETERMINE en HS con el mismo
motivo: la guia no define cuando una ofensa sexista individual constituye
discurso de odio. Textos (verbatim) y respuestas de los cuatro:

### P92C-05

Texto: "@juandeltoro08 T eres una pinche perra"

- A: NEGATIVE / NEGATIVE / POSITIVE (respuestas originales 0/0/1)
- B: NEGATIVE / NEGATIVE / NEGATIVE (respuestas originales 0/0/0; marca de fila: si)
- C: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insulto sexista directo; duda normativa odio vs. ofensa"
- Nota D: "Insulto sexista dirigido a una persona; la guia no distingue operativamente entre ofensa sexista individual y discurso de odio."

### P92C-09

Texto: "Tu no eres amiguera tu eres puta"

- A: NEGATIVE / NEGATIVE / POSITIVE (respuestas originales 0/0/1)
- B: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- C: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insulto sexista a una persona; duda normativa odio vs. ofensa"
- Nota D: "Insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa sexista individual constituye discurso de odio."

### P92C-11

Texto: "@lusiluu24 @muriRooster @SurgicalGolem @IKERABUSOS @chozasasas @Davicito07_CR ERES UNA PUTA GUARRA"

- A: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- B: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- C: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insulto sexista a una persona; duda normativa odio vs. ofensa"
- Nota D: "Insulto sexista dirigido a destinatarios identificados por menciones; la guia no establece si este tipo de ofensa individual constituye discurso de odio."

### P92C-13

Texto: "@vayakuadro pero t eres un poco zorra"

- A: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- B: NEGATIVE / NEGATIVE / NEGATIVE (respuestas originales 0/0/0; marca de fila: si)
- C: UNABLE_TO_DETERMINE / POSITIVE / UNABLE_TO_DETERMINE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insulto leve ('un poco zorra'), podria ser broma entre conocidos; duda agresivo vs. vulgar"
- Nota D: "Insulto sexista dirigido a una persona; la guia no distingue operativamente entre ofensa sexista individual y discurso de odio."

### P92C-14

Texto: "@isabellagiehl si la puta q te pario virgen de mierda no digas nada no opines callate nunca em tu vida probaste miche con seven up aa pero ron con coca cola si babieca balurdaaaa"

- A: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- B: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- C: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insultos fuertes dirigidos a una persona; duda normativa odio vs. ofensa"
- Nota D: "Ataque personal con insultos sexistas y vulgares y ordenes hostiles; la guia no ofrece un criterio operativo para decidir si constituye discurso de odio."

### P92C-15

Texto: "@daniellemx_ Te mereces mas la portadabde la revista playboy.. Por zorra.. Y mentirosa.."

- A: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- B: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- C: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insulto sexista y humillacion ('Por zorra'); duda normativa odio vs. ofensa"
- Nota D: "Insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa sexista individual constituye discurso de odio."

Pregunta para estas seis filas (Q1): debe anotarse HS = POSITIVE cuando el
insulto sexista se dirige a una persona concreta? Cual opcion (O1, O2 u O3)
describe mejor la definicion oficial? Si la discusion concluye que la norma no
alcanza, la pregunta se eleva al organizador (Q2).

## 4. Casos secundarios que precisan la guia

### P92C-01 - TR (C: UTD) y AG (D: UTD) - contexto insuficiente (relacionado con Q6)

Texto: "tu conoces al nio polla? y no hablo de jordi  xd estais tocaisimos quin a ver https://t.co/UsPceGCxUl"

- A: NEGATIVE / NEGATIVE / NEGATIVE (respuestas originales 0/0/0)
- B: NEGATIVE / NEGATIVE / NEGATIVE (respuestas originales 0/0/0; marca de fila: si)
- C: NEGATIVE / UNABLE_TO_DETERMINE / NEGATIVE
- D: NEGATIVE / POSITIVE / UNABLE_TO_DETERMINE
- Nota C: "Apodo 'nio polla' sin contexto; tono burlon ('estais tocaisimos'), no se ve a quien apunta ni si hay agresion real"
- Nota D: "La referencia 'nio polla' no es clara sin contexto; 'estais tocaisimos' se dirige al interlocutor, pero puede ser burla coloquial o insulto agresivo."
- B: marco la fila como dificil.

### P92C-03 - HS (D: UTD) - misma duda normativa que las seis (relacionado con Q1)

Texto: "@lortegadiaz Perra maldita t eres la primera en la@hoguera por crmenes de lesa humanidad perra maldita"

- A: NEGATIVE / POSITIVE / POSITIVE (respuestas originales 0/1/1)
- B: POSITIVE / POSITIVE / POSITIVE (respuestas originales 1/1/1)
- C: POSITIVE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Insulto sexista ('perra maldita') mas deseo de que la quemen en la hoguera"
- Nota D: "Insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa sexista individual constituye discurso de odio. La alusion a la hoguera anade hostilidad violenta."

### P92C-04 - HS (D: UTD) - contexto insuficiente + duda normativa (relacionado con Q1, Q6)

Texto: "@marialozano777 Cuidadn que llamar hijo puta a un hijo puta te condenan y no lo digo por nadie que no merezca una guillotina. Al revs. Arriba la cuchilla y debajo la rugosidad oxidada. Sucio. Lo merece el gnomo."

- A: NEGATIVE / NEGATIVE / NEGATIVE (respuestas originales 0/0/0; marca de fila: si)
- B: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- C: POSITIVE / POSITIVE / POSITIVE
- D: UNABLE_TO_DETERMINE / POSITIVE / POSITIVE
- Nota C: "Reclama guillotina/violencia contra una persona; el odio no es por pertenencia a un grupo, duda normativa odio vs. ofensa"
- Nota D: "'El gnomo' no puede identificarse sin contexto; hay hostilidad y una referencia explicita a la guillotina, pero no queda claro si el texto constituye discurso de odio."
- A: marco la fila como dificil para cualquier clase (motivo no especificado).

### P92C-06 - AG (D: UTD) - generalizacion hostil sin insulto (relacionado con Q3)

Texto: "Los inmigrantes vienen a quitarnos el trabajo a los espaoles https://t.co/DNlA7pYb5E"

- A: POSITIVE / POSITIVE / NEGATIVE (respuestas originales 1/1/0)
- B: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- C: POSITIVE / POSITIVE / NEGATIVE
- D: POSITIVE / POSITIVE / UNABLE_TO_DETERMINE
- Nota C: "Generalizacion xenofoba contra inmigrantes; el tono no es agresivo"
- Nota D: "Generalizacion negativa contra inmigrantes que los presenta como amenaza laboral. La agresividad es discutible porque no contiene insulto ni amenaza explicita."

### P92C-07 - HS (C: UTD) y AG (D: UTD) - sarcasmo/cita sin contexto (relacionado con Q4, Q5, Q6)

Texto: "@NexusSep @ximicomics @Barbijaputa Pues aki tenemos a los nietos y tataranietos de un genocida viviendo de puta madre"

- A: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- B: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- C: UNABLE_TO_DETERMINE / POSITIVE / NEGATIVE
- D: NEGATIVE / POSITIVE / UNABLE_TO_DETERMINE
- Nota C: "Hostil hacia descendientes de alguien (falta contexto de quien es el 'genocida'); tono mas acusatorio que agresivo"
- Nota D: "Critica o sarcasmo sobre descendientes de una persona descrita como genocida; sin el contexto enlazado, la agresividad no es inequivoca. 'De puta madre' parece funcionar como expresion idiomatica."

### P92C-02 - flag de fila de A (notas=1, motivo no especificado) (relacionado con Q7)

Texto: "T los traes por tus santos ovarios y luego pretendes cargrselos a otros. Anda y vete a la m...  https://t.co/RmyBnjGnJn"

- A: NEGATIVE / NEGATIVE / NEGATIVE (respuestas originales 0/0/0; marca de fila: si)
- B: NEGATIVE / POSITIVE / NEGATIVE (respuestas originales 0/1/0)
- C: NEGATIVE / POSITIVE / POSITIVE
- D: NEGATIVE / POSITIVE / POSITIVE
- Nota C: "Dirigido a una persona ('t los traes'); 'vete a la m...' es agresivo; 'santos ovarios' es una expresion coloquial, no basta para odio"
- A: marco la fila como dificil para cualquier clase (motivo no especificado).

## 5. Informacion normativa faltante y consulta propuesta (Q2)

Falta, como minimo: (1) criterio para distinguir ofensa sexista individual de
discurso de odio; (2) criterio agresivo frente a vulgar y tratamiento de la
generalizacion hostil; (3) tratamiento de citas, sarcasmo y discurso reproducido;
(4) relacion formal (si existe) entre HS y TR, y entre HS y AG.

Redaccion propuesta de la consulta (para revision de los responsables; NO
enviada por el agente):

```text
Estimado/a responsable de la competencia Tweet Toxic 2026 PUJ:

Durante un piloto de calibracion de anotacion (4 anotadores independientes, 15
textos) encontramos una ambiguedad sistematica en las definiciones oficiales.
Agradeceriamos una aclaracion sobre:

1. HS: cuando un insulto o agresion sexista se dirige a una persona concreta por
   su identidad, sin referirse a un grupo, debe anotarse HS = 1, HS = 0, o no
   existe criterio?
2. AG: como debe interpretarse "agresivo" frente a "vulgar" (por ejemplo, un
   insulto leve o lenguaje soez sin agresion dirigida)? Las generalizaciones
   hostiles hacia un grupo sin insulto explicito cuentan como AG?

Las dudas quedaron registradas sin forzar decisiones. Gracias.
```

## 6. Como se registran las decisiones

Cada pregunta tiene una entrada en `reports/p92_guideline_decision_record.md`
con los campos: question_id, issue, official_rule, observed_evidence,
options_considered, human_decision, rationale, remaining_uncertainty,
decision_owner, decision_date, guideline_change. Los campos de decision humana
estan vacios y los completan las personas responsables durante la discusion.

## 7. Que NO hace este paquete

- No decide ninguna etiqueta ni cuestion normativa por si mismo.
- No muestra etiquetas originales, probabilidades OOF, predicciones, umbrales ni
  motivos de seleccion de los casos.
- No contacta a los anotadores ni al organizador.
- No reescribe los resultados de la calibracion: siguen interpretados bajo la guia
  v1; cualquier guia posterior no se aplica retroactivamente sin documentarlo.

---

## NOTA ADITIVA P92-T-R (2026-10-12): estado de la codificacion A/B

Donde este paquete menciona que la codificacion 0/1 de A y B "quedo confirmada", debe
leerse con el estado corregido: ENCODING_UNRESOLVED al estandar estricto (instruccion
del responsable registrada en la sesion pero sin confirmacion directa de A/B, sin
persistencia previa en el proyecto y sin enumeracion literal de la direccion del mapeo).
Las respuestas de C y D y los valores crudos de A y B siguen siendo validos; las
estadisticas que dependen de la codificacion de A/B son condicionales. Ver
reports/p92tr_encoding_provenance_audit.md y la solicitud de confirmacion
(reports/p92tr_ab_confirmation_request.md; no enviada).
