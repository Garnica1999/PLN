# P92 - Paquete de discusion de la ronda de calibracion

Para: los cuatro participantes de la calibracion (A, B, C y D).
Fecha: 2026-10-12. Estado: HUMAN_DISCUSSION_PENDING.

La primera ronda independiente ya cerro: cada persona anoto los mismos 15 textos
sin comunicarse con las demas. Esta discusion es la etapa posterior prevista en el
protocolo. Reglas de la discusion:

- No se muestran etiquetas originales del dataset ni informacion de modelos.
- No se decide aqui quien tiene razon; no hay adjudicacion automatica.
- La mayoria simple no resuelve por si sola una cuestion normativa.
- Resultado esperado: registro de decisiones humanas y cuestiones pendientes.
- Las notas citadas se transliteran a ASCII; los originales con acentos estan en
  los archivos de respuesta de cada persona.

## 1. Resumen de los desacuerdos observados

Acuerdo exacto entre cada par (de 15 filas):

| etiqueta | A-B | A-C | A-D | B-C | B-D | C-D |
|---|---|---|---|---|---|---|
| HS | 10/15 | 5/15 | 6/15 | 6/15 | 6/15 | 12/15 |
| TR | 11/15 | 9/15 | 9/15 | 11/15 | 11/15 | 14/15 |
| AG | 12/15 | 11/15 | 8/15 | 10/15 | 7/15 | 11/15 |

- Las tres etiquetas juntas coinciden exactamente en los 4 anotadores en 1/15 filas.
- Uso de UNABLE_TO_DETERMINE: HS (C: 7/15, D: 8/15); TR (C: 1/15); AG (C: 1/15,
  D: 3/15). A y B registraron su duda como marca de fila en notas (A: 2 filas, B: 3).
- Donde el desacuerdo es binario puro (sin UTD), C y D coinciden en todas las
  celdas binarias de las tres etiquetas.

## 2. Las seis filas HS con incertidumbre coincidente de C y D

En las seis, C y D registraron UNABLE_TO_DETERMINE en HS con el mismo motivo: la
guia no define cuando una ofensa sexista individual dirigida a una persona
constituye discurso de odio. Textos (verbatim) y notas:

### P92C-05

Texto: "@juandeltoro08 T eres una pinche perra"

- C (HS = UNABLE_TO_DETERMINE): "Insulto sexista directo; duda normativa odio vs. ofensa"
- D (HS = UNABLE_TO_DETERMINE): "Insulto sexista dirigido a una persona; la guia no distingue operativamente entre ofensa sexista individual y discurso de odio."
- A: respuestas 0/0/0
- B: respuestas 0/0/0 y marca de fila

### P92C-09

Texto: "Tu no eres amiguera tu eres puta"

- C (HS = UNABLE_TO_DETERMINE): "Insulto sexista a una persona; duda normativa odio vs. ofensa"
- D (HS = UNABLE_TO_DETERMINE): "Insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa sexista individual constituye discurso de odio."
- A: respuestas 0/0/0
- B: respuestas 1/1/1

### P92C-11

Texto: "@lusiluu24 @muriRooster @SurgicalGolem @IKERABUSOS @chozasasas @Davicito07_CR ERES UNA PUTA GUARRA"

- C (HS = UNABLE_TO_DETERMINE): "Insulto sexista a una persona; duda normativa odio vs. ofensa"
- D (HS = UNABLE_TO_DETERMINE): "Insulto sexista dirigido a destinatarios identificados por menciones; la guia no establece si este tipo de ofensa individual constituye discurso de odio."
- A: respuestas 0/0/0
- B: respuestas 1/1/1

### P92C-13

Texto: "@vayakuadro pero t eres un poco zorra"

- C (HS = UNABLE_TO_DETERMINE): "Insulto leve ('un poco zorra'), podria ser broma entre conocidos; duda agresivo vs. vulgar"
- D (HS = UNABLE_TO_DETERMINE): "Insulto sexista dirigido a una persona; la guia no distingue operativamente entre ofensa sexista individual y discurso de odio."
- A: respuestas 0/0/0
- B: respuestas 0/0/0 y marca de fila

### P92C-14

Texto: "@isabellagiehl si la puta q te pario virgen de mierda no digas nada no opines callate nunca em tu vida probaste miche con seven up aa pero ron con coca cola si babieca balurdaaaa"

- C (HS = UNABLE_TO_DETERMINE): "Insultos fuertes dirigidos a una persona; duda normativa odio vs. ofensa"
- D (HS = UNABLE_TO_DETERMINE): "Ataque personal con insultos sexistas y vulgares y ordenes hostiles; la guia no ofrece un criterio operativo para decidir si constituye discurso de odio."
- A: respuestas 0/0/0
- B: respuestas 1/1/1

### P92C-15

Texto: "@daniellemx_ Te mereces mas la portadabde la revista playboy.. Por zorra.. Y mentirosa.."

- C (HS = UNABLE_TO_DETERMINE): "Insulto sexista y humillacion ('Por zorra'); duda normativa odio vs. ofensa"
- D (HS = UNABLE_TO_DETERMINE): "Insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa sexista individual constituye discurso de odio."
- A: respuestas 0/0/0
- B: respuestas 1/1/1

Pregunta para estas seis filas: debe anotarse HS = POSITIVE cuando el insulto
sexista se dirige a una persona concreta? Cual de las opciones de la seccion 6
(o ninguna) describe mejor la definicion oficial? Si la discusion concluye que la
norma no alcanza, la pregunta se eleva al organizador.

## 3. Casos adicionales (AG, TR, contexto y marca de fila)

### P92C-01 - TR (C: UTD) y AG (D: UTD) (INSUFFICIENT_CONTEXT)

Texto: "tu conoces al nio polla? y no hablo de jordi  xd estais tocaisimos quin a ver https://t.co/UsPceGCxUl"

- C: "Apodo 'nio polla' sin contexto; tono burlon ('estais tocaisimos'), no se ve a quien apunta ni si hay agresion real"
- D: "La referencia "nio polla" no es clara sin contexto; "estais tocaisimos" se dirige al interlocutor, pero puede ser burla coloquial o insulto agresivo."
- B: marco la fila como dificil.

### P92C-03 - HS (D: UTD; misma duda normativa que las seis) (HATE_VS_OFFENSE)

Texto: "@lortegadiaz Perra maldita t eres la primera en la@hoguera por crmenes de lesa humanidad perra maldita"

- C: "Insulto sexista ('perra maldita') mas deseo de que la quemen en la hoguera"
- D: "Insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa sexista individual constituye discurso de odio. La alusion a la hoguera anade hostilidad violenta."

### P92C-04 - HS (D: UTD; contexto + duda normativa) (HATE_VS_OFFENSE; INSUFFICIENT_CONTEXT)

Texto: "@marialozano777 Cuidadn que llamar hijo puta a un hijo puta te condenan y no lo digo por nadie que no merezca una guillotina. Al revs. Arriba la cuchilla y debajo la rugosidad oxidada. Sucio. Lo merece el gnomo."

- C: "Reclama guillotina/violencia contra una persona; el odio no es por pertenencia a un grupo, duda normativa odio vs. ofensa"
- D: ""El gnomo" no puede identificarse sin contexto; hay hostilidad y una referencia explicita a la guillotina, pero no queda claro si el texto constituye discurso de odio."
- A: marco la fila como dificil para cualquier clase.

### P92C-06 - AG (D: UTD) (INSULT_VS_AGGRESSION)

Texto: "Los inmigrantes vienen a quitarnos el trabajo a los espaoles https://t.co/DNlA7pYb5E"

- C: "Generalizacion xenofoba contra inmigrantes; el tono no es agresivo"
- D: "Generalizacion negativa contra inmigrantes que los presenta como amenaza laboral. La agresividad es discutible porque no contiene insulto ni amenaza explicita."

### P92C-07 - HS (C: UTD) y AG (D: UTD) (INSUFFICIENT_CONTEXT)

Texto: "@NexusSep @ximicomics @Barbijaputa Pues aki tenemos a los nietos y tataranietos de un genocida viviendo de puta madre"

- C: "Hostil hacia descendientes de alguien (falta contexto de quien es el 'genocida'); tono mas acusatorio que agresivo"
- D: "Critica o sarcasmo sobre descendientes de una persona descrita como genocida; sin el contexto enlazado, la agresividad no es inequivoca. "De puta madre" parece funcionar como expresion idiomatica."

### P92C-13 - AG (C: UTD) (INSULT_VS_AGGRESSION)

Texto: "@vayakuadro pero t eres un poco zorra"

- C: "Insulto leve ('un poco zorra'), podria ser broma entre conocidos; duda agresivo vs. vulgar"
- D: "Insulto sexista dirigido a una persona; la guia no distingue operativamente entre ofensa sexista individual y discurso de odio."
- B: marco la fila como dificil.

### P92C-02 - marca de fila de A (notas=1, motivo no especificado) (OTHER)

Texto: "T los traes por tus santos ovarios y luego pretendes cargrselos a otros. Anda y vete a la m...  https://t.co/RmyBnjGnJn"

- C: "Dirigido a una persona ('t los traes'); 'vete a la m...' es agresivo; 'santos ovarios' es una expresion coloquial, no basta para odio"
- A: marco la fila como dificil para cualquier clase.

## 4. Preguntas operativas por resolver en la discusion

- Q1. Frontera odio vs. ofensa: se adopta O1 (estricta), O2 (amplia) u O3
  (literal + UTD)? Con que fundamento dentro de la definicion oficial?
- Q2. Se eleva la pregunta normativa al organizador? (redaccion propuesta en la
  seccion 7.)
- Q3. Frontera agresivo vs. vulgar: basta la agresion dirigida para AG = POSITIVE?
  Cuenta como AG una generalizacion hostil sin insulto?
- Q4. Tono de broma o registro coloquial: como afecta cada etiqueta?
- Q5. Citas, sarcasmo y discurso reproducido: que regla operativa se adopta?
- Q6. Contexto insuficiente: se confirma UTD + nota (no adivinar)?
- Q7. Formato de la ronda principal: se confirma el uso exclusivo de
  POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE (los 0/1 quedan invalidos)?
- Q8. Duda de fila: se adopta marcar UTD en las etiquetas afectadas + nota?

## 5. Opciones alternativas de definicion (cuando hay mas de una lectura)

- HS (odio vs. ofensa): O1 estricta / O2 amplia / O3 literal + UTD.
- AG (agresividad): A1 = agresion dirigida; A2 = incluir tambien generalizaciones
  hostiles sin insulto.
- TR: sin alternativas documentadas; se mantiene la definicion oficial.

## 6. Informacion normativa que sigue faltando

1. Criterio para distinguir ofensa sexista individual de discurso de odio.
2. Criterio para distinguir agresivo de vulgar y para la generalizacion hostil.
3. Tratamiento de citas, sarcasmo y discurso reproducido.
4. Relacion formal (si existe) entre HS y TR, y entre HS y AG.

Pregunta propuesta para el organizador: "Cuando un insulto o agresion sexista
se dirige a una persona concreta por su identidad, sin referirse a un grupo, debe
anotarse HS = 1, HS = 0, o no existe criterio?".

## 7. Registro de decisiones (plantilla para completar en la discusion)

| pregunta | decision | fundamento | fecha | estado |
|---|---|---|---|---|
| Q1 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q2 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q3 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q4 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q5 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q6 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q7 | (pendiente) | (pendiente) | (pendiente) | pendiente |
| Q8 | (pendiente) | (pendiente) | (pendiente) | pendiente |

Cuestiones que queden sin decision se registran como pendientes y, si son
normativas, se elevan al organizador antes de congelar la guia.

## 8. Que NO hace este paquete

- No muestra etiquetas originales del dataset ni motivos de seleccion de los casos.
- No muestra predicciones, probabilidades, umbrales ni informacion de modelos.
- No decide automaticamente ninguna etiqueta ni senala a nadie como incorrecto.
- No contacta a los anotadores ni al organizador por si mismo.
