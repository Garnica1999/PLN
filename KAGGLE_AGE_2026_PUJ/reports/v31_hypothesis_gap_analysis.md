# V31+ - Gap analysis de las familias de representacion

**Baseline congelado:** V30 = 0.6134471 corrected_cv, LB 0,63302. El 0,6130051 que se publico en V30 no era reproducible; el operativo es 0,6134471 y es el que se envio y el que casa con el leaderboard.

## La distincion que manda

Una familia esta **cerrada** si el historial la cerro por equivalencia funcional, aunque el nombre suene nuevo. Y una esta **abierta** si el mecanismo no esta implementado en ningun sitio, aunque el nombre suene viejo.

## Familias

| ID | Track | Familia | Probada antes | Cerrada | Prioridad | Formulacion nueva |
|---|---|---|---|---|---|---|
| F01 | A | URL: presencia y estructura superficial | A1-A3 | **OPEN** | 1 | taxonomia de URL como canal separado, y comoplaceholder estructural |
| F11 | I | familias de modelo nuevas | LightGBM, MLP, lineal, NB-SVM, BM25, kNN, ExtraTrees, SGD | **OPEN** | 1 | solo ejecutar un modelo nuevo si la representacion nueva aporta senal demostrablemente distinta |
| F13 | N | integracion en el meta-espacio de V30 | integraciones previas | **OPEN** | 1 | anadir solo la representacion que ya tenga evidencia independiente, con controles de atribucion |
| F03 | B | EDA lexica basica | advanced_* y fase03 | **OPEN** | 2 | perfil composicional por clase: ratios, cuantiles y enriquecimiento de familias funcionales |
| F06 | D | ortografia y estilo de escritura | features de estilo existentes | **OPEN** | 2 | elongacion, identidad del caracter repetido, bloques caps, repeticiones de puntuacion |
| F10 | H | representaciones sparse nuevas | word y char ngrams | **CLOSED** | 2 | - |
| F12 | J | mineria linguistica guiada por errores | error analysis de V12-V22 | **OPEN** | 2 | perfil de los errores de alta confianza de V30 por dimension linguistica |
| F05 | C | verbos y tiempo verbal | no medida como composicion | **OPEN** | 3 | verbos auxiliares, imperativos, marcas de pasado/presente/futuro |
| F07 | E | discurso y pragmatica | no medida | **OPEN** | 3 | comillas, direccion directa, preguntas, risas, intensificadores |
| F08 | E | estructura de citas y mencion | posicion de mencion | **OPEN** | 4 | posicion de la cita y de la mencion como categoria combinatoria, no como coordenada |
| F09 | F | canales por posicion | posicion de keywords | **CLOSED** | 5 | - |
| F04 | B | palabras funcionales: pronombres | v10 y familias de pronouns | **CLOSED** | 6 | - |
| F02 | A | URL: recuperacion del contenido del tweet | A5-A8 | **QUARANTINED** | 9 | - |

## Las que quedan abiertas, con su mecanismo esperado

### F01 · URL: presencia y estructura superficial

- **Historial:** nunca medida: no hay columna de URL en el inventario de 279 features
- **Por que no es un cambio de hiperparametro:** taxonomia de URL como canal separado, y comoplaceholder estructural
- **Mecanismo esperado:** si el modelo usa la identidad textual de la URL en vez de su estructura, un canal de dominio y categoria aporta lo que la superficie no expresa

### F11 · familias de modelo nuevas

- **Historial:** V29 demostro que la capacidad del meta-modelo si es un cuello
- **Por que no es un cambio de hiperparametro:** solo ejecutar un modelo nuevo si la representacion nueva aporta senal demostrablemente distinta
- **Mecanismo esperado:** primero controles lineales, despues no lineal

### F13 · integracion en el meta-espacio de V30

- **Historial:** V28 mostro que anadir al espacio de V30 sin senal nueva no rinde
- **Por que no es un cambio de hiperparametro:** anadir solo la representacion que ya tenga evidencia independiente, con controles de atribucion
- **Mecanismo esperado:** complementariedad real frente a capacidad extra

### F03 · EDA lexica basica

- **Historial:** hay auditorias lexicales previas, pero orientadas a seleccionar features, no a caracter composicional por edad
- **Por que no es un cambio de hiperparametro:** perfil composicional por clase: ratios, cuantiles y enriquecimiento de familias funcionales
- **Mecanismo esperado:** la edad puede manifestarse en como se compone el texto mas que en que se dice

### F06 · ortografia y estilo de escritura

- **Historial:** si: hay 87 features estructuradas que incluyen longitud, mayusculas y ratios de puntuacion
- **Por que no es un cambio de hiperparametro:** elongacion, identidad del caracter repetido, bloques caps, repeticiones de puntuacion
- **Mecanismo esperado:** la ortografia de tecleo es una huella fuerte de edad y es totalmente determinista

### F12 · mineria linguistica guiada por errores

- **Historial:** si, hay analisis de error extensos
- **Por que no es un cambio de hiperparametro:** perfil de los errores de alta confianza de V30 por dimension linguistica
- **Mecanismo esperado:** un error con alta confianza apunta a una limitacion de representacion, no de umbral

### F05 · verbos y tiempo verbal

- **Historial:** no
- **Por que no es un cambio de hiperparametro:** verbos auxiliares, imperativos, marcas de pasado/presente/futuro
- **Mecanismo esperado:** el uso del auxiliar y la marca temporal varia mucho entre generaciones y es determinista

### F07 · discurso y pragmatica

- **Historial:** no
- **Por que no es un cambio de hiperparametro:** comillas, direccion directa, preguntas, risas, intensificadores
- **Mecanismo esperado:** los marcadores discursivos cambian de forma sistematica entre generaciones

### F08 · estructura de citas y mencion

- **Historial:** parcial: hay features de posicion de mencion en el bloque estructurado
- **Por que no es un cambio de hiperparametro:** posicion de la cita y de la mencion como categoria combinatoria, no como coordenada
- **Mecanismo esperado:** la estructura del turno de habla es generacional

## La familia en cuarentena

**F02 · URL: recuperacion del contenido del tweet** queda en **QUARANTINED**.

- las reglas de la compilacion no estan disponibles: sin el texto no se puede establecer que sea legal, y recuperarlo puede ser informacion externa o fuga de metadatos

No se implementa, no se hace ninguna llamada de red y ningun candidato que dependa de contenido recuperado puede llegar a una submission. El resto del Track A, que usa solo el texto del train, no se ve afectado.

## Lo que ya no se puede reabrir

| Familia | Decision que la cerro |
|---|---|
| feature selection supervisada | V25, D-060, -0,000629 |
| escalado afil de la evidencia | V26-01, D-061, -0,000001 |
| transformacion percentil | V27-05, MARGINAL +0,000272 |
| fiabilidad por modelo y clase | V27-01, H12, delta 0 |
| restricciones monotonas | V27-02, H11, -0,000355 |
| reponder blends equivalentes | V27, H14, equivalencia funcional |
| calibracion | H01, D-029 |
| ponderacion de modelos | H04 |
| apilado de segundo nivel | H06 |
| familias ordinales | H10, CORAL, D-058 |
| bagging del meta-modelo | V28-02, A, -0,000312 |
| descomposicion pairwise | V28-03, C, -0,067553 |
| geometria del consenso | V28, B, +0,0001 historico |
| fusion de evidencia | V28, D, equivalente a H04+H06 |
| incertidumbre explicita | V28, E, duplicacion de B |
| subespacio aleatorio | V28, F, ya en colsample_bytree |
| capacidad del meta-modelo | V29 y V30, +0,002461 y luego equivalente |
| diversidad de modelos historicos | V29, 0 de 47 con rescate neto positivo |

Y una regla de la propia fase: **cualquier familia que solo cambie `C`, `seed`, `threshold` o `max_features` esta cerrada por V25**, y no se reabre.

## Lo que dice la URL, en diagnostico

| Medida | Valor |
|---|---|
| Prevalencia de URL en train | **40.20%** (68,075 filas) |
| Twitter status | 0 |
| Solo URL | 4,022 |
| Texto + URL | 64,053 |
| Multiples URL | 5,388 |
| Acortadores | 54 |
| URL externa | 1,917 |
| URL con query string | 38 |
| Longitud media del texto | 90.1 |
| Longitud media con URL | 94.8 |
| Longitud media sin URL | 87.0 |
| Rango de enriquecimiento por clase | de 0.745 a 1.244 |

**Esto es diagnostico y no causal.** Que la URL este distribuida por edad no dice si el modelo la esta leyendo. La unica forma de saberlo es el experimento de enmascararla, que es A4.
