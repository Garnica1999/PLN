# P32+ - P01, gap analysis de las hipotesis

**Baseline:** V30 = 0.6134471, leaderboard 0.63302. **Ruido de entrenamiento medido en P00:** 0.0008876, que es la escala contra la que hay que comparar cualquier delta de esta fase.

## El hallazgo de este analysis

Leyendo `experiments.csv` en vez de fiarme de la memoria, aparecieron **seis precedentes directos** que P32+ iba a reabrir sin saberlo. Cinco de ellos son la misma hipotesis con otro nombre y otro baseline:

| Precedente | Que midio | Resultado | Decision |
|---|---|---|---|
| **PH39-FUSION-LEX** (D-039) | fundir un **bloque lexico** en V12 contra un control de permutacion | +0,0001, y el control daba **+0,0001** | CERRADA |
| **PH15-CALIB** (D-029) | promedio en **log-probabilidades** frente a probabilidades | 0,6007 (**+0,0003**, 3/5) | CERRADA |
| **PH32-CORRECCION** (D-037) | 14 capas de correccion, incluida **calibracion por confianza** | mejor +0,0002, 3/5 | CERRADA |
| **PH33-CORRECTOR** (D-037) | un **corrector minimo** sobre P(V12) con 5 variantes | **ninguna positiva** | CERRADA |
| **V28-03** | descomposicion **pairwise** en pares adyacentes, sustituyendo al multiclase | **-0,067553**, 0/5, neto -10.996 | C CLOSED |
| **V27-01** (H12) | fiabilidad por miembro y clase | delta **exactamente 0,0**, 0 filas cambian | H12 CLOSED |

## Lo que eso significa, sin adornos

`PH39-FUSION-LEX` es casi exactamente la tesis central de P32+: *fundir una fuente lexical en el baseline y compararla contra un control de permutacion*. Dio **+0,0001 con un control de +0,0001**, es decir **cero aporte sobre el azar**.

La mecanica de P32+ no es la misma en un punto importante: V28-03 **sustituyo** el clasificador multiclase por cinco binarios y por eso destruyo la estructura de la clase completa, mientras que P32 propone **anadir** una correccion de margen conservando la probabilidad de V30 para las demas clases. Eso si es genuinamente distinto. Y V12 no es V30: la fuente de P32 es mejor (+0,006820 en banco) y la base es mas fuerte.

**El priori honesto de esta fase es bajo.** Dos hipotesis son genuinamente nuevas, una es casi nueva, y las demas tienen precedentes medidos que las refutan. Se ejecutan porque son baratas y porque el brief manda no detenerse en el primer negativo, pero la expectativa razonable es que la fase **cierre lineas**, no que produzca un finalista.

## Clasificacion completa

| ID | Track | Hipotesis | Clasificacion | Precedente | Que se ejecuta | Prioridad |
|---|---|---|---|---|---|---|
| H-P32-04 | P6/P6.1 | Correccion pairwise localizada entre bandas adyacentes | **PARTIAL_PRECEDENT** | V28-03 (tecnica C) | SI, y es la hipotesis mas genuinamente nueva de la fase. Con el test de rescate por par que exig | 1 |
| H-P32-05 | P8 | Residualizacion: quitar al candidato la parte que replica a V30 | **GENUINELY_NEW** | ninguno | SI, con prioridad alta: es la formulacion mas limpia de la tesis central de P32 | 1 |
| H-P32-02 | P4.1 | Fusion en log-probabilidad (media geometrica) en vez de aritmetica | **FUNCTIONAL_DUPLICATE** | PH15-CALIB (D-029) | SI, porque el brief la pide y es casi gratis de medir. Si reproduce el +0,0003 de D-029, se cier | 2 |
| H-P32-01 | P4 | Fusion en espacio de probabilidad entre V30 y el canal linguistico | **FUNCTIONAL_DUPLICATE** | PH39-FUSION-LEX (D-039) | SI, una sola vez, con el control barajado. Si el control iguala, la linea se cierra y no se abre | 3 |
| H-P32-03 | P5 | Fusion con alpha por clase (alpha_c), un alpha compartido primero | **PARTIAL_PRECEDENT** | H14 y PH33-CORRECTOR (D-037) | SI, pero solo el alpha compartido primero. Seis alphas libres quedan prohibidos sin evidencia | 4 |
| H-P32-07 | P10 | Correccion condicionada por la longitud del texto | **GENUINELY_NEW** | ninguno como mecanismo de fusion | SI, pero con la forma gamma0 + gamma1 * log_len * residual, que es lo unico permitido | 5 |
| H-P32-06 | P9 | Correccion condicionada por la confianza de V30 | **PARTIAL_PRECEDENT** | PH32-CORRECCION (D-037) | SOLO si H-P32-05 deja residuo. Si no, no se ejecuta | 6 |
| H-P32-09 | P2.2 | Posicion de la URL dentro del texto y longitud relativa | **PARTIAL_PRECEDENT** | V31-03 canal `url` | SI, como extension acotada, y solo si el gap analysis de URLs muestra cobertura suficiente | 6 |
| H-P32-08 | P2.5/P2.7 | Representaciones estructurales de URL y canal de dominio | **FUNCTIONAL_DUPLICATE** | V31-02 y V31-03 | **NO**: duplicado exacto de V31-02 y V31-03 con el mismo baseline; se reutilizan sus resultados en vez de repetir la medicion | 7 |
| H-P32-11 | P7 | Fusion en score-space de una fuente linguistica y una fuente de URL | **PARTIAL_PRECEDENT** | PH39-FUSION-LEX (D-039) | SOLO si H-P32-04 y H-P32-05 producen rescates unicos en pares y clases distintos. Si no, no se e | 7 |
| H-P32-10 | P2.8 | Recuperacion del contenido textual de la URL | **QUARANTINED** | P0 de V31+ | **NO**: las reglas de la competicion no estan disponibles en este entorno, asi que el retrieval no puede clasificarse como ALLOWED y queda en cuarentena | 9 |

## Las dos genuinamente nuevas, y por que son las que valen la pena

**H-P32-05, residualizacion.** Ajustar `z_C[c] = a_c + b_c * z_V30[c]` y sumar solo el resto. No existe en el historico. Y es la formulacion que mejor encaja con lo que V31+ midio: la senal linguistica es real pero **casi toda se solapa con lo que V30 ya dice**, asi que la pregunta correcta no es *cuanto pesa el candidato* sino *que parte del candidato no es V30*.

**H-P32-07, correccion condicionada por longitud.** Condicionar la fuerza de la correccion a la longitud no se ha hecho: V31 y PH5-L2 usaron la longitud como corte diagnostico, y E-018 y V14A como feature. Es un mecanismo distinto de tener cuatro modelos por longitud, que es justo lo que la forma `gamma0 + gamma1 * log_len * residual` evita.

**H-P32-04, correccion pairwise localizada.** Con el precedente mas duro de la fase, pero con un mecanismo distinto: V28-03 sustituyo, P32 anade. El brief exige el test de rescate por par, y con el precedente de V28-03 a la vista la vara esta muy alta: sin rescate neto positivo por par, se cierra.

## El orden de contingencia que se va a seguir

```text
H-P32-05  residualizacion              (mas limpia, primera)
H-P32-04  correccion pairwise por par     (mas nueva, segunda)
H-P32-01  fusion de probabilidad + control (precedente fuerte, tercera)
H-P32-02  fusion en log-probabilidad       (casi gratis, cuarta)
H-P32-03  alpha por clase, uno compartido  (solo si hay rescate por clase)
H-P32-07  correccion por longitud          (solo si hay residuo)
H-P32-06  correccion por confianza         (solo si hay residuo)
H-P32-09  extension acotada de URL         (si la cobertura lo justifica)
```

Y **fuera de la lista**, sin excepcion:

- `H-P32-10`, retrieval de contenido de URL: **QUARANTINED**. Las reglas de la competicion no estan disponibles en este entorno, asi que no puede clasificarse como ALLOWED. Cierra P2.8, P2.9, P2.10 y P12 de un golpe.
- `H-P32-08`, representaciones estructurales de URL: **duplicado exacto** de V31-02 y V31-03, con el mismo baseline. Se reutilizan sus resultados.

## La calibracion que P00 deja mas importante que cualquier hipotesis

El ruido de **entrenamiento con otro reparto** es de 0.0008876, y el gate de PROMISING pide +0,0015: solo **1.7 veces** ese valor. Cualquier delta de esta fase por debajo de unos 0.0008876 esta dentro del ruido de la propia tecnica, por limpio que sea su desglose por folds. Ese es el liston contra el que se juzga todo lo que viene.
