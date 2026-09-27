# P45 - Estructura conjunta HS x TR x AG y error de V03

Generado por `experiments/v45_01_joint_label_structure.py` (V45-01 (P1 estructura conjunta HS x TR x AG y error de V03)).

Fecha 2026-09-27T23:52:51+00:00. **0 modelos, 0 submissions.**

> El holdout no se usa para ninguna decision. Las cifras de train global son descriptivas; todas las decisiones se toman sobre dev y el OOF canonico.

Pregunta: Hay evidencia de que los errores de V03 dependan de la combinacion conjunta de etiquetas de una forma que no capturen sus tres decisiones binarias independientes?

## 1. Distribucion de los ocho estados

Variable `LABEL_STATE = 4*HS + 2*TR + AG`.

| estado | HS | TR | AG | train n | train % | dev n | dev % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 949 | 54.824 | 813 | 54.821 |
| 1 | 0 | 0 | 1 | 26 | 1.502 | 22 | 1.483 |
| 2 | 0 | 1 | 0 | 0 | 0.000 | 0 | 0.000 |
| 3 | 0 | 1 | 1 | 0 | 0.000 | 0 | 0.000 |
| 4 | 1 | 0 | 0 | 92 | 5.315 | 79 | 5.327 |
| 5 | 1 | 0 | 1 | 145 | 8.377 | 124 | 8.361 |
| 6 | 1 | 1 | 0 | 28 | 1.618 | 24 | 1.618 |
| 7 | 1 | 1 | 1 | 491 | 28.365 | 421 | 28.388 |

Estados observados en train: [0, 1, 4, 5, 6, 7]. No observados: [2, 3].

Casos con `TR=1` y `HS=0` en train: **0**.

## 2. Matriz de 8 x 8: estado verdadero contra estado predicho

Filas = estado verdadero, columnas = estado predicho por V03.

| true \ pred | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **000** | 627 | 9 | 6 | 5 | 66 | 34 | 10 | 56 | 813 | **<-**
| **001** | 8 | 7 | 0 | 0 | 0 | 6 | 0 | 1 | 22 | **<-**
| **010** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **011** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **100** | 36 | 0 | 0 | 0 | 27 | 15 | 0 | 1 | 79 | **<-**
| **101** | 45 | 5 | 0 | 0 | 31 | 40 | 0 | 3 | 124 | **<-**
| **110** | 5 | 2 | 1 | 0 | 2 | 2 | 1 | 11 | 24 | **<-**
| **111** | 40 | 1 | 2 | 6 | 14 | 16 | 2 | 340 | 421 | **<-**

Filas con el estado entero correcto: **1042** (70.263 %).

## 3. Categorias de error de V03

| categoria | n | % |
|---|---:|---:|
| `A_tres_correctas` | 1042 | 70.263 |
| `B_falla_una` | 223 | 15.037 |
| `C_falla_dos` | 120 | 8.092 |
| `D_falla_tres` | 98 | 6.608 |

## 4. Errores por etiqueta, para cada estado verdadero

| estado | n | err HS | tasa | err TR | tasa | err AG | tasa |
|---|---:|---:|---:|---:|---:|---:|---:|
| 000 | 813 | 166 | 0.204 | 77 | 0.095 | 104 | 0.128 |
| 001 | 22 | 7 | 0.318 | 1 | 0.045 | 8 | 0.364 |
| 100 | 79 | 36 | 0.456 | 1 | 0.013 | 16 | 0.203 |
| 101 | 124 | 50 | 0.403 | 3 | 0.024 | 76 | 0.613 |
| 110 | 24 | 8 | 0.333 | 11 | 0.458 | 15 | 0.625 |
| 111 | 421 | 49 | 0.116 | 71 | 0.169 | 58 | 0.138 |

## 5. Contraste de independencia de los errores

Si los tres indicadores de error fueran independientes dado el estado verdadero, los ocho patrones de error seguirian el producto de las marginales. Se contrasta esa hipotesis nula con chi-cuadrado, celda a celda, agregando solo las celdas con esperado mayor que 5.

- chi2 = **343.156**, gl = 20, p = **0.000000**
- desviacion absoluta maxima de una celda: 116.20 casos
- rechazamos la independencia al 5 %: **True**

## 6. Consecuencia: correccion estructural dura

| regla | corrected_cv | delta |
|---|---:|---:|
| control V03 | 0.7772114525 | 0 |
| forzar TR=0 si HS=0 | 0.7765657080 | -0.0006457445 |
| forzar AG=0 si HS=0 | 0.7736697906 | -0.0035416619 |


## 7. Inventario de mecanismos conjuntos ya probados

Ninguna formulacion de la lista de la directiva queda sin probar.

| mecanismo | experimento | decision registrada |
|---|---|---|
| Label Powerset | `V4-E01-E03` | TRACK E cerrado como NO_SIGNAL (V4-E01, V4-E02 y V4-E03); se confirma que la dependencia e |
| Classifier chain HS->TR->AG | `E2-01-HS-TR-AG` | No adoptada (no supera la regla D-016) |
| Classifier chain HS->AG->TR | `E2-01-HS-AG-TR` | No adoptada (no supera la regla D-016) |
| Cadena con espacios normalizados | `E2-01-HS-TR-AG-STD` | No adoptada (no supera la regla D-016) |
| Espacio separado por etiqueta | `E2-21-ESPACIOS` | CERRADA sin mejora |
| Acoplamiento suave entre cabeceras | `V8-05` | **P4 cerrado como negativo, sin submission y sin variante.** El mecanismo esta descartado: |
| Meta-modelo sobre scores OOF | `E2-02-META` | ADOPTADA (Gate P1) |
| Meta-modelo con canales | `E2-18-META-BASES` | No adoptado / diagnostico |
| Correccion estructural dura | `P45-01 (esta fase)` | NEGATIVA en ambos casos |

Los dos unicos que produjeron mejora local (E2-02-META, 0.785320) ya fallaron en el leaderboard: `v08_meta_v2` scored 0.78163 frente a 0.78306 de V03.

## 8. Veredicto sobre la estructura conjunta

**Hay estructura conjunta real y es medible.** El contraste de independencia da chi2 343.2 con 20 grados de libertad, p = 0.00e+00. Los errores de HS, TR y AG no son independientes dado el estado verdadero.

**Pero la estructura no es explotable con ninguna de lasvias ya probadas.** La correccion estructural dura, que es la extraccion mas directa de esa estructura, es negativa en las dos formas (-0.0006457445 y -0.0035416619). Y las formulaciones de aprendizaje conjunto (label powerset, cadenas, meta-modelos, acoplamiento suave) ya estan todas registradas con resultado negativo o sin transferencia.

**`JOINT_LABEL_STRUCTURE = CLOSED`.** No se abre ninguna familia conjunta nueva.

