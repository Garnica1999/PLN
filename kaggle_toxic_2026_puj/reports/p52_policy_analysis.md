# P52 - Policy oracle, decision boundary y complementariedad entre canales

Generado por `experiments/v52_00_auditoria_y_techo.py` (V52-00 (auditoria de P51, utilidad y techo de politica)).

Fecha 2026-09-28T01:57:52+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

## 1. Auditoria del codigo real de P51

| componente | valor verificado en el codigo |
|---|---|
| target de S1-U | `(u > 0) con u = utilidad[v, c] y utilidad construida con los conteos del pliegue de entrenamiento` |
| target de S2-U | el mismo target binario; S2-U cambia la agregacion, no el target |
| **umbral de S1-U** | **0.5** |
| **equilibrio teorico** | **0.389157** |
| utilidad | FN->TP escala con TP+FP+FN, FP->TN solo con TP, forma cerrada de v19_01_valor_de_correccion.py |
| espacio de acciones | KEEP_V03 mas 20 canales, restringido a celdas en discrepancia |
| cross-fitting | 5 pliegues canonicos de TOXIC agrupados, objetivo y ajuste dentro del training fold |
| variables por clasificador | 15 |
| mezcla las dos direcciones | SI: P51 usa una mascara unica de discrepancia y un solo clasificador por canal, luego mezcla las dos direcciones. P52 las separa, que es lo que exige la seccion 7 |

**El hallazgo de la auditoria.** P51 uso el umbral 0,5 donde la teoria dice 0.389157 al subir y 0,6108 al bajar. Eso lo hace **mas estricto de lo necesario al subir y mas laxo al bajar**, luego sobre-cambia justo en la direccion que mas dano, y por eso su damage se concentro en `baja_1_a_0`. Ademas P51 usaba un unico clasificador por canal, mezclando las dos direcciones; P52 las separa.

## 2. Utilidad re-derivada de forma independiente

| magnitud | valor |
|---|---:|
| ganancia de subir (`FN -> TP`) | 1.402554e-03 |
| ganancia de bajar (`FP -> TN`) | 9.012335e-04 |
| razon entre ambas | **1.57x** |
| equilibrio al subir | **0.3892** |
| equilibrio al bajar | **0.6108** |
| suma de los dos equilibrios | 1.0000000000 |
| tasa base al subir | 17.86 pct |
| **elevacion necesaria al subir** | **2.18x** |
| tasa base al bajar | 37.61 pct |
| **elevacion necesaria al bajar** | **1.62x** |
| reproduce el equilibrio de P51 | True |
| reproduce la elevacion al subir | True |

Los dos equilibrios suman exactamente 1, luego no son dos numeros sino el mismo expresado desde cada lado, y la decision optima es un unico umbral sobre `P(y=1)`. Esa es una propiedad matematica de la decision, no un parametro ajustado.

## 3. Techos, y por que `policy oracle` no es un tercer objeto

Sin `y_true` no se puede construir un techo: una politica que estima `P(rescue | observable)` **es** una politica aprendida. Por eso se separan estos cuatro objetos, de abajo hacia arriba:

| objeto | corrected_cv | delta vs V03 | usa `y_true` |
|---|---:|---:|---|
| V03 (inaccion) | 0.7772114525 | +0,0000000000 | no |
| estrato, accion del training fold | 0.7549322355 | -0.0222792170 | solo en entrenamiento |
| **STRATUM_ORACLE** | **0.7753804469** | **-0.0018310056** | si |
| TRUE_ORACLE | 0.9870178614 | +0.2098064089 | si |

El techo usa 18 estratos definidos solo con la prediccion de V03, el bin de margen y el numero de canales que discrepan, y elige **una sola accion constante dentro de cada estrato**.

La descomposicion que responde a la pregunta de la directiva:

| tramo | magnitud | que mide |
|---|---:|---|
| TRUE_ORACLE -> STRATUM_ORACLE | +0.211637 | coste de obligar a la politica a ser constante en un estrato |
| STRATUM_ORACLE -> V03 | -0.001831 | | **lo que una politica estratificada podria ganar como maximo** |

## 4. Rescate y dano exclusivos

| magnitud | valor |
|---|---:|
| canales con `net_exclusivo` positivo | **2 de 20** |
| HHI del reparto de rescates | 0.0020 |
| rescate maximo de un canal | 337 |
| rescate minimo de un canal | 71 |
| celdas con rescate unico | 78 |

El HHI cercano a 1 significa que los rescates **no** estan concentrados: reparten entre los 20 canales.

| canal | rescate | exclusivo | dano | exclusivo | net | net excl. |
|---|---:|---:|---:|---:|---:|---:|
| `cita:NEG` | 337 | 45 | 2043 | 613 | -1706 | -568 |
| `emoc:EMO-EST@C=2.0` | 300 | 9 | 875 | 60 | -575 | -51 |
| `synt:SYN-POS` | 309 | 5 | 1014 | 29 | -705 | -24 |
| `synt:SYN-MORPH` | 297 | 5 | 1134 | 68 | -837 | -63 |
| `cita:CITA` | 186 | 4 | 285 | 4 | -99 | +0 |
| `synt:SYN-DEPS` | 302 | 3 | 1077 | 53 | -775 | -50 |
| `emoc:NB-UNION-SVC@C=1.0` | 217 | 2 | 274 | 7 | -57 | -5 |
| `emoc:NB-WORD-LR@C=1.0` | 168 | 2 | 214 | 5 | -46 | -3 |

## 5. Complementariedad por pares

- pares analizados: **190**
- lift maximo de la union sobre el mejor miembro: **1.0000**
- pares con exclusividad bilateral: **0**

| canal i | canal j | i solo | j solo | union | desacuerdo |
|---|---|---:|---:|---:|---:|
| `emoc:EMO-LEX@C=1.0` | `emoc:EMO-ALL@C=1.0` | 0 | 0 | 282 | 0 |
| `synt:SYN-DEPS` | `synt:SYN-3` | 0 | 0 | 221 | 0 |
| `synt:SYN-MORPH` | `synt:SYN-3` | 0 | 0 | 221 | 0 |
| `synt:SYN-POS` | `synt:SYN-DEPS` | 0 | 0 | 216 | 0 |
| `synt:SYN-POS` | `synt:SYN-3` | 0 | 0 | 210 | 0 |
| `emoc:NB-UNION-LR@C=1.0` | `emoc:NB-UNION-SVC@C=1.0` | 0 | 0 | 175 | 0 |
| `emoc:NB-CHAR-LR@C=1.0` | `emoc:NB-UNION-LR@C=1.0` | 0 | 0 | 164 | 0 |
| `synt:SYN-DEPS` | `synt:SYN-MORPH` | 0 | 0 | 163 | 0 |

## 6. Correlacion de error, solo en el regimen donde V03 falla

| magnitud | valor |
|---|---:|
| correlacion minima | -0.1598 |
| correlacion maxima | 1.0000 |
| correlacion media | 0.1859 |
| pares por encima de 0,5 | 14 |
| grupos a distancia 0,65 | 6 |

Esto importa porque la diversidad global de los canales no es la misma cosa que la diversidad **precisamente donde V03 falla**, que es el unico sitio donde un selector podria necesitar un canal distinto.

## 7. Curva de valor de la politica

| umbral | celdas cambiadas | net |
|---:|---:|---:|
| 0.1000 | 233 | +225 |
| 0.2000 | 233 | +225 |
| 0.3000 | 233 | +225 |
| 0.3892 | 233 | +225 |
| 0.4000 | 233 | +225 |
| 0.5000 | 203 | +203 |
| 0.6000 | 203 | +203 |
| 0.7000 | 203 | +203 |

La curva es **retrospectiva y de diagnostico**, calculada sobre training folds. No se usa para elegir umbral, y el valor derivado de la forma cerrada sigue siendo el unico umbral con justificacion matematica.

**CASE = C.**

