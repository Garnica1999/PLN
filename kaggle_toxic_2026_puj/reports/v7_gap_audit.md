# v7: auditoria de brechas (P0)

Generado por `experiments/v7_00_gap_audit.py` (V7-00 (P0, analisis)).
No entrena nada. Se ejecuta antes de gastar una sola unidad de la fase v7.

Cada afirmacion de la tabla se apoya en una busqueda programatica sobre el codigo
real del proyecto y en la lectura del numero medido en los artefactos existentes.
Un cierre sin cifra no es un cierre, y por eso la columna de mejor resultado se
trae de los ficheros JSON y CSV que los experimentos anteriores ya escribieron.

## 1. Inventario de lo ya experimentado

El registro tiene **184 experimentos** y **52 decisiones**. Por prefijo de identificador:

| prefijo | experimentos |
|---|---|
| `E2-*` | 58 |
| `EBD-*` | 20 |
| `V4-*` | 14 |
| `PRE-*` | 10 |
| `FIN-*` | 10 |
| `V6-*` | 10 |
| `MOD-*` | 9 |
| `REP-*` | 6 |
| `KAGGLE_TOXIC_2026_PUJ-*` | 6 |
| `E-*` | 5 |
| `ENS-*` | 4 |
| `HIB-*` | 4 |
| `DL-*` | 4 |
| `V2-*` | 4 |
| `AUD-*` | 3 |
| `E00-*` | 1 |
| `B1-*` | 1 |
| `TH-*` | 1 |
| `S02-*` | 1 |
| `S03-*` | 1 |
| `S04-*` | 1 |
| `B5-*` | 1 |
| `OPT-*` | 1 |
| `S05-*` | 1 |
| `R1-*` | 1 |
| `R2-*` | 1 |
| `S06-*` | 1 |
| `S07-*` | 1 |
| `PA-*` | 1 |
| `HOLD-*` | 1 |
| `S08-*` | 1 |
| `P8-*` | 1 |

## 2. El hallazgo que reencuadra la fase

Tres modelos con **mejor** metrica fuera de pliegue llegaron al final del proyecto

| experimento | OOF | leaderboard | diferencia |
|---|---|---|---|
| V03 | 0.777211 | **0.78306** | referencia |
| R1-C1, con C igual a 1.0 | 0.7929 | 0.77227 | **-0.011** |
| R2-MINDF5 | 0.7993 | 0.77034 | **-0.013** |
| OPT-01, busqueda con Optuna | **0.8012** | 0.77602 | **-0.007** |

La mejora en OOF de los tres era de mas de +0.015, muy por encima del gate de
+0.003, y los tres perdieron leaderboard. Ademas los tres aumentaban capacidad.
La conclusion de S05 lo dice sin rodeos: la busqueda por metrica fuera de pliegue premia capacidad que no
generaliza.

De ahi dos reglas que condicionan toda la fase:

1. **El unico movimiento positivo del leaderboard en toda la historia del proyecto
   vino de la capa de decision, no del modelo**: los umbrales por etiqueta pasaron
   el leaderboard de 0.77993 a 0.78306, es decir +0.0031.
2. **Toda mejora de capacidad de modelo ha restado leaderboard.** Por eso el freno
   de transferencia: por encima de +0.005 de OOF, un
   candidato solo llega a la mesa de finales si **no** aumenta la dimension ni la
   brecha entre ajuste y validacion. Es la regla que los tres fracasos violan y que
   V03 cumple.

Esto no dice que 0.80 sea imposible. Dice que hay que moverlo desde donde ya se ha
movido una vez, que es la decision, y que el unico gasto que justifica su precio es
medir bien.

## 3. La tabla de brecha completa

El detalle, con las once columnas que pide la especificacion, esta en
`reports/v7_gap_audit.csv`.

**Sobre la procedencia de las cifras de esta tabla**: la columna de mejor resultado
es un resumen. Las cifras con su origen exacto, leidas de los artefactos, estan en
la seccion 4 y en `reports/v7_gap_verificaciones.json`. Cuando una cifra no se pudo
leer, aparece el motivo del fallo en lugar de un numero, porque un hueco declarado
es informacion y un numero inventado es ruido.

| track | ya probado | mejor resultado | prioridad |
|---|---|---|---|
| P0.1 Umbral anidado | si, con un protocolo mas laxo | `0.776613 (umbral global anidado, es deci...` | media: se ejecuta, pero por rigor de medicion y no por upside |
| P0.2 Umbral por rebanada | si | `0.774954 (mejor de las cuatro variantes ...` | nula: cerrar |
| P0.3 Estimacion robusta de umbral | no | `sin medir` | baja-media: ejecutar solo si P0.1 muestra dispersion alta |
| P0.4 Calibracion de probabilidades | si | `0.001831 (mejora de la calibracion frent...` | nula: cerrar |
| P0.5 Umbral condicional por region | si, identico | `identico a P0.2` | nula: cerrar |
| P1.1 Ensemble de modelos por pliegue | la parte medible ya esta hecha | `el OOF canonico ya es el promedio por pl...` | aplazada: no es ejecutable en local |
| P2.1 Maquina de margen sobre V03 | si | `0.7715 frente a 0.777211 de V03, y 10 fi...` | nula: cerrar por techo |
| P2.3 Experto generativo disperso | si, cuatro variantes | `cerrada en un rango de -0.016 a -0.039` | nula: cerrar por techo |
| P3 CNN de caracteres desde cero | si, y con conclusion firme | `0.741036 (convolutional de caracteres en...` | nula: cerrar por duplicado |
| P4 Vista doble con normalizacion | si, identico | `-0.023836 (canal original mas canal norm...` | nula: cerrar por duplicado |
| P5.1 Alcance de negacion | si, y con la mayor perdida del proyecto | `-0.250519 (canal de negacion)` | nula: cerrar por duplicado |
| P6 Vista multiple en el momento de predecir | parcialmente, por el mismo mecanismo que P4 | `la doble vista perdio entre -0.0238 y -0...` | nula: cerrar |
| P7 Puerta de expertos por region | si, y con techo medido | `-0.002292 (modelo propio por rebanada)` | nula: cerrar por techo |
| P8 Prior de etiqueta del grupo | no: el mecanismo es distinto al de v6 | `nunca medido como caracteristica` | alta: mecanismo genuinamente nuevo |
| P9 Perdida orientada a la F1 | no | `nunca medido` | alta: primera linea de ejecucion |
| P10 Consenso de umbrales entre pliegues | no | `nunca medido` | media: ejecutar solo si P0.1 muestra dispersion alta |

## 4. Comprobaciones programaticas

| comprobacion | resultado | ficheros | cifra medida |
|---|---|---|---|
| `P0_1_umbral_anidado` | **presente** | 21_backlog_restante_v2.py (19), 05_ensembles.py (13) | `0.776613 (umbral global anidado, es decir, lea...` |
| `P0_2_umbral_por_rebanada` | **presente** | 21_backlog_restante_v2.py (196), 06_error_analysis.py (139) | `0.774954 (mejor de las cuatro variantes de umb...` |
| `P0_3_estimador_robusto` | **presente** | embedding_02_fasttext_baseline_v5.py (1), evaluation.py (1) | `sin artefacto asociado` |
| `P0_4_calibracion` | **presente** | 21_backlog_restante_v2.py (91), 06_error_analysis.py (4) | `0.001831 (mejora de la calibracion frente a no...` |
| `P1_1_oof_por_pliegue` | **presente** | 21_backlog_restante_v2.py (224), 09_optuna_tfidf_lr.py (119) | `1483 (el artefacto canonico ya lleva el pliegu...` |
| `P2_1_linearsvc` | **presente** | 07_submission_v04_ens04.py (10), 20_emocion_nbsvm_ortografia_v2.py (9) | `10 (filas que la maquina de margen acierta y V...` |
| `P2_3_generativo` | **presente** | 20_emocion_nbsvm_ortografia_v2.py (37), 22_artefactos_auditoria_v2.py (28) | `0.761203 (log-count ratio sobre la union de V0...` |
| `P3_1_cnn_caracteres` | **presente** | 11_dl_desde_cero.py (12), v6_00_gap_analysis.py (5) | `0.741036 (convolutional de caracteres entrenad...` |
| `P4_1_vista_duplicada` | **presente** | 20_emocion_nbsvm_ortografia_v2.py (42), 22_artefactos_auditoria_v2.py (35) | `-0.023836 (canal original mas canal normalizad...` |
| `P5_1_alcance_negacion` | **presente** | 19_sintaxis_cita_negacion_v2.py (203), 22_artefactos_auditoria_v2.py (106) | `-0.250519 (canal de negacion)` |
| `P7_1_especialistas` | **presente** | 21_backlog_restante_v2.py (101), v6_09_entregables.py (49) | `-0.002292 (modelo propio por rebanada)` |
| `P8_1_prior_grupo` | **ausente** | ninguno | `sin artefacto asociado` |
| `P9_1_perdida_f1` | **ausente** | ninguno | `sin artefacto asociado` |
| `P10_1_consenso_umbral` | **ausente** | ninguno | `sin artefacto asociado` |

Las lecturas que importan:

- `P8_1_prior_grupo` **ausente** y `P9_1_perdida_f1` **ausente**: son los dos
  unicos mecanismos de la especificacion que no existen en el proyecto. Uno crea
  una caracteristica a partir de la estructura de grupos repetidos, que es la
  senal que la fase v6 si encontro; el otro ataca la funcion de perdida, que es el
  objetivo real.
- `P0_1_umbral_anidado` **presente** con un numero peor que V03: la linea de
  umbrales no es nueva, pero el protocolo de la especificacion es **mas estricto**
  que el existente y ademas produce la medicion de dispersion que nadie tiene.
- `P1_1_oof_por_pliegue` **presente**: el artefacto canonico ya es el promedio por
  pliegue, de modo que la parte medible de P1 esta hecha y solo queda un cambio que
  no se puede medir en local.
- `P3_1_cnn_caracteres` **presente**: la CNN de caracteres ya se entreno desde cero
  y se descarto. La especificacion de v7 propone el mismo modelo y ademas le
  prohibe la unica entrada distinta que lo hacia util.

## 5. Lo que se cierra y por que no se ejecuta

| track | causa del cierre |
|---|---|
| P0.2 Umbral por rebanada | Las rebanadas con distribucion distinta se midieron con umbral propio y cuatro niveles de shrinkage, y ninguna supero al umbral global anidado. |
| P0.4 Calibracion de probabilidades | Se evaluo con ajuste, calibracion y evaluacion separados, mejoro frente a no calibrar pero perdio al.Reference en la fusion, y se cerro. |
| P0.5 Umbral condicional por region | Es exactamente lo que hizo E2-16: umbral por rebanada, con shrinkage, con soporte minimo y con calibracion separada. |
| P2.1 Maquina de margen sobre V03 | Es la referencia honesta del protocolo v2 y participo en cuatro ensembles. Ademas su techo esta medido: el mejor modelo alternativo acierta 9 a 12 filas de 1483 donde V03 falla, lo que limita la mejora a +0.0027. |
| P2.3 Experto generativo disperso | Se compararon el peso de Naive Bayes sobre palabras, sobre caracteres, el log-count ratio sobre la union de V03 y su margen con la maquina de margen. |
| P3 CNN de caracteres desde cero | La CNN de caracteres ya se entreno desde cero, con vocabulario de 91 caracteres, dimension 48 y longitud 200, y se descarto. Su brecha entre ajuste y validacion fue de 0.18, que es el sintoma de falta de datos, no de arquitectura. La especificacion de v7 propone el mismo modelo, y ademas prohibe usar los embeddings externos que si serian una entrada distinta. |
| P4 Vista doble con normalizacion | La doble vista se construyo exactamente asi: el canal original conserva el estilo y un canal normalizado recupera los lexemas ofuscados. Tres variantes, todas perdidas entre -0.0238 y -0.0286. |
| P5.1 Alcance de negacion | La negacion como canal, como patron de dependencia y unida a la cita ya se midieron. El canal de negacion fue el peor resultado registrado, muy por debajo de V03. |
| P6 Vista multiple en el momento de predecir | El mecanismo es la doble vista de P4, solo que en el momento de predecir. Si la vista normalizada no aporta en el ajuste, no hay razon para que aporte en la prediccion. Es el mismo argumento que cerro la linea de aumento de la fase v6. |
| P7 Puerta de expertos por region | El modelo por rebanada se midio y perdio. Ademas la interseccion de errores de la fase v6 demuestra que combinar dos clasificadores elige entre sus aciertos y no los suma, con un techo de +0.0027 para cualquier linea cuya unica via sea recuperar filas ajenas a V03. |

## 6. Lo que queda abierto de verdad

| track | por que es nuevo | upside esperado | coste |
|---|---|---|---|
| P0.1 Umbral anidado | Separar en tres niveles: ajuste, calibracion y evaluacion. El protocolo existente reutiliza las probabilidades fuera de pliegue de los otros cuatro pl | bajo: el mecanismo ya dio un resultado peor que V03 | medio: requiere reentrenar con particion de tres niveles |
| P0.3 Estimacion robusta de umbral | ya esta bien formulada: cambiar el estimador del agregado de umbrales por pliegue, manteniendo la evaluacion fuera del conjunto de calibracion | bajo, pero es la parte de P0 que si es nueva | bajo: reutiliza el protocolo de P0.1 |
| P1.1 Ensemble de modelos por pliegue | Reescribir la linea como cambio **solo de prueba**: promediar las probabilidades de los cinco modelos al predecir el test. Eso solo puede verificarse  | desconocido e inmedible en local | bajo de computationally, pero exige una sonda de leaderboard |
| P8 Prior de etiqueta del grupo | ya esta bien formulada. La distincion es mecanica y no nominal: la fase v6 cambio **pesos** de entrenamiento, que resta capacidad a un ejemplo repetid | medio: ataca la estructura que v6 si encontro con senal | bajo: es una caracteristica densa de tres columnas |
| P9 Perdida orientada a la F1 | ya esta bien formulada. Es la unica linea de la fase que ataca la funcion de perdida, es decir, el objetivo, y no la representacion ni el modelo. | medio: es el unico mecanismo de decision sin cerrar | medio: hay que elegir y justificar un sustituto estable |
| P10 Consenso de umbrales entre pliegues | ya esta bien formulada, y es la continuacion de P0.3 | bajo por si mismo, alto como garantia de estabilidad | bajo |

## 7. Lineas que se ejecutan con reservas expresas

- **P0.1 Umbral anidado**: Separar en tres niveles: ajuste, calibracion y evaluacion. El protocolo existente reutiliza las probabilidades fuera de pliegue de los otros cuatro pliegues como conjunto de calibracion, y por tanto ve mas datos de los que permite el protocolo de la especificacion. Ademas produce el informe de dispersion de umbrales que nadie tiene.
- **P6 Vista multiple en el momento de predecir**: ninguna que no repita P4

## 8. Riesgo de filtrado por linea

El riesgo que mas importa en esta fase no es estadistico sino de protocolo:
reutilizar el mismo conjunto fuera de pliegue para ajustar, para elegir umbrales y
para evaluar. La fase v6 ya lo midio: reajustar umbrales sobre el mismo conjunto
infla el resultado en +0.0036 sin cambiar el modelo. Por eso toda linea de v7 que
toque umbrales o perdidas debe declarar su esquema de tres niveles de forma
explicita en el registro.

| track | riesgo de filtrado |
|---|---|
| P0.1 Umbral anidado | bajo: es la unica linea que produce medicion fiable del sesgo |
| P0.3 Estimacion robusta de umbral | bajo: no toca el modelo, solo el punto de operacion |
| P1.1 Ensemble de modelos por pliegue | medio: no se puede validar antes de enviar |
| P8 Prior de etiqueta del grupo | alto de filtrado: el prior debe calcularse con las otras observaciones del grupo dentro del pliegue y nunca con la propia etiqueta en evaluacion |
| P9 Perdida orientada a la F1 | medio: las aproximaciones a F1 pueden ser inestables con 1731 filas |
| P10 Consenso de umbrales entre pliegues | bajo |

## 9. Orden de ejecucion que se deduce de esta auditoria

1. **P9** perdida orientada a la F1: unico mecanismo de decision sin cerrar, y
   ataca el objetivo real.
2. **P8** prior de etiqueta del grupo: unica senal nueva, y ataca la estructura
   que la fase v6 si encontro con senal.
3. **P0.1** umbral anidado con protocolo mas estricto: por rigor de medicion, no
   por upside. Su salida es el informe de dispersion que la fase v6 dejo pendiente.
4. **P0.3 y P10** estimador robusto y consenso: solo si P0.1 muestra dispersion
   alta, y son baratos porque reutilizan su protocolo.

Este orden no es el de la especificacion, y la razon es que la auditoria mostro que
los cuatro primeros tracks de aquel orden ya estaban medidos. El orden de la
especificacion era correcto **antes** de conocer el registro, no despues.

Presupuesto autorizado: 25 experimentos principales y 10 de reserva. Presupuesto
realista despues de esta auditoria: **4 a 8 experimentos**. No es un recorte, es lo
que queda abierto.

Fecha: 2026-09-26T02:01:28+00:00.
