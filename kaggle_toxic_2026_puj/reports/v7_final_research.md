# v7: informe de cierre de la fase

Generado por `experiments/v7_04_entregables.py` (V7-04 (cierre, analisis)).

## 1. Que buscaba la fase y que ha encontrado

La fase v7 buscaba una fuente de senal que V03 no capturara, con once tracks de
candidatura. La respuesta tiene dos partes.

**La parte facil: la mayoria de los tracks ya estaban medidos.** La auditoria de
brecha, con verificacion programatica sobre el codigo real y lectura de las cifras de
los artefactos, cerro ocho de los once: P0.2 y P0.5, P0.4, P2.1, P2.3, P3, P4, P5, P6 y
P7. Dos de ellos, P2.1 y P7, tenian ademas el techo medido de la interseccion de
errores, que es de +0.0027 y esta por debajo del gate. De los once solo **P8 y P9
eran mecanismos genuinamente nuevos**, y ambos se ejecutaron y ambosdieron un un
resultado negativo con su causa medida.

**La parte importante: lo que la fase deja es una medicion que v6 dejo pendiente y
una correccion de expectativas sobre el objetivo.**

## 2. Los tres tracks ejecutados

| track | resultado | causa medida |
|---|---|---|
| P9, perdida orientada a la F1 | -0.0022 frente al control BCE | optimizar el objetivo literalmente con 1.186 filas acopla el descenso a la muestra y sobreajusta |
| P8, prior de etiqueta del grupo | -0.0011 frente al control | solo el 7.08 % de las filas tiene evidencia de grupo, y donde la hay el prior es redundante con el texto |
| P0.1, umbral anidado estricto | -0.0019 frente al control de tres niveles | los umbrales estan debilmente identificados: desviaciones de 0.04 a 0.065 |

- variantes medidas: **9**, de las cuales **4** son controles de calidad
- variantes que superan el gate: **0**
- **V03 sigue congelada** en 0.777211, con sus tres umbrales, **por evidencia medida y
  no por inercia**
- **0 submissions y 0 sondas de leaderboard** en toda la fase

## 3. La correccion de expectativas sobre 0.80

El objetivo externo es superar 0.80. La fase aporta tres elementos para juzgarlo con
honestidad, y conviene tenerlos delante antes de gastar mas presupuesto.

| experimento | metrica local | leaderboard | diferencia |
|---|---|---|---|
| V03 | 0.777211 | 0.78306 | referencia |
| R1-C1 | 0.792900 | 0.77227 | -0.01079 |
| R2-MINDF5 | 0.799300 | 0.77034 | -0.01272 |
| OPT-01 | 0.801200 | 0.77602 | -0.00704 |

Tres consecuencias, y las tres estan medidas, no supuestas:

1. **Los tres mejores modelos locales de la historia del proyecto perdieron
   leaderboard.** Tenian entre +0.015 y +0.025 de metrica local, muy por encima del
   gate de +0.003, y perdieron entre 0.007 y 0.013. Una mejora local de +0.02 no es
   evidencia de una mejora externa: en este conjunto es evidencia de lo contrario.
2. **Los tres aumentaban capacidad.** Por eso el freno de transferencia que se
   adopto en la fase: por encima de +0.005 de mejora local, un
   candidato solo llega a la mesa de finales si **no** aumenta la dimension ni la
   brecha entre ajuste y validacion. Es la regla que incumplieron los tres fracasos y
   que V03 cumple.
3. **El unico movimiento positivo del leaderboard vino de la capa de decision**, de
   los umbrales por etiqueta, y no de un modelo mejor. Y esta fase acaba de medir
   **cuanto** cabe ahi: los umbrales estan debilmente identificados, con desviaciones
   de 0.04 a 0.065, y recalibrarlos empeora. La via que funciono una vez esta ahora
   medida, y no se ha cerrado por desinteres.

## 4. Que se cierra y que queda

Cerrados por la fase, con su cifra:

| track | cifra que lo refuta |
|---|---|
| P0.2 y P0.5, umbral por rebanada | 0.774954, cuatro variantes con shrinkage |
| P0.4, calibracion de probabilidades | +0.001831 frente a no calibrar, y se cerro |
| P2.1, maquina de margen | 10 filas de interseccion de 1.483 |
| P2.3, generativo | 0.761203 |
| P3, convolutional de caracteres desde cero | 0.741036 |
| P4, doble vista con normalizacion | -0.023836 |
| P5, canal de negacion | -0.250519 |
| P7, modelo por rebanada | -0.002292 |
| P9, perdida orientada a la F1 | -0.0022 frente al control BCE |
| P8, prior de etiqueta del grupo | -0.0011 frente al control |
| P0.1, umbral anidado estricto | -0.0019 frente al control de tres niveles |
| P0.3 y P10, estimador robusto y consenso | cerrados por evidencia: ver D-055 |
| P1, promedio de modelos en la prediccion | aplazado: solo medible en leaderboard |

**Queda abierto y sin ejecutar: P1** en su formulacion de solo prediccion, y la
verificacion de transferencia de un finalista en el leaderboard, que es el unico
lugar donde queda por medir si la mejora local se transfiere.

## 5. Las tres conclusiones que sobreviven

1. **Aproximar mejor el objetivo no es mejor.** La F1 suave, que es el sustituto
   literal de la metrica, fue la peor de las tres perdidas. Con 1.186 filas, optimizar
   el objetivo directamente acopla el descenso a la composicion de esa muestra. Es lo
   mismo que le paso a la busqueda con Optuna, que tuvo el mejor metrica local y el
   peor leaderboard de los tres intentos.
2. **Ser senal y ser senal nueva son dos cosas distintas.** La repeticion de tuits
   es senal, como concluyo la fase v6, pero es senal que **V03 ya captura**: dos filas
   del mismo grupo son casi el mismo texto, el modelo lineal ya les da la misma
   prediccion, y un prior casi determinista no aporta nada que el texto no tenga ya.
   La caracteristica no arreglo ni una sola fila de las que tenia evidencia.
3. **Los umbrales estan debilmente identificados.** El de toxicidad puede
   moverse de 0.37 a 0.60 sin que el dato lo impida. Los congelados de V03 caen
   dentro del rango por pliegue, asi que no son sobreajuste, pero recalibrarlos
   empeora. Eso explica por que reajustarlos sobre el mismo conjunto infla +0.0036.

## 6. Cumplimiento

- prohibiciones comprobadas por recuento sobre el codigo de la fase: 9, con **0 infracciones**
- 0 submissions, 0 sondas de leaderboard, 0 uso del holdout
- rama `proyecto-kaggle-toxic-2026-v7` con 4 commits de la fase, **sin push**

## 7. Entregables

| entregable | contenido |
|---|---|
| `reports/v7_final_research.md` | este informe |
| `reports/v7_gap_audit.md` y `.csv` | auditoria de brecha de los once tracks |
| `reports/v7_gap_verificaciones.json` | evidencia cruda de las comprobaciones |
| `reports/v7_master_results.csv` | una fila por variante, con metricas y gate |
| `reports/v7_candidate_comparison.csv` | las tres mas cercanas al gate |
| `reports/v7_error_intersection.csv` | interseccion de errores |
| `reports/v7_calibration_analysis.csv` | los quince umbrales con su dispersion |
| `reports/v7_compliance.md` | recuento programatico de las restricciones |
| `reports/v7_reproducibility.md` | como reproducir cada resultado |
| `reports/figuras/fig_66` a `fig_68` | tres figuras de la fase |

Fecha: 2026-09-26T02:19:27+00:00. Duracion de la consolidacion: 1.5 s.
