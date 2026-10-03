# V10-02: control de equivalencia y banda interna de ruido de particion

Generado por `experiments/v10_02_ruido_particion.py` (V10-02 (ruido de particion y control de equivalencia)).

## 1. Por que nace este experimento

P0b creo el holdout2 y el gate no se supero: la referencia de V03 recalculada en dev2
quedo en 0.763435, una desviacion de -0.013776
respecto de la referencia del desarrollo completo, por encima de la tolerancia
declarada de 0.005.

Ese numero admite tres explicaciones, y la regla 9 de la fase no permite afirmar
ninguna sin evidencia:

1. que la ruta de codigo de P0b no sea equivalente a la canonica, lo que seria un fallo;
2. que reducir el conjunto de ajuste de 1483 a 1236 filas cueste de verdad esa cantidad;
3. que la metrica tenga una banda de ruido de particion de ese orden.

Este script las separa con controles baratos.

## 2. Control de equivalencia de la ruta de codigo

| magnitud | valor |
|---|---|
| metrica con los pliegues canonicos | 0.7772114525165721 |
| referencia publicada | 0.7772114525165721 |
| diferencia | +0.000e+00 |
| equivalencia exacta | **True** |

Este control **descarta la explicacion 1**: el codigo de P0b es numericamente
equivalente al pipeline canonico, asi que su numero de 0.763436 es real y no un
artefacto de implementacion.

## 3. Banda de ruido por particion

Se recalculo la metrica sobre el desarrollo completo con 8
semillas de particion distintas, sin cambiar nada mas.

| semilla | metrica | frente a la canonica | celdas distintas |
|---|---|---|---|
| 42 | 0.777211 | +0.000000 | 0 |
| 7 | 0.772647 | -0.004565 | 253 |
| 123 | 0.772021 | -0.005190 | 250 |
| 2024 | 0.778237 | +0.001026 | 258 |
| 5150 | 0.779432 | +0.002221 | 278 |
| 20260926 | 0.789205 | +0.011993 | 245 |
| 31337 | 0.779585 | +0.002373 | 246 |
| 99 | 0.776682 | -0.000529 | 255 |

- recorrido: **0.017184**
- desviacion tipica: **0.005302**

## 4. Ruido por tamano de conjunto

Se repitio el calculo sobre subconjuntos de unas 1236 filas,
el tamano de dev2, para separar la contribucion del tamano de la de la particion.

| repeticion | semilla | filas | metrica |
|---|---|---|---|
| 0 | 42 | 1236 | 0.775172 |
| 1 | 7 | 1236 | 0.770735 |
| 2 | 123 | 1236 | 0.762910 |
| 3 | 2024 | 1236 | 0.772677 |
| 4 | 5150 | 1236 | 0.762169 |
| 5 | 20260926 | 1236 | 0.762500 |
| 6 | 31337 | 1236 | 0.782417 |
| 7 | 99 | 1236 | 0.769471 |

- recorrido: **0.020248**

## 5. Que explicacion gana

El valor de dev2, 0.763436, esta dentro de la banda de
particion del desarrollo completo: **False**.

## 6. Poder de deteccion del holdout2

Con las 247 filas de holdout2, un cambio de decision del 5 por ciento de las celdas da una diferencia observada de -0.028672, con intervalo al 95 por ciento [-0.050754, -0.008429].

- el intervalo incluye el cero: **False**
- 0.2 por ciento a favor de A de las repeticiones

Un cambio de decision del 5 por ciento, que es del orden de lo que mueven las lineas
de la via A, **no es distinguible de cero** con este tamano por comparacion pareada.

Fecha: 2026-09-26T07:43:25+00:00. Duracion: 26.2 s.
