# V19-02: la restriccion TR implica HS como regla de decision

Generado por `experiments/v19_02_restriccion_tr_implica_hs.py` (V19-02 (regla de estructura TR implica HS, con un parametro y disciplina de pliegues)).

**Un solo parametro, elegido por pliegues. Ninguna submission gastada.**

## 1. El hecho medido que sostiene la regla

No hay ni una sola fila con TR=1 y HS=0 en las 1483 del desarrollo, cuando por
independencia se esperarian 250. Y la implicacion es de una sola direccion, porque AG=1 con
HS=0 si ocurre. Eso es un hecho, no una intuicion sobre el significado de las etiquetas.

## 2. Por que la propagacion correcta es hacia arriba

| direccion | celdas | macro | delta |
| --- | ---: | ---: | ---: |
| V03 sin regla | 0 | 0.777211 | +0.000000 |
| HS=0 obliga a TR=0 | 20 | 0.776566 | -0.000646 |
| TR=1 obliga a HS=1 | 20 | 0.777896 | +0.000685 |

Gana la de arriba porque **el error esta en HS, no en TR**: en 9 de las 20 celdas que violan
la restriccion la verdad es (1,1,1), o sea V03 fallo HS y acerto TR. Forzar TR=0 ahi
destruye un acierto por culpa de un error de la otra etiqueta.

## 3. La regla con un parametro, elegida por pliegues

El gatillo es el score de TR a partir del cual se fuerza HS=1. Se elige en cuatro pliegues
y se evalua en el quinto, con los cinco rotados.

| pliegue | trigger | cambios | netos | macro sin regla | macro con regla | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.62 | 1 | +1 | 0.796010 | 0.797558 | +0.001548 |
| 2 | 0.48 | 5 | -1 | 0.794034 | 0.794134 | +0.000100 |
| 3 | 0.48 | 5 | -3 | 0.778860 | 0.776677 | -0.002183 |
| 4 | 0.48 | 9 | +1 | 0.738598 | 0.742763 | +0.004165 |
| 5 | 0.48 | 3 | -1 | 0.779056 | 0.778740 | -0.000317 |

- Delta medio fuera de muestra: **+0.000663**, con
  desviacion 0.002368.
- Pliegues que mejoran: **3 de 5**.
- Valor p empirico por permutacion de etiquetas: **0.7944**.

## 4. La traduccion a leaderboard, que es lo que decide

La regla cambia **26 celdas** de HS en todo el OOF:

| tipo de cambio | celdas | cambio neto |
| --- | --- | --- |
| FN->TP | 9 | +9 |
| TN->FP | 11 | -11 |
| TP->FN | 0 | +0 |
| FP->TN | 0 | +0 |

El cambio neto de celda es **-3**, y la escala del
proyecto es de **107 cambios netos** para mover el leaderboard 0.003. Una
regla competitive tendria que aportar al menos **27**.

## 5. No es un cambio de umbral disfrazado

Un umbral de HS mas bajo que 0.44 tomaria **805 celdas**. Esta
regla toca **26**, y las que deja intactas incluyen celdas por debajo de ese umbral en
filas donde el score de TR no llega al gatillo. La regla depende del score de la **otra
etiqueta**, luego no es una funcion monotona del score de HS y no se puede escribir como un
umbral.

## 6. Veredicto

**REGLA DESCARTADA**

- Delta medio: +0.000663, con 3 de 5 pliegues mejorando.
- Cambios netos: -3 frente a 27 necesarios.
- Submissions gastadas: **0**.


Fecha: 2026-09-27T03:20:05+00:00. Duracion: 0.1 s.
