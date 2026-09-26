# V10-08: confirmacion de integridad de holdout2 y filtro de detectabilidad (P3)

Generado por `experiments/v10_08_confirmacion_dev2.py` (V10-08 (P3, confirmacion e integridad)).

## 1. Por que esta fase es de verificacion y no de medicion

El gate de P0b no se supero, y por la regla pre-registrada **dev2 no se usa para
ninguna decision de adopcion**. V10-02 explico despues que la desviacion es ruido de
particion y de tamano, pero el veredicto del gate no se reinterpreta a posteriori.

Lo que queda es comprobar que lo congelado sigue congelado, confirmar que no hay nada
pendiente de juzgar, y responder a una pregunta distinta de la del gate: **aunque una
candidata hubiera mejorado, el externo de 1154 filas habria podido verlo?**

## 2. Integridad de los artefactos congelados

| comprobacion | resultado |
|---|---|
| los identificadores de holdout2 conservan su huella | **True** |
| huella de `config/holdout2.json` | `f71ad8ac4959dbdcccc7da23827a3692...` |
| huella de `data/processed/holdout2_ids.csv` | `f6c6ed958abf728b8210d15c26ec43fb...` |

## 3. La particion sigue siendo limpia

- particion limpia: **True**
- solape de filas: 0
- grupos compartidos: 0

## 4. Cuantas evaluaciones de holdout2 quedan

- gastadas: **0**
- disponibles: **3**

Las tres lineas de la via A se cerraron en sus propios gates, de modo que **ninguna llega
a la mesa de finales** y no hay nada que gastar. Las tres evaluaciones disponibles siguen
integras para el resto del proyecto.

## 5. Filtro de detectabilidad sobre las candidatas medidas

La regla de V8-02 dice que una candidata que cambia C celdas solo es resoluble por el
externo si su tasa de acierto entre esas celdas supera `0,5 + 19,2 / C`, porque con 1154
filas el leaderboard resuelve unos 19.2 aciertos netos.

| candidata | celdas | delta medio | mejoran | tasa minima | resoluble | enviable |
|---|---|---|---|---|---|---|
| V10-04 calibracion isotonica recortada | 0 | +0.000148 | 3/5 | no aplica | no | no |
| V10-05 regla F1 optima con prior simulado | 134 | -0.003921 | 1/5 | 0.6433 | si | no |
| V10-06 tres umbrales libres re-estimados por pliegue | 124 | +0.000131 | 3/5 | 0.6548 | si | no |

La prueba de mejora es la del addendum de protocolo, no la del signo: hacen falta las dos
cosas a la vez, delta medio de al menos 0.002 y al menos 4 de 5 particiones que mejoran.
Sin esa exigencia, V10-06 con su +0.000131 pareceria una candidata valida cuando es una
fraccion de la banda de ruido de 0.0053 que midio V10-02.

## 6. Veredicto

**ninguna candidata de la via A es enviable: las tres fallan sus gates de metrica, y ademas dos de ellas no serian resolubles por el externo ni aunque lo fueran.**

Lo relevante de la segunda condicion es que no es un tecnicismo: con 124 celdas, una
candidata necesitaria acertar mas del 65 por ciento de ellas para que el leaderboard la
separara del azar. Ninguna de las tres llega a ese punto, y dos de ellas ni siquiera
mueven celdas suficientes para que la medicion tenga potencia.

Fecha: 2026-09-26T08:02:00+00:00. Duracion: 0.0 s.
