# V13 · P16d — ¿Hay una puerta realizable?

## 1. Pregunta y respuestas previas

- El bloque de evidencia por token tiene señal propia: control *block-only* de 0.4000 frente a un suelo de azar de 0.1667.
- Su varianza es en parte accesible desde V12 (mejor R2 mediano 0,522), pero la reconstrucción lineal de V12 **no** conserva la información que discrimina clases (0,0485).
- El techo con puerta **perfecta** sería 0.6770 de exactitud, muy por encima de los 0.6005 de V12. Pero una puerta perfecta requiere saber qué filas se equivocan.
- La confianza del bloque separa mal sus propios aciertos de sus fallos: **AUC 0.7173** (0,5 sería azar).

## 2. La rejilla de puertas realizables

Regla: se usa el bloque cuando su confianza supera `t_b` **y** la de V12 no supera `t_v`. Solo dos umbrales sobre cantidades que el modelo produce en inferencia: nada oracular en la regla.

| Umbral bloque | Umbral V12 | Filas desviadas | macro-F1 | Delta |
|---|---|---|---|---|
| 0.0 | 0.0 | 0 (0.0 %) | 0.6005 | +0.0000 |
| 0.95 | 0.65 | 41 (0.02 %) | 0.6005 | -0.0000 |
| 0.75 | 0.0 | 0 (0.0 %) | 0.6005 | +0.0000 |
| 0.85 | 0.0 | 0 (0.0 %) | 0.6005 | +0.0000 |
| 0.85 | 0.35 | 26 (0.02 %) | 0.6005 | -0.0000 |
| 0.95 | 0.0 | 0 (0.0 %) | 0.6005 | +0.0000 |
| 0.95 | 0.35 | 0 (0.0 %) | 0.6005 | +0.0000 |
| 0.95 | 0.45 | 10 (0.01 %) | 0.6005 | -0.0000 |

## 3. Elección honesta de los umbrales

El máximo de la rejilla está elegido mirando el mismo OOF, así que está sesgado hacia arriba. Partir los folds en dos mitades, elegir los umbrales en una y medir en la otra es una primera aproximación a la segunda partición que exige §25.

| Elegido en | Umbrales | macro-F1 en la mitad elegida | macro-F1 en la otra | Delta en la otra |
|---|---|---|---|---|
| A | [0.0, 0.0] | 0.6029 | 0.5989 | +0.0000 |
| B | [0.0, 0.0] | 0.5989 | 0.6029 | +0.0000 |

## 4. Decision

- El máximo de la rejilla da +0.0000 y, elegido sin honestidad, +0.0000. **La pista se cierra**: ni siquiera una puerta—oracle-free construida sobre la confianza llega al gate de +0,003.
- La razón de fondo es que la confianza del bloque separa sus aciertos de sus fallos con un AUC de 0.7173, apenas por encima del azar: no hay señal de disparo que usar, y sin gatillo el techo de 0.6770 es inalcanzable.

