# V11-06: validacion de los estimadores de prevalencia bajo shift simulado (P2.3)

Generado por `experiments/v11_06_validacion_estimadores.py` (V11-06 (P2.3, validacion de estimadores bajo shift simulado)).

## 1. La pregunta que decide el resto de la fase

La auditoria de legalidad cerro el uso de la prueba para generar un submission, de modo
que la v11 solo puede estudiar la metodologia: si el mecanismo recupera un desplazamiento
de prevalencia que se conoce de antemano.

Antes de medir si la correccion de prior o la calibracion ayudan bajo shift, hay que
responder una pregunta mas basica y mas dura: **los estimadores funcionan?** Si se
implanta artificialmente un desplazamiento conocido y el estimador no lo recupera, todo
lo que se construya encima estara construido sobre arena, y es mucho mas barato saberlo
ahora que al final.

## 2. Los dos estimadores, y por que conviene tenerlos

- **EM de Saerens**, iteracion de punto fijo. Es el que uso V9-02. Necesita las
  probabilidades y un prior de partida.
- **Clasificar-y-contar ajustado**, de Forman y Crawford. No usa probabilidades sino
  **conteos**: si se predicen `m` positivos en `n` filas y se conocen la sensibilidad y la
  especificidad en origen, la prevalencia se despeja de una identidad algebraica.

Son ideas completamente distintas: uno ajusta una verosimilitud, el otro despeja una
ecuacion. Si los dos fallan, el problema no es el estimador sino la idea; si solo falla
uno, es el estimador. Por eso se miden los dos.

## 3. Como se simula el desplazamiento

Un cambio de prevalencia cambia `P(y)` manteniendo `P(x|y)`: las mismas probabilidades
por fila y distinto peso por clase. Reponderar positivos y negativos para que la
prevalencia ponderada sea la objetivo reproduce exactamente eso.

La simulacion es ponderada y **no** por submuestreo, porque lo que se mide es el sesgo
del estimador y no el ruido de muestreo que el submuestreo introduciria.

## 4. Metricas de la distribucion de origen, medidas sobre el OOF

| etiqueta | prevalencia origen | sensibilidad | especificidad |
| --- | --- | --- | --- |
| HS | 0.436952 | 0.7639 | 0.8024 |
| TR | 0.300067 | 0.7820 | 0.9306 |
| AG | 0.382333 | 0.7055 | 0.8788 |

## 5. Recuperacion del desplazamiento, etiqueta por etiqueta

| etiqueta | factor pedido | factor efectivo | EM estimada | error rel. EM | contar ajustada | error rel. contar |
| --- | --- | --- | --- | --- | --- | --- |
| HS | 0.70 | 0.7000 | 0.3120 | +0.0200 | 0.3059 | -0.0000 |
| HS | 0.80 | 0.8000 | 0.3543 | +0.0137 | 0.3496 | -0.0000 |
| HS | 0.90 | 0.9000 | 0.3967 | +0.0088 | 0.3933 | -0.0000 |
| HS | 1.00 | 1.0000 | 0.4391 | +0.0050 | 0.4370 | -0.0000 |
| HS | 1.10 | 1.1000 | 0.4816 | +0.0020 | 0.4806 | -0.0000 |
| TR | 0.70 | 0.7000 | 0.2104 | +0.0018 | 0.2100 | +0.0000 |
| TR | 0.80 | 0.8000 | 0.2399 | -0.0007 | 0.2401 | +0.0000 |
| TR | 0.90 | 0.9000 | 0.2694 | -0.0026 | 0.2701 | +0.0000 |
| TR | 1.00 | 1.0000 | 0.2989 | -0.0040 | 0.3001 | +0.0000 |
| TR | 1.10 | 1.1000 | 0.3284 | -0.0052 | 0.3301 | +0.0000 |
| AG | 0.70 | 0.7000 | 0.2719 | +0.0159 | 0.2676 | +0.0000 |
| AG | 0.80 | 0.8000 | 0.3092 | +0.0108 | 0.3059 | +0.0000 |
| AG | 0.90 | 0.9000 | 0.3465 | +0.0069 | 0.3441 | +0.0000 |
| AG | 1.00 | 1.0000 | 0.3839 | +0.0040 | 0.3823 | +0.0000 |
| AG | 1.10 | 1.1000 | 0.4213 | +0.0016 | 0.4206 | +0.0000 |

## 6. Gates pre-registrados

- **Gate A, recuperacion**: error relativo medio menor que 0.05.
  - EM: **0.006862**, supera: True
  - clasificar y contar: **0.000001**, supera: True
- **Gate B, no incorromper**: con shift nulo, el error debe seguir siendo pequeno. Un
  estimador que sin desplazamiento inventa una correccion es peor que no hacer nada,
  porque genera confianza falsa.
  - EM: **0.004341**, no corrompe: True
  - clasificar y contar: **0.000001**, no corrompe: True

**La fase puede continuar: True.**
**Estimador recomendado: EM de Saerens.**

## 7. Veredicto

**LOS ESTIMADORES FUNCIONAN.** El error relativo medio esta por debajo de la
tolerancia y el estimador no incorrumpe con shift nulo, de modo que la metodologia de
correccion de prevalencia tiene sobre que trabajar.

Conviene decir lo que esto **no** demuestra: que el estimador recupera un
desplazamiento que uno mismo ha implantado, sobre probabilidades que uno mismo ha
calibrado. No demuestra nada sobre la prevalencia real de la prueba, que es
precisamente lo que la auditoria de legalidad prohibits usar. Lo que demuestra es
que el mecanismo es correcto **en el caso en el que se puede comprobar**, que es lo
maximo que se puede pedir sin usar la prueba.

Fecha: 2026-09-26T21:55:32+00:00. Duracion: 0.8 s.
