# V26-02: Tsetlin Machine binaria, con control lineal de la misma matriz

Generado por `experiments/v26_02_tsetlin.py` (V26-02 (Tsetlin Machine binaria con control lineal de la misma matriz)).

**Ninguna submission.**

## 1. La pregunta

las conjunciones booleanas encuentran errores de V03 que un modelo lineal sobre la misma informacion no puede capturar

## 2. Controles

| control | resultado |
|---|---|
| aritmetica de clausulas | esperado conteo manual 4, prob 1.000000, obtenido conteo 4, prob 1.000000 |
| literal negativo saca la clausula positiva | esperado conteo 4.0 -> 3.0, obtenido conteo 4.0 -> 3.0 |
| polaridad en la clausula, no en los literales | esperado conteo positivo 4.0 sin cambio, obtenido conteo positivo 4.0 |
| entrada vacia -> probabilidad 1 | esperado 1.0, obtenido 1.000000 |
| determinismo misma semilla | esperado modelos identicos, obtenido identicos |
| K clausulas de cada polaridad | esperado 4 y 4, obtenido 4 y 4 |
| matriz word12 identica a TF-IDF | True |
| matriz word11 identica a TF-IDF | True |
| matriz charwb35 identica a TF-IDF | True |

## 3. El control lineal sobre la MISMA matriz binaria

La comparacion informativa es de tres vias: Tsetlin(binario) vs LR(binario) aisla el mecanismo; LR(binario) vs V03 mide el coste de la binarizacion; Tsetlin vs V03 es el efecto neto.

| representacion | LR(binario) OOF | dOOF vs V03 | train F1 | gap | neto |
|---|---:|---:|---:|---:|---:|
| word12 | 0.751393 | -0.025818 | 0.9903 | +0.2389 | -97 |
| word11 | 0.732509 | -0.044703 | 0.9632 | +0.2307 | -183 |
| charwb35 | 0.733128 | -0.044084 | 1.0000 | +0.2669 | -146 |

## 4. La progresion de clausulas

| K | TM OOF | dOOF vs V03 | train F1 | gap | clausulas activas | literales activos | rescates | danos | neto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.292547 | -0.484664 | 0.2216 | -0.0709 | 200 | 221627 | 387 | 1565 | -1178 |

**Corte en K=200** por gap -0.0709 o neto -1178.

