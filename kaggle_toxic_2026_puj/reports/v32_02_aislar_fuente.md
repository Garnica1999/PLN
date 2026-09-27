# V32-02: aislar la fuente de la ganancia de V31-04

Generado por `experiments/v32_02_aislar_fuente.py` (V32-02 (aislar la fuente de la ganancia de V31-04)).

**Estado `OOF_EVALUATED`. Alcance: tres variantes: URL aditivo, solo interacciones, y ambas. Sin combinar con otros bloques**

## 1. Las tres variantes

| variante | corrected_cv | delta vs V03 | rescates | danios | net |
|---|---:|---:|---:|---:|---:|
| V32-A URL aditivo | 0.7774199655 | +0.0002085130 | 35 | 29 | +6 |
| V32-B solo interacciones | 0.7795623565 | +0.0023509040 | 58 | 47 | +11 |
| V32-C URL + interacciones | 0.7813779888 | +0.0041665363 | 70 | 53 | +17 |

## 2. La descomposicion de la ganancia

| paso | delta |
|---|---:|
| V03 -> URL aditivo | +0.0002085130 |
| URL aditivo -> URL + interacciones | +0.0039580233 |
| V03 -> solo interacciones | +0.0023509040 |
| V03 -> URL + interacciones | +0.0041665363 |

**El 95.0 % de la ganancia se explica por las interacciones.**

## 3. Lo que este script no hace

- No combina con mencion ni con enfasis.
- No genera submission. No lee el test.

Coste: 15.5 s, memoria 228 MB, 4 modelos, 0 submissions.

