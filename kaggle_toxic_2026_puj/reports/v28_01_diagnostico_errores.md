# V28-01: diagnostico de errores de V03 por etiqueta

Generado por `experiments/v28_01_diagnostico_errores.py` (V28-01 (diagnostico de errores de V03 por etiqueta, condicionado por rasgos del texto)).

**Estado `DIAGNOSTIC_ONLY`. Alcance: no se entrena ningun modelo. Los rasgos se usan solo para DESCRIBIR los errores, no como features de un clasificador**

## 1. La estructura del error por etiqueta

| etiqueta | F1 V03 | errores | tasa | FN / positivos | FP | TP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| HS | 0.7617 | 316 | 0.2131 | 143 / 648 (0.2207) | 173 | 505 | 662 |
| TR | 0.8157 | 164 | 0.1106 | 82 / 445 (0.1843) | 82 | 363 | 956 |
| AG | 0.7542 | 277 | 0.1868 | 142 / 567 (0.2504) | 135 | 425 | 781 |

## 2. Los errores de las tres etiquetas, juntos y separados

| numero de etiquetas falladas | filas | fraccion |
|---:|---:|---:|
| 0 | 1042 | 0.7026 |
| 1 | 223 | 0.1504 |
| 2 | 120 | 0.0809 |
| 3 | 98 | 0.0661 |

El punto que decide si tiene sentido un especialista por etiqueta es cuanto se solapan
los errores. Si HS y TR fallan sobre los mismos textos, un especialista por etiqueta
no puede ganar nada, porque solo puede cambiar las decisiones donde ambos fallan.

| par | correlacion phi | ambos fallan | esperados si independientes | exceso | Jaccard |
|---|---:|---:|---:|---:|---:|
| HS_TR | +0.4203 | 115 | 35 | +80 | 0.3151 |
| HS_AG | +0.5069 | 179 | 59 | +120 | 0.4324 |
| TR_AG | +0.4930 | 120 | 31 | +89 | 0.3738 |

## 3. Que rasgo del texto distingue V03 wrong de V03 correct

Cohen d es la diferencia tipada de medias en desviaciones estandar. Se listan los cinco
rasgos de mayor efecto absoluto para cada etiqueta.

### HS  (F1 0.7617)

| rasgo | media si error | media si correcto | d | p |
|---|---:|---:|---:|---:|
| tokens | 19.4747 | 16.5938 | +0.2748 | 2.00e-05 |
| es_muy_corto | 0.0063 | 0.0480 | -0.2583 | 7.06e-04 |
| caracteres | 117.0696 | 101.3556 | +0.2570 | 2.55e-05 |
| alfabeticos | 92.1709 | 80.1011 | +0.2493 | 4.86e-05 |
| densidad_lexica | 0.9270 | 0.9454 | -0.2477 | 3.54e-05 |

### TR  (F1 0.8157)

| rasgo | media si error | media si correcto | d | p |
|---|---:|---:|---:|---:|
| es_muy_corto | 0.0000 | 0.0440 | -0.3032 | 6.18e-03 |
| longitud_media_token | 5.9967 | 6.4205 | -0.2848 | 1.49e-04 |
| urls | 0.2439 | 0.3434 | -0.2042 | 2.88e-02 |
| tiene_url | 0.2378 | 0.3177 | -0.1787 | 3.69e-02 |
| puntuacion | 3.0488 | 3.7218 | -0.1566 | 2.76e-01 |

### AG  (F1 0.7542)

| rasgo | media si error | media si correcto | d | p |
|---|---:|---:|---:|---:|
| tokens | 20.1227 | 16.5381 | +0.3372 | 1.03e-06 |
| alfabeticos | 95.6679 | 79.6882 | +0.3218 | 5.31e-06 |
| caracteres | 120.8881 | 100.9867 | +0.3197 | 3.29e-06 |
| es_muy_corto | 0.0072 | 0.0464 | -0.2444 | 2.41e-03 |
| densidad_lexica | 0.9269 | 0.9448 | -0.2437 | 3.49e-05 |

## 4. Regiones del espacio textual

Tamano de cada region y tasa de error de V03 dentro. Un subconjunto solo justifica un
especialista si es grande (al menos el 6 % del desarrollo) y ademas
difiere de la tasa global.

| region | filas | fraccion | error HS | error TR | error AG | suficiente |
|---|---:|---:|---:|---:|---:|:-:|
| densidad lexica alta (> 0.90) | 1087 | 0.7330 | 0.1914 | 0.1040 | 0.1610 | si |
| con mencion | 702 | 0.4734 | 0.1752 | 0.1182 | 0.1524 | si |
| con @ en el primer tercio | 639 | 0.4309 | 0.1674 | 0.1268 | 0.1487 | si |
| con URL | 458 | 0.3088 | 0.2445 | 0.0852 | 0.2227 | si |
| sin ninguna marca (@, #, URL) | 390 | 0.2630 | 0.2410 | 0.1179 | 0.2051 | si |
| tweets largos (>= 25 tokens) | 301 | 0.2030 | 0.2492 | 0.0963 | 0.2458 | si |
| con hashtag | 144 | 0.0971 | 0.2569 | 0.0833 | 0.2083 | si |
| alta puntuacion (> 0.10 por caracter) | 96 | 0.0647 | 0.1979 | 0.0833 | 0.1458 | si |
| casi todos mayusculas (frac > 0.7) | 72 | 0.0486 | 0.1944 | 0.1250 | 0.1944 | no |
| tweets muy cortos (<= 4 tokens) | 58 | 0.0391 | 0.0345 | 0.0000 | 0.0345 | no |

## 5. Lo que este diagnostico no hace

- No entrena ningun modelo. Los rasgos describen los errores, no alimentan un clasificador.
- No lee el test.
- No usa etiquetas para calcular ningun rasgo, de modo que no puede filtrar informacion.

Coste: 0.2 s, memoria 185 MB, 0 modelos, 0 submissions.

