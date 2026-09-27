# V30-01: perfil del error residual de V03

Generado por `experiments/v30_01_perfil_error_residual.py` (V30-01 (perfil del error residual de V03, orientado a encontrar el siguiente eje)).

**Estado `DIAGNOSTIC_ONLY`. Alcance: no se entrena ningun modelo, no se modifica V03, no se construye predictor y no se produce submission. Las caracteristicas describen el error, no alimentan un clasificador**

## 1. Las poblaciones historicas, reconstruidas sin redefinir

la de experiments/28_hardness_v4.py lineas 593-598: Posible ruido si cluster_contradictorio > 0; Dificil coherente si hardness_score >= percentil 80; Facil en otro caso. hardness_score = 1 - |(p - umbral) * (2y - 1)| por etiqueta con los umbrales congelados de V03.

| poblacion | filas | fraccion |
|---|---:|---:|
| Facil | 1151 | 0.7761 |
| Dificil coherente | 289 | 0.1949 |
| Posible ruido | 43 | 0.0290 |

Umbral de dureza (percentil 80): 0.8327.
Filas en cluster contradictorio: 43.

## 2. Se separan Facil y Dificil coherente en algo?

Cohen d es la diferencia de medias en desviaciones estandar. Se listan los ocho rasgos de
mayor efecto absoluto.

| rasgo | d (dificil - facil) | media facil | media dificil | p |
|---|---:|---:|---:|---:|
| longitud_media_token | -0.368 | 6.463 | 5.927 | 1.31e-09 |
| urls | -0.336 | 0.361 | 0.201 | 4.74e-06 |
| tiene_url | -0.309 | 0.332 | 0.197 | 8.83e-06 |
| es_muy_corto | -0.286 | 0.049 | 0.003 | 4.29e-04 |
| puntuacion | -0.234 | 3.818 | 2.848 | 7.76e-03 |
| densidad_lexica | -0.167 | 0.944 | 0.932 | 7.52e-03 |
| digitos | -0.162 | 1.166 | 0.879 | 1.68e-02 |
| frac_alfabeticos | -0.137 | 0.792 | 0.787 | 5.43e-03 |

## 3. Perfil por label

| label | errores | tasa | FP | FN | FP / predichos | FN / positivos |
|---|---:|---:|---:|---:|---:|---:|
| HS | 316 | 0.2131 | 173 | 143 | 0.2552 | 0.2207 |
| TR | 164 | 0.1106 | 82 | 82 | 0.1843 | 0.1843 |
| AG | 277 | 0.1868 | 135 | 142 | 0.2411 | 0.2504 |

Los tres rasgos de mayor efecto para cada label:

### HS

| rasgo | d | media si error | media si correcto | p |
|---|---:|---:|---:|---:|
| tokens | +0.275 | 19.475 | 16.594 | 2.00e-05 |
| es_muy_corto | -0.258 | 0.006 | 0.048 | 7.06e-04 |
| caracteres | +0.257 | 117.070 | 101.356 | 2.55e-05 |

### TR

| rasgo | d | media si error | media si correcto | p |
|---|---:|---:|---:|---:|
| es_muy_corto | -0.303 | 0.000 | 0.044 | 6.18e-03 |
| longitud_media_token | -0.285 | 5.997 | 6.420 | 1.49e-04 |
| urls | -0.204 | 0.244 | 0.343 | 2.88e-02 |

### AG

| rasgo | d | media si error | media si correcto | p |
|---|---:|---:|---:|---:|
| tokens | +0.337 | 20.123 | 16.538 | 1.03e-06 |
| alfabeticos | +0.322 | 95.668 | 79.688 | 5.31e-06 |
| caracteres | +0.320 | 120.888 | 100.987 | 3.29e-06 |

## 4. P(V03 wrong | region de texto)

Las regiones se cortan por cuantil para que tengan volumen comparable. El exceso es la
tasa de error dentro de la region menos la tasa fuera.

| region | filas | HS dentro | TR dentro | AG dentro |
|---|---:|---:|---:|---:|
| decil 1 de longitud (mas corto) | 167 | 0.0778 | 0.0359 | 0.0838 |
| decil 10 de longitud (mas largo) | 158 | 0.3038 | 0.1139 | 0.2911 |
| texto corto (tercio inferior de tokens) | 514 | 0.1615 | 0.0992 | 0.1362 |
| texto largo (tercio superior de tokens) | 499 | 0.2525 | 0.1122 | 0.2425 |
| con URL | 458 | 0.2445 | 0.0852 | 0.2227 |
| sin URL | 1025 | 0.1990 | 0.1220 | 0.1707 |
| con mencion | 702 | 0.1752 | 0.1182 | 0.1524 |
| sin mencion | 781 | 0.2471 | 0.1037 | 0.2177 |
| con hashtag | 144 | 0.2569 | 0.0833 | 0.2083 |
| sin hashtag | 1339 | 0.2084 | 0.1135 | 0.1845 |
| densidad lexica baja (tercio inferior) | 524 | 0.2576 | 0.1107 | 0.2328 |
| densidad lexica alta (tercio superior) | 730 | 0.1699 | 0.1055 | 0.1466 |
| casi todo en mayusculas (frac > 0,7) | 72 | 0.1944 | 0.1250 | 0.1944 |
| alta puntuacion (> 0,10 por caracter) | 96 | 0.1979 | 0.0833 | 0.1458 |
| con caracteres repetidos | 275 | 0.2982 | 0.0873 | 0.2327 |
| mencion en el primer tercio | 639 | 0.1674 | 0.1268 | 0.1487 |
| hashtag en el primer tercio | 64 | 0.2656 | 0.0312 | 0.1719 |

## 5. Cruces predefinidos

### longitud x URL

| tramo | condicion | filas | HS | TR | AG |
|---|---|---:|---:|---:|---:|
| corto | no | 396 | 0.1338 | 0.0934 | 0.1212 |
| corto | si | 118 | 0.2542 | 0.1186 | 0.1864 |
| medio | no | 323 | 0.2353 | 0.1455 | 0.1858 |
| medio | si | 147 | 0.2109 | 0.0680 | 0.1769 |
| largo | no | 306 | 0.2451 | 0.1340 | 0.2190 |
| largo | si | 193 | 0.2642 | 0.0777 | 0.2798 |

### longitud x mencion

| tramo | condicion | filas | HS | TR | AG |
|---|---|---:|---:|---:|---:|
| corto | no | 220 | 0.2273 | 0.1182 | 0.1864 |
| corto | si | 294 | 0.1122 | 0.0850 | 0.0986 |
| medio | no | 272 | 0.2353 | 0.1066 | 0.2022 |
| medio | si | 198 | 0.2172 | 0.1414 | 0.1566 |
| largo | no | 289 | 0.2734 | 0.0900 | 0.2561 |
| largo | si | 210 | 0.2238 | 0.1429 | 0.2238 |

### longitud x hashtag

| tramo | condicion | filas | HS | TR | AG |
|---|---|---:|---:|---:|---:|
| corto | no | 489 | 0.1534 | 0.0982 | 0.1329 |
| corto | si | 25 | 0.3200 | 0.1200 | 0.2000 |
| medio | no | 422 | 0.2370 | 0.1303 | 0.1919 |
| medio | si | 48 | 0.1458 | 0.0417 | 0.1042 |
| largo | no | 428 | 0.2430 | 0.1145 | 0.2360 |
| largo | si | 71 | 0.3099 | 0.0986 | 0.2817 |

### longitud x label

| tramo | condicion | filas | HS | TR | AG |
|---|---|---:|---:|---:|---:|
| corto | (tramo unico) | 514 | 0.1615 | 0.0992 | 0.1362 |
| medio | (tramo unico) | 470 | 0.2277 | 0.1213 | 0.1830 |
| largo | (tramo unico) | 499 | 0.2525 | 0.1122 | 0.2425 |

## 6. Estabilidad entre pliegues

Un fenomeno presente en un solo pliegue no es una estructura. Se cuenta en cuantos de los
cinco pliegues la tasa de error de la region supera a la del resto.

| region | pliegues con la misma direccion | delta medio |
|---|---:|---:|
| decil 1 de longitud (mas corto) | 0/5 | -0.1596 |
| decil 10 de longitud (mas largo) | 5/5 | +0.1227 |
| texto corto (tercio inferior de tokens) | 0/5 | -0.0792 |
| texto largo (tercio superior de tokens) | 5/5 | +0.0595 |
| con URL | 4/5 | +0.0473 |
| sin URL | 1/5 | -0.0473 |
| con mencion | 0/5 | -0.0720 |
| sin mencion | 5/5 | +0.0720 |
| con hashtag | 3/5 | +0.0456 |
| sin hashtag | 2/5 | -0.0456 |
| densidad lexica baja (tercio inferior) | 5/5 | +0.0679 |
| densidad lexica alta (tercio superior) | 0/5 | -0.0847 |
| casi todo en mayusculas (frac > 0,7) | 2/5 | -0.0008 |
| alta puntuacion (> 0,10 por caracter) | 3/5 | -0.0159 |
| con caracteres repetidos | 4/5 | +0.0994 |
| mencion en el primer tercio | 0/5 | -0.0808 |
| hashtag en el primer tercio | 4/5 | +0.0544 |

## 7. Lo que este script no hace

- No entrena ningun modelo.
- No modifica V03, sus umbrales ni su representacion.
- No construye un predictor con las caracteristicas.
- No lee el test.
- No produce submission.

Coste: 0.2 s, memoria 198 MB, 0 modelos, 0 submissions.

