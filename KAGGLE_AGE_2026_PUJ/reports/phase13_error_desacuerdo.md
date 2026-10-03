# Fase B — Análisis de errores y desacuerdo del modelo v12

Generado por `scripts/phase13_error_desacuerdo.py` sobre predicciones **OOF** (leave-one-fold-out, particiones agrupadas por clúster de near-duplicados).

## 1. Global

- macro-F1 OOF **0.6005** · accuracy 0.6007
- F1 por clase: {'13-17': 0.671, '18-24': 0.5059, '25-34': 0.4858, '35-49': 0.5593, '50-64': 0.6549, '65-xx': 0.7261}
- Error por distancia ordinal: {'0': 60.073, '1': 23.836, '2': 9.562, '3': 4.086, '4': 1.907, '5': 0.535}

## 2. Estructura de votos

- Todos los miembros fallan: **13.869%** · exactamente uno acierta: 11.387% · **rescatable (el ensamble falla y algún miembro acierta): 26.7%**
- Consenso medio: correctas 0.781 vs incorrectas 0.6839
- Entropía media: correctas 0.5855 vs incorrectas 0.7917
- Empate top-2: 1.752%

## 3. Rescate por familia

| familia     |   miembros |   exactitud_global |   pct_acierta_cuando_ensamble_falla |   unica_familia_que_acierta_pct |
|:------------|-----------:|-------------------:|------------------------------------:|--------------------------------:|
| ordinal     |          3 |             0.4569 |                               19.96 |                           2.499 |
| prototipos  |          1 |             0.404  |                               19.87 |                           2.151 |
| denso       |          1 |             0.4107 |                               19.27 |                           3.441 |
| bayes       |          1 |             0.5408 |                               16.07 |                           1.042 |
| prob_lineal |          2 |             0.5565 |                               15.32 |                           0.498 |
| svm_lineal  |          3 |             0.5637 |                               14.5  |                           0.102 |
| nbsvm       |          1 |             0.5635 |                               14.05 |                           0.373 |

## 4. Fronteras adyacentes

| frontera    |   filas |   exactitud_dentro |   errores |   pct_del_error_total |   pct_de_cruces_entre_la_pareja |
|:------------|--------:|-------------------:|----------:|----------------------:|--------------------------------:|
| 13-17|18-24 |   57539 |             0.5895 |     23622 |                 34.94 |                           18.99 |
| 18-24|25-34 |   57023 |             0.4963 |     28721 |                 42.48 |                           17.75 |
| 25-34|35-49 |   56577 |             0.4974 |     28438 |                 42.06 |                           12.82 |
| 35-49|50-64 |   56290 |             0.6124 |     21816 |                 32.27 |                           11.36 |
| 50-64|65-xx |   55220 |             0.7184 |     15550 |                 23    |                           10.26 |

## 5. Consenso y error

|   quintil_consenso |   filas |   error_pct |   rescatable_pct |
|-------------------:|--------:|------------:|-----------------:|
|                  0 |   29373 |       60.17 |            46.95 |
|                  1 |   18892 |       54.67 |            40.55 |
|                  2 |   25111 |       47.47 |            33.6  |
|                  3 |   60213 |       35.26 |            21.69 |
|                  4 |   35747 |       18.07 |             6.32 |

## 6. Segmentos (extracto)

| segmento   | valor             |   filas |   error_pct |   rescatable_pct |
|:-----------|:------------------|--------:|------------:|-----------------:|
| longitud   | L1                |   42917 |       47.7  |            32.99 |
| longitud   | L2                |   42110 |       41.48 |            28.29 |
| longitud   | L3                |   43444 |       36.35 |            23.95 |
| longitud   | L4                |   40865 |       33.96 |            21.38 |
| tokens     | T1                |   44555 |       45.65 |            31.68 |
| tokens     | T2                |   43834 |       38.44 |            25.85 |
| tokens     | T3                |   44754 |       35.27 |            22.66 |
| tokens     | T4                |   36193 |       40.44 |            26.6  |
| menciones  | 0                 |   61150 |       51.04 |            34.72 |
| menciones  | 1                 |   77427 |       37.76 |            25.01 |
| menciones  | 2+                |   30759 |       23.29 |            15.01 |
| hashtags   | 0                 |  135053 |       42.19 |            28.35 |
| hashtags   | 1+                |   34263 |       31    |            20.17 |
| hashtags   | nan               |      20 |       45    |            35    |
| url        | con_url           |   68562 |       39.6  |            26.1  |
| url        | sin_url           |  100774 |       40.15 |            27.11 |
| retuit     | no_rt             |  107741 |       42.09 |            28.6  |
| retuit     | rt                |   61595 |       36.14 |            23.38 |
| tipo       | hashtag_alto      |    2537 |       20.3  |            13.84 |
| tipo       | mencion_alta      |    5494 |       18.84 |            11.69 |
| tipo       | muy_corto         |    2677 |       50.8  |            36.72 |
| tipo       | rt_con_comentario |   49210 |       35.85 |            22.43 |
| tipo       | rt_puro           |   12376 |       37.29 |            27.18 |
| tipo       | solo_mencion      |     414 |       34.78 |            25.12 |
| tipo       | solo_texto        |   59221 |       43.48 |            29.73 |
| tipo       | solo_url          |    7661 |       59.89 |            40.07 |
| tipo       | url_alta          |   29746 |       40.21 |            27.09 |

## 7. Fuente ordinal continua

{'correlacion_con_indice_real': 0.7415, 'orientacion': 'creciente', 'desviacion_media_absoluta_en_clases': 38.7577}

_Duración: 35.5 s._
