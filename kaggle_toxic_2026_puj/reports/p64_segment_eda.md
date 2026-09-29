# P64 - Segmentacion determinista y EDA del perfil de segmento

Generado por `experiments/p64_02_segmentacion_eda.py`. Fecha 2026-09-29T07:18:33+00:00.

> **DIAGNOSTICO.** No entrena, no crea features, no usa test, holdout ni leaderboard.

## Por que la segmentacion es de esta manera

El gap analysis establecio que la segmentacion por puntuacion alcanza **5,29 veces mas**
filas que la ventana fija de P59-07, y que la tasa de error de V03 sube **8,35 puntos**
entre un segmento y cinco o mas. El encargo exige que la segmentacion sea determinista,
sin modelo, construida con reglas sobre el texto, y prohibe expresamente un parser
externo y spaCy en esta primera capa.

Hay cuatro esquemas, y son **cuatro hipotesis distintas** sobre donde esta un limite,
no cuatro parametros del mismo:

```text
S1  puntuacion fuerte, cualquier secuencia de . ! o ?
S2  puntuacion fuerte mas un salto de linea
S3  delimitadores fuertes: dos o mas de esos signos, o ; o :
S4  una mencion, un hashtag o una comilla de apertura abren unidad propia
```

La eleccion entre ellos **no es una busqueda sobre una rejilla**. Cada uno se reporta
con su cobertura y su perfil de error, y la lectura es descriptiva.

El primario se declara **antes de mirar nada**: **S1**.

## Las cuatro segmentaciones

| esquema | filas multi | porcentaje | segmentos medianos | maximos | error | error en multi |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 751 | 50.6% | 2 | 13 | 17.02% | 19.53% |
| S2 | 751 | 50.6% | 2 | 13 | 17.02% | 19.53% |
| S3 | 602 | 40.6% | 1 | 9 | 17.02% | 19.32% |
| S4 | 1018 | 68.6% | 2 | 31 | 17.02% | 18.47% |

**S4 parte por mencion, hashtag y comilla, luego da mas unidades que S1. Eso no es**
**mejor**: son unidades mas pequenas y mas homogeneas, y el objetivo de la fase es
evidencia local que compita con la global, no trocear mas. El primario declarado sigue
siendo S1 y S4 se reporta como sensibilidad para la linea de citas del encargo.

## La EDA del perfil

Siete rasgos acotados por fila: numero de segmentos, su logaritmo, indicador de multi,
fraccion del segmento mas largo y mas corto, desviacion de la longitud y entropia de la
distribucion de longitudes.

| estrato | n | `n_segmentos` | `log_n_segmentos` | `es_multi_segmento` | `fraccion_del_segmento_mas_largo` | `fraccion_del_segmento_mas_corto` | `desviacion_de_la_longitud` | `entropia_de_la_longitud` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| todas | 1483 | 1.9589 | 1.0108 | 0.5064 | 0.8623 | 0.5681 | 0.2619 | 0.1684 |
| CORTA | 167 | 1.3892 | 0.8404 | 0.2994 | 0.9584 | 0.7799 | 0.0412 | 0.0927 |
| MEDIA | 602 | 1.6163 | 0.9132 | 0.4103 | 0.9160 | 0.6564 | 0.1460 | 0.1152 |
| LARGA | 547 | 2.1024 | 1.0638 | 0.5941 | 0.8330 | 0.4892 | 0.3428 | 0.1913 |
| MUY_LARGA | 167 | 3.2934 | 1.3598 | 0.7725 | 0.6688 | 0.2962 | 0.6349 | 0.3603 |
| V03 correcto | 1042 | 1.8445 | 0.9755 | 0.4607 | 0.8826 | 0.6079 | 0.2338 | 0.1484 |
| V03 incorrecto | 441 | 2.2290 | 1.0943 | 0.6145 | 0.8145 | 0.4742 | 0.3281 | 0.2155 |
| error de alta confianza | 295 | 2.2034 | 1.0833 | 0.5966 | 0.8250 | 0.4867 | 0.3316 | 0.2066 |

## La separacion, que decide el paso siguiente

| rasgo | AUC para predecir si V03 falla |
|---|---:|
| n_segmentos | 0.588351 |
| log_n_segmentos | 0.588351 |
| es_multi_segmento | 0.576930 |
| fraccion_del_segmento_mas_largo | 0.413968 |
| fraccion_del_segmento_mas_corto | 0.419111 |
| desviacion_de_la_longitud | 0.578305 |
| entropia_de_la_longitud | 0.589847 |
| n_tokens_como_referencia | 0.600506 |

El mejor rasgo es **entropia_de_la_longitud**, con una ventaja de **-0.010659** sobre la
referencia trivial de longitud. Esa referencia importa: un AUC de un rasgo estructural
que solo mide longitud no aporta nada, y P63 ya mostro que la longitud explica una
parte sustancial del error.

## Veredicto

```text
el perfil segmental es PEOR que la referencia trivial de longitud, con una ventaja de -0.010659. Es una asociacion de la longitud disfrazada de estructura, y queda cerrado como rasgo

PERFIL CERRADO. Continuar con descomposicion de evidencia
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_02_segmentacion_eda.py`. Sin push.*