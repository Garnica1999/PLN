# P62 - EDA linguistica del corpus canonico

Generado por `experiments/p62_01_eda_linguistica.py`. Fecha 2026-09-29T05:47:07+00:00.

> **Esta EDA es descriptiva.** No entrena, no modifica V03, no selecciona
> hiperparametros, no usa test, ni holdout, ni leaderboard, ni ajusta thresholds. Su
> funcion es decidir si hay informacion representable que V03 no expresa, antes de
> gastar un experimento en buscarla.

## La anotacion

```text
herramienta   spaCy 3.8.13
modelo        es_core_news_md 3.8.0, idioma es, licencia GNU GPL 3.0
componentes   exclude=['ner']; vector, tensor y similarity prohibidos por D-003
documentos    1483 | media de 18.95 tokens
soporte min   8 ocurrencias, para no concluir sobre pocas observaciones
```

Casos limite verificados, sin excepcion en ninguno: texto vacio, solo puntuacion,
URL, mencion, elongacion, texto de un caracter y emoji.

## Composicion de POS por clase

| categoria | HS | TR | AG |
|---|---:|---:|---:|
| ADJ | 1.0216 | 0.8180 | 1.0229 |
| ADP | 1.7114 | 1.0404 | 1.5961 |
| ADV | 0.6173 | 0.5169 | 0.5908 |
| AUX | 0.7623 | 0.7213 | 0.7372 |
| CCONJ | 0.5617 | 0.4427 | 0.5379 |
| DET | 1.7531 | 1.2607 | 1.6384 |
| INTJ | 0.0231 | 0.0225 | 0.0194 |
| NOUN | 2.6605 | 1.9708 | 2.5026 |
| NUM | 0.0370 | 0.0270 | 0.0317 |
| PART | 0.0000 | 0.0000 | 0.0000 |
| PRON | 1.1914 | 1.0427 | 1.1340 |
| PROPN | 2.1142 | 2.1910 | 2.2628 |
| PUNCT | 0.0031 | 0.0045 | 0.0053 |
| SCONJ | 0.5525 | 0.4674 | 0.5238 |
| SYM | 0.0000 | 0.0000 | 0.0018 |
| VERB | 1.8009 | 1.4225 | 1.7513 |
| X | 0.0015 | 0.0000 | 0.0018 |

## Frecuencia contra enriquecimiento

El encargo dice que no se debe asumir que frecuencia alta equivale a senal
discriminativa. La medicion es directa:

```text
sustantivos con soporte  71
correlacion frecuencia contra log enriquecimiento  -0.0572
```

## La pregunta central de la seccion 24

> Los tweets HS, TR y AG se diferencian de los no toxicos principalmente por las
> palabras que contienen, o tambien por COMO esas palabras se combinan?

| magnitud | valor | que mide |
|---|---:|---|
| palabra sola, como elemento | +0.3773 | cuanto explica saber QUE palabra aparece |
| palabra sola, por tweet | +0.1549 | lo mismo, como proporcion |
| categoria sola | +0.0095 | cuanto explica la composicion gramatical |
| palabra y categoria juntas | +0.4525 | el canal combinado |
| **sinergia** | **+0.0657** | **lo que NO se puede escribir como funcion de las dos por separado** |
| synergy sobre el total de la palabra | 17.41 por ciento | cuanto de la senal es puramente interaccion |

**La lectura.** La synergy es la unica magnitud de la tabla que no se puede
descomponer en la palabra y la categoria por separado. Y es precisamente lo que
ninguna de las dos representaciones ya probadas en el proyecto puede expresar: el
canal categorial de E2-03 descarta la palabra, y el TF-IDF de V03 descarta la
categoria. La forma de la senal que queda por representar es la interaccion, no la
etiqueta.

## Perfiles linguisticos y donde falla V03

| perfil | etiqueta | n rico | F1 rico | F1 pobre | delta | err. alta conf. rico | pobre |
|---|---|---:|---:|---:|---:|---:|---:|
| adj_rico | HS | 169 | 0.8733 | 0.9023 | -0.0290 | 0.1420 | 0.0961 |
| adj_rico | TR | 121 | 0.8848 | 0.9346 | -0.0498 | 0.1405 | 0.0818 |
| adj_rico | AG | 146 | 0.8549 | 0.9075 | -0.0526 | 0.1507 | 0.0968 |
| clausula_simple | HS | 166 | 0.8227 | 0.9225 | -0.0998 | 0.1566 | 0.0886 |
| clausula_simple | TR | 77 | 0.7581 | 0.9471 | -0.1891 | 0.2857 | 0.0568 |
| clausula_simple | AG | 129 | 0.7536 | 0.9133 | -0.1597 | 0.2481 | 0.0973 |
| dependency_rico | HS | 209 | 0.9497 | 0.8428 | +0.1070 | 0.0335 | 0.1522 |
| dependency_rico | TR | 184 | 0.9364 | 0.8867 | +0.0497 | 0.0707 | 0.1504 |
| dependency_rico | AG | 195 | 0.9431 | 0.8340 | +0.1091 | 0.0462 | 0.1921 |
| discourse_rico | HS | 183 | 0.9009 | 0.8993 | +0.0016 | 0.1093 | 0.1013 |
| discourse_rico | TR | 132 | 0.8861 | 0.9372 | -0.0511 | 0.1439 | 0.0675 |
| discourse_rico | AG | 158 | 0.8714 | 0.9000 | -0.0286 | 0.1392 | 0.1055 |
| longitud_del_tweet | HS | 157 | 0.7876 | 0.9507 | -0.1630 | 0.1975 | 0.0385 |
| longitud_del_tweet | TR | 60 | 0.6813 | 0.9677 | -0.2864 | 0.4167 | 0.0433 |
| longitud_del_tweet | AG | 138 | 0.7042 | 0.9538 | -0.2495 | 0.2899 | 0.0465 |
| negation_rico | HS | 150 | 0.8281 | 0.8929 | -0.0648 | 0.1800 | 0.0937 |
| negation_rico | TR | 94 | 0.7792 | 0.9280 | -0.1488 | 0.1915 | 0.1086 |
| negation_rico | AG | 125 | 0.7805 | 0.8792 | -0.0987 | 0.1920 | 0.1422 |
| noun_rico | HS | 209 | 0.9337 | 0.8525 | +0.0812 | 0.0670 | 0.1600 |
| noun_rico | TR | 185 | 0.9607 | 0.8842 | +0.0765 | 0.0432 | 0.1509 |
| noun_rico | AG | 202 | 0.9368 | 0.8136 | +0.1233 | 0.0693 | 0.2143 |
| pronoun_rico | HS | 164 | 0.8919 | 0.9072 | -0.0153 | 0.0976 | 0.0680 |
| pronoun_rico | TR | 147 | 0.9191 | 0.9478 | -0.0286 | 0.0952 | 0.0709 |
| pronoun_rico | AG | 152 | 0.9104 | 0.8754 | +0.0350 | 0.0724 | 0.1420 |
| verb_rico | HS | 183 | 0.8704 | 0.9418 | -0.0714 | 0.1311 | 0.0550 |
| verb_rico | TR | 128 | 0.8673 | 0.9618 | -0.0945 | 0.1406 | 0.0429 |
| verb_rico | AG | 163 | 0.8561 | 0.9496 | -0.0934 | 0.1718 | 0.0565 |

El corte es el cuartil 75 dentro de cada rasgo y de cada etiqueta, y se declara
ANTES de mirar los resultados, que es lo que exige el parrafo 39 para que una
rebanada linguistica pueda usarse como criterio y no solo como descripcion.

**Aviso de lectura, y es una correccion.** Una version anterior de este informe
nombraba el perfil de longitud `estructura_corta` y concluyo que V03 fallaba en los
tweets cortos. Era al reves: su cuartil ALTO es lo largo, luego la delta negativa
dice que V03 es MEJOR en los largos. La comprobacion posterior por region lo
confirma de forma independiente, con AUC de 0,94 en los cortos frente a 0,85 a 0,92
en los largos, y F1 de 0,94 contra 0,72. El perfil se llama ahora
`longitud_del_tweet` para que el nombre no vuelva a inducir el error, y la region
corta se analiza aparte en `p62_techo_cortos.md`.

## Palabras funcionales, negacion, modalidad y discursivos

| grupo | HS | TR | AG |
|---|---:|---:|---:|
| negadores | 0.0168 | 0.0164 | 0.0159 |
| modales | 0.0027 | 0.0024 | 0.0025 |
| discursivos | 0.0481 | 0.0469 | 0.0465 |
| determinantes | 0.0892 | 0.0847 | 0.0839 |
| preposiciones | 0.0692 | 0.0497 | 0.0640 |
| conjunciones | 0.0618 | 0.0594 | 0.0600 |

Las listas son del proyecto, no lexicones externos (parrafo 50). Son las palabras
funcionales que el propio encargo nombra, y su inclusion es una decision
documentada.

## Composicion del corpus derivado

`p62_anotacion_spacy.json.gz` guarda el identificador de fila, el texto original, el lema, la
categoria POS, la relacion de dependencia, la categoria y el lema de la cabeza, y la
posicion. El dataset oficial no se modifica y las etiquetas HS, TR y AG no se tocan
(parrafo 46). El hash de cada texto queda disponible para auditar la derivacion.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_01_eda_linguistica.py`. Sin push.*