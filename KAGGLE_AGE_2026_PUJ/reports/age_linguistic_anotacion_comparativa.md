# P2.0 - anotacion linguistica determinista y comparativa de anotadores

**Fecha**: 2026-09-29 | **Artefactos**: `features/linguistic/anotacion_*_train.parquet` (hash en `experiments/age_linguistic_01_anotacion.json`)

## El proceso, en orden

```text
texto original de train (169.336 filas, ASCII, sin acentos)
    -> anotacion determinista, SIN etiquetas de edad y SIN test
    -> secuencias tipadas (pos, lemma_pos, morph, dep, ent)
    -> artefacto derivado con SHA256 en manifiesto
```

Regla de D-076: solo etiquetas; los vectores de los modelos no se leen ni se usan.

## Cobertura por anotador

| Anotador | Docs con tokens | Docs vacios | Media tokens | Media oraciones | Tokens totales |
|---|---|---|---|---|---|
| es_core_news_md | 169,232 (99.94%) | 104 | 15.3 | 1.541 | 2,590,877 |
| es_core_news_lg | 169,232 (99.94%) | 104 | 15.3 | 1.555 | 2,590,877 |

## Acuerdo md vs lg

| Campo | Secuencia exacta igual |
|---|---|
| pos_seq | 16.35% |
| lemma_pos_seq | 14.44% |
| morph_seq | 14.40% |
| dep_seq | 12.46% |
| ent_seq | 28.92% |

- Documentos con el mismo numero de tokens: **100.0%**
- Acuerdo POS por token (donde la longitud coincide): **86.269%**

## Degradacion por segmento de Twitter

| Corte | Docs | % con tokens (md) | % con tokens (lg) | Acuerdo POS exacto |
|---|---|---|---|---|
| con_http | 70,475 | 100.0 | 100.0 | 10.32% |
| con_hashtag | 34,283 | 100.0 | 100.0 | 4.16% |
| con_mencion | 108,186 | 100.0 | 100.0 | 8.49% |
| longitud<=40 | 27,533 | 99.62 | 99.62 | 37.06% |
| longitud_41-93 | 57,494 | 100.0 | 100.0 | 18.78% |
| longitud>93 | 84,309 | 100.0 | 100.0 | 7.92% |

## Distribucion de etiquetas (md)

| Etiqueta | Frecuencia |
|---|---|
| PROPN | 555,247 |
| NOUN | 367,687 |
| PUNCT | 298,169 |
| VERB | 259,041 |
| ADP | 256,109 |
| DET | 204,663 |
| ADJ | 153,005 |
| PRON | 128,176 |
| ADV | 81,202 |
| AUX | 70,702 |
| CCONJ | 60,364 |
| SCONJ | 54,393 |
| NUM | 50,600 |
| SPACE | 32,865 |
| INTJ | 10,349 |
| SYM | 7,373 |
| X | 484 |
| PART | 448 |

## Entidades por tipo (md)

| Tipo | Frecuencia |
|---|---|
| MISC | 349,625 |
| LOC | 92,681 |
| ORG | 86,279 |
| PER | 84,352 |

## Ejemplos de discrepancia entre anotadores

| fila | texto (inicio) | POS md | POS lg |
|---|---|---|---|
| 0 | Ud y su flia fueron las mas beneficiadas con Obredecht eso no es ninguna calumnia ud le debe decir l | PROPN CCONJ DET NOUN AUX DET ADV ADJ ADP PROPN PRON ADV AUX DET NOUN PRON PRON AUX VERB DET NOUN ADP PROPN PRO | PROPN CCONJ DET PROPN AUX DET ADV ADJ ADP PROPN PRON ADV AUX DET NOUN PROPN PRON AUX VERB DET NOUN ADP PROPN P |
| 1 | http://t.co/Jp4hFcwMF6 | PROPN | PRON |
| 2 | Hasta el 1 de oct Medellin estara respirando periodismo, incentivando la etica, innovacion y coheren | ADP DET NUM ADP NOUN PROPN AUX VERB NOUN PUNCT VERB DET NOUN PUNCT NOUN CCONJ NOUN PUNCT DET NOUN NUM | ADP DET NUM ADP NOUN PROPN AUX VERB NOUN PUNCT VERB DET NOUN PUNCT PROPN CCONJ NOUN PUNCT DET VERB NOUN |
| 3 | #TFZ  @M_MartinezRed  @RaisAlGhousia  @LuiguiSoutto  @followhelp1212  @NathalyIsabe  @Nijia2o  @elto | PROPN PROPN SPACE NOUN SPACE ADJ SPACE NOUN SPACE PROPN SPACE PROPN SPACE ADJ SPACE ADJ | AUX PROPN SPACE VERB SPACE NOUN SPACE NOUN SPACE PROPN SPACE PROPN SPACE VERB SPACE ADJ |

## Lectura

- La cobertura es alta en los dos modelos; las diferencias entre md y lg se concentran
  donde el corpus degrada el etiquetado (sin acentos, jerga, abreviaturas).
- El acuerdo por token mide cuanto cambia la anotacion al cambiar de tamano de modelo:
  es la cota superior de lo que puede mover el canal en P2.
- Ambos artefactos quedan congelados con hash; cualquier experimento posterior debe usar
  exactamente estos ficheros.
