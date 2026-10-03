# Cobertura del vocabulario TOXIC sobre el corpus de 10M tokens

Generado por `experiments/embedding_01_corpus_wikipedia_v5.py` (E-CORP-01).
Secciones 7, 8 y 9 de la spec. Los grupos se definen con listas y patrones **curados, sin usar labels**: son un instrumento de diagnostico, nunca una fuente de supervision para el embedding.

## 1. Cobertura global (secciones 7 y 9)

| Conjunto | tipos | tokens | cobertura tipo | cobertura token | OOV tipos | OOV tokens | word-OOV | representation-OOV |
|---|---|---|---|---|---|---|---|---|
| train | 6059 | 29646 | 0.7042 | 0.8831 | 0.2958 | 0.1169 | 1792 | 0 |
| test | 4503 | 19204 | 0.7213 | 0.8837 | 0.2787 | 0.1163 | 1255 | 0 |
| train+test | 8555 | 48850 | 0.6839 | 0.8833 | 0.3161 | 0.1167 | 2704 | 0 |

- `frequency_weighted_coverage` coincide con `token_coverage` por definicion: **0.8833** (ponderada por la frecuencia de cada tipo en TOXIC).
- **word-OOV**: 2704 tipos del train+test de TOXIC no aparecen en el corpus.
- **representation-OOV**: de esos, 0 reciben vector por composicion de subpalabras (3-6) en el probe fastText 50d; la tasa de indisponibilidad es 0.0000.
- El probe es un modelo de 50 dimensiones y 1 epoca, entrenado **solo** para responder esta pregunta de la seccion 9; no es el baseline de la fase (que es P2).

### 1.1 Que explica la falta de cobertura (y que no)

La hipotesis inicial era que la cobertura baja se debia a que los tweets de TOXIC estan en ASCII sin tildes mientras el corpus conserva las tildes. **Los datos la refutan**:

- Tipos de TOXIC ausentes del corpus: **2704**.
- Plegado ASCII: solo **230** (8.5%) coinciden con el plegado ASCII de un tipo del corpus. La cobertura por tipo proyectada pasaria de 0.6839 a 0.7108: **no explica la brecha**.
- Esqueleto vocalico (deformaciones que pierden vocales, como `cllate`, `violacin`): **1802** (66.6%), con cobertura proyectada 0.8946.
- Ejemplos de pares por esqueleto: `aadidle`~`ddl`, `abandononde`~`bndnnd`, `ablarme`~`blrm`, `abrazarte`~`brzrt`, `aburre`~`brr`, `acabarn`~`cbrn`, `acabas`~`cbs`, `aceptalo`~`cptl`, `aceptas`~`cpts`, `aclaren`~`clrn`.

La lectura correcta es que los tipos ausentes son de dos familias: (a) deformaciones ortograficas como la perdida de vocales, que **no** son pliegues de acento y (b) formas informales, elongaciones y lenguaje coloquial que un corpus enciclopedico no contiene.

### 1.2 Que resuelven los subpalabras (respuesta directa a la seccion 9)

De los 2704 tipos word-OOV se evaluaron 2704 con el probe 50d. Para cada uno se busco el tipo del corpus mas cercano al vector que fastText compone con subpalabras:

- coseno mediano al vecino mas cercano: **0.726** (media 0.730)
- proporcion con coseno >= 0.6: **99.6%**; >= 0.4: 100.0%; < 0.3: 0.0%

| tipo word-OOV (freq. TOXIC) | vecino del corpus | coseno centrado | segundo vecino |
|---|---|---|---|
| `callate` (350) | `hν` | 0.689 | `pulcherrima` (0.683) |
| `cllate` (279) | `lechuck's` | 0.758 | `sơn` (0.726) |
| `mereces` (135) | `lô` | 0.620 | `merecumbé` (0.613) |
| `rabes` (93) | `maoríes` | 0.741 | `idrisíes` (0.733) |
| `arabe` (70) | `alhárabe` | 0.747 | `διάλεκτος` (0.740) |
| `ensear` (69) | `sgbd` | 0.678 | `coger` (0.655) |
| `violacin` (58) | `rohirrim` | 0.687 | `lgbti` (0.664) |
| `espaa` (52) | `espagueti` | 0.676 | `owusu` (0.659) |
| `guarra` (47) | `salserín` | 0.749 | `guardabarranco` (0.746) |
| `inmigracin` (46) | `inmigrante` | 0.654 | `praecox` (0.651) |
| `mereca` (37) | `merecumbé` | 0.687 | `seneca` (0.676) |
| `pinche` (37) | `codorniz` | 0.753 | `ollagüe` (0.746) |
| `estn` (33) | `estupor` | 0.713 | `estupa` (0.657) |
| `imbcil` (31) | `¾` | 0.732 | `ית` (0.728) |
| `arabes` (30) | `idrisíes` | 0.769 | `azafíes` (0.758) |

**Nota metodologica sobre el coseno**: los vectores de fastText comparten un componente comun grande (la direccion de la frecuencia unigrama), por lo que el coseno crudo entre cualesquiera dos vectores sale alto (~0.94 con este probe) y el vecino mas cercano no significa nada. Por eso la tabla usa el coseno **centrado** (se resta la media de la matriz antes de comparar). Sin labels externos **no se puede afirmar** que un vecino sea el lexema correcto, y no se afirma: la tabla se muestra para que el lector juzgue.

Lo que si se puede afirmar con estos datos:

1. Todo tipo word-OOV obtiene una representacion no nula (los subpalabras siempre tienen materia prima), asi que el problema de cobertura **no** es de indisponibilidad de vector.
2. La deformacion ortografica afecta a 1802 de los 2704 tipos ausentes (66.6%) segun el esqueleto vocalico, que es exactamente el caso donde los subpalabras 3-6 pueden reconstruir el lexema.
3. El coseno centrado mediano es 0.726, con 100.0% de los tipos por encima de 0.4 y 0.0% por debajo de 0.3.
4. El grupo con peor cobertura sigue siendo `elongaciones` (0.035 por tipo) y `coloquial` (0.214): son formas que un corpus enciclopedico **no contiene**. Para esas la palanca no es el numero de tokens sino la **fuente** del corpus (registro web, mC4-es), que es lo que mide P4.
5. Los vecinos de la tabla **no son fiables como lexemas** con este probe (varios son claramente incorrectos: `callate` -> `pulcherrima`, `imbcil` -> `xẩm`). Es el resultado esperado en un modelo de 50 dimensiones y 1 epoca: los vectores de subpalabra estan subentrenados. La tabla no se usa para afirmar nada sobre la calidad del embedding.

La prueba que decide si los subpalabras aportan algo utilizable no es esta sonda, sino el embedding de P2 (200d, 5 epocas) medido downstream. Este probe de 50d y 1 epoca existe solo para responder la pregunta de disponibilidad de la seccion 9.

## 2. Cobertura por grupo relevante (seccion 8)

| Grupo | tipos | tokens | cobertura tipo | cobertura token | OOV | frec. media corpus |
|---|---|---|---|---|---|---|
| elongaciones | 113 | 127 | 0.0354 | 0.0394 | 0.9646 | 2.5 |
| formas_raras | 6518 | 7618 | 0.6359 | 0.6565 | 0.3641 | 185.9 |
| coloquial | 56 | 461 | 0.2143 | 0.6920 | 0.7857 | 463.2 |
| insultos | 72 | 2402 | 0.2222 | 0.7777 | 0.7778 | 6.6 |
| targeting | 11 | 1058 | 0.9091 | 0.9849 | 0.0909 | 7478.0 |
| colectivos | 19 | 177 | 0.9474 | 0.9944 | 0.0526 | 476.7 |
| negacion | 10 | 1051 | 1.0000 | 1.0000 | 0.0000 | 4805.4 |
| noticia | 15 | 234 | 1.0000 | 1.0000 | 0.0000 | 1815.2 |
| hashtags | 1 | 522 | 1.0000 | 1.0000 | 0.0000 | 2.0 |
| menciones | 1 | 2055 | 1.0000 | 1.0000 | 0.0000 | 783.0 |
| marcadores_web | 2 | 1365 | 1.0000 | 1.0000 | 0.0000 | 239807.5 |

## 3. Tipos word-OOV mas frecuentes en TOXIC (top 40)

| tipo | frecuencia en TOXIC | representable por subpalabras |
|---|---|---|
| `callate` | 350 | si |
| `cllate` | 279 | si |
| `mereces` | 135 | si |
| `rabes` | 93 | si |
| `arabe` | 70 | si |
| `ensear` | 69 | si |
| `violacin` | 58 | si |
| `espaa` | 52 | si |
| `guarra` | 47 | si |
| `inmigracin` | 46 | si |
| `mereca` | 37 | si |
| `pinche` | 37 | si |
| `estn` | 33 | si |
| `imbcil` | 31 | si |
| `arabes` | 30 | si |
| `jajaja` | 30 | si |
| `aqu` | 25 | si |
| `cmo` | 25 | si |
| `comemela` | 25 | si |
| `tambin` | 24 | si |
| `enseando` | 24 | si |
| `tonteras` | 23 | si |
| `despus` | 22 | si |
| `sudacas` | 21 | si |
| `jajajaja` | 20 | si |
| `geros` | 19 | si |
| `putos` | 19 | si |
| `vdeo` | 18 | si |
| `ensearle` | 17 | si |
| `joder` | 17 | si |
| `imbecil` | 16 | si |
| `slo` | 16 | si |
| `gabacho` | 16 | si |
| `ests` | 16 | si |
| `espaol` | 16 | si |
| `gilipollas` | 16 | si |
| `ahre` | 15 | si |
| `ojal` | 15 | si |
| `nios` | 15 | si |
| `ensearon` | 15 | si |

## 4. Lectura

La cobertura por tipo es la que importa para la representacion: un tipo ausente del vocabulario del corpus solo puede existir como composicion de subpalabras, y las subpalabras conservan el orden de las letras pero pierden el significado contextual. Los grupos con menor cobertura por token son los que mas dependen de los subwords.

## 5. Artefactos

- Tabla completa: `reports/embedding_vocabulary_coverage.csv`
- Figuras: `reports/figuras/fig_41_distribucion_frecuencias_v5.png`, `reports/figuras/fig_42_cobertura_grupos_v5.png`
- Sonda de representation-OOV: `data/external/corpora/sifts/probe_50d_sift.txt`
