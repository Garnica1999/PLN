# Evaluacion intrinseca de la fase v5 (corpus 10M)

Generado por `experiments/embedding_02_fasttext_baseline_v5.py` (EBD-01..05 (P2)).
Seccion 20 de la spec. **No se usa ningun label externo de hate speech**: los vecinos se
presentan como evidencia de la geometria aprendida, sin afirmar que un vecino sea
"correcto".

## 1. Modelos entrenados desde cero

| Modelo | Etiqueta | dim | subpalabras | semilla | vocabulario | sha256 del .vec | s |
|---|---|---|---|---|---|---|---|
| FASTTEXT-BASELINE | FASTTEXT-SG-SUBWORDS-3-6 | 200 | 3-6 | 42 | 100098 | `e54a363da986aeac` | 0.0 |
| FASTTEXT-SEED-CONTROL | FASTTEXT-SG-SUBWORDS-3-6 | 200 | 3-6 | 7 | 100098 | `afde390f208b1f51` | 0.0 |
| WORD2VEC-CONTROL | WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF | 200 | 0-0 | 42 | 100098 | `52f94d37e18cef72` | 0.0 |

## 2. Estabilidad entre semillas (seccion 20)

- Coseno centrado medio entre los vectores de `FASTTEXT-BASELINE` (semilla 42) y `FASTTEXT-SEED-CONTROL` (semilla 7) sobre 12 palabras: **-0.034**.
- Solapamiento medio de los 10 vecinos mas cercanos: **0.792**.

**Como leer estos dos numeros.** El coseno centrado entre semillas ~0 **no** significa que
la representacion sea ruido: las soluciones de word2vec/fastText solo son identicas salvo
rotacion y escala, de modo que las direcciones absolutas cambian con la semilla aunque la
geometria sea la misma. Lo que si es comparable entre semillas es la **estructura relativa**,
y por eso se reporta el solapamiento de vecinos. Si el solapamiento fuera bajo, la
representacion seria principalmente ruido de inicializacion y ningun resultado downstream de
la fase podria leerse como propiedad de la representacion.

## 3. Vecinos mas cercanos (coseno centrado)

### FASTTEXT-BASELINE

| palabra | vecinos |
|---|---|
| violencia | `violencias` (0.800), `violento` (0.723), `violenta` (0.704), `violentos` (0.676), `dolencia` (0.667) |
| inmigración | `emigración` (0.872), `inmigraciones` (0.834), `inmigratoria` (0.825), `inmigrante` (0.821), `inmigrantes` (0.815) |
| gobierno | `desgobierno` (0.753), `autogobierno` (0.708), `gobiernos` (0.703), `gobierna` (0.696), `gubernamental` (0.586) |
| mujer | `mujeres` (0.612), `mujeriego` (0.565), `hombría` (0.516), `niña` (0.469), `hombre` (0.457) |
| niño | `niñez` (0.576), `niña` (0.544), `niñera` (0.511), `niños` (0.474), `bebé` (0.410) |
| familia | `familiarde` (0.727), `superfamilia` (0.718), `familias` (0.713), `familie` (0.692), `subfamilia` (0.672) |
| iglesia | `anteiglesia` (0.772), `iglesias` (0.752), `católica` (0.725), `ortodoxa` (0.662), `apostólica` (0.635) |
| fútbol | `futbol` (0.693), `baloncesto` (0.678), `vóleibol` (0.660), `softbol` (0.656), `básquetbol` (0.642) |
| música | `musica` (0.718), `músicas` (0.668), `folclórica` (0.649), `musical` (0.649), `músicos` (0.618) |
| trabajo | `trabaje` (0.737), `trabajen` (0.730), `trabajos` (0.682), `trabaja` (0.657), `trabajado` (0.602) |
| ciudad | `ciudadela` (0.681), `torreciudad` (0.628), `ciudades` (0.577), `localidad` (0.558), `capital` (0.540) |
| guerra | `guerras` (0.672), `esguerra` (0.650), `postguerra` (0.600), `civil` (0.564), `guerrouj` (0.550) |

### WORD2VEC-CONTROL

| palabra | vecinos |
|---|---|
| violencia | `discriminación` (0.550), `delincuencia` (0.545), `conflictividad` (0.528), `lawfare` (0.520), `abuso` (0.518) |
| inmigración | `emigración` (0.679), `inmigrantes` (0.631), `migratorio` (0.528), `emigrantes` (0.513), `mestizaje` (0.498) |
| gobierno | `convocando` (0.595), `bukele` (0.595), `abdelfatah` (0.593), `semipresidencialista` (0.591), `sisi` (0.590) |
| mujer | `mujeres` (0.483), `ajusticiada` (0.462), `niña` (0.457), `dama` (0.428), `sentada` (0.428) |
| niño | `niña` (0.467), `prodigio` (0.455), `niños` (0.434), `adolescente` (0.431), `infidelidad` (0.408) |
| familia | `winteráceas` (0.623), `canellaceae` (0.615), `myristicaceae` (0.592), `passifloraceae` (0.591), `winteraceae` (0.586) |
| iglesia | `católica` (0.754), `ortodoxa` (0.681), `luterana` (0.651), `anglicana` (0.648), `reformada` (0.632) |
| fútbol | `baloncesto` (0.673), `balonmano` (0.665), `voleibol` (0.601), `americano` (0.589), `futsal` (0.577) |
| música | `folclórica` (0.672), `vallenata` (0.617), `folk` (0.575), `orquestal` (0.568), `musical` (0.565) |
| trabajo | `asalariado` (0.428), `trabajadores` (0.418), `trabajos` (0.410), `laboral` (0.397), `improductivo` (0.391) |
| ciudad | `villa` (0.578), `acámbaro` (0.566), `irapuato` (0.555), `tarata` (0.542), `sintra` (0.536) |
| guerra | `civil` (0.600), `ogaden` (0.577), `bóer` (0.574), `conflicto` (0.559), `vencedores` (0.539) |

## 4. Variantes ortograficas (tipos word-OOV del corpus TOXIC)

Estos tipos **no** estan en el vocabulario del corpus. FastText les compone un vector con
subpalabras; el control Word2Vec, entrenado con `minn=0, maxn=0`, no puede. La tabla
muestra el vecino mas cercano de cada variante.

| variante (freq. TOXIC) | FASTTEXT-BASELINE | WORD2VEC-CONTROL |
|---|---|---|
| `callate` (350) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `cllate` (279) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `violacin` (58) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `espaa` (52) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `imbcil` (31) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `arabe` (70) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `estn` (33) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `mereces` (135) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `pinche` (37) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `inmigracin` (46) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `guarra` (47) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `mereca` (37) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `estupido` (2) | (sin vector: no esta en el vocabulario y no hay subpalabras) | (sin vector: no esta en el vocabulario y no hay subpalabras) |
| `perra` (392) | `perra` (1.000) | `perra` (1.000) |

## 5. Limitaciones

1. El corpus es enciclopedico: los vecinos de palabras como `música` o `fútbol` son
   coherentes, pero eso dice poco sobre el registro informal que caracteriza al problema.
2. Los vecinos de variantes ortograficas son **indicativos**: con 10M tokens la señal
   subléxica es débil y no se afirma que el vecino sea el lexema correcto.
3. La normalizacion de este script se comparó token a token con la de P1 sobre los 2885 textos de TOXIC: 0 diferencias.
