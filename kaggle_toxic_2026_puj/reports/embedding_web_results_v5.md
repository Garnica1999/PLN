# Embedding sobre corpus web y mixto, y revision de la fusion (fase v5, P5)

Generado por `experiments/embedding_05_fasttext_web_v5.py` (EBD-10..12 (P5)).

## 1. Pregunta

P4 subio la cobertura del vocabulario TOXIC de 0.6839 a 0.8413 por tipo con la union
Wikipedia+web. Esta es la prueba de si **esa cobertura se traduce en aportacion**
complementaria: si la mecanica de P3 es correcta, la fusion con el embedding del corpus
web o del mixto debe superar a la fusion con el de Wikipedia.

## 2. Modelos (entrenados desde cero, misma arquitectura)

| modelo | corpus | vocabulario | tipos con vector | sha256 del .vec | s |
|---|---|---|---|---|---|
| FASTTEXT-WEB-10M | `data/external/corpora/mc4_es/emb_normal/tokens_10M.txt` | 106632 | 6059 de 6059 | `8f83703983d23a5e` | 0.0 |
| FASTTEXT-MIXTO-20M | `data/external/corpora/mixto/emb_normal/tokens_20M.txt` | 163228 | 6059 de 6059 | `18f8d10125c70b8d` | 0.0 |
| WORD2VEC-WEB-10M | `data/external/corpora/mc4_es/emb_normal/tokens_10M.txt` | 106632 | 4673 de 6059 | `c3781f69a2f2ad5b` | 0.0 |
| FASTTEXT-WIKI-10M | `data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt` | 100098 | 6059 de 6059 | `e54a363da986aeac` | 0.0 |

## 3. Resultados (umbrales congelados de V03, la metrica primaria)

| variante | corrected_cv | delta vs V03 | IC95 pareado | celdas + | celdas - | solo esta | solo V03 |
|---|---|---|---|---|---|---|---|
| FUSION-FASTTEXT-WIKI-10M | 0.7818 | +0.0046 | [-0.0032, +0.0124] | 33 | 20 | 8 | 5 |
| EMB-FASTTEXT-WIKI-10M | 0.6530 | -0.1242 | [-0.1481, -0.1006] | 87 | 160 | 16 | 88 |
| FUSION-FASTTEXT-WEB-10M | 0.7812 | +0.0040 | [-0.0032, +0.0112] | 32 | 19 | 5 | 5 |
| EMB-FASTTEXT-WEB-10M | 0.6638 | -0.1134 | [-0.1366, -0.0905] | 89 | 157 | 22 | 89 |
| FUSION-FASTTEXT-MIXTO-20M | 0.7776 | +0.0004 | [-0.0074, +0.0083] | 25 | 21 | 6 | 6 |
| EMB-FASTTEXT-MIXTO-20M | 0.6658 | -0.1114 | [-0.1348, -0.0885] | 87 | 152 | 20 | 79 |
| FUSION-WORD2VEC-WEB-10M | 0.7762 | -0.0010 | [-0.0090, +0.0070] | 29 | 25 | 4 | 7 |
| EMB-WORD2VEC-WEB-10M | 0.6412 | -0.1360 | [-0.1602, -0.1123] | 84 | 182 | 15 | 86 |

Mejor variante: **FUSION-FASTTEXT-WIKI-10M** (0.7818, delta +0.0046). Mejor fusion: **FUSION-FASTTEXT-WIKI-10M** (0.7818, delta +0.0046).

## 4. Complementariedad por rebanada

| rebanada | filas | V03 | F wiki | F web | F mixto |
|---|---|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.9199 | 0.9173 | 0.9182 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8925 | 0.8906 | 0.8909 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0657 | 0.0667 | 0.0648 |
| colectivos | 93 | 0.5740 | 0.5890 | 0.6028 | 0.5892 |
| insultos | 855 | 0.8357 | 0.8389 | 0.8386 | 0.8363 |
| elongaciones | 57 | 0.6853 | 0.7221 | 0.7059 | 0.7221 |
| coloquial | 214 | 0.8177 | 0.8230 | 0.8207 | 0.8253 |
| hashtags | 144 | 0.5311 | 0.5462 | 0.5415 | 0.5324 |
| menciones | 702 | 0.8462 | 0.8483 | 0.8502 | 0.8479 |
| urls | 458 | 0.6578 | 0.6782 | 0.6805 | 0.6638 |
| cita | 37 | 0.8449 | 0.8393 | 0.8460 | 0.8393 |

## 5. Vecinos de las variantes ortograficas

Sin labels externos no se afirma que un vecino sea el lexema correcto; la tabla se
muestra para comparar el comportamiento entre corpora.

| variante (freq. TOXIC) | wiki 10M | web 10M | mixto 20M |
|---|---|---|---|
| `callate` (212) | (sin vector) | `callate` (1.000) | `callate` (1.000) |
| `cllate` (159) | (sin vector) | (sin vector) | (sin vector) |
| `violacin` (34) | (sin vector) | (sin vector) | (sin vector) |
| `espaa` (34) | (sin vector) | `espaa` (1.000) | `espaa` (1.000) |
| `imbcil` (16) | (sin vector) | (sin vector) | (sin vector) |
| `arabe` (36) | (sin vector) | `arabe` (1.000) | `arabe` (1.000) |
| `estn` (27) | (sin vector) | `estn` (1.000) | `estn` (1.000) |
| `mereces` (77) | (sin vector) | `mereces` (1.000) | `mereces` (1.000) |

## 6. Decision

Se aplica el criterio pre-registrado de la seccion 29 sin modificarlo: `adopted` exige
delta >= +0.005 **y** IC95 pareado que excluya el 0.

- Mejor fusion: delta +0.0046, IC95 [-0.0032, +0.0124].
- Ninguna variante genera submission ni se envia al leaderboard.

## 7. Limites

1. El corpus web es un tramo del 13,9 % del shard 0 de C4, sin reordenacion interna.
2. La metrica primaria es el `corrected_cv`, que la leccion operativa de v4 demuestra que
   no es un selector fiable: cuatro mejoras locales consecutivas no transfirieron.
3. No hay holdout disponible (agotado 5/5), de modo que la unica evidencia externa posible
   seria el leaderboard, y enviar una sonda se decide de forma expresa.
