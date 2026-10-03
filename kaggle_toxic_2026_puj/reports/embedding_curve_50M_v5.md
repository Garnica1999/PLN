# Curva de aprendizaje 10M -> 50M y efecto del registro (fase v5, P7)

Generado por `experiments/embedding_08_fasttext_50M_v5.py` (EBD-13..14 (P7)).

## 1. Modelos (arquitectura fija de la §4 de la spec)

| modelo | corpus | subpalabras | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|---|---|
| FASTTEXT-WIKI-50M | 50M | 3-6 | 247863 | 576.8 | 6059 | `1045aba32c6440e3` |
| WORD2VEC-WIKI-50M | 50M | 0-6 | 247863 | 454.2 | 4412 | `53314bccc6c89369` |

## 2. Resultados de esta corrida

| variante | corrected_cv | delta vs V03 | IC95 pareado | celdas + | celdas - | solo esta | solo V03 |
|---|---|---|---|---|---|---|---|
| FUSION-FASTTEXT-WIKI-50M | 0.7803 | +0.0031 | [-0.0049, +0.0112] | 35 | 24 | 9 | 5 |
| EMB-FASTTEXT-WIKI-50M | 0.6687 | -0.1085 | [-0.1313, -0.0863] | 93 | 156 | 21 | 85 |
| FUSION-WORD2VEC-WIKI-50M | 0.7773 | +0.0001 | [-0.0087, +0.0088] | 37 | 28 | 10 | 6 |
| EMB-WORD2VEC-WIKI-50M | 0.6604 | -0.1168 | [-0.1402, -0.0939] | 90 | 154 | 20 | 72 |

## 3. La curva completa

Las cuatro filas comparten arquitectura, semilla, protocolo, umbrales y bootstrap. La
diferencia es unicamente el corpus.

| corpus | cobertura por tipo | cobertura por token | embedding solo | fusion con V03 | delta fusion |
|---|---|---|---|---|---|
| wikipedia 10M | 0.6839 | 0.8833 | 0.6530 | 0.7818 | +0.0046 |
| web 10M | 0.8256 | 0.9512 | 0.6638 | 0.7812 | +0.0040 |
| mixto 20M | 0.8413 | 0.9553 | 0.6658 | 0.7776 | +0.0004 |
| wikipedia 50M | 0.7703 | 0.9238 | 0.6687 | 0.7803 | +0.0031 |

## 4. Lectura

- **Embedding aislado**: 10M 0.6530 -> 50M 0.6687 (+0.0157).
- **Fusion con V03**: 10M 0.7818 -> 50M 0.7803 (-0.0015), con delta sobre V03 de +0.0031.

## 5. Complementariedad por rebanada

| rebanada | filas | V03 | fusion wiki 50M |
|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.9213 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8942 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0644 |
| colectivos | 93 | 0.5740 | 0.5891 |
| insultos | 855 | 0.8357 | 0.8406 |
| elongaciones | 57 | 0.6853 | 0.7067 |
| coloquial | 214 | 0.8177 | 0.8184 |
| hashtags | 144 | 0.5311 | 0.5575 |
| menciones | 702 | 0.8462 | 0.8479 |
| urls | 458 | 0.6578 | 0.6721 |
| cita | 37 | 0.8449 | 0.8460 |

## 6. Decision

Criterio pre-registrado de la §29 sin cambios: adopted exige delta >= +0.005 y
IC95 que excluya el 0. Mejor fusion de la corrida: **FUSION-FASTTEXT-WIKI-50M** con
0.7803 (delta +0.0031, IC95
[-0.0049, +0.0112]). Adoptado: **no**.

Si la aportacion complementaria no crece al quintuplicar los tokens, no se escala a 100M:
la conclusion de D-032 queda confirmada tambien en la dimension de tamano. Ninguna
variante genera submission ni sonda de leaderboard.
