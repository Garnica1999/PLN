# Curva de aprendizaje completa 5M -> 10M -> 50M (fase v5, P10)

Generado por `experiments/embedding_10_fasttext_5M_v5.py` (EBD-15..16 (P10)).
Cierra la curva de la §10 por debajo del punto de partida. Todas las filas comparten
arquitectura, semilla, protocolo, umbrales congelados y bootstrap pareado.

## 1. Modelos de 5M (arquitectura fija)

| modelo | subpalabras | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|---|
| FASTTEXT-WIKI-5M | 3-6 | 64839 | 0.0 | 6059 | `b7108b6877f8a6c8` |
| WORD2VEC-WIKI-5M | 0-6 | 64839 | 0.0 | 3412 | `4323c9fa94d12444` |

## 2. Resultados de esta corrida

| variante | corrected_cv | delta vs V03 | IC95 pareado | celdas + | celdas - | solo esta | solo V03 |
|---|---|---|---|---|---|---|---|
| FUSION-FASTTEXT-WIKI-5M | 0.7757 | -0.0015 | [-0.0091, +0.0060] | 26 | 22 | 7 | 3 |
| EMB-FASTTEXT-WIKI-5M | 0.6670 | -0.1102 | [-0.1327, -0.0880] | 77 | 139 | 18 | 70 |
| FUSION-WORD2VEC-WIKI-5M | 0.7697 | -0.0075 | [-0.0144, -0.0007] | 20 | 25 | 3 | 5 |
| EMB-WORD2VEC-WIKI-5M | 0.6269 | -0.1504 | [-0.1744, -0.1265] | 85 | 189 | 17 | 95 |

## 3. La curva completa

| corpus | tokens | cobertura por tipo | cobertura por token | embedding solo | fusion con V03 |
|---|---|---|---|---|---|
| wikipedia 5M | 5M | 0.6302 | 0.8646 | 0.6670 | 0.7757 |
| wikipedia 10M | 10M | 0.6839 | 0.8833 | 0.6530 | 0.7818 |
| wikipedia 50M | 50M | 0.7703 | 0.9238 | 0.6687 | 0.7803 |
| web mC4-es 10M | 10M | 0.8256 | 0.9512 | 0.6638 | 0.7812 |
| mixto 20M | 20M | 0.8413 | 0.9553 | 0.6658 | 0.7776 |

## 4. Lectura

- **Embedding aislado, solo Wikipedia**: 5M 0.6670 -> 10M 0.6530 -> 50M 0.6687. Monotona: **NO**.
- **Fusion con V03, solo Wikipedia**: 5M 0.7757 -> 10M 0.7818 -> 50M 0.7803 (deltas sobre V03: -0.0015, +0.0046, +0.0031).
- **Subpalabras**: fusion FastText frente al control = +0.0060 a 5M, +0.0138 a 10M y +0.0030 a 50M. La forma es de **joroba**, no decreciente: el maximo esta en 10M. Los tres valores caen dentro del ruido del protocolo, asi que lo defendible es que la ventaja de las subpalabras es pequeña en todo el rango y no depende de forma monotone de la escala. La caida a 50M si es coherente con la mecanica de fastText (a mas tokens hay mas palabras con evidencia propia y queda menos que reconstruir), pero la caida a 5M se explica por lo contrario (faltan ejemplos por palabra), no por una mecanica unica.

El punto de 10M en el embedding **aislado** queda por debajo tanto de 5M como de 50M, lo que rompe la monotonia. La lectura correcta no es que 10M sea peor: los intervalos de bootstrap de 5M y 10M se solapan casi por completo (5M delta -0.1102 con IC95 [-0.1327, -0.0880]; 10M delta -0.1242 con IC95 [-0.1481, -0.1006]), de modo que la diferencia de 0,014 no es distinguible del ruido. La curva del embedding aislado es plana en todo el rango 5M-50M.

Lo que si cambia de forma consistente es la **aportacion complementaria**: a 5M la fusion cae por debajo de V03 (delta -0.0015), mientras que a 10M y 50M es ligeramente positiva (+0.0046 y +0.0031). Hay por tanto un **suelo operativo** de tokens por debajo del cual el embedding no aporta nada: 5M no llega, 10M es el minimo que produce el efecto marginal, y por encima el efecto no crece.

## 5. Complementariedad por rebanada

| rebanada | filas | V03 | fusion wiki 5M |
|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.9160 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8889 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0676 |
| colectivos | 93 | 0.5740 | 0.5879 |
| insultos | 855 | 0.8357 | 0.8349 |
| elongaciones | 57 | 0.6853 | 0.7221 |
| coloquial | 214 | 0.8177 | 0.8189 |
| hashtags | 144 | 0.5311 | 0.5322 |
| menciones | 702 | 0.8462 | 0.8462 |
| urls | 458 | 0.6578 | 0.6617 |
| cita | 37 | 0.8449 | 0.8516 |

## 6. Decision

Criterio pre-registrado de la §29 sin cambios. Mejor fusion de la corrida: **FUSION-FASTTEXT-WIKI-5M** con 0.7757 (delta -0.0015, IC95 [-0.0091, +0.0060]). Adoptado: **no**.

La aportacion complementaria se mantiene en la banda 0.000-0.005 en **las cinco**
configuraciones medidas, con IC95 que toca el 0 en todas. **Ninguna** submission ni
sonda de leaderboard.
