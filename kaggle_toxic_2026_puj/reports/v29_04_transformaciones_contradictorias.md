# V29-04: que transformacion genera ejemplos casi identicos con etiquetas distintas

Generado por `experiments/v29_04_transformaciones_contradictorias.py` (V29-04 (que transformacion genera ejemplos casi identicos con etiquetas distintas)).

**Estado `DIAGNOSTIC_ONLY`. Alcance: analisis de pares. No produce ninguna senal para un predictor, no entrena nada y no genera submission. El conjunto de insultos se deriva del propio train, sin lexicon externo**

## 1. El control que evita la conclusion facil

una categoria es DISCRIMINANTE si aparece mas en los pares contradictorios que en los concordantes con razon de razones mayor que 1,5 y p de Fisher menor que 0,05, exigiendo al menos 10 pares de soporte. Sin ese control, contar categorias en los contradictorios seria concluir que el sol es causa de la lluvia.

El conjunto de insultos no viene de un lexicon externo: se deriva del propio train, con un minimo de 10 apariciones y una razon de razones sobre HS de 2.0. Se encontraron 31 tokens. Los mas frecuentes: perra, cllate, zorra, hija, guarra, novia, sos, subsaharianos, siempre, maldita, sudaca, ella.

## 2. Pares por franja de similitud

| umbral | pares | contradictorios | misma etiqueta | fraccion contradictoria |
|---:|---:|---:|---:|---:|
| 0.90 | 39 | 0 | 39 | 0.0000 |
| 0.85 | 39 | 0 | 39 | 0.0000 |
| 0.75 | 55 | 4 | 51 | 0.0727 |
| 0.70 | 62 | 7 | 55 | 0.1129 |
| 0.60 | 185 | 27 | 158 | 0.1459 |

## 3. Que categoria discrimina, y cual es ruido

Razon de razones mayor que 1 significa que la categoria aparece mas entre los pares
contradictorios. El p de Fisher dice si la diferencia es distinguible del azar.

| umbral | categoria | contradictorios | concordantes | razon | p Fisher | veredicto |
|---|---|---:|---:|---:|---:|---|
| 0.90 | negacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | insulto | 0 | 1 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | mencion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | hashtag | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | url | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | ortografia | 0 | 2 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | truncamiento | 0 | 20 | 0.00 | 1.00e+00 | **INVIRTIDA** |
| 0.90 | solo_puntuacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | repeticion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | numerico | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.90 | sin_clasificar | 0 | 18 | 0.00 | 1.00e+00 | **INVIRTIDA** |
| 0.85 | negacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | insulto | 0 | 1 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | mencion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | hashtag | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | url | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | ortografia | 0 | 2 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | truncamiento | 0 | 20 | 0.00 | 1.00e+00 | **INVIRTIDA** |
| 0.85 | solo_puntuacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | repeticion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | numerico | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.85 | sin_clasificar | 0 | 18 | 0.00 | 1.00e+00 | **INVIRTIDA** |
| 0.75 | negacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | insulto | 0 | 1 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | mencion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | hashtag | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | url | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | ortografia | 0 | 3 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | truncamiento | 0 | 20 | 0.00 | 2.85e-01 | **INVIRTIDA** |
| 0.75 | solo_puntuacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | repeticion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | numerico | 0 | 3 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.75 | sin_clasificar | 4 | 26 | 1.96 | 1.17e-01 | **RUIDO** |
| 0.70 | negacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | insulto | 0 | 1 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | mencion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | hashtag | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | url | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | ortografia | 0 | 5 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | truncamiento | 0 | 20 | 0.00 | 8.55e-02 | **INVIRTIDA** |
| 0.70 | solo_puntuacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | repeticion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | numerico | 0 | 3 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.70 | sin_clasificar | 7 | 28 | 1.96 | 1.55e-02 | **DISCRIMINANTE** |
| 0.60 | negacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | insulto | 1 | 15 | 0.39 | 4.74e-01 | **INVIRTIDA** |
| 0.60 | mencion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | hashtag | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | url | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | ortografia | 0 | 14 | 0.00 | 2.28e-01 | **INVIRTIDA** |
| 0.60 | truncamiento | 1 | 39 | 0.15 | 1.12e-02 | **INVIRTIDA** |
| 0.60 | solo_puntuacion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | repeticion | 0 | 0 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | numerico | 0 | 5 | 0.00 | 1.00e+00 | **SIN_SOPORTE** |
| 0.60 | sin_clasificar | 25 | 93 | 1.57 | 4.19e-04 | **DISCRIMINANTE** |

**Categorias discriminantes en todo el barrido: 2.**

## 4. Ejemplos de pares contradictorios, para leerlos

- similitud **0.8182**, etiquetas `[0, 0, 0]` -> `[1, 1, 1]`, categorias: sin clasificar
  - solo en A: (ninguno)
  - solo en B: mex
- similitud **0.7692**, etiquetas `[1, 1, 1]` -> `[0, 0, 0]`, categorias: sin clasificar
  - solo en A: (ninguno)
  - solo en B: de, ijo
- similitud **0.7500**, etiquetas `[0, 0, 0]` -> `[1, 1, 1]`, categorias: sin clasificar
  - solo en A: mi, tu
  - solo en B: t
- similitud **0.7500**, etiquetas `[1, 1, 0]` -> `[1, 1, 1]`, categorias: sin clasificar
  - solo en A: jeje
  - solo en B: v
- similitud **0.7273**, etiquetas `[1, 1, 1]` -> `[0, 0, 0]`, categorias: sin clasificar
  - solo en A: una
  - solo en B: mi
- similitud **0.7273**, etiquetas `[1, 1, 1]` -> `[0, 0, 0]`, categorias: sin clasificar
  - solo en A: una
  - solo en B: mi
- similitud **0.7097**, etiquetas `[1, 1, 1]` -> `[1, 1, 0]`, categorias: sin clasificar
  - solo en A: (ninguno)
  - solo en B: bien
- similitud **0.6923**, etiquetas `[0, 0, 0]` -> `[1, 1, 1]`, categorias: sin clasificar
  - solo en A: de, ijo, ya
  - solo en B: (ninguno)
- similitud **0.6923**, etiquetas `[1, 1, 1]` -> `[0, 0, 0]`, categorias: sin clasificar
  - solo en A: (ninguno)
  - solo en B: de, ijo, ya
- similitud **0.6923**, etiquetas `[1, 1, 1]` -> `[0, 0, 0]`, categorias: sin clasificar
  - solo en A: (ninguno)
  - solo en B: de, ijo, ya

## 5. Lo que este script no hace

- No entrena ningun modelo.
- No construye ninguna senal para un predictor.
- No lee el test.
- No produce submission.

Coste: 11.2 s, memoria 373 MB, 0 modelos, 0 submissions.

