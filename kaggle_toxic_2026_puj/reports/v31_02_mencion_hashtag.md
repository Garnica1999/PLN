# V31-02: Fase 2 de STRUCTURAL_TEXT_FEATURES, bloque mencion / hashtag

Generado por `experiments/v31_02_mencion_hashtag.py` (V31-02 (Fase 2 de STRUCTURAL_TEXT_FEATURES: bloque mencion / hashtag)).

**Estado `OOF_EVALUATED`. Alcance: solo el bloque mencion/hashtag, sin interacciones.**

## 1. Features

13 features, todas de posicion y contexto estructural. No se usa el
nombre del usuario ni el contenido semantico del hashtag.

- `mencion_existe`
- `mencion_pos_primera_rel`
- `mencion_pos_ultima_rel`
- `mencion_al_inicio`
- `mencion_al_final`
- `mencion_n`
- `mencion_densidad`
- `hashtag_existe`
- `hashtag_pos_primero_rel`
- `hashtag_pos_ultimo_rel`
- `hashtag_inicial`
- `hashtag_final`
- `hashtag_n`

## 2. Resultado

| etiqueta | F1 control | F1 challenger | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7605 | -0.0011 |
| TR | 0.8157 | 0.8130 | -0.0027 |
| AG | 0.7542 | 0.7559 | +0.0017 |

**corrected_cv 0.7772114525 -> 0.7764792007 | delta -0.0007322518**

train F1 0.9711 | gap +0.1946

## 3. Rescates y danos

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 30 | 32 | -2 | 62 |
| TR | 15 | 18 | -3 | 33 |
| AG | 27 | 29 | -2 | 56 |

**rescates 72 | danios 79 | neto -7**

## 4. Lo que este script no hace

- No prueba interacciones: eso es el Control B.
- No combina bloques.
- No genera submission. No lee el test.

Coste: 24.0 s, memoria 227 MB, 2 modelos, 0 submissions.

