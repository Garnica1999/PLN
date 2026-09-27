# V31-03: Fase 3 de STRUCTURAL_TEXT_FEATURES, bloque enfasis ortografico

Generado por `experiments/v31_03_enfasis_ortografico.py` (V31-03 (Fase 3 de STRUCTURAL_TEXT_FEATURES: bloque enfasis ortografico)).

**Estado `OOF_EVALUATED`. Alcance: solo el bloque enfasis, sin interacciones.**

## 1. Features

10 features de intensidad y repeticion. No se construye un lexicon
por palabra: son patrones de caracteres.

- `enfasis_n_repeticiones`
- `enfasis_long_max_repeticion`
- `enfasis_palabras_elongadas`
- `enfasis_ratio_elongadas`
- `enfasis_n_exclamaciones`
- `enfasis_n_interrogaciones`
- `enfasis_n_signos_mixtos`
- `enfasis_n_mayus_consecutivas`
- `enfasis_ratio_mayus`
- `enfasis_long_max_mayus`

## 2. Resultado

| etiqueta | F1 control | F1 challenger | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7616 | -0.0001 |
| TR | 0.8157 | 0.8121 | -0.0036 |
| AG | 0.7542 | 0.7496 | -0.0047 |

**corrected_cv 0.7772114525 -> 0.7744422516 | delta -0.0027692009**

train F1 0.9701 | gap +0.1957

## 3. Rescates y danos

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 14 | 16 | -2 | 30 |
| TR | 7 | 10 | -3 | 17 |
| AG | 28 | 35 | -7 | 63 |

**rescates 49 | danios 61 | neto -12**

## 4. Lo que este script no hace

- No prueba interacciones ni combina bloques.
- No genera submission. No lee el test.

Coste: 24.7 s, memoria 228 MB, 2 modelos, 0 submissions.

