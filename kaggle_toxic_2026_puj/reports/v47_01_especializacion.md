# V47-01: especialización por etiqueta con selección anidada

Generado por `experiments/v47_01_especializacion.py` (V47-01 (especializacion por etiqueta con seleccion anidada)).

**Estado `OOF_EVALUATED`. ¿Puede una estrategia OOF de entrenamiento robusto reducir el impacto de ejemplos potencialmente inconsistentes y mejorar V03?**

## 1. El control

corrected_cv del control: **0.7772114525**
Reproduce V03: **True**

## 2. La especialización

| etiqueta | C | word_range | char_range | threshold |
|---|---:|---|---|---:|
| HS | 4.0 | (1, 2) | (3, 5) | 0.46 |
| TR | 4.0 | (1, 2) | (3, 6) | 0.46 |
| AG | 1.0 | (1, 2) | (2, 5) | 0.46 |

corrected_cv con especialización: **0.7818097534**
delta vs control: **+0.0045983009**

## 3. Complementariedad contra V03

| magnitud | valor |
|---|---:|
| celdas modificadas | 107 |
| rescates | 62 |
| danios | 45 |
| neto | +17 |

**DECISION: NEW_SIGNAL**

## 4. Coste

400.3 s, memoria 239 MB, 2 modelos, 0 submissions.

