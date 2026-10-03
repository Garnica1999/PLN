# V32-00: reproducibilidad de V31-04

Generado por `experiments/v32_00_reproducibilidad.py` (V32-00 (reproducibilidad de V31-04 y control del nondeterminismo)).

**Estado `REPRODUCIBILITY_CONFIRMED`. Alcance: reproduccion exacta de V31-04 y control del nondeterminismo. No se prueba ninguna variante nueva**

## 1. La comparacion contra V31-04

| magnitud | V31-04 | reproduccion | coincide |
|---|---:|---:|:--:|
| corrected_cv | 0.7813779888 | 0.7813779888 | si |
| rescates | 70 | 70 | si |
| danios | 53 | 53 | si |
| neto | +17 | +17 | si |
| celdas modificadas | 123 | 123 | si |

## 2. El control de nondeterminismo

Se ejecuto con `PYTHONHASHSEED` en 0, 999 y las salidas son **identicas**: True.

Fuentes de nondeterminismo descartadas:

- orden de iteracion de set con PYTHONHASHSEED variable
- dos corridas en el mismo proceso
- el vectorizador TF-IDF se ajusta dentro de cada pliegue, como en V03

## 3. Lo que este script no hace

- No prueba ninguna variante nueva.
- No modifica V03.
- No genera submission.
- No lee el test.

Coste: 17.3 s, memoria 228 MB, 2 modelos, 0 submissions.

