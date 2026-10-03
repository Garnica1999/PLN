# V32-05: Fase 6 de V32, 4 bins de longitud en vez de 3

Generado por `experiments/v32_05_cuatro_bins.py` (V32-05 (Fase 6 de V32: 4 bins de longitud en vez de 3)).

**Estado `OOF_EVALUATED`. Alcance: solo 3 bins contra 4 bins, con cortes tomados del training fold**

cuartiles del training fold, nunca umbrales fijos ni elegidos mirando OOF.

## 1. Resultado

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0.7772114525 | 0 |
| 3 bins | 0.7817210522 | +0.0045095996 |
| 4 bins | 0.7774170350 | +0.0002055825 |

**4 bins sobre 3 bins: -0.0043040172**

Rescates y danios de 4 bins: 57 / 53 / neto +4.

## 2. Lo que este script no hace

- No hace un barrido de bins: solo 3 contra 4.
- No elige los cortes mirando el OOF.
- No genera submission. No lee el test.

Coste: 19.7 s, memoria 229 MB, 3 modelos, 0 submissions.

