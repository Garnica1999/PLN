# V22 · B1_SPECIALIST — Sustituir unicamente el primer discriminador

**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**.

## 1. Auditoria historica

Hecha buscando **codigo**, no nombres. Ningun experimento habia sustituido especificamente B1, entrenado un modelo dedicado a `13-17` frente a `>=18`, ni usado una representacion especifica para B1. El antecedente cercano **E-028 no es equivalente**: entrena un `LinearSVC` sobre el **par adyacente** (dos clases, 57.539 filas), no sobre el umbral **anidado** (todas las filas, 17/83).

## 2. V20 como control exacto

- Reconstruido: **0.6097782** frente a **0.6097782**. **Reproduce: True**.
- El script **se aborta** si no cuadra, porque sin ese control ninguna variante es interpretable.

## 3. Resultados, con las otras cuatro fronteras intactas

| Candidato | macro-F1 | Delta vs V20 | Folds | B1 balanceada | B1 AUC | Rescates | Danos | Neto | Cambios | Estado |
|---|---|---|---|---|---|---|---|---|---|---|
| V20 (control exacto) | 0.6097782 | +0.0000 | 0/5 | 0.7632 | 0.9284 | 0 | 0 | +0 | 0 | **CONTROL** |
| LinearSVC C 0.2 | 0.5992098 | -0.0106 | 0/5 | 0.6782 | 0.9178 | 6,226 | 8,683 | -2,457 | 24,075 | **NO_SIGNAL: B1 no mejora, mecanismo cerrado** |
| LinearSVC C 1.0 | 0.5992058 | -0.0106 | 0/5 | 0.6784 | 0.9178 | 6,228 | 8,685 | -2,457 | 24,076 | **NO_SIGNAL: B1 no mejora, mecanismo cerrado** |
| boosting especializado B1 | 0.6097377 | -0.0000 | 3/5 | 0.7606 | 0.9272 | 276 | 282 | -6 | 805 | **NO_SIGNAL: B1 no mejora, mecanismo cerrado** |

## 4. Delta por fold

| Candidato | Delta por fold |
|---|---|
| V20 (control exacto) | [0.0, 0.0, 0.0, 0.0, 0.0] |
| LinearSVC C 0.2 | [-0.010477, -0.011691, -0.010727, -0.009293, -0.010761] |
| LinearSVC C 1.0 | [-0.010485, -0.011644, -0.010657, -0.00936, -0.010823] |
| boosting especializado B1 | [4.6e-05, 3.1e-05, 0.000667, -0.000279, -0.000665] |

## 5. F1 por clase

| Candidato | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| V20 (control exacto) | 0.6818 | 0.517 | 0.4965 | 0.5687 | 0.6637 | 0.731 |
| LinearSVC C 0.2 | 0.6446 | 0.5161 | 0.4653 | 0.5686 | 0.6662 | 0.7343 |
| LinearSVC C 1.0 | 0.6447 | 0.5162 | 0.4653 | 0.5686 | 0.6662 | 0.7343 |
| boosting especializado B1 | 0.6815 | 0.517 | 0.4967 | 0.5686 | 0.6636 | 0.731 |
