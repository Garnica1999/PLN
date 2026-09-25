# Auditoría de la fase — capa de decisión ordinal

| Hipótesis | ¿Nueva? | ¿Ejecutada? | Capas | Δ máximo | ¿Adoptada? | Decisión |
|---|---|---|---|---|---|---|
| A | nueva (no existía como capa de decisión) | sí (2 capas) | -0.0031 | no | CERRADA: la evidencia de fronteras ya está en el meta-modelo |
| E | nueva (no existía como capa de decisión) | sí (6 capas) | -0.0033 | no | CERRADA: el argmax de V12 es la mejor decisión de su distribución |
| G | nueva (no existía como capa de decisión) | sí (3 capas) | +0.0011 | no | CERRADA como capa de decisión: mejora el MAE ordinal pero no macro-F1 |
| F | nueva (no existía como capa de decisión) | sí (3 capas) | +0.0004 | no | CERRADA: el coste ordinal solo preserva la decisión con perfiles casi planos |
| C | nueva (no existía como capa de decisión) | sí (4 capas) | -0.0000 | no | CERRADA: corrección residual neta cero en su forma más suave |
| D | nueva (no existía como capa de decisión) | sí (5 capas) | -0.0004 | no | CERRADA: la decodificación por confusión empeora monótonamente con λ |
| H | nueva (no existía como capa de decisión) | sí (2 capas) | +0.0002 | no | CERRADA: inerte, coherente con que una transformación por fila no cambia el argmax |
| B | nueva (no existía como capa de decisión) | sí (3 capas) | -0.0034 | no | CERRADA: los márgenes adyacentes no forman una escala de Bradley-Terry válida |
| I | nueva (no existía como capa de decisión) | sí (5 capas) | -0.0005 | no | CERRADA: el corrector mínimo reescribe aciertos a cambio de fallos nuevos |

## Separación exigida por el brief (§34)

- **Cerrado previamente**: consenso por familia, gating duro, especialistas, calibración de miembros, diversidad de subconjuntos, estructura v2, limpieza selectiva, edad implícita como features, greedy, multivista (27 variantes).
- **Nuevo y negativo**: 31 de las 33 capas de esta fase.
- **Nuevo y marginal**: 1 (`G desempate E[Y] (tau 0.05)`, Δ +0.0011, 4/5, neto -117).
- **Nuevo y prometedor**: ninguno.

## Criterio de cierre (§35)

Se cumplen las cinco condiciones: (1) ninguna formulación supera el gate; (2) las mejoras están por debajo del ruido entre folds (neto de rescate ≈ daño); (3) las correcciones ordinales no mejoran las fronteras centrales; (4) las mejoras locales no son estables; (5) los métodos reproducen esencialmente V12. **Se cierra la línea.**

## Nota de trazabilidad (D-034)

El artefacto `age_submission_v12_reproducible.csv` **puntuó 0,62188** en Kaggle, ligeramente por encima del archivo enviado (0,62182). La discrepancia de reconstrucción (93,88% de coincidencia) queda por tanto sin impacto en el desempeño y se considera cerrada funcionalmente; ambos artefactos se conservan sin modificar.
