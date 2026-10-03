# Búsqueda de señal residual — informe consolidado (fases A-J)

Generado por `scripts/phase20_informe_fase_residual.py` a partir de los artefactos en `experiments/`.

**Criterio de adopción declarado de antemano**: Δ ≥ +0,003 y ≥4/5 folds positivos, con confirmación independiente en el holdout congelado.

**Regla de integridad**: validación agrupada por clústeres de near-duplicados (`StratifiedGroupKFold`), comparación LOFO sobre las mismas particiones, holdout congelado intacto, ninguna decisión guiada por el leaderboard.

## A · Auditoría del estado

Artefacto: `experiments/phase12_auditoria_estado.json`

## B · Errores y desacuerdo

Artefacto: `experiments/phase13_error_desacuerdo.json`

## C · Consenso y gating

Artefacto: `experiments/phase14_consenso_familias.json`

| Variante | macro-F1 | Δ vs v12 | folds + | n features |
|---|---|---|---|---|
| +C1 | 0.5986 | -0.0018 | 1/5 | 290 |
| +C1+C2 | 0.6002 | -0.0002 | 3/5 | 297 |
| +C1+C2+C3 | 0.5998 | -0.0006 | 3/5 | 325 |

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## D · Calibración y logits

Artefacto: `experiments/phase15_calibracion_scores.json`

| Variante | macro-F1 | Δ vs v12 | folds + | n features |
|---|---|---|---|---|
| D1a log-prob (T=1) · promedio de combinadores | 0.6005 | +0.0001 | 3/5 | 279 |
| D1a log-prob (T=1) · promedio de combinadores | 0.6007 | +0.0003 | 3/5 | 279 |
| D1b log-prob + T in-fold · promedio de combinadores | 0.6003 | -0.0001 | 2/5 | 279 |
| D1b log-prob + T in-fold · promedio de combinadores | 0.6002 | -0.0002 | 2/5 | 279 |

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## D5 · Diversidad de subconjuntos

Artefacto: `experiments/phase15b_diversidad_subsets.json`

| Variante | macro-F1 | Δ vs v12 | folds + | n features |
|---|---|---|---|---|
| mlp_full | 0.5976 | -0.0028 | 0/5 | — |
| logreg_full | 0.5915 | -0.0089 | 0/5 | — |
| mlp_subset | 0.5851 | -0.0153 | 0/5 | — |
| logreg_subset | 0.5879 | -0.0125 | 0/5 | — |
| producción (MLP_full + LogReg_full) | 0.6004 | +0.0000 | 2/5 | — |
| 3 vías (+ MLP_subset) | 0.5987 | -0.0017 | 0/5 | — |
| 3 vías (+ LogReg_subset) | 0.5990 | -0.0014 | 1/5 | — |
| 4 vías | 0.5976 | -0.0028 | 0/5 | — |

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## G · Ordinal y greedy

Artefacto: `experiments/phase16_ordinal_greedy_g1g2.json`

- **g1_ordinal**: macro-F1 0.6001 (-0.0003, 1/5 folds, adoptado: False)

- **g2_greedy**: macro-F1 0.5770 (-0.0234, —/5 folds, adoptado: False)

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## E · Estructura v2 (bloques libres)

Artefacto: `experiments/phase17_estructura_v2_libres.json`

- **libres**: macro-F1 0.6006 (+0.0002, 2/5 folds, adoptado: False)

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## F · Limpieza selectiva

Artefacto: `experiments/phase18_limpieza_selectiva.json`

| Variante | macro-F1 | Δ vs v12 | folds + | n features |
|---|---|---|---|---|
| F2 clúster de near-duplicados (un representante) | 0.6003 | -0.0001 | 2/5 | — |
| F3 clúster selectivo en clases centrales | 0.6013 | +0.0009 | 3/5 | — |

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## §13 · Vista estructural y extremos

Artefacto: `experiments/phase21_vista_estructural.json`

| estructura | 0.6022 | +0.0018 | 4/5 | 285 |
| extremos | 0.6004 | +0.0000 | 2/5 | 285 |

Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## §13 · Refuerzo de la vista

Artefacto: `experiments/phase22_estructura_ampliada.json`

| Variante | macro-F1 | Δ vs v12 | folds + | n features |
|---|---|---|---|---|
| V1 estructura ampliada | 0.6009 | +0.0005 | 4/5 | 285 |
| V2 estructura + extremos | 0.6001 | -0.0003 | 2/5 | 291 |
| V3 ampliada + extremos | 0.6010 | +0.0006 | 3/5 | 291 |


Referencia: **0.6004** (por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

## H · Holdout congelado de la vista

Artefacto: `experiments/phase19_holdout_VistaEstructura.json`

| Configuración | macro-F1 en holdout |
|---|---|
| base | 0.6077 |
| VistaEstructura | 0.6108 |

## Veredicto

Ninguna variante alcanza el criterio. **La línea de señal residual se declara saturada**: las fuentes evaluadas no producen una mejora suficientemente grande y consistente respecto al modelo final v12 (CV 0,6004; holdout 0,6077; leaderboard 0,62182).

El modelo final **no cambia** y no se genera submission nueva.


## Líneas cerradas en esta fase

| Línea | Evidencia |
|---|---|
| Consenso/desacuerdo por familia y gating contextual | D-028 |
| Calibración por miembro y blending en logits | D-029 |
| Diversidad de subconjuntos de features en el segundo nivel | D-029 |
| Edad implícita/consistencia ordinal y selección greedy | D-030 |
| Estructura v2 (retuit, hashtag, extremos) | D-030 |
| Limpieza selectiva y representantes por clúster | D-031 |

