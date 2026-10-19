# Porte de los consensos V53-V56 y del analisis V57 (`repro/cadena/`)

Modulos: `repro/cadena/consenso_v2.py` (`poe_ponderado`, `CONSENSOS`, `pesos_geometricos`,
`regla_plug_in`) y `repro/cadena/analisis_v57.py` (4 bloques + `analisis_completo`).
Verificacion: `/tmp/opencode/verifica_consensos_v57.py` (fuera del repo).

## Especificaciones (miembros en orden de apilado; pesos pre-normalizacion)
| Version | Miembros | Pesos |
|---|---|---|
| V53 | v30, v43, v42, v40, v39_ordinal, v39_hgb, v53_dense | W_V48*0.975 + [0.025]; W_V48 = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549] |
| V54 | v30, v43, v47, v42, v40, v39_ordinal, v39_hgb, v53_dense | W_V48*0.98, Cramer 0.80 V43 + 0.20 V47, + [0.020] |
| V55 | v30, v43, v47, v42, v40, v39_ordinal, v39_hgb, v49_catboost | [0.52829, 0.07720, 0.02895, 0.17931, 0.07890, 0.05062, 0.04869, 0.00805] |
| V56 | v30, v43, v47, v42, v40, v39_ordinal, v39_hgb, v51_consistent, v53_dense | [0.55939, 0.06023, 0.02089, 0.16873, 0.06080, 0.04376, 0.05204, 0.01035, 0.02381] |

Regla plug-in [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]; eps 1e-12; `CONSENSOS[v]["pesos"]`
== `pesos_geometricos(v)` bit-exacto en las 4 versiones.

## Verificacion (169.336 OOF; y_true de `artifacts/p33_scores/todos/`; V30 = `features/age_plugin_segunda_particion_oof.npy`)
- V53 = **0.6188258751244** (doc 0.6188259) — V54 = **0.6188887497659** (doc 0.6188887)
- V55 = **0.6186427356931** (notebook ejecutado 0.6186427) — V56 = **0.6189929489712** (doc 0.6189929)
- V55 SI es re-ejecutable: existe `features/v49_oof_catboost.npy` (standalone 0.6066473); no se excluye.
- Con V30 = `artifacts/p33_scores/todos/v30_prob.npy` NO coinciden: 0.6182669 / 0.6182588 / 0.6183508 / 0.6183034.

V57 sobre el OOF de V56: `f1_base` 0.6189929489711685 bit-exacto; sensibilidad, Power-Mean (9 alphas),
deciles (10) y brecha de menciones con max|diff| = **0.000e+00** vs el JSON; folds coinciden
([0.6229813, 0.6203763, 0.6198614, 0.6192412, 0.6124917]).

## Dudas
1. La instruccion pedia `artifacts/p33_scores/todos/v30_prob.npy` para V30, pero solo la segunda particion reproduce lo documentado (fuente de `v57_analisis_matematico_estadistico.py:36` y secciones 49-53); el artefacto p33 es el "V30 canonico" antiguo.
2. V55 historico 0.6182436 usaba el baseline V48 antiguo; con los artefactos actuales da 0.6186427 (igual que el notebook ejecutado, seccion 51).
