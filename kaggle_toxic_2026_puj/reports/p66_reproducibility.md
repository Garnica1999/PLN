# P66 - Reproducibilidad

Fecha 2026-09-29T15:56:16+00:00.

## Nivel declarado: R2

Semilla fija, hash de V03 verificado al inicio de cada script, ejecucion real con
exit code capturado sin pipe, y regeneracion VERIFICADA contra el numero que cada
fase de origen reporto.

**No se declara R4.** R4 exige reproduccion independiente en otra sesion.

| magnitud | valor |
|---|---|
| commit | 9f75317 |
| seed | 42 |
| hash de V03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| corrected_cv | 0.7772114525165721 |
| particion | 5 pliegues canonicos por grupo |
| unidad de bootstrap | 1415 grupos |
| cortes de longitud | los cuatro de P63, heredados |
| umbrales | 0,44 / 0,51 / 0,46 congelados |
| mecanismo elegido | residual anidado con beta seleccionado en inner_validation |
| split interno | StratifiedKFold de 3 partes sobre el entrenamiento de cada pliegue |

## Los scripts, en orden

| orden | script | que hace |
|---:|---|---|
| 1 | `experiments/p66_00_audit_gap.py` | auditoria, inventario y clasificacion del espacio de scores |
| 2 | `experiments/p66_01_regenerar_scores.py` | regeneracion verificada de los 7 scores |
| 3 | `experiments/p66_02_fusion_espacio_scores.py` | cinco mecanismos anidados con control barajado |
| 4 | `experiments/p66_03_complementariedad_bootstrap.py` | rescate unico, alta confianza y bootstrap |
| 5 | `experiments/p66_04_cerrar.py` | cumplimiento, figuras e informe final |

## Artefactos

| fichero |
|---|
| `reports/p66_candidate_comparison.csv` |
| `reports/p66_candidate_inventory.csv` |
| `reports/p66_complementarity.csv` |
| `reports/p66_complementarity.json` |
| `reports/p66_complementarity.md` |
| `reports/p66_compliance.md` |
| `reports/p66_error_intersection.csv` |
| `reports/p66_final_research.md` |
| `reports/p66_final_summary.json` |
| `reports/p66_hypothesis_gap_analysis.csv` |
| `reports/p66_hypothesis_gap_analysis.json` |
| `reports/p66_hypothesis_gap_analysis.md` |
| `reports/p66_long_aware_correction.csv` |
| `reports/p66_master_results.csv` |
| `reports/p66_random_controls.csv` |
| `reports/p66_reproducibility.md` |
| `reports/p66_residual_correction.csv` |
| `reports/p66_score_fusion.csv` |
| `reports/p66_score_fusion.json` |
| `reports/p66_score_fusion.md` |
| `reports/p66_score_regeneration.json` |
| `figuras/p66/p66_01_v03_vs_fusion.png` |
| `figuras/p66/p66_02_delta_por_mecanismo.png` |
| `figuras/p66/p66_03_rescate_vs_dano.png` |
| `figuras/p66/p66_04_rescate_alta_confianza.png` |
| `figuras/p66/p66_05_rendimiento_por_longitud.png` |
| `figuras/p66/p66_06_residuo_vs_error.png` |
| `figuras/p66/p66_07_score_vs_margen.png` |
| `figuras/p66/p66_08_delta_por_pliegue.png` |
| `figuras/p66/p66_09_gates.png` |
| `figuras/p66/p66_10_segunda_particion.png` |
| `figuras/p66/p66_11_evolucion.png` |
| `figuras/p66/p66_12_verificacion_scores.png` |
| `figuras/p66/p66_13_bootstrap.png` |

> V03 remains the best verified leaderboard result.
