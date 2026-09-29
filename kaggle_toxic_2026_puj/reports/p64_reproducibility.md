# P64 - Reproducibilidad

Fecha 2026-09-29T07:31:17+00:00.

## Nivel declarado: R2

Semilla fija, hash de V03 verificado al inicio de cada script, ejecucion real con
exit code capturado sin pipe, y los bloques de evidencia reutilizados desde disco
en vez de recalculados, lo que hace que P64-04 no dependa de una ejecucion previa
en memoria.

**No se declara R4.** R4 exige reproduccion independiente en otra
sesion y eso no se ha hecho.

| magnitud | valor |
|---|---|
| commit | 2579b51 |
| seed | 42 |
| hash de V03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| corrected_cv | 0.7772114525165721 |
| particion | 5 pliegues canonicos, 1415 grupos, 0 partidos |
| cortes de longitud | los cuatro de P63, heredados y congelados |
| segmentacion | S1 declarada primaria; S2, S3 y S4 como sensibilidad |
| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |
| umbrales | 0,44 / 0,51 / 0,46 |

## Los scripts, en orden

| orden | script | que hace |
|---:|---|---|
| 1 | `experiments/p64_01_gap_analysis.py` | gap analysis contra P59 y medicion de cobertura |
| 2 | `experiments/p64_02_segmentacion_eda.py` | cuatro segmentaciones deterministas y EDA del perfil |
| 3 | `experiments/p64_03_evidencia_segmental.py` | descomposicion de la evidencia por segmento |
| 4 | `experiments/p64_04_leave_one_out.py` | leave-one-segment-out y perfil de contradiccion |
| 5 | `experiments/p64_05_especializacion_larga.py` | los bloques por estrato, contra el margen |
| 6 | `experiments/p64_06_cerrar.py` | registro, cumplimiento, figuras e informe final |

## Artefactos

| fichero |
|---|
| `reports/p64_candidate_comparison.csv` |
| `reports/p64_compliance.md` |
| `reports/p64_contradiction_analysis.csv` |
| `reports/p64_error_intersection.csv` |
| `reports/p64_especializacion_larga.json` |
| `reports/p64_especializacion_larga.md` |
| `reports/p64_evidence_distribution.csv` |
| `reports/p64_evidencia_segmental.json` |
| `reports/p64_evidencia_segmental.md` |
| `reports/p64_final_research.md` |
| `reports/p64_final_summary.json` |
| `reports/p64_hypothesis_gap_analysis.csv` |
| `reports/p64_hypothesis_gap_analysis.json` |
| `reports/p64_hypothesis_gap_analysis.md` |
| `reports/p64_hypothesis_registry.csv` |
| `reports/p64_leave_one_out.json` |
| `reports/p64_leave_one_out.md` |
| `reports/p64_master_results.csv` |
| `reports/p64_reproducibility.md` |
| `reports/p64_segment_analysis.csv` |
| `reports/p64_segment_eda.json` |
| `reports/p64_segment_eda.md` |
| `figuras/p64/p64_01_v03_vs_candidatos.png` |
| `figuras/p64/p64_02_delta_vs_v03.png` |
| `figuras/p64/p64_03_rendimiento_por_longitud.png` |
| `figuras/p64/p64_04_rendimiento_por_segmentos.png` |
| `figuras/p64/p64_05_distribucion_longitud_segmento.png` |
| `figuras/p64/p64_06_confianza_vs_conflicto.png` |
| `figuras/p64/p64_07_concentracion_evidencia.png` |
| `figuras/p64/p64_08_contribucion_leave_one_out.png` |
| `figuras/p64/p64_09_rescate_vs_dano.png` |
| `figuras/p64/p64_10_interseccion_errores.png` |
| `figuras/p64/p64_11_rescate_alta_confianza.png` |
| `figuras/p64/p64_12_consistencia_pliegues.png` |
| `figuras/p64/p64_13_evolucion_candidatos.png` |
| `features/p64_segment_profile.npz` |
| `features/p64_evidencia_cells.npz` |
| `features/p64_loo_cells.npz` |

> V03 remains the best verified leaderboard result.
