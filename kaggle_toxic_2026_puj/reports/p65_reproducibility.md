# P65 - Reproducibilidad

Fecha 2026-09-29T14:55:51+00:00.

## Nivel declarado: R2

Semilla fija, hash de V03 verificado al inicio de cada script, y ejecucion real con
exit code capturado sin pipe. Los controles de capacidad y de margen se calculan
dentro de cada pliegue, nunca antes.

**No se declara R4.** R4 exige reproduccion independiente en otra sesion.

| magnitud | valor |
|---|---|
| commit | fb2e422 |
| seed | 42 |
| hash de V03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| corrected_cv | 0.7772114525165721 |
| particion | 5 pliegues canonicos, 0 partidos |
| cortes de longitud | los cuatro de P63, heredados y congelados |
| puerta del especialista | 16 tokens, heredada de P63, no optimizada |
| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |
| umbrales | 0,44 / 0,51 / 0,46 |

## Los scripts, en orden

| orden | script | que hace |
|---:|---|---|
| 1 | `experiments/p65_00_audit_gap.py` | auditoria del baseline y clasificacion funcional de 16 familias |
| 2 | `experiments/p65_01_ngrams.py` | word (1,3) anadido a V03, con dos controles de capacidad |
| 3 | `experiments/p65_02_huecos_contraste.py` | huecos 1-2, 3-5 y 6-10, y alcance de contraste |
| 4 | `experiments/p65_03_cohesion.py` | diagnostico de cohesion y bloque con las mismas cantidades |
| 5 | `experiments/p65_04_especialista.py` | especialista de tweets largos y sus tres controles |
| 6 | `experiments/p65_05_cerrar.py` | registro, cumplimiento, figuras e informe final |

## Artefactos

| fichero |
|---|
| `reports/p65_candidate_comparison.csv` |
| `reports/p65_cohesion.csv` |
| `reports/p65_cohesion.json` |
| `reports/p65_cohesion.md` |
| `reports/p65_compliance.md` |
| `reports/p65_error_intersection.csv` |
| `reports/p65_especialista.json` |
| `reports/p65_especialista.md` |
| `reports/p65_final_research.md` |
| `reports/p65_final_summary.json` |
| `reports/p65_huecos_contraste.json` |
| `reports/p65_huecos_contraste.md` |
| `reports/p65_hypothesis_gap_analysis.csv` |
| `reports/p65_hypothesis_gap_analysis.json` |
| `reports/p65_hypothesis_gap_analysis.md` |
| `reports/p65_hypothesis_registry.csv` |
| `reports/p65_long_range.csv` |
| `reports/p65_master_results.csv` |
| `reports/p65_ngrams.json` |
| `reports/p65_ngrams.md` |
| `reports/p65_reproducibility.md` |
| `reports/p65_scope_analysis.csv` |
| `reports/p65_skipgram.csv` |
| `figuras/p65/p65_01_v03_vs_candidatos.png` |
| `figuras/p65/p65_02_delta_vs_v03.png` |
| `figuras/p65/p65_03_largo_alcance.png` |
| `figuras/p65/p65_04_skipgram_vs_cohesion.png` |
| `figuras/p65/p65_05_cohesion_por_estrado.png` |
| `figuras/p65/p65_06_confianza_vs_cohesion.png` |
| `figuras/p65/p65_07_especialista_por_estrado.png` |
| `figuras/p65/p65_08_rescate_vs_dano.png` |
| `figuras/p65/p65_09_interseccion_errores.png` |
| `figuras/p65/p65_10_rescate_por_estrado.png` |
| `figuras/p65/p65_11_consistencia_pliegues.png` |
| `figuras/p65/p65_12_brechas_controles.png` |
| `figuras/p65/p65_13_evolucion_candidatos.png` |

> V03 remains the best verified leaderboard result.
