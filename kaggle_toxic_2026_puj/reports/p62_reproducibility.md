# P62 - Reproducibilidad

Generado por `experiments/p62_05_cerrar.py`. Fecha 2026-09-29T05:24:55+00:00.

## Nivel declarado: R2

R2 significa semilla fija, artefactos en disco, hash del OOF y de la anotacion
verificados, y ejecucion real con exit code capturado sin pipe. **No se declara R4,
porque R4 exige reproduccion independiente en otra sesion y eso no se ha hecho.**

## Identificadores registrados

| magnitud | valor |
|---|---|
| commit | ab9fbf5 |
| seed | 42 |
| hash de V03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| corrected_cv de V03 | 0.7772114525165721 |
| hash del OOF | verificado al inicio de cada script |
| anotacion | features/p62_anotacion_spacy.json.gz, ids verificados contra el OOF |
| tagger | spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003 |
| particion | los 5 pliegues canonicos, 1415 grupos, 0 partidos |
| canales | inducidos por pliegue con min_df=2 |
| umbrales | 0,44 / 0,51 / 0,46, congelados |
| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |

## Artefactos

| fichero | que contiene |
|---|---|
| `reports/p62_compliance.md` | artefacto de la fase |
| `reports/p62_final_summary.json` | artefacto de la fase |
| `reports/p62_hypothesis_gap_analysis.csv` | artefacto de la fase |
| `reports/p62_hypothesis_gap_analysis.md` | artefacto de la fase |
| `reports/p62_linguistic_composition.csv` | artefacto de la fase |
| `reports/p62_linguistic_composition.md` | artefacto de la fase |
| `reports/p62_linguistic_eda.json` | artefacto de la fase |
| `reports/p62_linguistic_gap.csv` | artefacto de la fase |
| `reports/p62_master_results.csv` | artefacto de la fase |
| `reports/p62_p61_audit.json` | artefacto de la fase |
| `reports/p62_pista_l_word_pos.json` | artefacto de la fase |
| `reports/p62_pista_l_word_pos.md` | artefacto de la fase |
| `reports/p62_pista_m_dependencias.json` | artefacto de la fase |
| `reports/p62_pista_m_dependencias.md` | artefacto de la fase |
| `reports/p62_probability_analysis.csv` | artefacto de la fase |
| `reports/p62_reproducibility.md` | artefacto de la fase |
| `features/p62_anotacion_spacy.json.gz` | anotacion o celdas |
| `features/p62_l_word_pos_cells.npz` | anotacion o celdas |
| `features/p62_m_dep_cells.npz` | anotacion o celdas |

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_05_cerrar.py`. Sin push.*