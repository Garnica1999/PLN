# P62 - Reproducibilidad

Fecha 2026-09-29T05:50:51+00:00.

## Nivel declarado: R2

Semilla fija, artefactos en disco, hash del OOF y de la anotacion verificados, y
ejecucion real con exit code capturado sin pipe. **No se declara R4**, porque R4
exige reproduccion independiente en otra sesion y eso no se ha hecho.

| magnitud | valor |
|---|---|
| commit | e1acc44 |
| seed | 42 |
| hash de V03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| corrected_cv | 0.7772114525165721 |
| particion | 5 pliegues canonicos, 1415 grupos, 0 partidos |
| anotacion | features/p62_anotacion_spacy.json.gz, ids verificados contra el OOF |
| tagger | spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003 |
| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |
| umbrales | 0,44 / 0,51 / 0,46 |

## Guardias permanentes

Cada script de experimento verifica el F1 de control de V03 en el pliegue 1 antes
de entrenar nada, y para si no da 0,7786259541984732. Se anadio despues de que un
error de acumulacion produjera 0,2518 en todos los brazos.

## Artefactos

| fichero |
|---|---|
| `reports/p62_cobertura_vocabulario.csv` |
| `reports/p62_cobertura_vocabulario.json` |
| `reports/p62_cobertura_vocabulario.md` |
| `reports/p62_compliance.md` |
| `reports/p62_final_summary.json` |
| `reports/p62_hypothesis_gap_analysis.csv` |
| `reports/p62_hypothesis_gap_analysis.md` |
| `reports/p62_linguistic_composition.csv` |
| `reports/p62_linguistic_composition.md` |
| `reports/p62_linguistic_eda.json` |
| `reports/p62_linguistic_gap.csv` |
| `reports/p62_master_results.csv` |
| `reports/p62_notebook_integration.json` |
| `reports/p62_p61_audit.json` |
| `reports/p62_pista_l_word_pos.json` |
| `reports/p62_pista_l_word_pos.md` |
| `reports/p62_pista_m_dependencias.json` |
| `reports/p62_pista_m_dependencias.md` |
| `reports/p62_pos_o_no_pos.csv` |
| `reports/p62_pos_o_no_pos.json` |
| `reports/p62_pos_o_no_pos.md` |
| `reports/p62_presupuesto_canal.csv` |
| `reports/p62_presupuesto_canal.json` |
| `reports/p62_presupuesto_canal.md` |
| `reports/p62_probability_analysis.csv` |
| `reports/p62_reproducibility.md` |
| `reports/p62_techo_cortos.csv` |
| `reports/p62_techo_cortos.json` |
| `reports/p62_techo_cortos.md` |
| `features/p62_anotacion_spacy.json.gz` |
| `features/p62_cobertura_cells.npz` |
| `features/p62_l_word_pos_cells.npz` |
| `features/p62_m_dep_cells.npz` |
| `features/p62_pos_o_no_pos_cells.npz` |
| `features/p62_presupuesto_curve.npz` |

> V03 remains the best verified leaderboard result.
