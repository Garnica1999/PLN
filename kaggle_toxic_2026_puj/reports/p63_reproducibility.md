# P63 - Reproducibilidad

Fecha 2026-09-29T06:59:40+00:00.

## Nivel declarado: R2

Semilla fija, artefactos en disco, hash de la submission de V03 verificado al
inicio de cada script, ejecucion real con exit code capturado sin pipe, y

**No se declara R4.** R4 exige reproduccion independiente en otra sesion y eso no
se ha hecho. Declararlo seria la misma clase de error que el techo inflado.

| magnitud | valor |
|---|---|
| commit | 13eda02 |
| seed | 42 |
| hash de V03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| corrected_cv | 0.7772114525165721 |
| particion | 5 pliegues canonicos, 1415 grupos, 0 partidos |
| anotacion | features/p62_anotacion_spacy.json.gz, ids verificados contra el OOF |
| tagger | spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003 |
| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |
| umbrales | 0,44 / 0,51 / 0,46 |

## Guardias permanentes de la fase

Cada script verifica el sha256 de la submission de V03 antes de leer nada y para
si no coincide. Los que entrenan verifican ademas que el OOF canonico da
0.7772114525165721 exactamente.

## Los scripts, en orden de ejecucion

| orden | script | que hace |
|---:|---|---|
| 1 | `experiments/p63_01_gap_analysis.py` | inventario funcional y gap analysis |
| 2 | `experiments/p63_02_id_forensics.py` | forense del identificador |
| 3 | `experiments/p63_03_id_auditoria_predictiva.py` | id-only con capacidad y permutacion |
| 4 | `experiments/p63_04_estructura_relacional.py` | estructura de grupo con dos alcances |
| 5 | `experiments/p63_05_composicion_linguistica.py` | composicion en ratios |
| 6 | `experiments/p63_06_estructura_tweet_largo.py` | posicion y estructura interna |
| 7 | `experiments/p63_07_integracion_y_techo.py` | integracion y techo residual |
| 8 | `experiments/p63_08_residuo_de_v03.py` | los bloques contra el residuo |
| 9 | `experiments/p63_09_dataset_auxiliar.py` | dataset de registros, esquema y figuras |
| 10 | `experiments/p63_10_cerrar.py` | registro, cumplimiento y este informe |

## Artefactos

| fichero |
|---|
| `reports/p63_compliance.md` |
| `reports/p63_composicion_linguistica.csv` |
| `reports/p63_composicion_linguistica.json` |
| `reports/p63_composicion_linguistica.md` |
| `reports/p63_error_intersection.csv` |
| `reports/p63_estructura_relacional.csv` |
| `reports/p63_estructura_relacional.json` |
| `reports/p63_estructura_relacional.md` |
| `reports/p63_estructura_tweet_largo.csv` |
| `reports/p63_estructura_tweet_largo.json` |
| `reports/p63_estructura_tweet_largo.md` |
| `reports/p63_final_research.md` |
| `reports/p63_final_summary.json` |
| `reports/p63_hypothesis_gap_analysis.csv` |
| `reports/p63_hypothesis_gap_analysis.json` |
| `reports/p63_hypothesis_gap_analysis.md` |
| `reports/p63_hypothesis_registry.csv` |
| `reports/p63_id_auditoria_predictiva.csv` |
| `reports/p63_id_auditoria_predictiva.json` |
| `reports/p63_id_auditoria_predictiva.md` |
| `reports/p63_id_forensics.csv` |
| `reports/p63_id_forensics.json` |
| `reports/p63_id_forensics.md` |
| `reports/p63_id_forensics_blocks.csv` |
| `reports/p63_id_forensics_ocupacion.csv` |
| `reports/p63_integracion_y_techo.csv` |
| `reports/p63_integracion_y_techo.json` |
| `reports/p63_integracion_y_techo.md` |
| `reports/p63_master_results.csv` |
| `reports/p63_record_features.csv` |
| `reports/p63_record_features.md` |
| `reports/p63_record_structure.csv` |
| `reports/p63_reproducibility.md` |
| `reports/p63_residuo_de_v03.csv` |
| `reports/p63_residuo_de_v03.json` |
| `reports/p63_residuo_de_v03.md` |
| `reports/p63_techo_por_rebanada.csv` |
| `figuras/p63/p63_01_v03_vs_candidatos.png` |
| `figuras/p63/p63_02_delta_vs_v03.png` |
| `figuras/p63/p63_03_consistencia_pliegues.png` |
| `figuras/p63/p63_04_delta_candidatos.png` |
| `figuras/p63/p63_05_bloques_en_solitario.png` |
| `figuras/p63/p63_06_v03_mas_bloques.png` |
| `figuras/p63/p63_07_distribucion_id.png` |
| `figuras/p63/p63_08_id_rango_vs_fila.png` |
| `figuras/p63/p63_09_saltos_id.png` |
| `figuras/p63/p63_10_multiplicidad_id.png` |
| `figuras/p63/p63_11_distribucion_tamano_grupo.png` |
| `figuras/p63/p63_12_grado_duplicado.png` |
| `figuras/p63/p63_13_error_por_rebanada.png` |
| `figuras/p63/p63_14_error_por_rebanada_y_etiqueta.png` |
| `figuras/p63/p63_15_techo_por_rebanada.png` |
| `figuras/p63/p63_16_rescate_vs_dano.png` |
| `figuras/p63/p63_17_interseccion_errores.png` |
| `figuras/p63/p63_18_confianza_por_rebanada.png` |
| `figuras/p63/p63_19_margen_por_rebanada.png` |
| `figuras/p63/p63_20_id_bloques.png` |
| `figuras/p63/p63_21_evolucion_candidatos.png` |
| `features/toxic_record_profile.parquet` |
| `features/toxic_record_profile_schema.md` |

> V03 remains the best verified leaderboard result.
