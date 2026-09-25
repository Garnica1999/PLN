# Auditorías finales — KAGGLE_AGE_2026_PUJ

Generado por `scripts/fase23_auditorias.py` · resultado global: **APROBADO**

## 1. Auditoría de leakage

- `ningun_fit_sobre_test`: True
- `n_llamadas_fit_sobre_test`: 0
- `n_canales_representacion`: 6
- `vectorizadores_ok`: True
- `meta_entrenado_con_oof`: True
- `id_no_usado_como_feature`: True
- `test_coverage_near_dup_pct`: 4.2554
- `cv_agrupado_por_cluster`: True
- `memorizacion_con_cv_agrupado`: 0.0
- `sin_target_encoding`: True
- `normalizacion_determinista`: True
- `auditoria_ok`: True

## 2. Auditoría de reproducibilidad

- `archivos_con_semilla_o_random_state`: 73
- `semilla_global_definida`: True
- `requirements_existe`: True
- `n_paquetes_congelados`: 312
- `pin_scikit-learn`: True
- `pin_pandas`: True
- `pin_numpy`: True
- `pin_scipy`: True
- `rutas_absolutas_en_codigo`: []
- `coincidencia_pipeline_vs_v06`: 1.0
- `auditoria_ok`: True

## 3. Auditoría de submission

- `columns_ok`: True
- `columns_match_sample`: True
- `rows_ok`: True
- `ids_equal_test`: True
- `ids_unique`: True
- `ids_same_order`: True
- `no_nan`: True
- `labels_valid`: True
- `duplicados_id`: 0
- `etiquetas_usadas`: ['13-17', '18-24', '25-34', '35-49', '50-64', '65-xx']
- `nulos`: 0
- `auditoria_ok`: True

## 4. Auditoría académica

- `n_experimentos_registrados`: 67
- `todos_con_hipotesis`: True
- `todos_con_resultado`: True
- `n_decisiones`: 37
- `n_referencias`: 30
- `referencias_con_doi_o_url`: 30
- `anios_cubiertos`: [2006, 2021]
- `experimentos_negativos_registrados`: 8
- `experimentos_negativos`: ['B0', 'B5', 'E-005a', 'E-005b', 'E-005d', 'E-015', 'E-019a', 'E-019b']
- `catalogo_features`: True
- `hipotesis_gate_a`: True
- `notebooks_entregables`: ['01_data_audit.ipynb', '02_eda.ipynb', '03_linguistic_analysis.ipynb', '04_feature_engineering.ipynb', '05_baselines.ipynb', '06_classical_models.ipynb', '07_modelos_densos.ipynb', '08_error_analysis.ipynb', '09_final_pipeline.ipynb', '99_entrega_consolidado.ipynb']
- `auditoria_ok`: True
