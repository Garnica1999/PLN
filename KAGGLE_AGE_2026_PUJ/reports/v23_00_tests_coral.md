# V23 · 00 — Tests unitarios de CORAL

Ninguno de estos tests toca AGE. Validan la implementacion antes de usarla.

## Valores pre-registrados

- `lambda_w = 1.0` (ridge de libro, normalizado por el numero de columnas).
- `lambda_umbral = 0.0` (maximo verosimilitud de libro).

El segundo lo fija TEST B: a la escala de AGE (135.000 filas, 279 columnas y proporciones reales de clase) **no hay separacion**, luego no hace falta penalizar los separadores. En un sintetico pequeno y de senal debil si la hay, y por eso el modulo la tiene disponible.

## TEST A — sintetico perfectamente separable

- **filas**: 600
- **columnas**: 279
- **clases_presentes**: 6
- **macro_f1_entrenamiento**: 0.978052
- **violaciones_monotonia**: 0
- **convergio**: True
- **iteraciones**: 91
- **umbrales**: [-12.8132, -5.2446, 0.6578, 5.9422, 11.5064]
- **norma_pesos**: 8.82027
- **criterio**: macro-F1 >= 0.80, seis clases presentes y violaciones = 0
- **resultado**: PASA

## TEST B — senal compartida a la escala de AGE

- **filas**: 135000
- **columnas**: 279
- **proporciones_clase**: [0.05, 0.2, 0.25, 0.2, 0.15, 0.15]
- **convergio**: True
- **iteraciones**: 24
- **gradiente_maximo**: 0.0
- **macro_f1_validacion**: 0.419431
- **prior_validacion**: 0.065352
- **norma_eta_maxima**: 9.6102
- **cdf_min**: 5.953e-07
- **cdf_max**: 0.999995
- **fraccion_saturada**: 0.0
- **violaciones_monotonia**: 0
- **lambda_w_pre_registrado**: 1.0
- **lambda_umbral_pre_registrado**: 0.0
- **criterio**: converge, supera el prior y no satura (fraccion_saturada = 0)
- **resultado**: PASA

## TEST C — orden de los umbrales durante y despues

- **pasos**: 60
- **incumplimientos**: []
- **umbrales_al_final_del_descenso**: [-2.940976, -1.096785, 0.002615, 0.851052, 1.739878]
- **umbrales_tras_fit**: [-7.665251, -3.232574, -0.066331, 2.456967, 4.967228]
- **violaciones_tras_fit**: 0
- **traza_inicial**: [-2.940936, -1.096724, 0.002667, 0.851092, 1.739909]
- **traza_media**: [-2.940956, -1.096757, 0.00263, 0.851055, 1.739873]
- **criterio**: cero incumplimientos en los 60 pasos y tras el ajuste
- **resultado**: PASA

## TEST D — gradiente analitico contra diferencias finitas

- **error_relativo_maximo**: 3.651e-10
- **forma_gradiente_correcta**: True
- **perdida_y_gradiente_finitos**: True
- **forma_cdf_correcta**: True
- **diferencia_perdida_vs_referencia**: 4.441e-16
- **norma_gradiente_analitico**: 0.715133
- **norma_gradiente_numerico**: 0.715133
- **criterio**: error relativo < 1e-6 y diferencia de perdida < 1e-9
- **resultado**: PASA

## Decision

**Los cuatro tests pasan; CORAL puede ejecutarse sobre AGE**.
