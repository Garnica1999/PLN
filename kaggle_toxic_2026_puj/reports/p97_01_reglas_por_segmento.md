# P97-01 (H97-A): reglas de decision condicionadas por segmento

Pre-registro en `p97_independent_research.md` seccion 5. Sin entrenamiento: evaluacion nested sobre el OOF canonico (1483 filas, pliegues 1..5).

| Regla | corrected_cv | delta vs frozen | pliegues mejorados vs frozen |
|---|---|---|---|
| frozen | 0.7772114525 | +0.000000 | 0/5 |
| global_nested | 0.7781367700 | +0.000925 | 4/5 |
| segmento_nested | 0.7702913557 | -0.006920 | 1/5 |
| mencion_nested | 0.7699832924 | -0.007228 | 1/5 |
| hashtag_nested | 0.7752042825 | -0.002007 | 1/5 |

**Comparacion principal (segmento - global_nested)**: delta -0.007845, pliegues 1/5, IC95 por clusters [-0.019467, 0.003431], porcentaje de remuestreos positivos 0.0885.

**Segmento vs frozen**: delta -0.006920, pliegues 1/5, IC95 [-0.018586, 0.004664].

**Veredicto pre-registrado: FALSIFICADA** (SENAL exige delta > 0, >= 3/5 pliegues y IC95 sin cero contra la regla global anidada).

Nota: la comparacion contra `frozen` mezcla condicionamiento y re-optimizacion; la comparacion principal aisla el efecto de condicionar.
