# Informe de investigación avanzada — segunda generación

**Proyecto**: `KAGGLE_AGE_2026_PUJ` · **Fecha**: 2026-09-24 · **Candidato**: `AGE_ADVANCED_V01` (submission v09)

## 1. Baseline congelado

Se congeló el estado real del proyecto en `experiments/baseline_frozen/` (11 archivos con hash SHA-256, `MANIFEST.json` y `README.md`). El brief asumía v06 (CV 0,5851 / LB 0,60273); el baseline real es **v08** (CV 0,5866 / **LB 0,60376**), estrictamente mejor. La desviación está documentada en el manifiesto.

## 2. Gap analysis

`reports/advanced_feature_gap_analysis.md` responde las 10 preguntas del brief. Conclusión central: el sistema anterior explota la **forma superficial** (longitud, ratios, char n-grams) y la **bolsa de palabras** de sus canales léxicos, pero **no** la estadística supervisada de entidades, las distribuciones internas, la rareza/novedad léxica ni la posición. El hallazgo fuerte: los hashtags **no** aportan y las menciones **sí**.

`reports/signal_coverage_matrix.csv`: 18 fenómenos × 8 niveles de representación. Las brechas marcadas como *ausentes* eran posición, OOV/novedad, rareza, léxico por clase y perfil supervisado de entidades.

`reports/research_backlog.md`: 18 líneas priorizadas con justificación técnica, incluidas las **cerradas por resultado negativo previo** (especialistas por longitud, BM25, kNN, meta LightGBM, jerarquía gruesa) y las que **requieren verificación de reglas** (retrieval train->test, POS tagger preentrenado).

## 3. Nuevas features construidas

**160 features** en `features/age_structured_features.parquet` (87 locales, idénticas para train y test) y 73 supervisadas calculadas in-fold. Resumen por familia:

| Familia | Nº de features |
|---|---|
| A — distribución de longitud de palabra | 21 |
| I/M/U — posición de entidades y de palabras extremas | 20 |
| N/R/S — estructura, entropía y compresión | 17 |
| J — menciones supervisadas P(clase|mención) | 17 |
| L — hashtags supervisados P(clase|hashtag) | 17 |
| K — cuenta fuente de RT supervisada | 17 |
| E/Y — rareza y novedad léxica | 12 |
| B — distribución de caracteres | 10 |
| H/T — forma de palabra y repetición | 10 |
| F/X — distancia al léxico por clase (log-odds) | 10 |
| D — riqueza léxica avanzada | 9 |

## 4. Resultados

Evaluación con leave-one-fold-out sobre las mismas 5 particiones agrupadas:

| Configuración | Features del meta | macro-F1 | Δ vs baseline |
|---|---|---|---|
| Baseline congelado v08 | 107 | 0.5866 ± 0.0038 | — |
| + features estructuradas | 194 | 0.5871 ± 0.0046 | +0.0006 |
| + mencion | — | 0.5887 ± 0.0045 | +0.0016* |
| + hashtag | — | 0.5871 ± 0.0044 | -0.0001* |
| + fuente_rt | — | 0.5878 ± 0.0044 | +0.0007* |
| + rareza | — | 0.5875 ± 0.0043 | +0.0003* |
| + lexico | — | 0.5872 ± 0.0045 | +0.0001* |
| **+ todas (candidato)** | 267 | **0.5897** ± 0.0043 | **+0.0031** |

*Δ respecto a la configuración con features estructuradas.

**Estabilidad**: la mejora es positiva en **5/5 folds** ([0.003, 0.0041, 0.0027, 0.0038, 0.0021]), con mínimo +0.0021 y máximo +0.0041. No depende de un fold afortunado.

## 5. Holdout congelado (confirmación fuera del CV de desarrollo)

Se reservó el 15% de los clústeres de near-duplicados (25,204 filas en 24,602 clústeres, semilla 2026) y el meta-modelo se entrenó **solo con el 85% restante**:

| Configuración | macro-F1 en el holdout | accuracy |
|---|---|---|
| baseline | 0.5914 | 0.5924 |
| candidato | 0.5966 | 0.5975 |

**Δ = +0.0052 -> confirmado.** El holdout reproduce la mejora del CV de desarrollo e incluso la supera, lo que descarta que el resultado sea producto del ajuste al CV.

**Salvedad metodológica honesta**: las matrices OOF de los modelos base provienen del CV de 5 folds sobre todo train, por lo que las predicciones del holdout las generaron modelos base que vieron parte de esos datos. Eso afecta por igual a baseline y candidato (la comparación es válida), pero el nivel absoluto del holdout es ligeramente optimista. Se reporta junto al CV, nunca en su lugar.

## 6. Análisis de errores: qué sigue fallando

Los errores del ensamble siguen concentrados en las clases medias (`25-34` F1 0.449, `18-24` 0.476) y en los tweets cortos (52,5% de error en el quintil más corto). Las features nuevas atacan parte de eso: la rareza léxica es más informativa en tweets cortos, y el perfil supervisado de menciones aporta donde el TF-IDF no distingue la identidad de la cuenta.

Un benchmark ya disponible y no explotado: la precisión en las fronteras adyacentes (`reports/tables/e028_combinaciones.csv`) muestra que `18-24|25-34` es la frontera más difícil (0,6856), y ahí es donde queda el mayor margen.

## 7. Ablación

| Bloque añadido | Δ macro-F1 (LOFO) | Decisión |
|---|---|---|
| Features estructuradas (87) | +0,0006 | **se conserva** |
| Menciones supervisadas (17) | **+0,0016** | **se conserva** (mayor aporte) |
| Cuenta fuente de RT (17) | +0,0007 | **se conserva** |
| Rareza léxica (12) | +0,0003 | **se conserva** |
| Léxico por clase (10) | +0,0001 | se conserva (coste nulo) |
| Hashtags supervisados (17) | −0,0001 | se conserva solo en el conjunto completo |

## 8. Modelos y especialistas

Esta fase **no** añadió modelos base nuevos: el brief pedía buscar señal ausente, y la evidencia mostró que la ganancia está en el **espacio de features del meta-modelo**, no en más miembros. Los especialistas (longitud, pares adyacentes, ordinal acumulativo, jerarquía) ya existían y siguen en el ensamble; esta fase los alimentó con features nuevas.

## 9. Ordinalidad (segunda generación)

Ya cubierta antes de esta fase: acumulativo (E-021a), pares adyacentes (E-028a, precisión 0,7428 en fronteras) y jerarquía gruesa (E-028b, redundante). Esta fase añade la **distancia al léxico por clase** (familia X), que es una forma suave de ordinalidad léxica y aportó +0,0001.

## 10. Retrieval (auditoría de reglas)

**REQUIERE_VERIFICACIÓN — no implementado.** El brief lo autoriza solo si las reglas lo permiten. La competencia prohíbe usar el test para entrenar, para construir vocabularios o para seleccionar cualquier cosa; un lookup train->test en inferencia es técnicamente inferencia y no entrenamiento, pero la decisión D-001 (interpretación amplia) no cubre este caso y el riesgo académico es real. Se deja **explícitamente sin implementar** y documentado como línea abierta. Nota cuantitativa: el techo de esa vía está acotado por la cobertura de near-duplicados del test (4,26%) y su pureza (92,8%).

## 11. Techo del sistema

El candidato alcanza **0,5897 de CV** y **0,5966 de holdout**. Con la brecha de calibración medida (+0,0165 ± 0,0014 en 7 envíos), el LB esperado ronda **0,605-0,607**.

## 12. Líneas descartadas (registradas, no eliminadas)

- Especialistas por longitud (E-023/E-024): +0,0000.
- BM25 sin normalización L2 (E-025): −0,0002; la normalización por documento es esencial.
- kNN sobre SVD (E-026): 0,3088 individual, sin aporte.
- Meta-modelo LightGBM (E-022): 0,5858 vs 0,5860 de la regresión logística.
- Jerarquía gruesa->fina (E-028b): +0,0003 redundante con los pares adyacentes.
- Hashtags supervisados: neutros (−0,0001) individualmente.
- Deep learning denso como clasificador (E-019): 0,3381 / 0,4071.

## 13. Líneas futuras

- **Retrieval train->test** si se obtiene confirmación explícita de las reglas (techo acotado: 4,26% de cobertura × 92,8% de pureza).
- **Frontera `18-24|25-34`**: es el punto de mayor error; un modelo dedicado con features supervisadas de entidades (no solo TF-IDF) es la apuesta natural.
- **POS tagger** (spaCy, no transformer): zona gris de reglas (D-001); degradado por el corpus ASCII.
- Reentrenar periódicamente el holdout congelado con nuevas particiones para evitar sobreajuste al conjunto de confirmación.

## 14. Entregables de la fase

- `experiments/baseline_frozen/` (baseline congelado con hashes)
- `reports/advanced_feature_gap_analysis.md`
- `reports/signal_coverage_matrix.csv`
- `reports/research_backlog.md`
- `reports/advanced_feature_catalog.csv`
- `features/age_structured_features.parquet` + `_test.parquet`
- `features/age_supervised_features.npy` (OOF in-fold)
- `submissions/age_submission_v09.csv` (AGE_ADVANCED_V01)
- `reports/submissions_registry.csv`
- `experiments/advanced_02_resultados.json`, `advanced_03_holdout.json`

---

*Informe generado desde los artefactos de la fase. Experimentos totales del proyecto: 30.*