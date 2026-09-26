# Auditoría de target encoding y aprendizaje fold-local (FASE 87) — KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-25. Alcance: verificar que toda estadística aprendida en la fase v2
(target encoding, perfiles léxicos, umbrales, meta-modelos) se ajusta dentro del
pliegue de entrenamiento o en OOF anidado, y que el holdout congelado no participa
en el desarrollo (`docs/advanced_research_v2_spec.md`, sección 6 y FASE 87).
Método: revisión de código con cita de función y línea.

## Mecanismos auditados

| Mecanismo | Ubicación | Correcto | Evidencia | Riesgo residual |
|---|---|---|---|---|
| Target encoding de entidades (E2-13) | `experiments/21_backlog_restante_v2.py` | Sí | `tablas_target_encoding` (l.1195-1227) y `codificar_entidades` (l.1230-1257) construyen las tablas solo con `idx_ajuste`; `features_entidades` (l.1260-1300) aplica suavizado y bucket desconocido = prior del pliegue; `ejecutar_e2_13` (l.1436-1482) usa CV interna de 4 pliegues. | Ninguno adicional: cobertura informativa 0.5 % → línea cerrada por dispersión. |
| Perfiles léxicos condicionales (P2) | `experiments/16_targeting_v2.py` | Sí | `log_odds_informativo` (l.808-846) y `inducir_lexicos` (l.863-953) usan solo documentos del pliegue de entrenamiento. | Ninguno: el canal targeting no superó al global (0/5). |
| Umbrales anidados | `experiments/21_backlog_restante_v2.py` | Sí | `umbrales_anidados` (l.896-916) y `prediccion_con_umbrales_anidados` (l.919-937): el umbral se ajusta fuera del pliegue evaluado; E2-16 usa shrinkage `n/(n+m)` (l.1629-1673). | Sesgo optimista acotado y documentado; la variante global anidada fue la referencia. |
| Meta-modelo / stacking (P6) | `experiments/17_stacking_v2.py` | Sí | `evaluar_meta_por_pliegue` (l.587-627) entrena el meta solo con OOF; `probas_bases_anidadas` (l.748-812) genera las probabilidades base sin ver el pliegue evaluado. | Multiplicidad/winner's curse documentado (D-014, D-020); sin contaminación in-sample. |
| Holdout congelado | `config/holdout_v2.json`; `reports/holdout_evaluaciones_v2.csv` | Sí | 248 tweets (pliegue 0, sha256 `e5b62e5f…`); 5 de 5 evaluaciones usadas en total (P7-01 V03 0.8285; P7-02 meta 0.8465; FIN-02 0.8364; FIN-04 0.8336; P8-C-meta36 0.8163, rechazado); todas con umbrales congelados y registradas en reports/holdout_evaluaciones_v2.csv. | Dos evaluaciones restantes sin usar; sin impacto. |

## Hallazgos

- No se detectó ninguna estadística aprendida con datos del pliegue evaluado, del
  holdout o del test.
- El único sesgo conocido (reajuste de umbral de fase 1) está marcado con
  `advertencia_sesgo_optimista` en `reports/fase13a_refit_umbral_metricas.json` y
  no participa en las decisiones v2 (D-016 exige umbrales anidados).
- La discrepancia honesta detectada en AUD-01 (el meta falla las 3 etiquetas en
  103 tweets frente a 98 de V03) es consistente con su mayor varianza en el núcleo
  duro y no con leakage.

## Veredicto

**Sin leakage de aprendizaje detectable.** Todos los mecanismos auditados ajustan
sus estadísticas fold-localmente o en OOF anidado; el holdout permanece congelado
con 2/5 evaluaciones usadas.
