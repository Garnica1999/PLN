# Cumplimiento de la especificación v2 «Advanced Research» — KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-25. La especificación maestra está archivada en
`docs/advanced_research_v2_spec.md` (113 secciones; FASES 0–99). Esta auditoría
verifica cada FASE contra evidencia en disco: scripts, reportes, `experiments.csv`
(118 filas), `DECISIONS.md` (D-001…D-022) y los artefactos generados el 2026-09-25.

Convención de estados:

- **COMPLETA**: ejecutada con evidencia verificable.
- **COMPLETA (variante)**: cubierta por una implementación con otro nombre o
  empaquetado distinto al nombrado en la spec.
- **PARCIAL**: cubierta solo en parte; lo restante se lista en brechas.
- **NO EJECUTADA**: sin evidencia; se indica la razón.
- **N/A**: no aplicable al corpus o impedida por las reglas de la competencia.

## Matriz FASE por FASE

| FASE | Tema | Estado | Evidencia | Resultado |
|---|---|---|---|---|
| 0 | Recuperar conocimiento existente | COMPLETA | `reports/advanced_gap_analysis_v2.md`; `reports/contexto.md` | Artefactos leídos y verificados en disco antes de experimentar (paquete P0). |
| 1 | Gap analysis del feature space | COMPLETA | `reports/advanced_gap_analysis_v2.md` | Matriz fenómeno → representación → brecha → hipótesis construida. |
| 2 | Dependencias de etiquetas (chains, soft) | COMPLETA (variante) | `experiments/15_cadenas_etiquetas_v2.py`; `reports/cadenas_etiquetas_v2.json`; D-017 | Soft chains con P(otra etiqueta) OOF; meta 0.7853 local; sin transferencia al LB (D-020). |
| 3 | Modelo del espacio de etiquetas | COMPLETA (variante) | `reports/error_by_label_combination_v2.csv`; D-017/D-019 | Dependencia modelada vía meta/cadenas sobre las combinaciones válidas; powerset directo no (clases raras tratadas por jerarquía). |
| 4 | EDA de segunda generación orientada a error | COMPLETA | `reports/error_by_label_combination_v2.csv` (AUD-02); `reports/error_analysis.md` | Señales estructurales medias por combinación verdadera (000…111). |
| 5 | Interacciones lingüísticas (insulto × X) | COMPLETA (variante) | `reports/targeting_v2.json`; `reports/emocion_nbsvm_ortografia_v2.json` | Pares insulto × mención/colectivo/risas/afecto/negación/cita evaluados dentro de canales P2/P4/P5; cerrados sin mejora. |
| 6 | Broma / afiliación | COMPLETA (variante) | P5 emoción (`reports/emocion_nbsvm_ortografia_v2.json`) | Canal de emoción/afiliación 0.6330; no superó al global. |
| 7 | Distancia target–insulto | COMPLETA | `experiments/16_targeting_v2.py` (P2) | Distancias token/carácter, orden y densidad construidas y evaluadas; sin mejora. |
| 8 | Target profiling | COMPLETA | `experiments/16_targeting_v2.py` (P2) | Tipos de target (persona/colectivo/segunda persona) y scores; sin mejora. |
| 9 | Citation / attribution (H13) | COMPLETA | `experiments/19_sintaxis_cita_negacion_v2.py` (P4); `reports/sintaxis_cita_negacion_v2.json` | Canal cita/atribución 0.7657 (−0.0115, 0/5); H13 cerrada. |
| 10 | Negación | COMPLETA | `experiments/19_sintaxis_cita_negacion_v2.py` (P4) | Negación por dependencias (advmod+Polarity=Neg) y bajo insulto/target; sin ganancia. |
| 11 | POS y morfología | COMPLETA (variante) | P3 (canal `SYN`); `features/v2_oof_syntax.npz` | POS simbólico → representación sparse; morfología vía spaCy (ver FASE 13). |
| 12 | Dependency features | COMPLETA | P3; `features/v2_syntax_annotations.json.gz` | Patrones `PRON_VERB`, `NOUN_ADJ`, `NEG_VERB`, `TARGET_ADJ` como tokens simbólicos → TF-IDF; sin ganancia. |
| 13 | Patrones morfológicos | PARCIAL | P3 | Sufijos/prefijos/terminaciones no materializados como canal dedicado; cobertura parcial vía morfología de spaCy. Brecha. |
| 14 | Evasión ortográfica | COMPLETA | P5 (`reports/emocion_nbsvm_ortografia_v2.json`); canales ORT | Multicanal raw/colapsado/sin puntuación interna/leet 0.7534; sin ganancia. |
| 15 | Profanity / insult detection | COMPLETA (variante) | P2 léxico inducido fold-local (37/36/33/36/36 lemas) | Sin léxico externo (D-007); canales fold-local cerrados. |
| 16 | Lexicón de emociones (H5/E04) | COMPLETA | P5 emoción 0.6330 | Primera ejecución real de la hipótesis; sin mejora transferible. |
| 17 | Sentimiento local al target | COMPLETA (variante) | P5 (ventanas locales dentro del canal de emoción) | Polaridad/densidad local al target evaluadas; sin ganancia propia. |
| 18 | Textometría avanzada | COMPLETA (variante) | Fase 1 (features densas, negativas); EDA | Estadísticos descriptivos completos; el canal denso no se replicó en v2 (ver FASES 19–20). |
| 19 | Entropía y concentración | PARCIAL | EDA (descriptivo) | Descriptivo sí; el canal dedicado (E2-12) no se ejecutó. Brecha. |
| 20 | Rareza léxica | PARCIAL | EDA (descriptivo) | Descriptivo sí; sin canal fold-local de rareza (E2-12). Brecha. |
| 21 | Class-conditional lexical profiles | COMPLETA (variante) | E2-05/E2-21 (log-odds fold-local) | Cobertura vía léxicos inducidos in-fold; sin fila propia (E2-11 subsumida). |
| 22 | Target encoding de entidades | COMPLETA | P2 + E2-13 (`experiments/21_backlog_restante_v2.py`) | Cobertura informativa 0.5 % → cerrada por dispersión. |
| 23 | Mention profiling | COMPLETA | P2 | Menciones × contexto evaluadas; sin mejora. |
| 24 | Hashtag profiling | COMPLETA | P2 | Perfiles y densidades evaluados; sin mejora. |
| 25 | URL y noticias | COMPLETA (variante) | Análisis de rebanadas (fase 1) y P2/P4 | Régimen noticia caracterizado; modelo dedicado no entrenado (condición no cumplida). |
| 26 | Retweets | N/A | Auditoría del corpus | 0 retuits con prefijo canónico `RT @`; 8 tweets contienen el token `RT`. Fenómeno casi inexistente. |
| 27 | Contenido citado vs contenido del autor | COMPLETA (variante) | P4 (in-quote/out-quote) | Separación cita/autor evaluada; sin ganancia. |
| 28 | word + char + structure (A–F) | COMPLETA (variante) | Secuencia P3 → P5 → P6 | Concatenación densa degrada (fase 1); integración resuelta por late fusion/stacking (P6). |
| 29 | POS como representación sparse | COMPLETA | P3 (`SYN-3`) | Sin mejora incremental ni complementariedad. |
| 30 | Dependency paths sparse | COMPLETA | P3 | Sin mejora; línea cerrada. |
| 31 | N-gramas alternativos | COMPLETA | Fase 1 `reports/variantes_representaciones.csv` | Barrido controlado; la unión word(1,2)+char_wb(3,5) sigue siendo óptima. |
| 32 | Representaciones de peso alternativas | PARCIAL | Fase 1 + P5 | TF-IDF/sublinear/union probados y NB-SVM sí; Count/binario/delta/BM25 no ejecutados. Brecha menor. |
| 33 | NB-SVM | COMPLETA | P5 NB-SVM 0.7612 | IC del delta incluye 0; cerrada. |
| 34 | Normalización multicanal | COMPLETA | P5 (raw + normalizado) | Sin recuperar señal ofuscada con ganancia; cerrada. |
| 35 | Feature engineering de patrones | COMPLETA (variante) | P2/P4 (binario, count, densidad, distancia) | Patrones estructurados evaluados dentro de canales; cerrados sin mejora. |
| 36 | Modelo por etiqueta | COMPLETA | E2-21 (P8) | Espacios por etiqueta −0.0253 (0/5); se mantiene espacio compartido. |
| 37 | Features específicas por etiqueta | COMPLETA | E2-21 (P8) | Ídem: peor que el espacio compartido. |
| 38 | Threshold tuning de segunda generación | COMPLETA | E2-14/E2-16 (P8) | Umbrales anidados y calibración sin mejora de F1. |
| 39 | Thresholds por rebanada (H15) | COMPLETA | E2-16 (P8) | Shrinkage monótono; H15 cerrada con evidencia. |
| 40 | Calibración | COMPLETA | E2-14 (P8) | Brier 0.1583→0.1199; F1 invariante con umbrales reajustados. |
| 41 | Soft label dependency | COMPLETA | P1/P6; `reports/cadenas_etiquetas_v2.json` | Scores OOF como features; dependencia capturada localmente. |
| 42 | Meta-modelo | COMPLETA (variante) | `experiments/17_stacking_v2.py` | Ridge 0.7874; variantes LR/Ridge evaluadas; MLP superficial no registrado. |
| 43 | Error de las tres etiquetas | COMPLETA | `reports/error_intersection_v2.csv`, `reports/all_three_errors.csv` (AUD-01) | 98 (V03) / 103 (meta) sobre 1.483, IC bootstrap; histórico 101/1.731 contrastado. |
| 44 | Errores de una sola etiqueta | COMPLETA | `reports/error_intersection_v2.csv` (AUD-01) | Patrones solo-HS/TR/AG cuantificados con IC. |
| 45 | False positives | COMPLETA (variante) | Fase 1 `reports/error_analysis.md`; P2/P5 | Taxonomía FP documentada; sin artefacto v2 dedicado. |
| 46 | False negatives | COMPLETA (variante) | Fase 1 (colectivo: 45.5 % FN vs 13.9 % TP); E2-06 | Fenómeno cuantificado y atacado en P2; sin mejora. |
| 47 | Error by label combination | COMPLETA | `reports/error_by_label_combination_v2.csv` (AUD-02) | P/R/F1 por etiqueta y señales estructurales por combinación. |
| 48 | Especialización por rebanada | COMPLETA | E2-20 (P8) | 11/21 celdas significativas tras BH, pero −0.045 de media; cerrada. |
| 49 | Late fusion | COMPLETA | P6 + `reports/model_complementarity_v2.csv` (AUD-03) | Canales → meta; complementariedad medida por pares. |
| 50 | Diversidad de modelos | COMPLETA | AUD-03 (496 parejas de 32 canales) | Disagreement, correlación de errores y corrección mutua calculados. |
| 51 | Peor pero complementario | COMPLETA | P6 (adopción por complementariedad, D-019) + AUD-03 | Criterio aplicado y documentado. |
| 52 | Bootstrap de diferencias | COMPLETA | P6/P7/P8 y AUD-01 | IC 95 % a nivel tweet en las comparaciones clave. |
| 53 | Ablation | COMPLETA (variante) | Paquetes P1–P8 + fase 1; `experiments.csv` | Ablación secuencial por bloques documentada por experimento. |
| 54 | Feature redundancy | NO EJECUTADA | — | Sin MI/Pearson/Spearman de features densas; la familia densa ya estaba cerrada (fase 1) y el catálogo v2 existe. Brecha documentada. |
| 55 | Dataset final de features | COMPLETA | `features/v2_structured_features.parquet` (2.885 × 123) + `reports/feature_catalog_v2.csv` (225 features) | Generados y verificados el 2026-09-25 (script 22). |
| 56 | Versiones | COMPLETA (variante) | Paquetes E2-* y canales (EMO/SYN/TGT/CYT/ORT/NBSVM/META) | Nomenclatura adaptada a los paquetes reales; V03 intacta. |
| 57 | Research log | COMPLETA | `RUNLOG.md`; `experiments.csv` (118 filas) | Registro acumulativo por experimento y por paquete. |
| 58 | Decisions | COMPLETA | `DECISIONS.md` D-001…D-022 | Cada decisión con observación/hipótesis/resultado/decisión/evidencia. |
| 59 | Research backlog | COMPLETA | `reports/research_backlog_v2.md` (estados cerrados 2026-09-25); D-021 | 24/24 líneas evaluadas y cerradas. |
| 60 | Control computacional | COMPLETA | Reutilización de cachés (`reports/v2_structured_features_manifest.md`) | Sin matrices redundantes; el paquete de artefactos corrió en 7.1 s. |
| 61 | Tests unitarios | COMPLETA (variante) | Verificación a nivel dataset: 2.885 filas, 0 nulos/inf en el consolidado | Sin suite dedicada por feature (observación). |
| 62 | Data augmentation | N/A | Decisión por defecto (FASE 62 de la spec) + E2-24 | Sin augmentation; regla respetada. |
| 63 | External lexicons | N/A | D-007 | No permitidos/descartados por reglas; la variante autocontenida es la única implementada. |
| 64 | Normalización del léxico | COMPLETA (variante) | Léxicos inducidos normalizados a ASCII | Sin léxico externo que normalizar (D-007). |
| 65 | Representación del insulto | COMPLETA (variante) | P5 (canales ORT/EMO) + E2-05 (`<INSULT>`) | Canal abstracto evaluado dentro de E2-05; sin ganancia. |
| 66 | N-gramas sobre patrones estructurales | COMPLETA | P3 (`SYN-3`) | Sin ganancia; línea cerrada. |
| 67 | Distancia / orden / densidad | COMPLETA | P2 | Métricas no binarias construidas; sin ganancia. |
| 68 | Modelo régimen noticia | COMPLETA (variante) | Análisis de rebanadas; `reports/error_analysis.md` | Especialista no entrenado: la condición P(error|news) no lo justificó. |
| 69 | Agresividad contra grupos | COMPLETA | P2 (colectivo + predicado negativo); hallazgo del 45.5 % | Recall de la rebanada colectiva mejoró en desarrollo pero no transfirió. |
| 70 | HS implícito | PARCIAL | Fase 1 (taxonomía de FN de HS) | Sin features dedicadas por subfenómeno (estereotipo, deshumanización, amenaza). Brecha. |
| 71 | Detección de estereotipos | COMPLETA (variante) | P2 (grupo + predicado/adjetivo/generalización) | Cobertura vía patrones de targeting; sin ganancia. |
| 72 | Patrones de generalización | COMPLETA (variante) | P2 (cuantificadores, plural, genéricos) | Dentro del canal de targeting; cerrado. |
| 73 | Targeted vs general | PARCIAL | P2 | El continuo no se materializó como scores separados (`target_specificity/directness/collectivity`). Brecha. |
| 74 | Agresividad como continuo | PARCIAL | P5 | `aggression_score` dedicado no construido; intensidad cubierta en parte por emoción/estilo. Brecha. |
| 75 | HS score / target score | PARCIAL | P2/P5 | Scores dedicados no construidos; canales cubren parte del concepto. Brecha. |
| 76 | Tres scores latentes | COMPLETA (variante) | P6 (meta sobre scores de canales) + AUD-03 | Independencia analizada vía complementariedad y meta. |
| 77 | Interpretabilidad | NO EJECUTADA | — | Sin artefacto v2 de coeficientes/permutation importance. Brecha documentada. |
| 78 | Modelo final por etiqueta | COMPLETA (variante) | E2-21 | Peor que el modelo compartido; decisión de no adoptar registrada. |
| 79 | Meta-ensemble | COMPLETA | P6 (`experiments/17_stacking_v2.py`) | Meta regularizado sobre OOF; 0.7874 local. |
| 80 | Stacking correcto | COMPLETA | `17_stacking_v2.py` (`probas_bases_anidadas`, `evaluar_meta_por_pliegue`) | Base OOF por pliegue y meta solo sobre OOF; sin predicciones in-sample. |
| 81 | Bootstrap del ensemble | COMPLETA | P6/P7 (IC 95 % del delta por tweet) | Mejora local con IC>0; no transfirió al LB. |
| 82 | Criterio de adopción | COMPLETA | D-016 aplicada en P1–P8 | Regla ≥+0.005 y ≥3/5 pliegues (o complementariedad con IC>0) aplicada en cada adopción. |
| 83 | Research holdout | COMPLETA | P0/P7; `config/holdout_v2.json` (sha256); `reports/holdout_evaluaciones_v2.csv` | 248 tweets congelados; 2/5 evaluaciones usadas. |
| 84 | Leaderboard | COMPLETA | `submissions/leaderboard_log.csv` (9 sondas) | Envío solo con hipótesis + evidencia local + candidato distinto. |
| 85 | Gap local ↔ leaderboard | COMPLETA | `reports/advanced_research_v2.md`; `submissions/leaderboard_log.csv` | Gap dev-LB cuantificado; winner's curse documentado. |
| 86 | Test prevalence | COMPLETA | Auditoría de uso del test | Solo `transform`/`predict`; sin pseudo-labeling ni calibración contra el test. |
| 87 | Auditoría final | COMPLETA | `reports/leakage_audit.md`; `reports/auditoria_final.md`; `reports/rule_audit_v2.md`; `reports/target_encoding_audit_v2.md` | Leakage, reglas y reproducibilidad auditadas con evidencia. |
| 88 | Estructura de artefactos | COMPLETA | Los 9 artefactos exigidos existen (los faltantes se generaron el 2026-09-25) | `advanced_research_v2.md`, `advanced_gap_analysis_v2.md`, `research_backlog_v2.md`, `error_intersection_v2.csv`, `error_by_label_combination_v2.csv`, `model_complementarity_v2.csv`, `feature_catalog_v2.csv`, `target_encoding_audit_v2.md`, `rule_audit_v2.md`. |
| 89 | Feature datasets | COMPLETA (variante) | `features/v2_structured_features.parquet` + manifest | Consolidado único en lugar de un parquet por bloque; exclusiones documentadas. |
| 90 | Experiment log | COMPLETA (variante) | `experiments.csv` (118 filas; schema con campos extra) | Campos adaptados al registro real del proyecto. |
| 91 | Prioridad de investigación | COMPLETA | `tasks/plan.md` (sección de reordenamiento) | Orden reajustado por evidencia: P6 antes que P3–P5 al degradar la concatenación. |
| 92 | No repetir experimentos negativos | COMPLETA | `experiments.csv` + `DECISIONS.md` consultados antes de cada paquete | Sin repeticiones de líneas ya cerradas. |
| 93 | Significancia vs utilidad | COMPLETA (variante) | Cierres con effect size/IC en `backlog_restante_v2.csv` | Cierres basados en F1 incremental y complementariedad, no solo p-valores. |
| 94 | Correlación vs causalidad | COMPLETA | Redacción de `FINAL_REPORT.md` y reportes v2 | Lenguaje asociativo («proporciona señal», «asociado con»). |
| 95 | Respuesta a resultados negativos | COMPLETA | Protocolo de cierre aplicado en P1–P8 | Cada fallo se analizó por hipótesis/representación/integración antes de cerrar. |
| 96 | Conclusión de cada experimento | COMPLETA | `reports/*.json` + filas de `experiments.csv` | Observación→hipótesis→método→resultado→decisión por experimento. |
| 97 | Criterio de cierre | COMPLETA | D-021; `reports/backlog_restante_v2.csv` | Cierres con variantes razonables evaluadas y resultado estable. |
| 98 | No destruir el baseline | COMPLETA | `experiments/baseline_v03_frozen/` (hashes) | V03 intacta y reconstruible byte a byte tras toda la fase v2. |
| 99 | Condición de finalización | COMPLETA | Ver veredicto abajo | Condición B cumplida con observaciones acotadas. |

## Secciones transversales (0–9 y 110–112)

- **0–9** (rol, competencia, baseline congelada, evidencia, protocolo corregido,
  reglas anti-leakage, reglas de modelos, objetivo y «no irreducible»): aplicadas
  como restricciones del proyecto; la baseline se congeló en P0
  (`experiments/baseline_v03_frozen/`) y el protocolo corregido (grupos
  near-duplicado ∪ pseudo-autor) fue el estándar de toda la fase v2 (D-015/D-016).
- **110** (primera acción): ejecutada íntegra en P0–P1 (leer artefactos, congelar,
  reproducir, corrected CV, baseline por etiqueta, intersección de errores,
  error por combinación —los dos últimos completados como AUD-01/AUD-02 el
  2026-09-25—, gap analysis y backlog).
- **111** (regla absoluta): el cierre de cada línea usa el patrón
  «esta representación no añadió información bajo esta integración» y no
  «las features lingüísticas no funcionan» ni «el error es irreducible».
- **112** (objetivo final): ver mapeo abajo.

## Brechas restantes

1. **E2-12** (canal distributional de rareza/entropía): no ejecutado como canal;
   el descriptivo existe en el EDA y la familia densa fue negativa en fase 1.
2. **E2-22** (umbral condicionado a incertidumbre): no ejecutado; E2-14 (calibración)
   y E2-16 (shrinkage por rebanada) cubren el mecanismo general y fueron negativos.
3. **E2-24** (augmentation determinista): no ejecutado por la regla por defecto
   de la FASE 62 de la spec.
4. **FASE 13** (patrones morfológicos como canal), **FASE 54** (redundancia/MI de
   features densas), **FASE 77** (interpretabilidad v2) y los scores dedicados de
   las FASES 70/73/74/75: sin artefacto dedicado; documentados como evidencia
   negativa acotada, sin fabricar resultados.

Ninguna brecha afecta al entregable (V03) ni a la conclusión de la fase.

## Veredicto de la condición de terminación (FASE 99-B)

**CUMPLIDA CON OBSERVACIONES.** Las fuentes de señal razonables fueron evaluadas
y documentadas: 24/24 líneas del backlog con resultado numérico y decisión
(D-021), 9 sondas de leaderboard, 118 experimentos registrados. No se encontró
mejora que transfiriera al test (V03 0.78306 vs. meta 0.78163, dentro del error
estándar ≈0.013). Las observaciones son las cuatro brechas acotadas de arriba.

## Mapeo del objetivo FASE 112 (TOXIC_ADVANCED_V01)

- **Materialización**: el candidato v2 («meta de dependencia de etiquetas») es
  `models/v2_meta_final.joblib` con la submission
  `submissions/toxic_submission_v08_meta_v2.csv` (LB 0.78163); no existe un
  artefacto con el nombre literal `TOXIC_ADVANCED_V01`.
- **Qué demuestra**: la dependencia entre etiquetas es señal adicional real en
  desarrollo (+0.0081) y en el holdout congelado (+0.0180 pareado); corrige
  errores de AG en la rebanada colectiva (recall 0.452 → 0.574-0.589).
- **Qué errores nuevos introduce**: en el núcleo duro falla las tres etiquetas en
  103 tweets (6.95 %) frente a 98 (6.61 %) de V03 (AUD-01).
- **Por qué no es el modelo final**: la mejora local no se transfirió al test
  (gap dev-LB −0.0058 frente a +0.0059 implícito de V03; D-020). El modelo final
  entregable sigue siendo V03 (LB 0.78306).
