# P95 - Criterios de reapertura y decision estrategica final

Fecha: 2026-10-13. Fase estrategica documental. Este documento fija las condiciones
bajo las cuales podria reabrirse la experimentacion y registra la decision vigente.

## 1. Condiciones de reapertura (P6)

Volver a autorizar experimentos exige AL MENOS UNA de estas condiciones:

1. Una fuente nueva de evidencia metodologicamente valida (P95-P4: N1/N2/N3; hoy
   ninguna supera el umbral de valor esperado -> NO_ACTIONABLE_NEW_EVIDENCE_SOURCE).
2. Una autorizacion explicita que habilite una linea antes bloqueada (G01-G07 de
   p95_governance_options.csv; ninguna recomendada hoy).
3. Un defecto de implementacion DEMOSTRADO en un experimento historico (via N2; hoy
   no existe ninguno demostrado).
4. Una hipotesis genuinamente nueva que no duplique una familia cerrada (P94: 0 de
   17; inexistente hoy).

## 2. Plantilla obligatoria de cualquier reapertura

Ninguna reapertura puede ejecutarse sin un acto de autorizacion explicito y posterior
a esta fase que complete, campo a campo:

| Campo | Contenido requerido (valores por defecto obligatorios) |
|---|---|
| hypothesis_id | identificador nuevo del acto (no reutilizar ids de familias cerradas) |
| mechanism | mecanismo exacto; debe declarar por que NO es equivalente a una familia del inventario P94 |
| historical_overlap | experimentos historicos mas cercanos y por que no los duplica |
| data_allowed | train/dev y grupos; nada mas |
| data_prohibited | test, holdout historico, holdout2 (salvo instrumento final con presupuesto), leaderboard como seleccion, pseudo-labeling, recursos externos, preentrenados |
| minimum_experiment | una sola configuracion principal + control; sin rejillas nuevas sobre puntos ya medidos |
| primary_metric | corrected_cv canonico (umbrales congelados 0.44/0.51/0.46, OOF agrupado; definicion congelada por escrito, D-255.2) |
| validation_protocol | StratifiedGroupKFold con grupos near-duplicate UNION pseudo-autor; consistencia por pliegue >= 4/5; bootstrap pareado con IC95; segunda particion confirmatoria para candidatos; holdout2 solo como instrumento final y con su presupuesto |
| stopping_rule | parada inmediata si el delta no alcanza el gate o si dos iteraciones no muestran senal; cierre documentado; prohibido ampliar rejillas o cambiar la metrica tras ver resultados |
| resource_budget | <= 30 min CPU para un diagnostico; GPU solo con justificacion explicita y autorizacion separada; presupuesto fijado antes de ejecutar |
| adoption_gate | dOOF > +0.0376 con neto de celdas positivo en la misma configuracion + reproducibilidad bit a bit + revision de transferencia (freno vigente: los deltas locales pequenos no transfieren) + fase de sonda solo con autorizacion |

Estado del gate hoy: **no existe ninguna hipotesis que cumplimentar**; la plantilla
queda documentada para el futuro y no autoriza nada por si misma.

## 3. Clasificacion de vias futuras (P7)

| Via | Clase | Motivo |
|---|---|---|
| Modelado exploratorio adicional (cualquier familia nueva) | CLOSED | P94: 0 hipotesis genuinamente nuevas tras 24 familias y 17 propuestas auditadas |
| Correccion de las 14 etiquetas | REQUIRES_GOVERNANCE_DECISION | Sin evidencia humana independiente (N1) y sin decision del propietario no se toca el train; efecto esperado bajo ruido |
| Propagacion de near-duplicados | CLOSED | Techo medido nulo (0 celdas); la puerta administrativa existe pero el valor es ~0 |
| Candidato cost-sensitive (F14) | PRESERVE_FOR_LATER | Congelado en gobernanza; no es nuevo; su sonda externa perdio (0.77522) |
| Preentrenados/transformers | CLOSED | Prohibido por regla del profesor; el cambio de regla esta fuera del control del proyecto |
| N1: revision humana ciega independiente | REQUIRES_GOVERNANCE_DECISION | Opcional, valor documental; requiere decision del propietario; no mejora el score |
| N2: auditoria dirigida de negativos | PRESERVE_FOR_LATER | Coste ~0 y util si algun dia se quiere atacar la condicion 3; hoy sin defectos conocidos |
| N3: clarificacion oficial de TR | REQUIRES_GOVERNANCE_DECISION | Opcional y documental; requiere que el propietario decida contactar |

**Via prioritaria seleccionada: NINGUNA.** No existe hoy una via que supere el
criterio de valor esperado para mejorar el resultado de Kaggle; se conservan las
puertas documentadas y los artefactos, sin solicitar autorizaciones y sin contacto
automatico con nadie.

## 4. Decision estrategica final

```text
MODEL_EXPLORATION_PAUSED
V03 = FROZEN
NO_TRAINING_AUTHORIZED
NO_SUBMISSIONS_AUTHORIZED
REOPENING_CRITERIA_DOCUMENTED
```

La pausa NO es prueba de que ninguna mejora futura sea posible: significa que el
historial actual (24 familias medidas, 17 propuestas auditadas, 0 hipotesis nuevas,
transferencia local->LB no demostrada y techo de anotacion acotado) no justifica otro
experimento en las condiciones presentes. Conservar el mejor modelo verificado es
preferible a aumentar complejidad sin expectativa razonable de transferencia.

Estado del proyecto: V03 sigue siendo el mejor resultado verificado
(corrected_cv 0.7772114525165721; LB 0.78306). Las 14 candidatas de etiquetas quedan
PRESERVED_NOT_CONFIRMED; los 5 casos humanos sin resolver permanecen sin resolucion.
