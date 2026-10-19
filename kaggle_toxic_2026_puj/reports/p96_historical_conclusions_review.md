# P96 - Revision de las conclusiones historicas frente a la auditoria

Fecha: 2026-10-13. Pregunta: que resultados siguen siendo evidencia fiable,
comparable y reproducible de que una tecnica no mejora V03.

Estados posibles por familia: INTACTA (evidencia re-verificada o no
dependiente de los hallazgos), MATIZADA (sigue en pie con una nota), SUSPENDIDA
(no se encontro; ninguna familia queda suspendida).

## 1. Resumen

- Ninguna conclusion historica se suspende.
- Dos matices documentales: (a) el criterio IC de la fase V4 se aplico sobre un
  proxy (F96-17); (b) los umbrales de V03 tienen optimismo in-sample conocido
  (F96-05). Ambos con direccion de cierre preservada.
- El benchmark interno queda reforzado: V03, folds, hashes y 20 comparaciones
  curadas reproducen exactamente desde los artefactos guardados.

## 2. Revision por familia

| Familia | Cierre historico | Verificacion P96 | Estado |
|---|---|---|---|
| V03 (referencia) | LB 0.78306; corrected_cv 0.7772114525165721 | Reproduccion EXACTA (metrica, etiquetas, pliegues, hashes; proba M2 identica entre artefactos) | INTACTA |
| v2 candidatos (metas/correcciones; D-024) | Cerrada; V03 se mantiene | Variantes recomputadas desde artefactos (targeting 2, syntax 8, cita 5, emocion 11; meta) a 0.5; convenciones de la columna f1 documentadas (D-128) | INTACTA (nota F96-07) |
| v4 reweighting/weighting/joint/meta_shift | Cerradas por medicion (D-016..D-029) | 13+14+12+12 variantes recomputadas; valores documentados reproducen (meta_control/shift/cuantil/platt en dos protocolos) | INTACTA |
| v4 candidatos A/B/C/D (36_candidatos_v4) | Solo C superaba los tres criterios; C rechazado por holdout (-0.0122); A quedo PENDING | A-D recomputados exactos (0.7854160387 / 0.7827736230 / 0.7881730927 / 0.7904096156); IC recomputado con la metrica primaria (F96-17) | MATIZADA (frase "unico C" dependiente del estimando; adopcion sin cambios) |
| v6 (label factorization, ponderacion, char weight, skipgrams, bagging, ruido, aumento) | Cerrada; controles en 0.0; variantes negativas | JSON con deltas verificados (control B0 = 0.0); sin matrices OOF para recomputo independiente (F96-11/REP-96-01) | INTACTA (hueco de provenance declarado) |
| v7 (perdidas) | F1-suave la peor; control BCE fiel | Codigo del control y de atribucion leido; sin defecto | INTACTA |
| v5/v5.1 y EBD (embeddings) | Cerrada por medicion + directiva (D-032..D-043; P85) | Artefactos p84_04/p85_01 recomputan; auditorias P80/P81/P84/P85 previas citadas; IC por filas -> nota F96-06 | INTACTA |
| V25 FM / V26 / V27 / V43 (geometria) | Cerrada (D-190); control bit-a-bit declarado (D-291) | Sin probas guardadas para recomputo independiente; documentacion consistente | INTACTA (no re-auditada numericamente; declarado) |
| P61 LightGBM | DUPLICATE/cerrada (D-143) | Referencias del registro consistentes; sin re-ejecucion | INTACTA |
| P73-P77 (textcnn, null aleatorio, complementariedad, calibracion, selector) | Cerradas con auditorias dedicadas | Auditorias de fase citadas; artefactos parciales en orden | INTACTA |
| P84-01..04 / P85 (DL corregido) | Cerradas por medicion | P84-01 semillas 42/43/44, P84-02 control x2, P84-03 control=V03, P84-04 agregado: recomputados y coincidentes | INTACTA |
| P93-R5 (etiquetas) | 0 errores confirmados; 14 candidatas preserved | Hashes intactos; no tocadas | INTACTA |
| P94/P95 (frontera) | NO_NEW_MODELING_HYPOTHESIS; MODEL_EXPLORATION_PAUSED | Auditados en P0 (entregables y hashes intactos) | INTACTA |

## 3. Impacto sobre P94/P95

- "No existe una hipotesis de modelado genuinamente nueva": sin cambios. La
  auditoria no encontro ningun defecto que reabra una familia; el residuo no
  explorado sigue siendo el mismo (instance-based/kNN, con evidencia en contra).
- Gates de P95: se mantienen; se anaden dos notas de higiene para cualquier
  reapertura: IC de la metrica primaria por clusters (F96-06/F96-17) y
  trazabilidad de numeracion de pliegues 1..5 (F96-04).
- El unico ajuste de redaccion sugerido al registro historico (sin editarlo):
  donde dice que solo C superaba los tres criterios de D-016, precisar que el
  criterio de IC de esa fase uso un estimando proxy.

## 4. Que haria falta para suspender algo

Nada de lo auditado lo exige. Las condiciones que si exigirian una revision
serian: un NUMERIC_MISMATCH reproducible en una comparacion curada, un defecto
demostrado en el codigo de una familia cerrada (via REP-96-01 o equivalente), o
una autorizacion de gobernanza que cambie el espacio experimental (P95-P2).
Ninguna se cumple hoy.
