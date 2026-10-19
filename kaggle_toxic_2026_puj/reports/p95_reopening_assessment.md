# P95 - Evaluacion de reapertura: que limita la mejora, las 14 candidatas y la distancia al leaderboard

Fecha: 2026-10-13. Fase estrategica documental: sin entrenamiento, sin predicciones,
sin modificacion de etiquetas, sin submissions, sin accesos a test, holdout, holdout2
ni leaderboard.

## 0. Auditoria del diagnostico P94 (P0, resultados reales)

- Git: rama `main`, HEAD `e43c594`, 180 entradas sin rastrear y 1 modificada (solo el
  notebook consolidado); sin commit ni push; sin cambios locales perdidos.
- Los cinco entregables de P94 existen y sus hashes coinciden con los registrados en
  la seccion 154 del notebook: inventario `11f376e1d4c1545c`, sota `245cf119df408b4b`,
  registro `17917ce2a1a132d9`, recomendacion `7a09af6710b863af`, cumplimiento
  `588c7664e8b7ddf4`.
- Re-lectura del registro P94: 17 propuestas auditadas, 0 GENUINELY_NEW, 0
  recomendadas. **Confirmado: P94 concluyo con cero hipotesis nuevas.** No existe
  ninguna propuesta posterior a P94 en el arbol (la seccion 154 es la ultima del
  notebook; el ultimo commit es P90-00).
- Discrepancias P94 vs registro experimental: ninguna nueva. Las dos conocidas siguen
  documentadas sin alterar resultados historicos: (i) P83 se ejecuto en completo y su
  aviso quedo superado (P94-F21); (ii) la columna `f1` del registro mezcla protocolos
  de fases (D-128), lo que P94 ya trato de forma cualitativa.

## 1. P1 - Que limita actualmente la mejora

Cinco limitaciones, con evidencia a favor y en contra. "Familias cerradas" no
significa "imposible": significa que, con los protocolos probados, las variantes
medidas no funcionaron.

### A. Representacion insuficiente

- evidence_supporting: el techo con oraculo sobre los errores de V03 es +0.046271 y
  ningun mecanismo lo captura (D-091/D-103); los regimenes de error (hashtag,
  pregunta, referencia colectiva, longitud Q4) muestran interacciones que el modelo
  aditivo de bolsas no representa (FINAL_REPORT seccion 20); el odio implicito sin
  insulto lexico escapa; H11 (registro bromista/afiliacion) fallo en su formulacion
  probada (-0.0213) ; los embeddings desde cero mejoraron la cobertura de
  representacion pero no la tarea (D-041/D-043).
- evidence_against: mas de 50 configuraciones no superaron word+char+LR (F01); las
  familias de representacion alternativas estan medidas y cerradas (F07-F11, F16-F18,
  F22): kernels y espectro (V26/V27), PLS (V38-V44), FM (V25/P86), SVD300-RBF
  (V4-G/P8), hashing (V36), BM25 y ponderacion supervisada (V37/v4-C), PLS densa;
  V32-05 (interaccion URL x longitud) mejoro el OOF (+0.0045) y produjo un test
  **identico byte a byte a V03** (delta LB exactamente 0).
- historical_experiments: F01, F07, F08, F09, F10, F11, F16, F17, F18, F22; E2-03/04/
  07/08; EBD-00..27; V4-C/G; V25-V27; V31-V33.
- remaining_uncertainty: la formulacion especifica de conjunciones insulto x
  afiliacion (H11 "otra formulacion") no se cerro con una implementacion
  definitiva; los mecanismos candidatos para atacarla (regularizacion por bloque,
  calibracion por subgrupo) ya pertenecen a familias cerradas (D-086/087; F05).
- is_the_family_closed: empirically_closed_under_tested_protocols. Residuo
  genuinamente_no_explorado: clasificadores instance-based tipo kNN (P94 H-P94-03,
  INSUFFICIENT_EVIDENCE, con geometria memorizadora en contra).
- what_new_evidence_would_change_the_conclusion: datos nuevos autorizados (no
  existen), un defecto demostrado en una medicion clave (N2), o una regla nueva que
  reabra representaciones (G04/G05).

### B. Objetivo de entrenamiento o decision

- evidence_supporting: la F1-suave (sustituto literal del objetivo) fue la PEOR de las
  perdidas orientadas (-0.0033 vs control BCE con hueco -0.000046, 3/5 pliegues,
  V7-01); focal -0.0024; asimetrica, numero efectivo, polinomial, transporte,
  alineacion de gradiente y pares, todas cerradas (P59-P60, V12); los umbrales estan
  debilmente identificados (D-055) y las re-optimizaciones no aportan (F05).
- evidence_against: el montaje del control es fiel (hueco de -0.000046: la
  implementacion no explica el negativo); la unica mejora pareada de LB del proyecto
  vino justamente de la capa de decision (umbrales v02->v03, +0.0031).
- historical_experiments: F05, F22; V7-01; P59-10; P60-01/02/04/06/07/08; V12-01..05.
- remaining_uncertainty: si una mejora del objetivo entrenara, el freno de
  transferencia sugiere que no llegaria al LB sin mas evidencia; nada de esto esta
  abierto con los protocolos actuales.
- is_the_family_closed: empirically_closed_under_tested_protocols.
- what_new_evidence_would_change_the_conclusion: un defecto demostrado en V7-01/V12
  (N2); ningun otro camino plausible.

### C. Variabilidad y transferencia de la validacion

- evidence_supporting: R1-C1 (+0.0157 local, -0.0108 externo), R2-MINDF5 (+0.0221,
  -0.0127) y OPT-01 (+0.0240, -0.0070) perdieron los tres en el LB (D-013 refutada);
  S08 (meta) 0.78163 < 0.78306; la banda de particion corregida es ~0.017 (D-070);
  la familia campeona en el LB forma un plateau de ~1 error estandar
  (0.77034-0.78306); el pseudo-autor explicaba +0.0088 de optimismo (D-015), hallado
  despues de la seleccion original; winner's curse documentado.
- evidence_against: el protocolo corregido (grupos fusionados, D-015/D-016) redujo el
  gap local-LB del modelo final a -0.0039 y el arnes fue validado en los dos sentidos
  (V10-03); V33-03 verifico que una candidata OOF no cambio ni una fila del test.
- historical_experiments: F02, F03, F06; V9-V11; D-013/D-015/D-064/D-067/D-070.
- remaining_uncertainty: el mecanismo del gap local-LB no tiene explicacion formal ni
  predictor validado; para tenerlo harian falta mas observaciones externas (sondas),
  gobernadas y limitadas, que hoy no se justifican sin candidato.
- is_the_family_closed: no es una familia de modelado; es una propiedad medida y una
  restriccion vigente (empirically_closed_under_tested_protocols como bloqueo de
  cualquier candidato pequeno).
- what_new_evidence_would_change_the_conclusion: un instrumento de evaluacion externo
  nuevo (no disponible bajo las reglas) o sondas pareadas de un candidato futuro que
  pase el gate interno.

### D. Calidad y definicion de las etiquetas

- evidence_supporting: el 40.49% de la region "insulto interpersonal con mencion y sin
  referencia colectiva" esta etiquetada (0,0,0) (FINAL_REPORT seccion 20); las 171
  celdas con consenso humano binario de P93-R5 difieren TODAS del gold; 14 candidatas
  con justificacion documentada (AG 8), todas del componente enriquecido; D-005 (TR)
  sigue sin resolver; las intervenciones medidas son +-0.0005 y el techo historico de
  la via (~+0.009) queda bajo la banda de ruido 0.017 (P87).
- evidence_against: 0 errores confirmados; los humanos de ronda 2 vieron A/B (no
  independientes); la muestra es enriquecida por discrepancia (sin prevalencia
  poblacional inferible); P93-R5 no encontro ninguna candidata justificada en la
  muestra aleatoria de referencia.
- historical_experiments: F23, F24; D-005; D-090..D-104; V4-A; V28-03; P87-00;
  P91-P93; P93-R5.
- remaining_uncertainty: la tasa real de error/ambiguedad del gold es desconocida y no
  es estimable desde la muestra enriquecida; la semantica oficial de TR no esta
  confirmada documentalmente.
- is_the_family_closed: blocked_by_governance (la modificacion del train no esta
  autorizada y las reglas son silenciosas) + insufficient_evidence (no hay tasa
  independiente; N1/N3 podrian aportarla).
- what_new_evidence_would_change_the_conclusion: revision humana CIEGA independiente
  de las 14 mas una sonda de ambiguedad aleatoria (N1); clarificacion oficial de TR
  (N3).

### E. Restricciones academicas y de la competicion

- evidence_supporting: la prohibicion de transformers/preentrenados es una regla
  explicita del profesor (docs/competition_rules.md, "Restriccion adicional");
  el estado del arte actual en odio en espanol es transformer (R14 de P94); el DL
  desde cero quedo por debajo de V03 con 1.731 filas (F12); sin datos externos
  etiquetados, sin pseudo-labeling, sin corpus sin auditar.
- evidence_against: las restricciones no impidieron alcanzar el mejor resultado
  clasico verificado (0.78306), comparable al mejor sistema sin transformers de
  HatEval espanol (macro-F1 0.73, Argota Vega 2019); la auditoria aprobo el
  cumplimiento integro.
- historical_experiments: F10-F12; V8-00 (D-057); D-002/D-003/D-007; D-118.
- remaining_uncertainty: cuanto aportaria exactamente un transformer aqui es
  desconocido y medirlo exigiria cambiar la regla (fuera de nuestro control).
- is_the_family_closed: blocked_by_governance (regla externa del curso).
- what_new_evidence_would_change_the_conclusion: solo un cambio de regla del
  organizador (G04) o una autorizacion nueva del marco experimental.

### Sintesis P1

Las limitaciones A y B estan agotadas por medicion (empiricamente cerradas bajo los
protocolos probados, con controles fieles). La limitacion C es una propiedad medida
del sistema (los deltas locales pequenos no transfieren) y actua como restriccion, no
como familia. Las limitaciones D y E son las unicas con puertas que hoy NO estan
cerradas por medicion, sino por gobernanza (etiquetas) o por regla externa
(preentrenados). Ninguna de esas dos puertas, sin embargo, ofrece hoy una mejora
medible: D tiene techo bajo la banda de ruido y E esta fuera del control del proyecto.
Residuo genuinamente no explorado del espacio permitido: los clasificadores
instance-based (una celda, con evidencia adyacente en contra).

## 2. P3 - Las 14 candidatas de etiquetas

Estado: PRESERVED_NOT_CONFIRMED; 0 errores confirmados; HS 2 / TR 4 / AG 8; todas del
componente enriquecido; ninguna de la muestra aleatoria de referencia; los cinco casos
humanos sin resolver siguen sin resolucion.

Checklist de lo que haria falta para justificar un experimento de correccion:

| Criterio | Estado actual | Comentario |
|---|---|---|
| clear annotation rationale | PARCIAL | Existe rationale de C1/C2, pero cita la guia v2.1; no verifica el gold original |
| independent evidence | NO | Ronda 2 vio A/B; C1/C2 no son independientes de la evidencia que selecciono las candidatas |
| adjudication requirements | NO | La tercera revision no se realizo ni esta autorizada (P93-R5: NOT_REQUIRED_FOR_THIS_PHASE) |
| group-aware validation | PENDIENTE | Cualquier experimento exigiria grupos near-dup UNION pseudo-autor |
| pre-registered experiment | PENDIENTE | Exigiria pre-registro congelado con presupuesto y gate |
| expected measurable effect | NO | 14 celdas de ~5193; techo historico ~+0.009 < banda 0.017; evaluar sobre las mismas celdas seria circular |

Clasificacion: REQUIRES_MORE_HUMAN_EVIDENCE.

Justificacion: la linea no se cierra por imposibilidad, sino por falta de evidencia
independiente; sin una revision ciega que nunca haya visto A/B, cualquier correccion
seria un eco de la propia evidencia que la selecciono. Aun obteniendola, el efecto
medible esperado sobre el score es indistinguible del ruido, y el diseno honesto
(reservar grupos corregidos fuera de la evaluacion) diluiria la senal todavia mas.
Los artefactos se conservan integros (KEEP_PRESERVED de facto); como palanca de
Kaggle: NOT_WORTH_PURSUING. No se exige continuar la linea por el trabajo humano ya
invertido.

## 3. P5 - Distancia respecto al leaderboard

Tres registros que no deben mezclarse:

- internal_generalization_estimate: corrected_cv de V03 = 0.7772114525165721 (OOF
  agrupado con umbrales congelados 0.44/0.51/0.46; caveat D-126: los umbrales vienen
  de un protocolo mas debil que el de la metrica reportada).
- external_transfer_observation: LB de V03 = 0.78306 (mejor de 23 sondas historicas;
  la unica mejora pareada positiva del proyecto fue la capa de umbrales, v02->v03
  +0.0031). Observaciones clave: deltas locales de +0.015 a +0.024 NO transfirieron
  (R1-C1, R2-MINDF5, OPT-01); un candidato con test identico byte a byte a V03 dio
  delta LB exactamente 0 (V32-05/V33-03); la banda del LB de la familia campeona
  (0.77034-0.78306) equivale a ~1 error estandar de muestreo; ninguna sonda posterior
  a v03 supero 0.78306.
- leaderboard_optimization: prohibida como criterio de seleccion (D-014; winner's
  curse documentado). No se estiman modelos ocultos de competidores ni se usan sus
  scores; ninguna sonda nueva se deriva de este analisis.

Implicacion operativa: el gate de un candidato futuro debe exigir, antes de
considerar una submission: (1) delta sobre corrected_cv consistente por pliegue
(>= 4/5); (2) incertidumbre estadistica (bootstrap pareado cuyo IC95 excluya cero) y
neto de celdas positivo; (3) reproducibilidad (script pre-registrado, re-ejecucion
bit a bit); (4) segunda particion confirmatoria; (5) magnitud compatible con el gate
vigente (dOOF > +0.0376 con neto positivo). No se inventa un techo teorico de F1 ni
una prediccion de score futuro.

## Referencias de politica

- Reglas verbatim: docs/competition_rules.md (silencio sobre correccion de etiquetas,
  near-dup y corpus; prohibicion explicita de transformers/LLM).
- Gobernanza interna: D-002/D-003/D-007/D-012/D-013/D-014/D-015/D-016/D-057/D-069/
  D-074/D-080/D-118/D-288; V6-06; V10-08; P87-00; P93-R5; P94.
