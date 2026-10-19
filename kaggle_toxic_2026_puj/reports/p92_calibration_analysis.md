# P92-T - Analisis de la calibracion humana (4 anotadores, 15 tweets)

Fecha: 2026-10-12. Fase diagnostica: sin entrenamiento, sin predicciones, sin
modificacion de etiquetas, sin acceso a test/holdout, sin leaderboard, sin
submissions. Las respuestas humanas originales no se modificaron.

Estados:

```text
CALIBRATION_ANALYZED
ENCODING_AB = ENCODING_CONFIRMED (aclaracion del responsable del proyecto)
GUIDELINE_V2_PROPOSED_PENDING_DISCUSSION
HUMAN_DISCUSSION_PENDING
MAIN_ANNOTATION_NOT_DISTRIBUTED
V03 = FROZEN | CANONICAL_OOF = FROZEN | TEST = UNTOUCHED | LEADERBOARD = UNUSED
NO NEW MODELING PHASE
```

## 0. Capas de datos (no mezcladas)

```text
raw response            -> bytes originales de los 4 archivos de respuesta
normalized response     -> derivados de A/B con la codificacion confirmada
annotation interpretation -> uso de las respuestas como tokens en el analisis
adjudicated decision    -> NO EXISTE todavia (etapa posterior y separada)
```

## 1. Auditoria de las cuatro respuestas

| anotador | archivo | sha256 (16) | filas | ids | texto | esquema |
|---|---|---|---|---|---|---|
| A | `artifacts/human_annotation_v1/calibration_responses/calibration_response_A.csv` | `a20521989aa3fea9` | 15 | P92C-01..15 | integro (identico en los 4 archivos) | 0/1 en clases y notas (desviacion de formato) |
| B | `artifacts/human_annotation_v1/calibration_responses/calibration_response_B.csv` | `41e50c1f1d48760e` | 15 | P92C-01..15 | integro (identico en los 4 archivos) | 0/1 en clases y notas (desviacion de formato) |
| C | `artifacts/human_annotation_v1/calibration_responses/calibration_response_C.csv` | `8de76e21046d5091` | 15 | P92C-01..15 | integro (identico en los 4 archivos) | tokens + notas libres |
| D | `artifacts/human_annotation_v1/calibration_responses/calibration_response_D.csv` | `763229f9a9caa176` | 15 | P92C-01..15 | integro (identico en los 4 archivos) | tokens + notas libres |

- sha256 completo de las respuestas originales:
  - A: `a20521989aa3fea94e059f0666713b437a333d2cead31f9e4198d24dde9c9596`
  - B: `41e50c1f1d48760eaf293a8e5f51b9dd0d036b19e8d48f92a2e1ce34c024a49f`
  - C: `8de76e21046d5091bfa360cce6fb5299d2ef9951685b46ae85178ce45e477d0f`
  - D: `763229f9a9caa176df7b9ee1ff247269aa6ea51259a9ce3f8ca658dc3432f8a4`
- Dominios crudos observados: A y B = {0, 1} en las cuatro columnas; C y D = {POSITIVE, NEGATIVE, UNABLE_TO_DETERMINE} en las columnas de clase y texto libre en uncertainty_notes.
- Los archivos crudos no cambiaron respecto a lo recibido (raw_annotation_files_unchanged = True).
- Las copias del handoff de A y B coinciden byte a byte con las respuestas (editaron en el sitio): hash identico verificado.
- Los textos de las cuatro respuestas son identicos entre si para los 15 review_id (textos_identicos = True; sha256 del bloque de textos: `21fe77e8bf21a02f`).
- `guideline_version` = v1 en los cuatro archivos; la seleccion de calibracion no cambio (calibration_sample_unchanged = True).
- Desviaciones de formato detectadas (documentadas, no corregidas en el origen):
  1. A y B usaron 0/1 en lugar de los tokens exigidos por el README y la guia v1.
  2. A y B conservaron BOM de Excel (los archivos C y D no lo tienen).
  3. A y B usaron la columna de notas con un valor 0/1; C y D usaron texto libre.

## 2. Codificacion de A y B: resultado de la auditoria

Resultado: **ENCODING_CONFIRMED**.

Regla establecida: En las columnas de clase: 0 = NEGATIVE, 1 = POSITIVE. En uncertainty_notes: 1 = dificil de determinar para cualquier clase (flag de fila), no UTD por etiqueta.

Fuente: aclaracion explicita del responsable del proyecto (Carlos, 2026-10-12), solicitada tras detectar que A y B devolvieron 0/1 en lugar de los tokens exigidos por el README y la guia v1.

Evidencia revisada:
- El README del handoff y la guia v1 exigen exclusivamente POSITIVE/NEGATIVE/UNABLE_TO_DETERMINE; ningun documento define 0/1 (por eso la correspondencia requeria verificacion, no suposicion).
- La aclaracion registrada define el significado de 0/1 en las columnas de clase y en la columna de notas, resolviendo ambas ambiguedades de esquema.
- La interpretacion es consistente con el esquema binario observado en los archivos de A y B (dominios {0,1} en las cuatro columnas).

La coincidencia con C o D NO se uso como fundamento de la codificacion (seria circular). Se reporta solo como observacion de consistencia posterior.

Tratamiento en metricas: Las respuestas de clase de A y B se usan como binarias interpretadas; los flags de fila se mantienen como capa separada y NO se convierten en UTD (no existe base documental para equiparar 'dificil para cualquier clase' con UTD por etiqueta).

Flags de fila registrados (notas=1): A = ['P92C-02', 'P92C-04']; B = ['P92C-01', 'P92C-05', 'P92C-13'].

Observacion de consistencia (NO fundamento de la codificacion): 4 de las 5 filas
marcadas por A o B coinciden con filas donde C o D usaron UNABLE_TO_DETERMINE;
la unica fila marcada sin UTD de C/D es P92C-02. El contraste de los archivos
normalizados derivados contra la interpretacion coincide en todas las celdas
de clase (45 por anotador): A=True, B=True.

## 3. Acuerdo entre anotadores (respuestas interpretadas)

Denominador: 15 filas por etiqueta. UTD se conserva como categoria diferenciada
para C y D; A y B no emitieron UTD en las columnas de clase (solo flags de fila).

### 3.1 Acuerdo exacto (3 categorias) y por par

| etiqueta | AB | AC | AD | BC | BD | CD |
|---|---|---|---|---|---|---|
| HS | 10/15 | 5/15 | 6/15 | 6/15 | 6/15 | 12/15 |
| TR | 11/15 | 9/15 | 9/15 | 11/15 | 11/15 | 14/15 |
| AG | 12/15 | 11/15 | 8/15 | 10/15 | 7/15 | 11/15 |

### 3.2 Acuerdo solo sobre celdas binarias (UTD excluido) y acuerdo positivo/negativo

| etiqueta | par | binarias | acuerdo bin | dice positivo | dice negativo |
|---|---|---|---|---|---|
| HS | AB | 15/15 | 10/15 | 6/11 | 14/19 |
| HS | AC | 8/15 | 5/8 | 4/9 | 6/14 |
| HS | AD | 7/15 | 6/7 | 4/7 | 8/15 |
| HS | BC | 8/15 | 6/8 | 6/12 | 6/11 |
| HS | BD | 7/15 | 6/7 | 4/10 | 8/12 |
| HS | CD | 6/15 | 6/6 | 6/8 | 6/7 |
| TR | AB | 15/15 | 11/15 | 18/22 | 4/8 |
| TR | AC | 14/15 | 9/14 | 18/23 | 0/6 |
| TR | AD | 15/15 | 9/15 | 18/24 | 0/6 |
| TR | BC | 14/15 | 11/14 | 22/25 | 0/4 |
| TR | BD | 15/15 | 11/15 | 22/26 | 0/4 |
| TR | CD | 14/15 | 14/14 | 26/27 | 2/2 |
| AG | AB | 15/15 | 12/15 | 12/15 | 12/15 |
| AG | AC | 14/15 | 11/14 | 14/17 | 8/12 |
| AG | AD | 12/15 | 8/12 | 14/18 | 2/9 |
| AG | BC | 14/15 | 10/14 | 14/18 | 6/11 |
| AG | BD | 12/15 | 7/12 | 14/19 | 0/8 |
| AG | CD | 11/15 | 11/11 | 20/21 | 2/5 |

### 3.3 Kappas y combinacion exacta de los 4

| etiqueta | kappa CD (bin) | kappa CD (3cat) | exacto 4/15 | alpha UTD=categoria | alpha UTD=faltante |
|---|---|---|---|---|---|
| HS | 1.000 | 0.683 | 4/15 | 0.243 | 0.466 |
| TR | 1.000 | 0.643 | 8/15 | 0.104 | 0.093 |
| AG | 1.000 | 0.444 | 6/15 | 0.354 | 0.491 |

Matrices completas por par (celdas 3x3) y kappas de los seis pares: `reports/p92_calibration_agreement.csv`. Las alfas coinciden con las
registradas en P92-S2 (reconciliacion: HS TR AG = True/True/True en ambos tratamientos).

Lectura descriptiva de esta muestra (15 filas; no extrapolable):
- El par C-D tiene el acuerdo mas alto en las tres etiquetas y kappa binario 1.000
  en HS, TR y AG: cuando ambos emiten respuesta binaria, nunca discrepan; sus
  diferencias son solo por el uso de UTD (que si coincide en HS: 6 filas).
- Los pares con D en TR dan kappas bajos o negativos: D respondio POSITIVE en las
  15 filas de TR (varianza cero en su margen), lo que hace inestable el kappa de
  esos pares; el acuerdo bruto sigue siendo alto (9-14 de 15).
- A y B, entre si, tienen acuerdo alto en TR (11/15) y AG (12/15) y menor en HS (10/15).
- Combinacion exacta de los 4 anotadores (las tres etiquetas identicas): 1/15.

### 3.4 UTD por etiqueta y anotador

| etiqueta | A | B | C | D |
|---|---|---|---|---|
| HS | 0 | 0 | 7 | 8 |
| TR | 0 | 0 | 1 | 0 |
| AG | 0 | 0 | 1 | 3 |

A y B no usaron UTD en las columnas de clase: registraron la duda con el flag de
fila (A: 02, 04; B: 01, 05, 13). Los flags NO se convierten en UTD (seccion 2).

## 4. Las seis filas HS con UTD coincidente de C y D

Verificadas en los archivos reales: P92C-05, P92C-09, P92C-11, P92C-13, P92C-14,
P92C-15. En las seis, C y D registraron UNABLE_TO_DETERMINE en HS con notas que
apuntan al mismo vacio normativo: la guia v1 no define cuando una ofensa sexista
individual dirigida a una persona constituye discurso de odio.

| review_id | C | D | motivo agrupado |
|---|---|---|---|
| P92C-05 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |
| P92C-09 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |
| P92C-11 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY (blanco por menciones) |
| P92C-13 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY + INSULT_VS_AGGRESSION (tono) |
| P92C-14 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |
| P92C-15 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |

Los seis casos comparten un mismo tipo de ambiguedad (odio vs. ofensa), con un
matiz adicional en P92C-13 (agresivo vs. vulgar / posible tono de broma). No se
fuerza una categoria unica para 13: sus dos notas sustentan dos tipos.

Casos relacionados fuera de la interseccion:
- P92C-03 y P92C-04: D marco UTD en HS con la misma duda normativa (C respondio
  POSITIVE en ambas, con la duda anotada en 04).
- P92C-07: C marco UTD en HS por contexto insuficiente (blanco no identificable).

La coincidencia C-D se reporta como hallazgo descriptivo de la muestra; NO se
interpreta como prueba de que las etiquetas originales esten equivocadas, y no se
usa como fundamento unico de ninguna regla.

## 5. Otras ambiguedades identificadas (AG, TR, contexto, flag de fila)

- AG: P92C-06 (generalizacion hostil sin insulto: agresividad discutible, D UTD),
  P92C-13 (agresivo vs. vulgar, C UTD), P92C-07 (sarcasmo/contexto, D UTD).
- TR: P92C-01 (blanco no identificable sin contexto, C UTD; D respondio POSITIVE).
- Contexto: P92C-01, P92C-04 y P92C-07 dependen de informacion externa ausente.
- Flag de fila: P92C-02 (A, sin motivo especificado) es la unica senal de duda sin
  UTD asociado de C o D; se clasifica como OTHER.

Detalle por caso (respuestas de los 4, tipo de ambiguedad, vacio de guia, regla
candidata): `reports/p92_calibration_case_review.csv` (15 filas).

## 6. Limitacion muestral (explicita)

- La calibracion consta de 15 tweets. Todas las fracciones de este informe son
  resultados de ESTA muestra, con denominadores visibles (n/15, n/den parcial).
- No se estima con esta muestra la tasa global de error de anotacion del dataset.
- 6/15 filas de esta muestra tienen UTD coincidente de C y D en HS: es un hallazgo
  de calibracion sobre las instrucciones, no una tasa del dataset completo.
- Los coeficientes de acuerdo no se extrapolan a la muestra principal de 250 tweets.

## 7. Guia v2 propuesta

Ver `reports/p92_annotation_guidelines_v2_proposal.md` (10 secciones, changelog
completo v1 -> v2, criterios operativos separados en OFFICIAL_DEFINITION /
PROPOSED_OPERATIONAL_RULE / UNRESOLVED_POLICY_QUESTION).

Decision de version: **V2_PROPOSED_PENDING_DISCUSSION**. La guia no se congela y
la muestra principal no se distribuye hasta: resolver lo procedimental, registrar
las decisiones humanas, documentar lo normativo abierto, aprobar la version
aplicable y registrar version y hash de la guia.

## 8. Discusion humana pendiente

Ver `reports/p92_calibration_discussion_pack.md`: resumen de desacuerdos, las seis
filas HS, casos adicionales, preguntas operativas, opciones alternativas y el
registro de decisiones. El resultado esperado es un registro de decisiones humanas
y cuestiones pendientes, no una etiqueta correcta inferida.

## 9. Trazabilidad y auditoria de integridad

```text
raw annotation files unchanged     = True
original labels unchanged          = True
original texts unchanged           = True
calibration sample unchanged       = True
main sample not distributed        = True
V03 unchanged                      = True
canonical OOF unchanged            = True
holdouts untouched                 = True
test not accessed                  = True
leaderboard not used               = True
no models trained                  = True
no predictions generated           = True
no new submissions                 = True
no pseudo-labeling                 = True
no external classifier             = True
no external labeled dataset        = True
```

Hashes canonicos verificados (prefijos): data/raw/toxic_train.csv=OK, features/oof_v03_final.npz=OK, submissions/toxic_submission_v03_lr_union_thr.csv=OK, experiments/baseline_v03_frozen/model_v03.joblib=OK, data/processed/holdout_v2_ids.csv=OK, artifacts/human_annotation_v1/annotator_A.csv=OK, artifacts/human_annotation_v1/annotator_B.csv=OK

## 10. Estado final y siguientes pasos

1. La discusion humana de la calibracion queda abierta (paquete de discusion).
2. Tras la discusion: registrar decisiones, decidir v1 vs v2 y congelar la guia
   aplicable con version y hash ANTES de distribuir la muestra principal.
3. La muestra principal (250 tweets) permanece sin distribuir; los paquetes no se
   regeneran hasta que exista guia congelada (cualquier regeneracion sera trazable).
4. Sin fase de modelado nueva: V03 y el OOF canonico permanecen congelados.

---

## NOTA ADITIVA P92-T-R (2026-10-12): procedencia de la codificacion A/B

La atribucion "ENCODING_CONFIRMED" de la seccion 2 queda CORREGIDA en su interpretacion.
La auditoria de procedencia (reports/p92tr_encoding_provenance_audit.md) encontro que la
aclaracion del usuario existe como instruccion registrada en la sesion de trabajo (cita
verbatim en el addendum), pero (1) no estaba persistida en ningun artefacto del proyecto,
(2) no enumera literalmente la direccion del mapeo 0/1 y (3) es de nivel coordinador, sin
confirmacion directa de A ni de B.

Estado corregido: ENCODING_UNRESOLVED al estandar estricto. Consecuencias:
- Las estadisticas que dependen de la codificacion de A/B (pares AC, AD, BC, BD,
  combinacion exacta de los 4, alfas con A/B) quedan CONDICIONALES a esa codificacion.
- Las observaciones de C y D, sus UNABLE_TO_DETERMINE y los valores crudos 0/1 de A y B
  permanecen validos; el acuerdo A-B (10/15, 11/15, 12/15) es valido como igualdad de
  valores crudos.
- Los archivos normalizados de A/B quedan marcados como derivados CONDICIONALES.
- Se preparo una solicitud breve de confirmacion directa a A y B
  (reports/p92tr_ab_confirmation_request.md; no enviada).
Ver el addendum para el detalle completo y el impacto por niveles de evidencia.
