# P91 - Protocolo de revision humana ciega de etiquetas (disenado, NO ejecutado)

Estado: **PENDIENTE DE AUTORIZACION DE GOBERNANZA**. Este documento permite ejecutar la
revision de forma independiente y reproducible cuando exista autorizacion. El agente no
sustituye la adjudicacion humana. Material: desarrollo autorizado (1483 filas); el
holdout, holdout2 y el test no se usan jamas.

## 1. Poblacion, marco y unidad

- `population_definition`: tweets del desarrollo autorizado (1483) con etiquetas
  originales; `eligible_partitions`: dev (excluye holdout/holdout2/test).
- `sampling_frame`: ids de dev en `data/processed/holdout_v2_ids.csv` (particion dev).
- `sampling_unit`: tweet (el anotador ve el texto completo y anota HS, TR y AG).
- `stratification`: ninguna obligatoria; opcional por combinacion HS/TR/AG para cubrir
  las combinaciones raras 001 (22) y 110 (24) si la gobernanza lo pide.
- `group_handling`: las unidades en grupos near-dup/fusionados son evidencia dependiente;
  para prevalencias se usa el tweet como unidad (los grupos se reportan aparte); en la
  cola enriquecida se incluyen las filas divergentes con su identificador de grupo.

## 2. Componentes de la muestra (separadas, no mezcladas para prevalencias)

**A. Muestra probabilistica de referencia.** Muestreo aleatorio simple de 100 tweets de
los 1483 (semilla fija 20261010; probabilidad de inclusion conocida 100/1483;
las 3 etiquetas de cada tweet entran como celdas). Permite estimar la frecuencia de
problemas con ponderaciones correctas. Tamano justificado en la seccion 3.

**B. Muestra enriquecida con candidatos.** Cola de 882 celdas
(`reports/p91_label_review_candidates.csv`) con motivos verificables:
`HIGH_CONFIDENCE_OOF_DISAGREEMENT` (418),
`POSSIBLE_INSTANCE_LABEL_INCONSISTENCY` (129),
`DOCUMENTED_RULE_CONFLICT` (22),
`LABEL_SPECIFIC_MARGIN` (84),
`RANDOM_REFERENCE_SAMPLE` (300).
Topes por componente documentados (cap 150); ninguna
celda se presenta como error confirmado (`annotation_status = pending_authorization`).

## 3. Tamano muestral (justificacion, correccion por poblacion finita)

Para una prevalencia esperada p en celdas (4449), con nivel 95 por ciento y FPC:
`n = N*z^2*p(1-p) / (z^2*p(1-p) + (N-1)*e^2)`.

- p=0.05, e=0.03, N=4449: n ~ 194 celdas; con efecto de diseno 1.2 ~ 233 celdas; los 100
  tweets de referencia aportan 300 celdas -> semi-ancho esperado +-2.6 puntos porcentuales.
- A nivel tweet (N=1483, p~0.10, n=100): semi-ancho ~ +-5.7 puntos porcentuales (FPC
  0.933); suficiente para una primera tasa, no para estimaciones finas.
- `precision_assumptions`: p desconocida; se reportan ambos escenarios; cualquier cambio
  de tamano debe fijarse ANTES de ver resultados.
- `limitations`: dependencia por grupos y pseudo-autor; desacuerdo humano no es error.

## 4. Cegamiento

Se muestra al anotador: texto original completo; definiciones oficiales verbatim
(`docs/competition_rules.md`) y ejemplos de referencia autorizados; los campos de
anotacion; la opcion `unable_to_determine` (si la gobernanza la aprueba). NO se muestra:
etiquetas originales, probabilidades OOF, predicciones, distancia al threshold, features
del logit, motivo de seleccion, identidad de otros anotadores. Orden de ejemplos
aleatorizado por anotador. Los textos NO se modifican ni se recortan.

## 5. Anotacion y adjudicacion

- Dos anotadores independientes como minimo; cada etiqueta por separado
  (positive / negative / unable_to_determine); sin discusion antes de cerrar la ronda.
- Adjudicacion separada para discrepancias, con justificacion escrita y sin acceso a las
  etiquetas originales.
- Con un solo anotador, la revision es preliminar y no se presenta como consenso.
- Estados (no equivalentes): `possible label issue` -> `confirmed annotation error`
  (adjudicado) -> `adjudicated label` -> `approved dataset correction` (requiere
  autorizacion separada).

## 6. Medicion del acuerdo

Por etiqueta: acuerdo bruto; acuerdo positivo y negativo; Cohen kappa (dos anotadores) o
Krippendorff alpha nominal (mas de dos; admite falta de datos); matrices de confusion
entre anotadores; frecuencia de `unable_to_determine`; acuerdo exacto de la combinacion
HS/TR/AG. **Limitaciones declaradas**: kappa se degrada con prevalencias extremas
(TR=1 en 30 por ciento; AG=1 con HS=0 es raro) - se reporta junto a los acuerdos
positivo/negativo, nunca solo. Desacuerdo humano != etiqueta original incorrecta
(puede indicar definicion ambigua).

## 7. Flujo y trazabilidad

1. Autorizacion de gobernanza (documentada). 2. Construccion del paquete ciego
(`review_id` por celda; textos desde `toxic_train.csv`; clave de evaluacion SEPARADA con
gold y evidencia). 3. Anotacion independiente. 4. Adjudicacion. 5. Analisis de acuerdo y
tasas. 6. Informe con estados por celda. 7. Solo con autorizacion separada: politica de
correccion versionada (solo train; nunca test/holdout).

## 8. Criterios de decision (P6)

Seis preguntas separadas: (1) hay desacuerdo entre anotadores; (2) la etiqueta original
contradice una regla oficial; (3) hay error de anotacion confirmado; (4) los errores
afectan una fraccion relevante; (5) una correccion podria cambiar el aprendizaje;
(6) podria evaluarse de forma valida. Ninguna implica la siguiente.

Gates para una eventual correccion: autorizacion; anotacion ciega; adjudicacion
independiente; politica documentada; cambios SOLO en train; validacion anidada o por
grupos; analisis por pares (celda a celda); bootstrap/incertidumbre; reproducibilidad.
Criterios numericos fijados ANTES de mirar resultados (p. ej., proceder solo si la tasa
de errores adjudicados supera el umbral que la banda de particion 0.017 exige y el analisis
por pares es favorable).

## 9. Requisitos pendientes

Autorizacion de gobernanza (no solicitada; P87-00 la declaro LOW_EXPECTED_VALUE); reglas
operativas de anotacion (no disponibles); decision sobre el uso de
`unable_to_determine`. Sin ellos, este protocolo no se ejecuta.

## 10. Nota de consistencia P91-R (aditiva, 2026-10-09)

Correccion documental aditiva solicitada por la auditoria P91-R; el texto anterior se
conserva integro (registro original: sha256 ffbe1113622b42b5795b8135d70fd3b6cbcffcd2f260cca2c4a64bfacf40ccd0).

- **Precisiones por magnitud (no intercambiables)**, con formulas y unidades:
  - Tasa AGREGADA de celdas (3 etiquetas juntas; N=4449 celdas; n=300 celdas; p=0.05;
    FPC=sqrt((N-n)/(N-1))=0.9658; DEFF=1.2): SE=sqrt(p(1-p)/n)*FPC*sqrt(DEFF)=0.01331;
    margen 95 por ciento = **+-2.61 puntos porcentuales**. El "+-2.6" del texto original es
    correcto para esta magnitud.
  - Tasa POR ETIQUETA a nivel tweet (N=1483 tweets; n=100 tweets; p=0.05; FPC=0.9660;
    DEFF=1.2): SE=0.02306; margen 95 por ciento = **+-4.52 puntos porcentuales** (no estaba
    explicitado en el texto original).
  - Tasa de tweets a p=0.10: +-5.68 sin DEFF (el "+-5.7" original **omitia el efecto de
    diseno**); con DEFF 1.2 = +-6.22 puntos porcentuales. Correccion: toda cifra a nivel
    tweet debe reportarse CON el efecto de diseno declarado.
- **Semantica de motivos**: `DOCUMENTED_RULE_CONFLICT` es un CODIGO DE SELECCION de la cola
  (celdas AG=1 con HS=0, la tension estructural documentada en D-001), NO la afirmacion de
  una violacion de regla: P91 concluyo DOCUMENTATION_INSUFFICIENT y ninguna regla oficial
  aplica a esos casos. Su clasificacion correcta es EMPIRICAL_STRUCTURE_ANOMALY (ver
  `reports/p91r_methodology_consistency.md`). Los motivos NO son mutuamente excluyentes:
  las 882 celdas unicas portan 953 asignaciones de motivo (69 celdas con mas de un motivo).
