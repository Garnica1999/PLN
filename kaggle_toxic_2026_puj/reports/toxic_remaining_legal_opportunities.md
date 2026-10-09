# TOXIC - Auditoria adversarial de oportunidades legales restantes

Fecha: 2026-10-08.
Alcance: auditoria documental y de artefactos. **Sin entrenamiento, sin GPU, sin
submissions, sin test, sin holdout y sin consultas al leaderboard.**
Formulacion canonica del proyecto: **V03 remains the best verified leaderboard
result.**

## 0. Resumen ejecutivo

La pregunta de la auditoria: existe alguna intervencion permitida por las reglas,
no ejecutada previamente y con un mecanismo de mejora defendible, que pueda
justificar una nueva fase experimental?

**Respuesta: no. Caso B.** Las dos vias pendientes quedan bloqueadas o en
cuarentena, las cinco hipotesis de la Puerta 2 estan ejecutadas y cerradas, las
cinco preguntas abiertas de P57 no cumplen el umbral de reapertura y ninguna
linea del mapa alcanza el estado `ACTIONABLE`. La investigacion de modelado
permanece cerrada y no se propone ninguna fase nueva.

**No se identifica actualmente una via experimental nueva, legal y
suficientemente justificada dentro de las restricciones vigentes del proyecto.**

Reparto de las 20 lineas auditadas (`reports/toxic_remaining_legal_opportunities.csv`):

```text
ACTIONABLE          0
BLOCKED_BY_RULES    5
DUPLICATE           1
CLOSED             14
```

Las dos vias que el encargo pedia auditar expresamente:

| via | evidencia principal | estado |
| --- | --- | --- |
| A. Frontera de anotacion de TR | D-005; material de revision humana listo y nunca completado; V4-A (2,8 por ciento de contradiccion, el modelo acierta los ambiguos) | `BLOCKED_BY_RULES` (silencio normativo sobre correccion de etiquetas) |
| B. Propagacion de near-duplicados train -> test | leakage_audit (6 pares >= 0,90 y 10 en 0,80-0,90); V4-F04 (techo medido nulo); sonda v18 (0 celdas) | `QUARANTINED` / `BLOCKED_BY_RULES` (autorizacion del organizador; techo <= 0,002) |

**Hipotesis Twitter/X (Word2Vec de dominio).** La fila 18 del CSV y la seccion 6.6 explicitan su decision: la linea **se ejecuto** con un corpus legal autorizado (mmaguero, CC BY-SA 4.0, periodo cerrado, solapamiento cero con train) y quedo **`CLOSED` por medicion** (P72/P75/P78; P73/P74 invalidados; P84-04 negativo sobre corpus general). El corpus de 597M del encargo original sigue bloqueado (sin licencia y con periodo abierto), y la combinacion SGNS corregido x corpus Twitter no se ejecuto (cambio de corpus: no reabre por directiva).

## 1. Estado de partida verificado

| elemento | valor | verificacion |
| --- | --- | --- |
| V03 `corrected_cv` | 0.7772114525165721 | `features/oof_v03_final.npz` sha256 `55926581...` (verificado hoy) |
| V03 submission | sha256 `7d7478f2...` | `submissions/toxic_submission_v03_lr_union_thr.csv` (verificado hoy) |
| train oficial | sha256 `3d40e734...` | `data/raw/toxic_train.csv` (verificado hoy) |
| Candidato downweight | `PROMISING_UNCONFIRMED` / `NEGATIVE` / `EXPERIMENTAL_RESEARCH_CLOSED` / `PENDING`; LB externo 0.77522 | `reports/a_cost_sensitive_downweight_final_decision.md`; post-mortem de transferencia |
| Registro | 455 filas; decisiones D-001 a D-287 | `experiments.csv`; `DECISIONS.md` |
| git | `main` @ `e0fcff8`, arbol limpio | `git status` en F0 |

No se modifica ninguno de estos estados. El candidato no se reajusta, no se
generan variantes nuevas y no se demuestra ni refuta con el leaderboard.

## 2. Metodo

Fuentes leidas (inspeccion de codigo y artefactos, no solo nombres):
`PROJECT_STATE.md`, `DECISIONS.md`, `RUNLOG.md`, `FINAL_REPORT.md`,
`experiments.csv`, `reports/` (TOXIC_COMPLETE_RESEARCH_REPORT,
TOXIC_FORENSIC_AUDIT_FINAL, post_p68_remaining_mechanisms, p48_research_gap,
v11_02_auditoria_legalidad, leakage_audit, audit_new_hypotheses,
p57_strategic_closure, p57_open_questions, sugerencias_080,
embedding_*_v5_1, p59_confident_learning, label_consistency_v4,
research_backlog_v4, p66/p67/p68, p84), `docs/competition_rules.md`,
`experiments/` y `notebooks/`.

Criterios: clasificacion de cada familia en `EXECUTED_AND_CLOSED`,
`INVALIDATED`, `DUPLICATE`, `PARTIALLY_EXPLORED`, `GENUINELY_OPEN`,
`BLOCKED_BY_RULES`; y las siete condiciones simultaneas para recomendar
ejecucion (mecanismo nuevo, evidencia, normativa, datos legales, control,
evaluacion sin test/holdout/leaderboard, coste razonable).

## 3. Mapa de familias

| familia / eje | estado | evidencia de cierre |
| --- | --- | --- |
| Umbrales y calibracion | EXECUTED_AND_CLOSED | P55/P56 (anidado honesto -0.00266); V16 (maximo del LB en V03); P76 (calibrar empeora); V8-09 |
| Capacidad / Optuna | EXECUTED_AND_CLOSED | v05 0.8039 -> LB 0.77602; OPT-01 +0.024 -> -0.0070; P61-01; D-057 |
| Meta / stacking / ensambles | EXECUTED_AND_CLOSED | 4 envios de meta a la baja; P61-02 (el control negativo no colapsa: sesgo de seleccion); E2-18; ENS-01..04 |
| Term weighting / NB / features estructurales | EXECUTED_AND_CLOSED | V4-C01..C06; E2-15; V32-05 (delta de LB exactamente 0); V37 |
| Perdidas y ponderacion | EXECUTED_AND_CLOSED | V4-B01..B05; P60 A/J/L; P68-H2 (DUPLICATE de V4-B03); P84-02 |
| Cadenas / powerset | EXECUTED_AND_CLOSED | V4-E01..E03; E2-01 (mejor 0.7744); D-116 |
| SGD / hashing / FTRL / kernels / FM / Tsetlin / BM25 | EXECUTED_AND_CLOSED | p57_family_status; V25/V26/V37/V42/V43; v23 (rechazado por LB) |
| Representacion linguistica (POS, sintaxis, registro) | EXECUTED_AND_CLOSED | P62; P63; V29/V32/V33 |
| Tweet largo y largo alcance | EXECUTED_AND_CLOSED | P63; P64; P65 (la senal existe y la adicion la rechaza) |
| Fusion tardia (MI_arbol) | EXECUTED_AND_CLOSED | P66 (PROMISING no finalista; IC95 incluye 0; 6 de 53 rescates de alta confianza); P67 (residuo MIXED; 4,01 celdas) |
| Enrutado por estructura / seleccion | INVALIDATED / CLOSED | P68-H1 (AUC dificultad 0,608 vs ganador 0,549); P50-P53; P77 (0/5) |
| Ruido de etiqueta / reweighting | EXECUTED_AND_CLOSED | V4-A (2,8 por ciento de contradiccion, no explica el error); V30-00; P59-C (MARGINAL) |
| Retrieval / propagacion por vecinos | CLOSED (techo nulo) | V4-F01..F04; 0 coincidencias literales; sonda v18 con 0 celdas |
| Embeddings desde cero (v5, v5.1, P73-P84) | EXECUTED_AND_CLOSED | v5.1 (3 de 5 condiciones de escalado fallan); P73/P74 (INVALIDATED por contaminacion); P78/P79/P80/P81; P84-02; P84-04 (S vs V03 -0.151120) |
| Downstream secuencial sobre embeddings (H1) | EXECUTED_AND_CLOSED | V9-04 (0,6943 vs 0,6636 de la media); V10-11 (congelado y afinado; "H1 cerrada") |
| Proyeccion supervisada de bajo rango (H3) | EXECUTED_AND_CLOSED | v9 SDR 0,7066 vs SVD 0,6531; V43-02 verdadero -0,01374 |
| Correccion de prevalencia EM (H4) | EXECUTED_AND_CLOSED + BLOCKED | v9 (+0,0006 anidado); V11-07; v16-ch5 (-0,00811); P84-05 (-0,008394); paso B ambiguo de V11-02 |
| Sensibilidad de fugas (H5) | EXECUTED_AND_CLOSED | v9: delta exacto 0,0 |
| RNN/CNN desde cero y DL | EXECUTED_AND_CLOSED | P75; P82 (atencion indistinguible de cero); P84-01 (delta -0,018910) |
| Bayesiano disperso | EXECUTED_AND_CLOSED | P83 (20/20 lecturas negativas; M1 0,7155 descriptivo) |
| Word2Vec de dominio Twitter/X | CLOSED (ejecutada y medida) | Corpus legal autorizado (mmaguero, CC BY-SA 4.0, periodo cerrado, 10.000.029 tokens) y solapamiento cero con train; P72 sin ventaja, P73/P74 INVALIDADOS, P75 sin transferencia, P78 sin transferencia util (brecha 0.117450 vs V03); P84-04 negativo sobre corpus general; ver seccion 6.6 |
| Expansion de datos etiquetados | BLOCKED_BY_RULES | P47: 0 filas legales adicionales; v5.1 senala el tamano etiquetado como unica variable abierta |
| Transformers, LLM, embeddings preentrenados, etiquetas externas, pseudo-labeling | BLOCKED_BY_RULES | `docs/competition_rules.md`; D-002; D-030; D-057 |

**Ninguna familia queda `GENUINELY_OPEN`.** La unica pregunta que sobrevive en
el registro (informacion residual que ningun mecanismo probado supo explotar) no
es una tecnica: es una pregunta sin mecanismo de alcance, y asi queda declarada.

## 4. Via A - Frontera de anotacion de TR

### 4.1 Evidencia disponible

- **D-005.** La definicion oficial ("dirigido contra una persona o grupo") y la
  definicion HatEval (rango del target) son compatibles con los mismos ejemplos;
  los datos muestran TR=1 solo dentro de HS=1 (519/519) y 26 casos AG=1 con
  HS=0. La caracterizacion empirica se hizo; la revision humana quedo pendiente.
- **EDA de la fase 2 (2026-09-22).** El material de revision esta listo y nunca
  se completo: `reports/eda/tr_ejemplos_para_revision.csv` (40 TR=1 + 40 TR=0)
  tiene **las tres columnas humanas vacias en las 80 filas** (verificado hoy), y
  `reports/eda/fase2_tr0_con_referencia_colectiva.csv` conserva los casos limite.
  La evidencia preliminar: TR=1 se asocia a mencion inicial y segunda persona
  (V de Cramer 0,512); TR=0 a referencia colectiva (57,0 por ciento frente a
  10,0 por ciento).
- **V4-A (consistencia de etiquetas).** 15 clusters con 2+ miembros; solo 2 con
  etiquetas contradictorias (2,8 por ciento de los ejemplos). El modelo **acierta
  los ambiguos**: la contradiccion textual no explica el error.
- **V28-03 / P57.** La hipotesis de ambiguedad quedo registrada sin medir; P57 la
  cuantifica como BAJA: techo de 42 celdas no rescatables (0,95 por ciento), por
  debajo de la banda de particion de 0,017.
- **P59-C (confident learning).** Estimador de ruido de etiqueta con protocolo
  propio: MARGINAL; no supera al control de dificultad (AUC 0,9730 frente a
  0,9741), luego ponderar no tendria justificacion mecanica.

### 4.2 Reglas y clasificacion normativa

`docs/competition_rules.md` dice, en lo que aqui importa: **"El dataset de prueba
no debe ser usado para entrenamiento"**. No dice nada sobre revisar o corregir
etiquetas de entrenamiento, y no autoriza anotaciones externas. La revision de
las etiquetas oficiales de desarrollo es analisis de datos permitidos; **la
correccion de etiquetas para reentrenar cambia el objetivo de la tarea y no
tiene cobertura explicita**. Esa ambiguedad es la que bloquea la via: sin
autorizacion escrita del organizador no hay camino legal a una mejora.

Ademas, el uso modelable de la revision (confirmar la lectura de TR para disenar
features de targeting) cae en familias ya cerradas: E2, P62, P63 y P65.

### 4.3 Protocolo propuesto, condicionado a autorizacion

Si el organizador autorizara por escrito revisar y, en su caso, corregir
etiquetas de entrenamiento:

1. Alcance: solo las 1483 filas de desarrollo. Nunca holdout, nunca test.
2. Muestra prioritaria: los 80 ejemplos preparados, los casos limite de
   referencia colectiva y los 2 clusters contradictorios de V4-A.
3. Doble revision humana ciega con acuerdo medido (kappa); los desacuerdos pasan
   a una tercera revision. Ninguna etiqueta se cambia automaticamente.
4. Si se autorizara reentrenar: particion agrupada D-016, gates de siempre,
   identidad V03 como control, sin tocar test ni holdout.
5. Coste: horas de revision humana mas una corrida CPU.

### 4.4 Decision

**`BLOCKED_BY_RULES`.** Motivo: silencio normativo sobre la correccion de
etiquetas; la revision-como-analisis es legal pero su consecuencia modelable
(features de targeting) pertenece a familias cerradas. No se ejecuta.

## 5. Via B - Propagacion de near-duplicados train -> test

### 5.1 Evidencia

- `reports/leakage_audit.md`: 6 pares train-test con similitud >= 0,90 y 10 en
  la banda 0,80-0,90 (maximo 1,0); 4 grupos cruzan train y test (grupos 112:
  2 test / 3 train; 338: 1/3; 1272: 1/1; el grupo 1899 es solo de test). D-002
  registra 17 textos de test con near-duplicados en train.
- **V4-F04**: la propagacion afecta a ~10 tweets de test y su **techo medido es
  nulo**; `spec_compliance_v4.md` lo cierra con "0 coincidencias literales entre
  test y train".
- **Sonda v18** (`toxic_submission_v18_near_dup_propagacion.csv`): preparada y
  **CUARENTENADA** por transductiva; ademas "cambia 0 celdas".
- **C4 de la hoja de ruta 0.80**: <= 48 celdas de 3462, por debajo de la banda de
  deteccion del leaderboard; "Ask first"; nunca ejecutada.

### 5.2 Autorizacion necesaria y procedimiento

Que haria falta: autorizacion escrita del organizador para aplicar una regla de
decision **transductiva** (que depende de la similitud entre textos de test y de
train) sobre predicciones congeladas, sin usar el texto del test para entrenar,
ajustar el vectorizador ni calibrar.

Procedimiento que separaria una evaluacion final legal de un desarrollo
transductivo:

1. La regla se pre-registra y se congela antes de ver nada del test.
2. Se aplica **una sola vez** sobre las predicciones congeladas de V03.
3. El resultado se registra como `EXTERNAL_TRANSFER_RESULT`, nunca como
   instrumento de tuning; no genera variantes ni busquedas.
4. Verificacion manual par a par de los duplicados antes de aplicar.
5. El texto del test no se usa para construir features, vocabulario ni umbrales.

Aun con autorizacion, el techo medido (<= 48 celdas, <= 0,002 de LB, por debajo
de la banda de deteccion) no justifica la apuesta.

### 5.3 Decision

**`QUARANTINED`** (estado legal) y **`BLOCKED_BY_RULES`** (decision): sin
autorizacion suficiente no se ejecuta, y con ella el techo medido la desaconseja.
No se elimina la evidencia historica ni se genera ninguna submission.

## 6. Barrido focalizado de otras vias

### 6.1 Puerta 2: las cinco hipotesis estan ejecutadas y cerradas

`reports/audit_new_hypotheses.md` (auditoria forense) dejo cinco hipotesis; la
fase v9 y las fases posteriores las ejecutaron todas:

| hipotesis | mecanismo | ejecucion | resultado | estado |
| --- | --- | --- | --- | --- |
| H1 | downstream secuencial sobre embeddings propios | V9-04, V10-11 | 0,6943 frente a 0,6636 de la media; congelado y afinado no compiten con V03 | CLOSED |
| H2 | propagacion de near-duplicados train -> test | sonda v18 | 0 celdas cambiadas; cuarentenada por transductiva | QUARANTINED |
| H3 | proyeccion supervisada de bajo rango (PLS/SDR) | v9; V42-01; V43-02; V44-02 | SDR 0,7066 frente a SVD 0,6531; valor verdadero -0,01374 | CLOSED |
| H4 | correccion de prevalencia EM | v9; V11-07; v16-ch5; P84-05 | +0,0006 anidado; -0,00811 externo; -0,008394 en simulacion | CLOSED (uso real ambiguo) |
| H5 | revalidacion de sensibilidad de fugas | v9 | delta exacto 0,0 | CLOSED |

### 6.2 Preguntas abiertas de P57

`reports/p57_open_questions.csv` deja cinco preguntas con mecanismo. Ninguna
cumple el umbral de reapertura; dos tienen veredicto empirico (V13-01 +0,000534;
V15-01 -0,0014) y el resto son restricciones del instrumento o hipotesis
cuantificadas como bajas.

### 6.3 Post-P68: sin mecanismo nuevo

`reports/post_p68_remaining_mechanisms.md`: los ocho ejes del encargo tienen
cita historica, 0 sin recorrer, y la condicion de cierre se cumplio. Lo que
queda abierto es una pregunta (informacion residual que ningun mecanismo supo
explotar; techo oracle +0,0391 no alcanzado por cuatro mecanismos), y para
reabrir haria falta un mecanismo fuera de la lista conocida y una razon de
alcanzabilidad desde el entrenamiento; esa razon no existe hoy.

### 6.4 v5.1, P47, P48/P49

- v5.1: 3 de las 5 condiciones de escalado fallan (curva plana; la cobertura no
  mueve la aportacion). No se escala el corpus.
- P47: no existe fuente legal adicional de filas etiquetadas.
- P48/P49: `EDA_RESEARCH_GAP = NONE`; ninguno de los 20 canales del pool tiene
  net positivo.

### 6.5 Propuestas genericas rechazadas

No se admiten como linea: mas ensembles, mas features, mas tuning, mas
calibracion, mas thresholds, mas modelos clasicos ni otra busqueda de
embeddings. Cada una equivale a una familia `CLOSED` o `DUPLICATE` del mapa de la
seccion 3.

### 6.6 Hipotesis Twitter/X (Word2Vec de dominio): decision explicitada

La fila del CSV describia esta hipotesis solo por su primer P0 (cerrado por
indisponibilidad de corpus). La reconstruccion completa de la linea (RUNLOG del
2026-09-29/30, `reports/twitter_word2vec_*`, `docs/p73_provenance_autorizacion_corpus.md`
y secciones 116 a 118 del notebook) fija la decision y sus cinco distinciones:

1. **Corpus generales ya evaluados.** v5/v5.1 (wikipedia, mC4, mixto; escalado
   cerrado en 10M tokens) y P84-04 (SGNS corregido y anclado sobre wikipedia
   10M: -0.151120 frente a V03). Esa variante queda `CLOSED` / `DUPLICATE`.
2. **Corpus genuinamente de Twitter/X.** Si, ejecutado. La linea se reabrio
   (correccion de criterio: la licencia no era una regla) con el corpus
   mmaguero/covid19-spanishtweets (CC BY-SA 4.0, periodo cerrado del 29-mar al
   1-may-2020, sin etiquetas, 361.994 documentos / 10.000.029 tokens) y
   embeddings 100d/200d/300d entrenados desde cero. Resultado: sin ventaja util
   (P72), sin transferencia (P75), senal fragil frente al azar y brecha 0.117450
   frente a V03 (P78). `CLOSED` por medicion.
3. **Fuente legal, no etiquetada y con procedencia verificable.** La del corpus
   ejecutado si: licencia explicita, periodo cerrado, autorizacion escrita en
   `docs/p73_provenance_autorizacion_corpus.md` (resuelve D-246) y solapamiento
   cero con train. La del encargo original (597M de pysentimiento) no: sin
   licencia declarada y con recoleccion por timelines sin periodo acotado; ese
   corpus concreto queda bloqueado.
4. **Riesgo de contaminacion del test.** Declarado como riesgo residual formal y
   no demostrable por regla (tres escenarios improbables). Lo verificado: cero
   solapamiento exacto y normalizado con train, y near-duplicados con Jaccard
   maximo 0.142857. El test no se inspecciona.
5. **Diferencia metodologica.** Corpus de dominio frente a corpus generales;
   corpus pareado Twitter vs general con la misma tokenizacion (EMB-NORMAL-TW,
   que conserva el contenido del hashtag; v5 lo colapsaba); P78 es el control
   anidado limpio del embedding de dominio frente a una matriz aleatoria pareada
   por forma y norma; P84-04 mide el objetivo SGNS corregido sobre el corpus
   general, no sobre el de dominio.

**Incertidumbre documental conservada.** La combinacion objetivo SGNS corregido
x corpus de Twitter/X no se ejecuto; por la directiva vigente, un cambio de
corpus no reabre la familia. No genera linea accionable.

## 7. Decision final: Caso B

Ninguna linea cumple las siete condiciones simultaneas:

| linea | 1 mecanismo | 2 evidencia | 3 normativa | 4 datos | 5 control | 6 evaluacion | 7 coste |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. TR (correccion) | si (pregunta) | parcial y baja | **NO** | si | posible | posible | bajo |
| B. Near-duplicados | no (duplicado) | techo nulo | **NO** | si | posible | posible | bajo |
| H1-H5 y familias | **NO** (ejecutadas) | n/a | n/a | n/a | n/a | n/a | n/a |
| Expansion de datos | si | si | **NO** | **NO** | posible | posible | n/a |

**Conclusion: la investigacion de modelado permanece cerrada y se detiene la
experimentacion.** No se propone hipotesis prioritaria porque no existe una via
defendible; inventar un metodo para evitar el cierre seria el error que esta
auditoria existe para prevenir. El siguiente avance exigiria una fuente de senal
o un mecanismo legal realmente distinto, y hoy no existe ninguno identificado.

> V03 remains the best verified leaderboard result.

## 8. Cumplimiento verificado

```text
V03 FROZEN                      si
test labels = 0                 si
test usado para desarrollo = 0  si
holdout reutilizado = 0         si
leaderboard tuning = 0          si
etiquetas externas = 0          si
pseudo-labeling = 0             si
embeddings preentrenados = 0    si
transformers = 0                si
LLM = 0                         si
submissions nuevas = 0          si
push = 0                        si
commit = 0 durante la auditoria  si (cierre posterior: un unico commit local, sin push)
candidato downweight            intacto (0.77522 registrado)
```

## 9. Fuentes internas citadas

- `docs/competition_rules.md`; `DECISIONS.md` (D-002, D-004, D-005, D-030,
  D-057, D-070, D-116, D-281..D-287); `PROJECT_STATE.md`; `RUNLOG.md`.
- `reports/TOXIC_COMPLETE_RESEARCH_REPORT.md`;
  `reports/TOXIC_FORENSIC_AUDIT_FINAL.md`; `reports/audit_new_hypotheses.md`;
  `reports/post_p68_remaining_mechanisms.md`; `reports/p48_research_gap.md`;
  `reports/v11_02_auditoria_legalidad.md`; `reports/leakage_audit.md`;
  `reports/p57_strategic_closure.md`; `reports/p57_open_questions.csv`;
  `auditoria/sugerencias_080.md`; `reports/embedding_scaling_conclusion_v5_1.md`;
  `reports/p59_confident_learning.md`; `reports/label_consistency_v4_ejemplos.md`;
  `reports/eda_resumen.md`; `reports/eda/tr_ejemplos_para_revision.csv`;
  `reports/eda/fase2_tr0_con_referencia_colectiva.csv`;
  `reports/a_cost_sensitive_downweight_final_decision.md`;
  `reports/a_cost_sensitive_downweight_governance_note.md`;
  `reports/research_backlog_v4.md`; `reports/p66_final_research.md`;
  `reports/p67_final_research.md`; `reports/p68_*`; `reports/p84_*`.
