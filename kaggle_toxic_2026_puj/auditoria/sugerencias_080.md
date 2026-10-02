# Sugerencias para acercarse a 0.80 en el leaderboard externo

Fecha: 2026-10-02. Autor: arquitecto ML/IA (subagente). Destino: orquestador del
proyecto `kaggle_toxic_2026_puj`.

Alcance y límites de este documento:

- Es una **auditoría de recomendación**, no una ejecución. No se entrenó nada, no
  se envió nada, no se tocó el holdout ni el test, y no se modificó ningún archivo
  existente.
- **No se promete alcanzar 0.80.** La evidencia acumulada indica que la meta exige
  un movimiento que ningún candidato permitido tiene hoy por sí solo.
- **V03 remains the best verified leaderboard result.** (LB 0.78306,
  `corrected_cv` 0.7772114525).
- Las cifras que se citan proceden de los artefactos del proyecto listados al
  final; las expectativas se marcan como **juicio del arquitecto**, no como
  medición.

---

## 0. Resumen ejecutivo

1. El objetivo es **+0.01694** de LB sobre V03. D-063 lo sitúa en **1,1 veces el
   ruido** del leaderboard y en unos **21 aciertos netos**; el error estándar del
   LB con 1154 filas registrado en el proyecto es del orden de **0,012**.
2. El historial es inequívoco: **ninguna ganancia local de alta capacidad
   transfirió** (0.0157 a 0.0240 local terminaron en −0.007 a −0.0127 de LB). El
   único movimiento externo positivo del proyecto (+0.00313) fue la optimización
   de umbrales v02 → v03, y esa capa ya está cerrada por P55/P56 y por la curva
   v16 (el máximo del LB está exactamente en la tasa de V03).
3. La auditoría P80–P81 dejó **dos hilos abiertos de verdad**: (a) re-ejecutar
   DL-01 con el protocolo corregido (su 0.7949 usó el pool de 1731 filas, que
   incluye el holdout, y selección de época no anidada), y (b) reentrenar la
   representación desde cero con el objetivo **SGNS corregido** (el artefacto
   congelado promedia el término negativo con peso 1/K, no lo suma, y nunca se
   ancló contra una implementación externa).
4. Los dos candidatos obligatorios restantes están, a juicio de esta auditoría,
   **cerrados en lo esencial**: la corrección de prevalencia EM falló en su
   formulación corregida con umbrales congelados (V11-07) y en el reajuste de
   umbrales por prevalencia estimada (v16-ch5, −0.00811 externo); el
   meta-ensamble restringido es la familia mejor medida y la que más veces
   transfirió a la baja (4 envíos, ninguno por encima de V03).
5. La recomendación es gastar un presupuesto pequeño y ordenado: **primero el
   SGNS corregido con ancla externa**, **segundo DL-01 con protocolo corregido**,
   **tercero un cierre barato de la capa de decisión (EM + shrinkage) por
   simulación**. Nada de esto promete 0.80; juntos compran información y,
   en el mejor caso, un candidato finalista honesto.
6. La única vía con evidencia publicada de un salto grande está **fuera de las
   reglas vigentes** (transformers, embeddings preentrenados, etiquetas
   externas); queda documentada como NO EJECUTABLE.

---

## 1. Estado y techo medido

### 1.1 Cifras duras

| Elemento | Valor | Fuente |
| --- | --- | --- |
| Campeón V03 (LB externo) | **0.78306** | `submissions/leaderboard_log.csv` |
| V03 `corrected_cv` (referencia congelada) | **0.7772114525** | `docs/p78_preregistro.md`; D-016 |
| Objetivo | ≥ 0.80 | D-063 |
| Salto necesario | **+0.01694** ≈ 1,1 × ruido del LB | D-063 |
| Salto para 0.81 | +0.027 ≈ 1,8 × ruido | D-063 |
| Error estándar del LB (n=1154) | ≈ 0.01219 | registro de v08_p1 |
| Banda de la métrica corregida entre particiones equivalentes | ≈ 0.017 | D-070 (citado en el estado de familias) |
| Envíos registrados con LB | 19–20 sondas; mejor = V03 | `leaderboard_log.csv` |
| Corpus externo no etiquetado | 42 808 docs / 95.0M tokens, 0 contaminación con test | `reports/p47_*`, manifiesto v5 |
| Filas etiquetadas disponibles | 1731 oficiales; 0 filas legales sin usar | P47 |
| Holdout v2 | registro con 5 evaluaciones, más usos no registrados; **agotado** | `auditoria/revision_nucleo.md` A5; D-016 |

### 1.2 El techo medido, en una frase

El proyecto ya tocó el techo de lo que su protocolo local puede prometer: el
mejor OOF honesto de una candidata fue 0.7904 (SVD 300 + RBF, 5/5 pliegues,
McNemar p=0.0452) y el LB cayó −0.00368; el OOF más alto (0.8039, Optuna) cayó
−0.0070; el meta con mejor holdout (+0.0180) cayó en los cuatro envíos. La banda
completa de 19 sondas con local y LB medidos es 0.7703–0.7831, y el gap
local-LB mediano es ≈ −0.0037, con los casos de mayor capacidad en el extremo
peor del gap. **El instrumento local tiene una resolución de ≈0.017 (D-070), es
decir, el objetivo entero de la fase cabe dentro del ruido del instrumento.**

### 1.3 El cuello de botella documentado

E00 y v16-00 fijan el hecho central: **el test tiene ~7 % menos positivos que
train** (factor k ≈ 0.93), y la curva v16 del LB contra la tasa predicha de
positivos tiene su **máximo en V03** (0.3645). Cualquier candidato que mueva la
tasa de positivos hacia arriba o hacia abajo pierde (v16-ch1 −0.00478,
v16-ch5 −0.00811). El cuello de botella no es de representación: es de
**distribución y de decisión**, y la decisión de V03 ya está en el óptimo
observado de esa curva.

---

## 2. Lecciones de la auditoría que condicionan cualquier propuesta

- **L1. Las ganancias locales de capacidad no transfieren.** Tres mediciones
  independientes: R1-C1 +0.0157 → −0.0108; R2-MINDF5 +0.0221 → −0.0127;
  OPT-01 +0.0240 → −0.0070 (D-057). Y el mejor perfil interno del proyecto
  (+0.0132, 5/5, McNemar) → −0.00368 (v23-01). Cualquier candidato que aumente
  dimensión o brecha ajuste-validación debe pasar el freno de D-057 antes de ser
  finalista.
- **L2. La tasa de positivos ya está en el óptimo externo.** No se propone nada
  que suba o baje la tasa predicha de V03 salvo que lo haga por reasignación
  (mismo número de positivos) y con evidencia propia.
- **L3. El holdout está agotado y contaminado.** Cinco evaluaciones registradas
  más al menos un uso no registrado (v33-01) y usos literales de las 248 filas
  como entrenamiento (`18_p7`, `25_submissions`). El orquestador resume los usos
  reales en 6 de 5. **No se evalúa nada más en el holdout y no se crea otro.**
- **L4. El 0.7949 de DL-01 no es comparable con V03.** P80 A18/A19
  CONFIRMADOS: el pool de CV de la fase DL era de 1731 filas (1483 + 248) y la
  época se elegía sobre el mismo pliegue que se puntuaba. Su número es optimista
  por dos causas independientes y no puede leerse como mejora.
- **L5. El artefacto Word2Vec congelado no implementa SGNS.** El término
  negativo pesa 1/K en lugar de sumarse K veces (P79 confirmado en
  `p71_04:214-216`); P81 refutó la dirección "suma" en el control plantado, y el
  **ancla contra una implementación externa quedó SIN HACER** por declaración
  explícita del propio informe.
- **L6. El meta es la familia mejor medida y la peor transferente.** S08 +0.0180
  en holdout; cuatro envíos por debajo de V03, el mejor a −0.00156. La métrica
  local no puede resolver deltas de ese tamaño (D-070).
- **L7. La calibración es portante al revés.** P76: calibrar V03 lo empeora de
  0.7772 a 0.7686; su sobreconfianza sostiene el punto de operación.
- **L8. La estimación es extremadamente inestable entre particiones.** D-064:
  desacuerdo medio 0.9175/0.9671/0.9256 (HS/TR/AG); solo 2 de 1483 filas tienen
  las tres etiquetas con desacuerdo < 0.4. **Un delta local de una sola
  partición por debajo de ~0.005 es ruido de partición.**

---

## 3. Candidatos permitidos, rankeados

Ranking por **valor de información y probabilidad ajustada por riesgo**, no por
magnitud prometida. Ninguno promete 0.80.

| # | Candidato | ¿Viola reglas? | Coste | Expectativa honesta |
| --- | --- | --- | --- | --- |
| C1 | SGNS corregido desde cero + ancla externa + downstream | No (D-030) | 1–2 días | Baja; cierra el hilo P80–P81 |
| C2 | DL-01 con protocolo corregido (pool 1483 + época anidada) | No | <1 día | Baja–media de pasar el gate; cierra la cifra 0.7949 |
| C3 | EM + shrinkage en la capa de decisión, por simulación OOF | No (sin test) | <1 día | Muy baja; cierre barato |
| C4 | Propagación de near-duplicados train → test (H2) | Dudosa ("Ask first") | <1 día | Muy baja; techo ≤ 0.002 de LB |
| C5 | Meta restringido con features de shift | No | 1 día | Muy baja; no recomendado |
| C6 | Repetición corregida P7 (5 modelos, umbrales por cuantil) | No | 1 día + 1 envío | Muy baja; no recomendado como apuesta |

### C1. Reentrenar la representación desde cero con el objetivo SGNS corregido

- **Mecanismo.** El entrenador actual agrega el término negativo con peso 1/K
  (promedio) en lugar de sumar K contribuciones, que es la ecuación (4) de
  Mikolov et al. (2013) y la ecuación (1) de Levy & Goldberg (2014). Corregir
  esa agregación cambia la matriz implícita que el embedding factoriza (PMI
  desplazada por −log k): no es un reajuste cosmético, es otro objetivo.
  Después se re-mide downstream con el protocolo congelado y se contrasta contra
  una implementación externa de referencia.
- **Por qué no está cerrado.** P80/P81 CONFIRMAN el defecto y **dejan sin hacer**
  (i) la corrida con el objetivo corregido y (ii) el anclaje contra una
  implementación externa. La familia "embeddings" está cerrada por resultados
  del objetivo defectuoso, no por el objetivo correcto. Es el único hilo del
  proyecto con un bug confirmado y una medición pendiente.
- **Protocolo falsable (sin test, sin holdout).**
  1. Pre-registro con sha256, antes de correr nada.
  2. Corpus EXTERNAL-ONLY del manifiesto v5 (95.0M tokens, 0 contaminación);
     nada nuevo se descarga sin registrar.
  3. Tres brazos con idéntico vocabulario y corpus: (A) SGNS corregido (suma),
     (B) implementación actual (1/K) como control, (C) embeddings aleatorios
     pareados (control negativo estilo P78).
  4. Ancla externa: `fastText` C++ oficial compilado localmente (autorizado por
     D-030) sobre un corpus plantado con familias separables, más una
     comprobación numérica del objetivo sobre pares plantados. El ancla se
     aprueba si el objetivo corregido es monótono en la estructura plantada y el
     control 1/K no lo es.
  5. Downstream fold-local sobre las 5 particiones canónicas del desarrollo
     (1483), cabeza (i) media + LR y (ii) la misma cabeza de P73/P82 solo si la
     primera da señal. Embeddings aprendidos **solo** con corpus externo; sin
     etiquetas del desarrollo dentro del embedding (la variante in-fold, si se
     quiere, se declara aparte).
  6. **Gate:** Δ mayor o igual a +0.005 sobre V03 con al menos 4/5 pliegues e IC
     bootstrap pareado que excluya cero; o, alternativamente, `ensemble_gain`
     mayor o igual a 0.0015 con IC > 0 (criterio de complementariedad de D-016).
     El brazo A debe superar a B y a C en la misma dirección.
- **Coste.** El corpus ya está en disco y auditado; el entrenamiento del
  embedding es CPU multihilo (32 hilos) y las cabezas usan la RX 9070. Del orden
  de 1–2 jornadas con el ancla y los brazos.
- **Riesgo.** El nivel absoluto histórico de los modelos basados en W2V es
  ~0.60–0.65, muy por debajo de V03; aunque el objetivo corregido mejore el
  embedding, la vía más probable es el cierre documentado, no la mejora. Riesgo
  de fuga: bajo por diseño (corpus verificado, fold-local, sin etiquetas
  externas).
- **Expectativa honesta (juicio).** Probabilidad baja de superar el gate local
  (~10–20 %); probabilidad aún menor de que un candidato así supere el ruido del
  LB. Su valor principal es **cerrar P80–P81 con una medición en lugar de con una
  inferencia** y descartar definitivamente que la representación sea el cuello
  de botella.

### C2. Re-ejecutar DL-01 (MLP sobre SVD 300) con protocolo corregido

- **Mecanismo.** El MLP TF-IDF union → SVD 300 → 256-128 es la **única cifra
  interna del proyecto en el rango del objetivo** (0.7949, +0.0177 sobre V03).
  Re-medirla con (a) pool de desarrollo de 1483 filas (sin las 248 del holdout)
  y (b) selección de época **anidada dentro del pliegue de entrenamiento**
  convierte una cifra contaminada en una cifra comparable.
- **Por qué no está cerrado.** P80 A18/A19 CONFIRMADOS: pool de 1731 y época
  elegida sobre el pliegue evaluado. La cifra corregida **nunca se midió**.
  Además, `reports/audit_new_hypotheses.md` la usa como evidencia de "capacidad
  explotable"; mientras no se re-mida, contamina las decisiones.
- **Protocolo falsable.**
  1. Réplica exacta de la arquitectura y configuración históricas (110 339
     parámetros, SVD 300, MLP 256-128, AdamW lr 1e-3, wd 1e-4, batch 32,
     dropout 0.3, clip 1.0, `pos_weight` por etiqueta, paciencia 5).
  2. Dos correcciones de protocolo: pool = 1483; partición interna agrupada
     dentro de `outer_train` para elegir la época. Nada de early stopping sobre
     el pliegue evaluado.
  3. Semillas 42/43/44 y tres adicionales; media ± desviación, pliegues
     mejorados, IC bootstrap pareado y perfil de celdas (rescates/daños).
  4. Umbrales de V03 congelados (primario); umbral por etiqueta anidado solo
     como brazo secundario declarado.
  5. **Gate:** Δ mayor o igual a +0.005 y al menos 4/5 pliegues e IC que excluya
     cero. Si lo pasa, se aplica el **freno de D-057**: un candidato > +0.005
     solo llega a finalista si no aumenta dimensión ni la brecha
     ajuste-validación (la brecha histórica es +0.185, y el precedente
     SVD+RBF del proyecto terminó en −0.00368 de LB).
- **Coste.** La corrida histórica completa tomaba ~19 s; con partición interna y
  6 semillas, menos de una jornada de CPU/GPU.
- **Riesgo.** El riesgo principal no es el coste, es la **mala lectura** de un
  resultado local positivo: el proyecto ya pagó ese error con v23-01. La regla
  debe quedar escrita antes de correr: aunque pase el gate local, un MLP con
  brecha +0.18 no se envía sin control de capacidad y sin una segunda partición.
- **Expectativa honesta (juicio).** Es probable que la cifra corregida caiga por
  debajo de V03 (las dos causas de optimismo actuaban en la misma dirección).
  Probabilidad moderada-baja de que aun así supere +0.005; en ese caso tendría
  derecho a **una** sonda pre-registrada, no a la fase entera. No promete 0.80.

### C3. Corrección de prevalencia EM con shrinkage en la capa de decisión

- **Mecanismo.** Estimar por EM de Saerens la prevalencia del destino sobre
  probabilidades sin etiquetas, aplicar la corrección de odds en logit
  (`logit(p) + logit(pi_destino) − logit(pi_origen)`) **encogida** hacia cero
  (shrinkage m o prior jerárquica), y decidir con umbrales que maximicen la F1
  esperada bajo un intervalo de prevalencia (robusto), no bajo un punto.
- **Por qué no está cerrado (parcialmente).** V11-06 validó el estimador (error
  relativo medio 0.69 %; clasificar-y-contar exacto bajo shift puro), pero
  V11-07 midió la corrección con umbrales congelados bajo shift simulado y dio
  **negativa** (HS −0.001032; TR 0.000000; AG +0.000154; ningún gate superado), y
  el informe cierra la vía con la frase "no se reintenta con más variantes de la
  corrección". V9-02 (EM sobre test, con umbral reoptimizado) había dado una
  ganancia anidada de solo +0.000574 (HS +0.00284, TR −0.00084, AG −0.00028), y
  v16-ch5 reajustó umbrales a la prevalencia estimada (k=0.9296) y perdió
  −0.00811 en el LB. Lo único no medido es el **shrinkage explícito con decisión
  robusta**, que es una variante matemática distinta pero no un mecanismo nuevo.
- **Protocolo falsable (solo simulación sobre OOF; sin test).**
  1. El test no se usa ni para estimar ni para decidir: el shift se simula sobre
     el OOF del desarrollo por reponderación de filas (como V11-06/V11-07).
  2. Rejilla pequeña y pre-registrada: m de shrinkage en {0, 25, 50, 100};
     niveles de shift en {0.85, 0.90, 0.94, 1.00, 1.05} por etiqueta; umbral
     robusto que minimice la F1 esperada en el peor punto del intervalo de
     prevalencia estimado.
  3. **Gate:** Δ mayor o igual a +0.002 con IC pareado que excluya cero y 4/5
     repeticiones antes de siquiera considerar una sonda. Además, la tasa
     predicha de positivos debe quedar a ±0.005 de la de V03 (L2): una mejora
     que suba la tasa no es admisible con la curva v16 en la mano.
  4. Si no pasa, la familia se cierra **definitivamente** y no se reintenta con
     más variantes (coherente con V11-07).
- **Coste.** Menos de una jornada de CPU; no requiere modelo nuevo.
- **Riesgo.** Reabrir una línea cerrada multiplica las comparaciones y el riesgo
  de falso positivo; por eso el pre-registro debe fijar una sola corrida y una
  sola rejilla.
- **Expectativa honesta (juicio).** Negativa. Probabilidad muy baja (<10 %) de
  superar el gate, y aunque lo superara, el efecto esperado es de milésimas, muy
  por debajo del ruido del LB. **No hay camino a 0.80 por aquí.**

### C4. Propagación de near-duplicados train → test (H2), como control

- **Mecanismo.** Para las filas de test con near-duplicado en train (6 pares con
  similitud ≥ 0.90 y 10 en la banda 0.80–0.90 según el EDA), sustituir la
  predicción por la etiqueta de la fila de train cuando el duplicado es
  unánime.
- **Por qué no está cerrado.** Los near-duplicados se usaron solo para agrupar la
  validación; nunca como regla de predicción. Es una sonda barata y de techo
  conocido.
- **Protocolo.** Verificación manual par a par; propagar solo duplicados
  unánimes; medir el efecto máximo posible (≤ 48 celdas de 3462, es decir, muy
  por debajo de la banda de detección del LB) y comparar contra el control de no
  hacer nada.
- **Riesgo y legalidad.** Usa texto de test en el momento de decidir; no usa
  etiquetas de test ni entrena con test, pero la regla "el dataset de prueba no
  debe ser usado para entrenamiento" exige una lectura del organizador. **Ask
  first.**
- **Expectativa honesta.** ≤ +0.001–0.002 de LB en el mejor caso, probablemente
  0. No cierra la brecha de 0.017. Se incluye por rigor (dejar el EDA sin
  cerrar), no como apuesta.

### C5. Meta-ensamble restringido con features de shift

- **Mecanismo.** Un meta de segundo nivel sobre los scores de V03 más features
  que describan el desplazamiento (densidad local train/test, longitud,
  marcadores de registro, entropía), con pesos ajustados de forma anidada.
- **Por qué no está cerrado.** Literalmente, "features de shift" no estaban en
  el meta de 44 columnas ni en los metas restringidos EMO/SYN. Es una clase de
  feature nueva.
- **Por qué no se recomienda ejecutarlo.** (i) El meta es la familia mejor
  medida y la que más veces transfirió a la baja: S08 +0.0180 en holdout y
  cuatro envíos por debajo de V03 (mejor: −0.00156); (ii) la banda del
  instrumento (0.017) es mayor que cualquier delta esperado, de modo que el
  resultado local no sería interpretable; (iii) con 1483 filas, añadir features
  de segundo nivel aumenta la varianza más rápido que la señal; (iv) el
  precedente de construcciones de segundo nivel es negativo (v20-01 cross-score,
  −0.00643). Es una clase de feature nueva dentro de una **clase de mecanismo
  agotada**.
- **Si el orquestador insiste (protocolo mínimo).** Pesos y features anidados;
  control con features barajadas (como P66-02) para atribución; gate Δ mayor o
  igual a +0.005, 4/5 pliegues, IC que excluya cero y tasa predicha a ±0.005 de
  V03; una sola sonda si todo se cumple. Expectativa: negativa.
- **Veredicto.** **No recomendado**; no asignar presupuesto en esta fase.

### C6. Repetición corregida de P7 (5 modelos, uno por partición)

- **Mecanismo.** Promediar las probabilidades de cinco particiones repetidas
  (una por repetición, no 25 modelos) con umbrales fijados por cuantil para que
  la contracción de probabilidades no mueva celdas.
- **Por qué no está cerrado.** El envío v12-p7 promedió 25 modelos (5 pliegues ×
  5 repeticiones) y no el ensemble de 5 que se validó localmente (+0.005066,
  4/5 pliegues); hay un error de diseño reconocido en el propio registro.
- **Veredicto.** **No recomendado como apuesta.** El promedio de modelos de
  pliegue ya perdió (−0.00836, v08_p1), el IC local incluía cero y la línea
  completa está dentro del ruido. Solo tendría sentido como **una** sonda de
  cierre si C1 y C2 fallan y el orquestador quiere liquidar la línea de
  varianza de partición con identidad exacta entre lo validado y lo enviado.

---

## 4. Descartados con motivo (para no repetirlos)

| Familia | Evidencia de cierre |
| --- | --- |
| Capacidad / Optuna | v05 0.8039 → 0.77602; OPT-01 +0.024 → −0.0070; D-057 |
| Umbrales (reoptimización) | P55/P56: el anidado honesto es −0.00266; reoptimizar sobre el mismo OOF infla +0.0036; V16: máximo del LB en V03 |
| Calibración | P76: la mejor configuración −0.00177; calibrar V03 lo empeora 0.0086; V11-07 |
| Selección contextual (P50–P53) | Techo oráculo +0.2098, pero ninguna política observable supera a V03; el espacio real de acciones cae a 2 |
| Operador selectivo (P77) | 0/5 pliegues, IC excluye cero, TR daña más |
| TextCNN / Word2Vec (P73–P75) | Signo estable, magnitud inestable; complementariedad negativa; P78 +0.0249 con IC que contiene cero |
| RNN con atención (P82) | B1 0.6397 local → 0.6815 LB; por debajo de V03 y con gap invertido por desajuste de dominio |
| Bayesiano (P83) | 20/20 lecturas negativas; M1 0.7155 descriptivo |
| PLS / embeddings supervisados | V42-01/V43-01 invalidados por implementación; V43-02 verdadero −0.01374 |
| Ruido de etiqueta / reweighting | D-026; V30-00; el +0.0036 era un error de comparación (D-072) |
| Aumentación de texto | V6-08 −0.0061; V41-01 cierra ortográfica y mixup; la literatura de aumentación no ofrece mecanismo nuevo aplicable (Bayer et al. 2022) |
| Etiquetas conjuntas | D-116: estructura real pero no explotable |
| Retrieval / propagación por vecinos | Techo medido nulo antes de gastar (V4-F01–F04) |
| Expansión de datos | P47: no existe fuente legal adicional de filas etiquetadas |
| Features estructurales | V32-05: Δ de LB exactamente 0 |
| Ensambles simples | Tres envíos negativos; la contracción mueve celdas |
| FM / Tsetlin / BM25 / hashing / SGD | `reports/p57_family_status.csv`, cierres tipo A |

---

## 5. Candidatos que exigirían cambiar las reglas: NO EJECUTABLES hoy

1. **Transformers o LLM (BERT, BETO, RoBERTa, mDeBERTa, ChatGPT, etc.).**
   Prohibidos por `docs/competition_rules.md` y ratificados por D-057. La
   literatura de odio en español (HatEval; Pérez, Miguel y Cotik 2025) muestra
   que los codificadores preentrenados afinados son la referencia fuerte, pero
   quedan fuera. **NO EJECUTABLE.**
2. **Embeddings preentrenados** (word2vec/fastText/GloVe preentrenados,
   sentence-transformers). Prohibidos por D-030 y por la regla "entrenarse desde
   cero". **NO EJECUTABLE.**
3. **Pseudo-etiquetado del test o de corpus externo.** Prohibido por D-030 y por
   la regla de no usar el test para entrenamiento; además, la revisión de estado
   del arte ya documentó que las variantes de self-training dependen de
   pseudo-etiquetado. **NO EJECUTABLE.**
4. **Etiquetas externas** (HatEval train, DETOXIS u otras). Prohibidas por D-002
   y D-030. **NO EJECUTABLE.**
5. **Reutilizar las 248 filas del holdout para entrenar o evaluar.** El holdout
   está agotado, con usos no registrados y usos de entrenamiento detectados; no
   admite más evaluaciones y no se crea otro. **NO EJECUTABLE.**
6. **Ajustar la decisión con las probabilidades del test** (EM real, umbrales
   estimados sobre test, calibración sobre test) sin pasar la auditoría de
   legalidad de v11-02. La postura vigente del proyecto es estudiarlo **solo por
   simulación sobre el OOF**. **Ask first / hoy no ejecutable.**

---

## 6. Recomendación: las tres primeras, con orden y presupuesto

| Orden | Candidato | Presupuesto | Gate que debe superar | Política de envío |
| --- | --- | --- | --- | --- |
| 1 | C1. SGNS corregido + ancla externa + downstream | 1–2 jornadas (CPU 32 hilos + GPU RX 9070) | Δ ≥ +0.005, ≥4/5 pliegues, IC > 0, o `ensemble_gain` ≥ 0.0015 con IC > 0; A > B > C | Si supera: 1 sonda pre-registrada. Si no: cierre documentado de la familia |
| 2 | C2. DL-01 protocolo corregido | <1 jornada | Δ ≥ +0.005, ≥4/5 pliegues, IC que excluya cero; freno D-057 sobre brecha/dimensión | Si supera el gate **y** el freno: 1 sonda. Si no: actualizar la narrativa (0.7949 = artefacto) |
| 3 | C3. EM + shrinkage en decisión, simulación OOF | <1 jornada | Δ ≥ +0.002, IC > 0, 4/5 repeticiones, tasa ±0.005 de V03 | No genera sonda salvo que pase todo; en la práctica, cierre |

Reglas transversales de estos tres:

- Pre-registro con sha256 antes de cada corrida; una sola rejilla y una sola
  lectura por candidato (control de multiplicidad).
- `corrected_cv` sobre el desarrollo (1483) como métrica primaria y referencia
  V03 = 0.7772114525; holdout y test intactos.
- Identidad de pipeline verificada antes de cualquier envío (el proyecto ya sabe
  cómo: comparación celda a celda contra el OOF canónico).
- Cualquier resultado local positivo se interpreta contra la banda de 0.017
  (D-070): +0.005 local **no** es una promesa de +0.005 externo.
- **No se promete alcanzar 0.80.** Con la evidencia de 19 sondas, la expectativa
  honesta de que este paquete produzca un candidato que cierre +0.01694 es baja.
  El paquete compra información y, si algo pasa los gates, un finalista con
  derecho a una sonda; no compra la meta.

Secuencia de decisión:

1. Si C1 o C2 produce un Δ ≥ +0.005 estable (≥4/5, IC > 0, dos particiones):
   congelar, verificar identidad y **una** sonda de LB pre-registrada.
2. Si ninguno lo produce: no gastar más sondas en familias cerradas; el
   entregable honesto es el notebook con el mapa completo de cierres y la
   auditoría P80–P81.
3. C4 y C6 solo como controles de cierre, nunca como apuestas.
4. C5 no se ejecuta.

---

## 7. Referencias verificables

Verificadas en la sesión del 2026-10-02 (título, autores, año, venue, URL/DOI):

1. Saerens, M., Latinne, P., Decaestecker, C. (2002). *Adjusting the Outputs of a
   Classifier to New a Priori Probabilities: A Simple Procedure*. Neural
   Computation 14(1):21–41. DOI 10.1162/089976602753284446.
2. Alexandari, A., Kundaje, A., Shrikumar, A. (2020). *Maximum Likelihood with
   Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation*. ICML
   2020, PMLR 119:222–232. URL https://proceedings.mlr.press/v119/alexandari20a.html
   (arXiv:1901.06852).
3. Lipton, Z. C., Elkan, C., Naryanaswamy, B. (2014). *Optimal Thresholding of
   Classifiers to Maximize F1 Measure*. ECML PKDD 2014, LNCS 8725:225–239.
   DOI 10.1007/978-3-662-44851-9_15.
4. Forman, G. (2008). *Quantifying counts and costs via classification*. Data
   Mining and Knowledge Discovery 17(2):164–206. DOI 10.1007/s10618-008-0097-y.
5. Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M.,
   Rosso, P., Sanguinetti, M. (2019). *SemEval-2019 Task 5: Multilingual
   Detection of Hate Speech Against Immigrants and Women in Twitter*. SemEval
   2019, ACL Anthology S19-2007.
6. Bayer, M., Kaufhold, M.-A., Reuter, C. (2022). *A Survey on Data Augmentation
   for Text Classification*. ACM Computing Surveys 55(7):1–39.
   DOI 10.1145/3544558.
7. van der Laan, M. J., Polley, E. C., Hubbard, A. E. (2007). *Super Learner*.
   Statistical Applications in Genetics and Molecular Biology 6(1):25.
   DOI 10.2202/1544-6115.1309.
8. Levy, O., Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix
   Factorization*. NIPS 27:2177–2185. URL
   https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization

Citadas desde la revisión ya verificada del propio proyecto:

9. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013).
   *Distributed Representations of Words and Phrases and their Compositionality*.
   arXiv:1310.4546 (usada por P81 para la ecuación (4)).
10. Castillo-López, G., Riabi, A., Seddah, D. (2023). *Analyzing Zero-Shot
    transfer Scenarios across Spanish variants for Hate Speech Detection*.
    VarDial 2023. DOI 10.18653/v1/2023.vardial-1.1.
11. Pérez, J. M., Miguel, V., Cotik, V. (2025). *Exploring Large Language Models
    for Hate Speech Detection in Rioplatense Spanish*. Findings NAACL 2025.
    DOI 10.18653/v1/2025.findings-naacl.400.

---

## 8. Preguntas abiertas y qué queda sin hacer

- **El ancla externa del SGNS sigue sin hacerse** hasta que se ejecute C1. Es la
  única forma de saber si el objetivo corregido es realmente mejor y no solo
  distinto.
- **La cifra corregida de DL-01 sigue sin medirse**; mientras tanto, cualquier
  documento que la use como "capacidad explotable" debe llevar la advertencia de
  P80.
- **La legalidad de la propagación de near-duplicados** (C4) requiere respuesta
  del organizador antes de generar cualquier submission.
- **La formulación robusta de la corrección de prevalencia** (shrinkage +
  decisión minimax sobre un intervalo) no se ha medido; se propone solo como
  cierre barato.
- **No hay ninguna línea permitida con evidencia publicada de un salto de
  +0.017.** Las que la tienen (transformers, preentrenados) están prohibidas.
- Sin hacer en esta auditoría: no se ejecutó ningún script, no se generó ninguna
  submission, no se tocó el holdout, no se modificó ningún archivo existente y
  no se hizo commit.
