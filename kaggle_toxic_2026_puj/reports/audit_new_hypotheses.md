# Puerta 2 - Nuevas hipotesis tras la auditoria (KAGGLE TOXIC 2026 PUJ)

Regla aplicada: solo se conserva una hipotesis si responde "que hace este
mecanismo que ninguno de los experimentos anteriores hace". Categorias:
A = genuinamente no probada; B = parcialmente probada; C = probada con
protocolo defectuoso; D = probada y cerrada correctamente. Solo A y C son
candidatas. Sin transformers, sin LLM, sin recursos externos nuevos, sin
pseudo-etiquetado y sin usar test/holdout para decidir.

## H1 - Downstream SECUENCIAL sobre los embeddings propios (categoria A)

- Mecanismo: alimentar los vectores FastText entrenados desde cero (10M-50M) a
  un modelo con estructura de secuencia (BiGRU/CNN 1D sobre tokens) en lugar del
  pooling por media + modelo lineal que se uso en toda la fase v5.
- Por que no se ha probado: el hallazgo del §24 de la auditoria; v5 solo uso
  agregacion por media (mean pooling) + LR/Ridge; la DL de v2 entreno embeddings
  dentro de la red (otro mecanismo) y las LSTM/CNN de v2 no vieron FastText.
- Evidencia historica: curva de escala plana (5M-50M), complementariedad nula con
  V03 en pooling; DL-01 (MLP sobre SVD) alcanzo 0.7949 OOF en el protocolo
  historico, senal de que hay capacidad explotable.
- Efecto esperado: cerrar la pregunta "la representacion o el uso de la
  representacion"; magnitud incierta (probablemente pequena).
- Complementariedad esperada: baja-media con V03 (familia de error distinta).
- Riesgo de leakage: bajo (corpus verificado limpio; protocolo corregido).
- Coste: GPU moderado (minutos por variante).
- Protocolo: OOF corregido + gates D-016; sin sondas LB salvo adopcion.
- Modo de fallo: sobreajuste con 1731 filas (regularizacion fuerte y early
  stopping estricto obligatorios); si no supera el gate, cierra la familia.
- Prioridad: media.

## H2 - Propagacion de etiquetas train->test por near-duplicados (categoria A)

- Mecanismo: para las filas de test con similitud >= 0.85/0.90 contra filas de
  train (6 pares >= 0.90 y 10 en banda 0.80-0.90 documentados en el EDA),
  sustituir la prediccion del modelo por la etiqueta de la fila de train copiada
  (propagacion determinista con regla de desempate).
- Por que no se ha probado: los near-duplicados se usaron SOLO para agrupar la
  validacion; nunca como mecanismo de prediccion.
- Evidencia historica: reports/eda/similitud_resumen.csv (pares train-test con
  similitud 1.0); E00 sugiere ~7% menos positivos en test.
- Efecto esperado: muy pequeno (afecta <=16 filas de 1154; <=0.2 %), pero medible
  en LB si coincide con el sesgo de copias.
- Complementariedad: n/a (es una capa de decision sobre V03).
- Riesgo de leakage: ninguno por etiquetas externas; es informacion de TRAIN
  aplicada a test (transductivo estructural permitido: no usa las etiquetas de test).
- Coste: minutos (sin reentrenar).
- Protocolo: aplicar a la submission campeona (V03) con verificacion de filas
  afectadas y comparacion local del subconjunto; una sonda LB opcional.
- Modo de fallo: que las copias esten etiquetadas distinto (revisar manualmente
  los pares antes de aplicar).
- Prioridad: alta como sonda barata; baja como fuente de mejora estructural.

## H3 - Representacion supervisada de bajo rango (PLS/SVD supervisada) (categoria A)

- Mecanismo: comprimir el TF-IDF union con una proyeccion que use Y dentro del
  pliegue (PLS-DA o SVD de la matriz centrada por clase), y entrenar LR sobre
  pocas componentes; contrasta "direcciones de varianza" (SVD no supervisada, ya
  usada en v2 DL) contra "direcciones discriminantes".
- Por que no se ha probado: v2 uso SVD NO supervisada + MLP; v5.1 vario el
  embedding, no la compresion del TF-IDF; nunca se probo una proyeccion
  supervisada fold-local como entrada del modelo lineal.
- Evidencia historica: la familia de capacidad (optuna) sobreajusta sin
  transferir; una compresion supervisada podria reducir varianza manteniendo
  sesgo bajo (hipotesis falsable).
- Efecto esperado: pequeno; riesgo de sobreajuste por usar Y (por eso fold-local
  estricto y comparacion contra PCA del mismo rango).
- Complementariedad: posible con V03 (errores por varianza).
- Riesgo de leakage: medio si el ajuste no es fold-local; protocolo obligatorio.
- Coste: bajo (CPU).
- Protocolo: OOF corregido; control = SVD no supervisado del mismo rango.
- Modo de fallo: PLS sobreajusta con 1483 filas; el control lo detecta.
- Prioridad: media-baja.

## H4 - Correccion de prior shift por EM (Saerens) sobre las probabilidades finales (categoria C)

- Mecanismo: estimar la prevalencia del test por EM sobre las probabilidades de
  V03 (sin etiquetas de test) y ajustar los umbrales para maximizar la F1
  esperada bajo esa prevalencia; variante por etiqueta y con shrinkage.
- Por que es C (protocolo defectuoso historico): la familia "shift" se cerro en
  v4 (D-028) midiendo el shift de calibracion del META y con recuperacion por
  cuantiles/Platt sobre el meta; nunca se aplico una correccion de prevalencia EM
  al modelo final V03 con protocolo anidado.
- Evidencia historica: E00 (~7 % menos positivos en test); v05/v07 sobrepredicen
  y pierden en LB ("predecir por encima de la prevalencia del test penaliza");
  v7 midio que los umbrales estan debilmente identificados.
- Efecto esperado: puede recuperar parte del gap local-LB (-0.016) sin cambiar el
  modelo; techo acotado por la varianza del estimador de prevalencia.
- Complementariedad: n/a (capa de decision).
- Riesgo de leakage: ninguno (usa solo test sin etiquetas; permitido por reglas).
- Coste: muy bajo.
- Protocolo: validar la correccion por simulacion de shift en el OOF corregido
  (reweighting de filas) y con el holdout ya gastado NO; una sonda LB opcional.
- Modo de fallo: el estimador EM se sesga con modelos muy confiados (V03 lo es).
- Prioridad: alta (relacion directa con el cuello de botella documentado).

## H5 - Revalidacion de sensibilidad de las dos fugas corregidas (categoria A de metodo)

- Mecanismo: re-ejecutar V03 con (a) grupos near-dup construidos SOLO con train
  (B-07) y (b) el detector de sospecha fold-local (B-06) no aplica a V03 (solo a
  v6-07), de modo que la unica re-ejecucion necesaria es (a).
- Por que no se ha probado: la auditoria midio particiones identicas (ARI=1.0) y
  predice delta 0.0; falta la ejecucion completa del pipeline con la correccion.
- Evidencia: F-05; audit/p04b.
- Efecto esperado: 0.000000 (prediccion falsable).
- Coste: minutos (ya casi cubierto por P6).
- Protocolo: comparacion exacta de 0.777211.
- Prioridad: media (cierra la seccion de fugas del informe final).

## Descartadas y por que

- kNN supervisado y TextCNN de similitud: cerradas por cota medida (+0.0027, v6).
- Ponderaciones de grupo / bagging / skip-grams / aumento / focal / prior de
  grupo: probadas y cerradas (v6/v7) con controles exactos.
- FastText con mas escala o ejes: cerradas por inercia y curva plana (v5.1).
- Calibracion por tasa/quantiles: medida y refutada (v8 P5-P7, D-064/D-065).
- Pseudo-etiquetado, transformers, corpus externo supervisado, recursos de
  HatEval: prohibidos por las reglas y por D-002/D-030.

## Orden recomendado de la siguiente fase

1. H5 (sensibilidad de fugas; cierra auditoria con evidencia ejecutada).
2. H4 (prior shift EM; unico mecanismo nuevo ligado al cuello de botella local-LB).
3. H2 (sonda barata de near-duplicados test; verificar pares a mano).
4. H1 (secuencial sobre embeddings; cierra la pregunta de representacion).
5. H3 (PLS fold-local; solo si H1 deja capacidad sin usar).
