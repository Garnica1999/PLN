# P84: refutacion adversarial final de las conclusiones de fase

- Fecha: 2026-10-02
- Alcance: conclusiones de cierre de la fase P84 (EXP-01..EXP-06), artefactos
  reports/p84_01..05, DECISIONS D-281..D-286, experiments.csv filas 452-456 y los
  dos QA de fase.
- Metodo: solo lectura. Recomputo independiente desde los npz con el venv del
  proyecto (Python 3.14.7, numpy 1.26.4), lectura de scripts y verificacion de
  mtimes/git. No se reentreno, no se modifico codigo ni datos; la unica escritura
  es este informe. Veredictos: CONFIRMADA, PARCIAL, REFUTADA, NO VERIFICABLE.
- Hallazgo principal: la sub-afirmacion de P84-03 "predicciones binarias
  identicas (0/0 rescate/dano)" es REFUTADA; el contador rescate/dano del script
  tiene un bug (experiments/p84_03_ros_equivalencia.py:927-929) que propago 0/0 a
  JSON, MD, CSV, D-283, plan y QA. La decision de cierre no cambia.

## 1. Tabla de veredictos

| # | Conclusion auditada | Veredicto | Evidencia principal | Ajuste requerido |
|---|---|---|---|---|
| 1 | Ningun candidato nuevo supera a V03; V03 remains the best verified leaderboard result | CONFIRMADA con matices de alcance | gates P84-01..05 todos False (D-281:10056, D-282:10074, D-283:10091, D-284:10106, D-285:10119; D-286:10131); max de leaderboard_log.csv = 0.78306 (v03, linea 5) sobre 23 entradas | Decir "ninguna mejora verificable" y acotar a las 1483 filas de desarrollo; B1 tiene delta puntual +0.000266 no significativo; "verified" = registro interno |
| 2 | DL-01 corregido cierra el 0.7949 historico como artefacto de protocolo (delta -0.018910, 1/5, IC95 [-0.033910, -0.004150]) | PARCIAL | numeros exactos (p84_01_dl01.md:23,44-49; recomputo); pero "artefacto" es interpretacion causal no descompuesta; el protocolo corregido cambia tambien el procedimiento (E* anidado, dispersion 1.41-4.50) | "no comparable/no reproducible en protocolo limpio; queda por debajo de V03 en las tres lecturas y 3/3 semillas", sin cuantificar cuanto del 0.7949 era fuga |
| 3 | fastText supervisado se cierra: brazo principal degenerado con los flags preregistrados y control en -0.039441 | PARCIAL (cierre procedimental OK; pregunta de subwords sin responder) | principal proba constante 0.5 (std 0.0000, 1483/1483 celdas; RUNLOG:5066); nested 0.583398/-0.193813; control 0.737771/-0.039441 (p84_02_fasttext.md:43-48); E* 5 vs 20 (md:28-37) | No citar "aporte sublexico -0.155385" (CSV:453) como efecto de subwords: el brazo colapso y ademas difiere en E*; el diagnostico aislado solo consta en RUNLOG, sin artefacto JSON |
| 4 | ROS es equivalente a reweighting/class_weight y no promovible (refutacion parcial en probabilidad; B1 +0.000266) | PARCIAL con sub-afirmacion REFUTADA | equivalencia: p84_03_ros.md:111-114, JSON max_delta_proba 0.071883489308, peso rel 0.556222, max_iter 5; B1 +0.0002656 recomputado; "predicciones binarias identicas (0/0)" REFUTADA: B1 cambia 61 celdas (32 rescate/29 dano), B2 446 (207/239), B3 63 (34/29) | Corregir D-283:10087 y plan:266: no son identicas; citar 61 celdas y neto +3 para B1. Arreglar ok_cand (linea 928) si se reutiliza el artefacto |
| 5 | SGNS corregido se cierra: S vs M +0.003204 con IC que incluye 0; S peor que R (-0.025734); corpus externo no aporta; ancla lograda | PARCIAL | numeros exactos (p84_04_sgns.md:22-34; JSON por_semilla); S vs M confundido por magnitud de gradiente (md:42); S vs R valido solo para el ensemble de proba media: por semilla -0.0032/-0.0035/+0.0062 (media -0.00015); R son 3 matrices, una por semilla (script:985-999); ancla en alcance declarado (md:11-14) | Acotar "S peor que R" al ensemble de 3 semillas (bagging); declarar R como 3 matrices; el corpus externo no aporta "a este downstream" (cobertura dev 65.85%) |
| 6 | EM+shrinkage se cierra: delta medio -0.008394, 2/5 LOFO, guardarrail falla en AG | CONFIRMADA | p84_05_em.md:59-67,79-85,109-124; JSON delta -0.0083944710, pliegues 2, AG -0.014835 | Anadir la sanidad rho=1 fallida (-0.005043) como motivo de cierre y aclarar que la componente EM +0.020019 no es evidencia end-to-end |
| 7 | EXP-06 se cierra sin ejecucion porque ningun candidato fue elegible | CONFIRMADA | no existen p84_06 script/report/fila; gates False; ensemble_gain: P84-01 -0.001679 (IC incluye 0), P84-02 control -0.028430 y principal 0.0 vacuo, P84-04 -0.049004 | Precisar que el test de complementariedad de fastText principal fue vacuo (proba constante), como ya advierte QA parcial:84-86 |
| 8 | No hay submissions nuevas y no se toco test ni holdout en la fase | CONFIRMADA | contadores 1483/0/0 en los 5; QA parcial:39-43 y QA final:24,38-39; find submissions sin cambios tras 12:25; data/raw con mtime 2026-09-25; leaderboard_log modificado a las 05:26 (auditoria previa) | Matizar: el holdout viaja dentro de toxic_train.csv y se aparta por indice (no se usa); el archivo parasito "=" se creo en la ventana, no es submission |

## 2. Detalle

### 2.1 Conclusion 1 (V03 sigue en pie)

- Verificado por recomputo: V03 con umbrales 0.44/0.51/0.46 =
  0.7772114525165721 exacto desde features/oof_v03_final.npz.
- Todos los brazos P84 quedan por debajo en F1 congelada: P84-01 0.758301
  (semilla 42; -0.0189104), P84-02 principal 0.583398 / control 0.737771,
  P84-04 S 0.626091 / M 0.622888 / R 0.651825, P84-05 0.772169 a rho=1.
- Matiz 1: P84-03 B1 tiene delta puntual POSITIVO vs A (=V03) de +0.0002656 con
  IC95 [-0.004077, +0.004838], 3/5 pliegues y 61 celdas cambiadas. "Ningun
  candidato supera a V03" debe leerse como "ninguna mejora verificable", no como
  "ningun delta puntual positivo".
- Matiz 2: "verified" cubre solo el registro interno (23 filas; max 0.78306 en
  submissions/leaderboard_log.csv:5). R4 de reports/p84_auditoria_premisas.md:49-52
  ya advierte que un score externo no registrado no puede descartarse.
- Matiz 3: la conclusion es local (desarrollo 1483). No hubo submission, luego la
  brecha +0.01694 y la meta 0.80 siguen sin evidencia nueva de fase.

### 2.2 Conclusion 2 (DL-01 como artefacto)

- Numeros confirmados: delta -0.0189104006, 1/5 pliegues, IC95
  [-0.033910, -0.004150]; semillas 43/44 -0.022313/-0.015441; anidada 0.756968
  (0/5) y congelada 0.761659 (1/5) tambien por debajo de V03 (p84_01_dl01.md:23-31).
- Lo que no sostiene el texto actual: (a) "artefacto de protocolo" atribuye
  causalmente la diferencia al protocolo sin descomponerla; el run corregido
  cambia pool (1731 -> 1483), seleccion de epoca (anidada, E* = [2,5,4,3,10] con
  dispersion 1.41-4.50) y refit; (b) la lectura primaria compara candidato a 0.5
  contra V03 con umbrales congelados; a igualdad de 0.5 el delta es -0.013477.
- Redaccion corregida: "la re-medicion limpia no reproduce el 0.794854 historico
  y queda por debajo de V03 en las tres lecturas y en las 3 semillas; la cifra
  historica (pool 1731 + epoca elegida sobre el pliegue puntuado) queda marcada
  como no comparable; no se cuantifica que fraccion era fuga".

### 2.3 Conclusion 3 (fastText)

- La degeneracion es real y extrema: proba principal constante en 0.5 (media
  0.5000, std 0.0000, 100 % de celdas en (0.45, 0.55)); con umbrales congelados
  predice 1/0/1 constante (p84_02_fasttext.md:45; JSON tasa [1.0, 0.0, 1.0]).
- El control esta bien medido y por debajo de V03 (-0.039441, 0/5).
- La pregunta de EXP-02 ("aportan los subwords errores complementarios?") queda
  SIN RESPONDER: el brazo con subwords colapso, y ademas difiere del control en
  E* (5 en 5/5 pliegues vs 20). El campo "aporte sublexico principal vs control
  delta -0.155385" (experiments.csv:453) confunde subwords con degeneracion y
  epocas; no debe citarse como efecto de subwords.
- El "diagnostico aislado" que descarta bug del harness consta solo en
  RUNLOG:5066/D-282; no hay artefacto JSON/MD que lo documente.

### 2.4 Conclusion 4 (ROS) - incluye REFUTACION

- Equivalencia algebraica y verificacion: max|delta proba| 7.188e-02 > 1e-3
  (p84_03_ros.md:111), causa redondeo entero (peso relativo max 0.556222,
  liblinear <= 5 iteraciones, 0 ajustes sin converger; JSON diagnostico_causas).
- B1: delta F1 congelada +0.0002656 (recomputado), 3/5, IC95 [-0.004077,
  +0.004838]; B2 -0.044992; B3 +0.000210. No promovible: correcto.
- REFUTADO: "predicciones binarias identicas (0/0 rescate/dano)" (D-283:10087,
  plan:266, CSV:454, QA parcial:22). Recomputo: B1 cambia 61 celdas de 4449
  (32 rescates, 29 danos, neto +3), B2 446 (207/239, neto -32), B3 63 (34/29).
  Causa: en experiments/p84_03_ros_equivalencia.py:927-928, ok_cand vuelve a usar
  `pred` en lugar de `pred_ref`, de modo que rescue=~ok_ref&ok_cand y
  dano=ok_ref&~ok_cand son identicamente vacios en todas las comparaciones.
  Ningun gate depende del contador, por lo que la decision CLOSED BY EQUIVALENCE
  se mantiene; el texto de evidencia no.
- Redaccion corregida: "equivalencia practica a nivel de decision, no exacta en
  probabilidad (redondeo entero); B1 modifica 61 celdas con neto +3 y delta F1
  +0.000266 no significativo; no promovible".

### 2.5 Conclusion 5 (SGNS)

- Ancla: confirmada en su alcance (algebra 1.3e-16, gradiente 4.4e-08, pureza
  1.0 en 16 palabras, corpus sintetico; binario oficial inestable con NaN y
  reintento). No prueba equivalencia de vectores entrenados, solo del objetivo.
- S vs M +0.003204 (3/5, IC95 [-0.012606, +0.018844]) coincide con lo publicado,
  pero el propio md:42 advierte que confunde correccion del objetivo con magnitud
  del gradiente: no permite concluir "la correccion no cambia el downstream".
- S vs R -0.025734 (1/5, IC95 [-0.046887, -0.004500]) es correcto para la lectura
  primaria, que promedia probabilidades de 3 semillas (script:3120-3145). Por
  semilla los deltas son -0.003186, -0.003461 y +0.006204 (media -0.00015): a
  nivel de semilla individual S y R empatan. "S peor que la matriz aleatoria" es
  un efecto del ensemble, no de cada embedding.
- R no es una matriz unica: se genera una por semilla (script:985-999); el cierre
  debe decir "3 matrices aleatorias".
- "El corpus externo no aporta al downstream": operativamente sostenible (S 0.626
  vs V03 0.777; cobertura dev 65.85 %), pero el diseno no aisla el corpus (no hay
  brazo con corpus interno sin etiquetas); la afirmacion correcta es "los
  embeddings SGNS de este corpus y regimen no superan a una proyeccion aleatoria
  en este downstream".

### 2.6 Conclusion 6 (EM)

- Delta medio -0.0083944710, 2/5 LOFO, IC95 [-0.015763, -0.001231], guardarrail
  AG -0.014835: todo confirmado (p84_05_em.md:59-85; JSON).
- Falta citar en el resumen: sanidad sin shift rho=1 delta -0.005043 (falla) y
  descomposicion +0.020019 (EM) -0.028414 (umbral reubicado). La componente EM
  positiva no es evidencia end-to-end; el gate principal cae por el umbral.

### 2.7 Conclusiones 7 y 8

- EXP-06: no existe script, report ni fila p84_06; gates False en los cinco y
  ningun ensemble_gain positivo. P84-02 principal con gain 0.0 es vacuo
  (proba constante), no un cero informativo (QA parcial:84-86).
- Sin submissions/test/holdout: verificado por contadores, contenido de scripts
  (solo comentarios), mtimes de data/raw (2026-09-25) y find en submissions sin
  cambios tras las 12:25; leaderboard_log.csv se modifico a las 05:26 (fase
  previa). El unico artefacto extrano es el fichero "=" de 0 bytes (12:56), no
  relacionado con submissions.

## 3. Limitaciones que faltan declarar en el cierre

1. N = 1483 de desarrollo; holdout v2 de 248 agotado (6 usos frente a 5,
   premisas:21); test de 1154 intocable.
2. Ruido del LB publico: ~808 filas (70 %) -> cota binomial ~0.0146 y banda
   escalada ~0.0182; privado ~346 -> ~0.0223/~0.0278 (docs/p84_spec.md:115-130).
   Con esa vara, B1 (+0.000266) y S vs M (+0.003204) estan muy por debajo de la
   detectabilidad de una sonda publica; los cierres por deltas >= 0.026 son
   robustos en magnitud, pero ninguno se valido externamente (no hubo sondas).
3. fastText: sin rejilla por preregistro; brazo con subwords degenerado -> la
   pregunta de subwords queda abierta; control con E*=20 (sobreajuste) frente a
   E*=5 (subentrenamiento): las epocas confunden la comparacion.
4. SGNS: S vs M confundido; R solo ensemble; 3 semillas; sin corpus interno de
   control; ancla sobre corpus sintetico y 16 palabras.
5. P84-01: el protocolo limpio no es "el mismo modelo sin fuga", es otro
   procedimiento de seleccion de epoca; la cifra -0.0189 es entre protocolos.
6. P84-05: la conclusion negativa es solida, pero la "componente EM positiva"
   puede leerse mal; declarar siempre la sanidad rho=1 negativa.
7. V03: "mejor resultado verificado" = mejor registro interno; el ranking final
   depende del 30 % privado, desconocido.
8. Erratas menores: D-285:10114 dice "15.511 s GPU" cuando son 15511.17 s (~4.3
   h; plan:300, CSV:456); el contador rescate/dano de P84-03 (seccion 2.4).
9. P84-05 es el unico cierre sin recomputo independiente en los QA (solo
   consistencia CSV/JSON/MD/D-284/plan); este informe verifico su consistencia
   numerica, no re-ejecuto la simulacion EM.

## 4. Redaccion corregida de la conclusion de fase

"Ningun candidato nuevo mostro una mejora verificable sobre V03 en las 1483
filas de desarrollo bajo el protocolo preregistrado de P84 (gates de los cinco
experimentos no superados; unico delta puntual positivo, ROS B1 +0.000266, no
significativo y no promovible). V03 sigue siendo el mejor resultado registrado en
el leaderboard publico del proyecto (0.78306, maximo de 23 entradas). No hubo
submissions ni uso de test ni de holdout en la fase. Todas las conclusiones de
P84 son locales: la brecha externa de V03 (+0.01694) y la meta 0.80 no quedaron
cerradas por esta fase."

## 5. Trazabilidad del auditor

- Recomputado desde npz (coincide con informes): V03 0.7772114525165721; P84-01
  semilla 42 0.7583011/delta -0.0189104; P84-02 nested 0.5833981/-0.1938134 y
  control 0.7377708/-0.0394407; P84-03 B1 +0.0002656 y max|dproba| 0.0718835;
  P84-04 S 0.6260910 / M 0.6228875 / R 0.6518248, deltas +0.0032035 / -0.0257338
  y valores por semilla; max LB 0.78306 sobre 23 filas.
- No recomputado: superficie EM de P84-05 (solo consistencia); internals del
  ancla (se aceptan p84_04_ancla.json y QA final:20-21).
- Bug detectado por lectura de experiments/p84_03_ros_equivalencia.py:927-929 y
  confirmado por recomputo de celdas.
