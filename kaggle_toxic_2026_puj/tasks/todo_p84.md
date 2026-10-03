# TODO P84 - ejecucion de las specs EXP-01 a EXP-06

Proyecto: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj
Spec: docs/p84_spec.md (normativa; cualquier desvio requiere enmienda hasheada)
Preregistro: docs/p84_preregistro.md
Auditoria: reports/p84_auditoria_premisas.md
Estado: LISTA DE TAREAS. No autoriza ejecutar. Sin submissions.

Leyenda: [ ] pendiente, [~] en curso, [x] hecho, [!] cerrado con evidencia negativa,
[S] saltado con razon documentada.

## Checkpoints

| Id | Cuando | Criterio de paso |
|---|---|---|
| C0 | Antes de cualquier corrida | Hashes del preregistro verificados; dev 1483 / holdout 248 / test 0; folds [297,297,296,296,297] sin fuga; identidad V03 reproducida con tolerancia 1e-12; V03@0.5 = 0.771778476256122 recalculado; enmiendas hasheadas |
| C1 | Tras EXP-01 | Medicion valida de DL-01 con control exacto, 3 semillas y freno D-057; decision promover a Gate 2 o cerrar DL-01 como artefacto de protocolo |
| C2 | Tras EXP-02 | Gate standalone o de complementariedad evaluado; determinismo verificado; control sin subwords publicado |
| C3 | Tras EXP-03 | Equivalencia ROS verificada o refutada con causa; familia ROS cerrada como no promovible |
| C4 | Tras EXP-04 | Ancla documentada; S vs M vs R pareados; decision de linea (gate o cierre) |
| C5 | Tras EXP-05 | Gate LOFO evaluado con descomposicion EM vs umbral y guardarrail de tasa; familia EM/threshold cerrada o abierta |
| C6 | Tras EXP-01..05 | Puerta EXP-06: lista de elegibles congelada; ejecutar combinacion o documentar puerta cerrada |
| C7 | Gate 2 | Segunda particion (rs=20261020) con receta congelada: delta >= +0.002, >= 4/5, IC > 0; freno D-057 |
| C8 | Cierre | Matriz de cierres, RUNLOG y DECISIONS al dia; GO/NO-GO de sonda documentado; ningun envio sin autorizacion de Carlos |

## P0 - Gate 0 y preparacion (dias 1-2)

- [ ] Revalidar hashes del preregistro (train, oof_v03_final.npz, grupos, corpus 10M,
      binario fastText) y registrar el resultado (criterio: los 5 coinciden).
- [ ] Reconstruir particiones y contadores (criterio: dev 1483, holdout 248, interseccion 0,
      folds [297,297,296,296,297], 0 grupos compartidos, 0 lecturas de test).
- [ ] Reconstruir V03 con el harness comun y comparar con el npz (criterio:
      max |delta proba| <= 1e-12 y |delta F1| <= 1e-12).
- [ ] Recalcular V03@0.5 y corrected_cv desde el npz (criterio: 0.771778476256122 y
      0.7772114525165721 exactos).
- [ ] Escribir y hashear la enmienda de preregistro de EXP-01 (fija E*, semillas,
      lectura primaria a 0.5 y tolerancias) (criterio: sha256 y commit previo a codigo).
- [ ] Verificar versiones del entorno (venv, sklearn 1.8.0, torch 2.14.0+rocm7.2,
      GPU disponible) y registrar device por experimento (criterio: log del entorno).
- [ ] Reconfirmar en sesion autenticada el reparto publico/privado 70/30 del leaderboard
      (riesgo R6 de la auditoria) (criterio: nota fechada en docs/p84_spec.md seccion 1.5
      o enmienda si cambia).

## P1 - EXP-01 (P84-01) DL-01 corregido (dias 3-5)

- [ ] Enmienda hasheada de EXP-01 (criterio: incluye particiones internas, semillas 42/43/44,
      E* mediana, tolerancias y criterio de cierre).
- [ ] Implementar `experiments/p84_01_dl01_corregido.py` (criterio: control de identidad
      exacto, contadores, E* solo interno, reajuste fijo, ASCII, docstrings, un solo script).
- [ ] Revision de @ml-reviewer (criterio: sin fuga; vectorizador/SVD/scaler/pos_weight/
      epoca/umbral ajustados solo en outer_train o particiones internas; sin test/holdout;
      sin editar experimentos cerrados).
- [ ] Smoke test de 1 pliegue 1 semilla (criterio: corre sin excepciones, escribe JSON
      parcial, tiempos publicados).
- [ ] Corrida completa con hito a los 40 minutos (criterio: 5 pliegues x 3 semillas;
      prediccion unica de outer_val; contador verificado; aborta si se supera).
- [ ] Calcular comparaciones vs V03 (criterio: delta por pliegue, delta medio, 4/5, IC
      bootstrap 10.000/20260926, rescate/dano, tasas, F1 por etiqueta, brecha, parametros).
- [ ] Evaluar C1 (criterio: Gate 1 + estabilidad entre semillas + freno D-057, o cierre
      documentado de DL-01 como artefacto de protocolo).
- [ ] Escribir `reports/p84_01_dl01.json` y `.md`; fila P84-01 en experiments.csv;
      RUNLOG; DECISIONS si hay cierre/apertura (criterio: artefactos completos y fila
      idempotente).

## P2 - EXP-02 (P84-02) fastText supervisado (dias 6-7)

- [ ] Enmienda hasheada de EXP-02 (criterio: fija dim 100, lr 0.05, minCount 2, bucket,
      loss softmax, wordNgrams 2, minn/maxn, thread 1, seed, rejilla {5,10,20} y rejilla
      de umbral; declara la expectativa nula y la equivalencia funcional con la familia
      lineal).
- [ ] Implementar `experiments/p84_02_fasttext_supervisado.py` (criterio: archivos de
      entrenamiento por pliegue/etiqueta solo con outer_train; control sin subwords;
      probabilidades P(__label__POS)).
- [ ] Verificar determinismo (criterio: dos corridas iguales en un pliegue producen
      probabilidades identicas, tolerancia 0).
- [ ] Revision de @ml-reviewer (criterio: 0 uso de vectores preentrenados; 0 corpus
      externo; 0 texto de validacion en el archivo de entrenamiento; sin editar cerrados).
- [ ] Corrida completa (criterio: 5 pliegues x 2 brazos, E* y umbral anidados, tres
      lecturas publicadas).
- [ ] Evaluar C2 (criterio: Gate 1 o complementariedad; control sin subwords publicado;
      correlaciones y solapamiento de errores).
- [ ] Escribir `reports/p84_02_fasttext.json` y `.md`; `features/p84_02_oof_fasttext.npz`;
      fila P84-02; RUNLOG; DECISIONS (criterio: artefactos completos).
- [ ] Si no pasa: registrar cierre de fastText supervisado sin abrir rejilla (criterio:
      decision [!] con numero).

## P3 - EXP-03 (P84-03) ROS: verificacion de equivalencia (dia 8)

- [ ] Enmienda hasheada que sustituye el diseno del plan (criterio: incluye la derivacion
      algebraica, los brazos A/B1/B2/B3, la no promocion y la tolerancia de equivalencia).
- [ ] Implementar `experiments/p84_03_ros_equivalencia.py` (criterio: vectorizador en
      outer_train antes de duplicar; duplicacion sembrada; C_efectivo publicado).
- [ ] Revision de @ml-reviewer (criterio: sin SMOTE/ADASYN/undersampling; sin duplicar
      validacion; sin aplicarlo antes de dividir).
- [ ] Corrida (criterio: equivalencia B1-A dentro de tolerancia, o refutacion con causa;
      todos los deltas publicados).
- [ ] Evaluar C3 (criterio: familia ROS cerrada como no promovible; ningun brazo va a
      Gate 2 ni EXP-06).
- [ ] Escribir `reports/p84_03_ros.json` y `.md`; fila P84-03 marcada CLOSED BY
      EQUIVALENCE; RUNLOG; DECISIONS (criterio: cierre registrado).
- [ ] Decision de presupuesto (Ask first): micro-verificacion ejecutada o cierre por
      demostracion sin ejecutar (criterio: una de las dos registrada con su razon).

## P4 - EXP-04 (P84-04) SGNS corregido y anclado (dias 9-12)

- [ ] Enmienda hasheada de EXP-04 (criterio: fija corpus wikipedia_es 10M, dim 200,
      ventana 5, min_count 3, negativos 10, epocas 5, lote 256, lr 0.025*lote, 3 semillas,
      formula del objetivo y tolerancias del ancla).
- [ ] Implementar `experiments/p84_04_sgns_corregido_anclado.py` (criterio: ancla antes
      de entrenar; S/M con flujo de datos identico; R congelado; EXTERNAL-ONLY verificado
      por contador).
- [ ] Chequeo algebraico y de gradiente (criterio: loss_suma = 10 * loss_media con tol
      1e-9; gradiente por diferencias finitas con error relativo <= 1e-5).
- [ ] Ancla ejecutable contra el binario oficial en corpus plantado (criterio:
      pureza_nuestra >= pureza_oficial - 0.05, o reporte ANCLA NO LOGRADA).
- [ ] Revision de @ml-reviewer (criterio: sin texto de la competencia en el corpus; sin
      rejillas de dim/ventana/pooling; sin usar el artefacto 100d como control; sin
      editar p71_04/p71_08/p71_10).
- [ ] Corrida completa con hito a los 40 minutos por brazo-semilla (criterio: 3 brazos x
      3 semillas; tiempos publicados; embeddings con sha256; budget < 1 dia).
- [ ] Downstream fijo y comparaciones pareadas (criterio: mean pooling + LR en folds
      canonicos; S vs M, S vs R, S vs V03; ensemble_gain; IC 10.000/20260926).
- [ ] Evaluar C4 (criterio: gate de complementariedad o cierre de la linea con el defecto
      resuelto; secundaria de fusion solo si hay senal, marcada exploratoria).
- [ ] Escribir `reports/p84_04_sgns.json`, `reports/p84_04_ancla.json`, `.md`;
      `features/p84_04_oof_emb.npz`; `modelos/p84/`; fila P84-04; RUNLOG; DECISIONS.

## P5 - EXP-05 (P84-05) EM + shrinkage simulado (dia 13)

- [ ] Enmienda hasheada de EXP-05 (criterio: fija LOFO, rejilla m {0,25,50,100}, t 0.20-0.80
      paso 0.01, rho {0.85,0.90,0.94,1.00,1.05}, guardarrail de tasa a rho=1.00,
      descomposicion EM vs umbral, bootstrap con pesos/20260926).
- [ ] Implementar `experiments/p84_05_em_shrinkage_simulado.py` (criterio: F1 ponderada,
      EM con shrinkage y n_eff de Kish, seleccion LOFO, superficie completa publicada).
- [ ] Revision de @ml-reviewer (criterio: 0 lecturas de test; 0 seleccion sobre las filas
      evaluadas; 0 reentrenamiento de modelos).
- [ ] Corrida (criterio: 5 pliegues LOFO x 5 rho x rejillas; tiempos publicados;
      resultado por etiqueta y por rho).
- [ ] Evaluar C5 (criterio: Gate del 6.7 con descomposicion positiva y guardarrail; o
      cierre de EM/threshold).
- [ ] Escribir `reports/p84_05_em.json` y `.md`; fila P84-05; RUNLOG; DECISIONS.

## P6 - EXP-06 y Gate 2 (dias 14-15)

- [ ] Congelar la lista de elegibles con sus OOF y sha256 (criterio: elegibles = Gate 1
      standalone o ensemble_gain >= 0.0015 con IC > 0; P84-03 excluido por diseno).
- [ ] Evaluar C6: si no hay elegibles, escribir `reports/p84_06_puerta_cerrada.md` y fila
      P84-06 "NO EJECUTADO" (criterio: razon y lista de candidatos en el informe).
- [ ] Si hay elegibles: implementar y correr la regla R1 (media a 0.5) y R2 (media
      ponderada con w elegido solo en particiones internas); sin stacking ni calibracion
      (criterio: Gate 1 con la receta congelada; matriz de errores publicada).
- [ ] Gate 2 para cada candidato que pase Gate 1: segunda particion rs=20261020, misma
      receta, sin reajuste; criterio delta >= +0.002, >= 4/5, IC > 0 y freno D-057.
- [ ] Registrar resultado de Gate 2 y decision de linea (criterio: promover a propuesta
      de sonda o cerrar sin abrir rejilla).

## P7 - Cierre (dias 16-17)

- [ ] Matriz de cierres: familia, mecanismo, resultado, gate, decision, ruta del informe
      (criterio: cubre EXP-01..06 y las familias cerradas citadas en docs/p84_spec.md
      seccion 9).
- [ ] Actualizar RUNLOG.md y DECISIONS.md con todas las decisiones de la fase (criterio:
      cada experimento con su id, resultado y decision).
- [ ] Actualizar `experiments.csv` y verificar las filas P84-01..P84-06 (criterio:
      idempotencia, 23 columnas, sin duplicados).
- [ ] Documento GO/NO-GO de sonda: recomendacion razonada con el ruido corregido
      (criterio: sin envio; la decision es de Carlos).
- [ ] Verificacion final: 0 envios, 0 commits no autorizados, holdout/test intactos
      (criterio: contadores y hashes de cierre).

## Dependencias y decisiones que requieren consulta (Ask first)

- EXP-02 y EXP-04 comparten el mecanismo de particiones internas con EXP-01: si EXP-01
  cambia su enmienda de semillas, EXP-02 debe rehashear la suya antes de correr.
- EXP-03: ejecutar la micro-verificacion o cerrar por demostracion (coste bajo, decision
  de Carlos).
- EXP-04: corpus y dimension fijados en la enmienda; cualquier cambio de corpus requiere
  hash nuevo y enmienda previa.
- EXP-05: aplicar la regla a probabilidades de test queda fuera de P84; requiere
  autorizacion explicita.
- Cualquier submission, commit o PR: solo con autorizacion explicita de Carlos.

## Fuera de alcance (decidido)

- Envios automaticos o sondas sin autorizacion.
- Transformers, LLM, embeddings preentrenados, pseudoetiquetado, etiquetas externas.
- Reabrir familias cerradas (docs/p84_spec.md seccion 9).
- Nuevo holdout, segunda evaluacion del holdout, uso de test para cualquier ajuste.
- Optuna, rejillas grandes o busqueda de hiperparametros no preregistrada.
