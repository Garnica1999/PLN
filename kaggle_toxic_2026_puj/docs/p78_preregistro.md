# Pre-registro P78 - Control anidado Word2Vec frente a azar, con normalizacion alineada

Fecha de escritura: 2026-09-30
Rama: `proyecto-kaggle-toxic-2026-p67`
Script previsto: `experiments/p78_01_control_anidado.py`
Estado: PRE-REGISTRADO. Este texto se hashea y se commitea ANTES de escribir una linea de codigo.

## 0. Por que existe este experimento

P73 reporto +0.0316259 para Word2Vec frente a azar, y P74 reporto 10/10 semillas
positivas con delta +0.02427. La auditoria posterior encontro que ambos eligen el
checkpoint del TextCNN usando el pliegue EXTERNO de validacion:

- `experiments/p73_02_textcnn_vs_random.py:768-779` reserva una mascara interna `m_in`
  dentro de `outer_train`, pero llama a `entrenar_pliegue` (`:393-483`) pasando
  `ids[ib]` y `y[ib]`, que son las filas del pliegue externo.
- `experiments/p74_01_null_replicado.py:259-270` repite el mismo patron via
  `entrenar_con_semilla` (`:120-197`).
- `experiments/p73_03_segunda_particion.py:84-131` reutiliza el mismo camino.

Ademas, el texto de entrada del CNN se tokeniza SIN aplicar la normalizacion que si
se aplico al corpus de Word2Vec (`experiments/p71_03_preparar_corpus.py:67-96` frente a
`experiments/p73_02_textcnn_vs_random.py:148-177`). La cobertura declarada fue 68.6361
por ciento con 31.3639 por ciento de OOV, y el id 0 se comparte entre padding y OOV
(`p73_02:160-177`).

P78 no intenta beating records. P78 responde una sola pregunta, con control pareado y
sin fuga:

> En condiciones de early stopping estrictamente anidados, el embedding Word2Vec
> aporta senal sobre el objetivo, en comparacion con una matriz aleatoria de identica
> forma y identicas normas de fila, cuando el texto de entrada se normaliza igual que
> el corpus?

Los numeros historicos de P73 y P74 NO se modifican. P78 los reinterpreta, no los
reemplaza.

## 1. Pregunta, hipotesis y comparaciones

Hiptesis nula H0: la diferencia macro-F1 entre el brazo Word2Vec y su control aleatorio
es cero dentro del ruido de semillas.

Cuatro brazos, todos con el mismo early stopping estrictamente anidado:

| Brazo | Matriz de embeddings | Normalizacion de entrada |
|---|---|---|
| A | Word2Vec 100d congelado | NO (tokenizado actual, sin `normalizar`) |
| B | aleatoria pareada | NO |
| C | Word2Vec 100d congelado | SI (`normalizar` de P71-03) |
| D | aleatoria pareada | SI |

Comparaciones, todas sobre el delta macro-F1 pareado por fold:

- **P1, PRIMARIA: C - D.** Efecto del embedding con la normalizacion alineada.
- **P2: A - B.** Efecto del embedding con el tokenizado actual. Es la comparacion que
 P73 y P74 miden, y por lo tanto la que se confronta directamente con sus numeros.
- **P3: C - A.** Efecto de alinear la normalizacion en el brazo Word2Vec.
- **P4: D - B.** Mismo efecto en el brazo aleatorio. Es el control de confusion: si P3 y
  P4 se mueven por igual, el efecto de la normalizacion no es especifico del embedding.

## 2. Representacion y arquitectura: NO se cambian

- Matriz de embeddings: `modelos/twitter_domain/w2v_twitter_100d.npz`, la misma de P73.
  sha256 `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`.
  sha1 de la matriz con la fila de padding incluida
  `c1b774e52e0e1dc131311bddea967e2a72a96e29`.
- Arquitectura: `TextCNN` de `experiments/p73_02_textcnn_vs_random.py:194-240`, sin
  cambios. Embeddings congelados.
- Hiperparametros: sin cambios. `MAXLEN=60`, `FILTROS=64`, `KERNELS=(2,3,4)`,
  `HIDDEN=64`, `DROPOUT=0.5`, `LR_PICO=1e-3`, `LR_MIN=1e-4`, `WARMUP=3`, `EPOCAS=40`,
  `BATCH=32`, `PACIENCIA=6`, `VENTANA_ES=3`, `TOL_ES=1e-3`, `CLIP=1.0`, `WD=1e-4`.
- Umbrales: los de V03, congelados, `[0.44, 0.51, 0.46]`.
- V03 es FROZEN. `corrected_cv = 0.7772114525165721`. LB historico `0.78306`.
  No se reentrena, no se toca, no se consulta el leaderboard.

## 3. Protocolo por pliegue externo, sin fuga

Para cada pliegue externo `k` de 1 a 5, con indices `ia` (outer_train) e `ib`
(outer_validation):

1. `epoca_estrella` de `experiments/p75_01_complementariedad_textcnn.py:102-160` se
   ejecuta con `ia` y produce `E*` como mediana de tresplits internos por clusters. Los
   tres splits se obtienen con `partir_interna(grupos[ia], y[ia], 20261101 + 100*k + 7*j)`.
   El early stopping ocurre sobre `ia[m_in]` contra `ia[~m_in]`. **Ninguna fila de `ib`
   participa.**
2. `entrenar_fijo` (`p75_01:163-211`) reentrena sobre `ia` COMPLETO durante `E*` epocas
   exactas, y predice `ib` **una sola vez**.
3. `outer_validation` se evalua exactamente una vez por (brazo, semilla, pliegue). Se
   lleva un contador explicito y el script aborta si se supera.

Este es el mismo aislamiento de P75. La diferencia con P73 y P74 es que alli el
checkpoint se elegia con `ib`.

## 4. Semillas pareadas

Tres semillas: `20261130`, `20261131`, `20261132`. Para una misma condicion de entrada
(A contra B, y C contra D), la misma semilla produce:

- el mismo `init` de pesos via `init_de_semilla` (`p74_01:101-117`),
- el mismo generador de permutaciones de batch, luego el mismo orden de mini-batches,
- el mismo reparto de filas, el mismo `E*` si se alcanza con la misma curva,
- la misma matriz de embeddings, al menos en forma y en norma de fila.

La matriz aleatoria se construye con `genera_aleatoria(semilla, W, normas)` de
`p74_01:357-373`, que conserva las normas de fila de la matriz Word2Vec. Sin eso, el
control mediria una diferencia de escala y no de informacion.

Se reporta la media y la desviacion tipica sobre las tres semillas, y se separa
explicitamente variabilidad de semillas de variabilidad de datos.

## 5. Metricas

- **Primaria:** macro-F1 con `f1_media_columnas` y umbrales V03 congelados.
- Por etiqueta: F1 positivo de HS, TR y AG. Se usa la funcion de F1 real del proyecto
  (`src/evaluation.py:64-123`). **No** se usa `TP / positivos`; ese es recall y fue el
  defecto de P77.
- De probabilidad, y separadas de las de F1: Brier score por etiqueta y log-loss medio.
- Deltas por pliegue: los 5 deltas de cada comparacion, con su signo.
- Bootstrap pareado por clusters, 2000 replicas, semilla `20261102`, sobre el delta
  macro-F1 de la comparacion primaria y de P2. Se reutiliza `bootstrap_clusters`
  (`p73_02:486-511`), que remuestrea grupos, no filas.
- IC95 del delta pareado sobre las tres semillas.
- Conteo de unidades tweet-etiqueta por las que se cambia la decision. Se deja
  constancia de que `rescue/damage/net` es DESCRIPTIVO y no sustituye al macro-F1.

## 6. Diagnostico de cobertura (sin modelo)

Se mide, para la entrada con y sin normalizacion, sobre el conjunto dev:

- proporcion de tokens con id distinto de cero;
- proporcion de OOV;
- numero de tipos distintos en dev ausentes del vocabulario;
- filas con cero tokens;
- fraccion de secuencias truncadas por `MAXLEN=60`;
- numero de filas cuya longitud real es menor que el kernel mayor, que es el caso
  degenerado del max-pooling enmascarado.

Ademas se separa padding de OOV en el informe, aunque la matriz siga usando el id 0
compartido, que es la convencion de `TextCNN` con `padding_idx=0`. No se cambia la
arquitectura, pero el deficit queda cuantificado.

## 7. Criterio de decision, fijado antes de correr

Definamos `d(P) = macro_F1(P) - macro_F1(control)` en macro sobre el OOF pareado.

- Si `P1 >= +0.003` y `P1` es positivo en al menos 4 de 5 pliegues: **el embedding
  aporta senal con normalizacion alineada.** La linea Word2Vec tiene fundamento y la
  Fase 4 (curva de corpus) se justifica.
- Si `P1 <= 0` y su IC95 contiene al cero: **el embedding no aporta.** Se cierra la linea
  y se documenta que P73 y P74 estaban sobreestimados. No se abre ni corpus ni
  hiperparametros.
- Si `P2 > 0` pero `P1 <= 0`: **el efecto medido por P73 y P74 era un artefacto del
  tokenizado**, no informacion del embedding.
- Si `P3` y `P4` son de magnitud parecida: la alineacion de normalizacion actua sobre
  ambos brazos por igual, y no es un mecanismo especifico del embedding.
- Si `P1` y `P2` son ambos claramente positivos: el tokenizado era un cuello de botella
  adicional y hay senal que investigar.

El umbral `+0.003` es el mismo `EPS_GATE_DELTA` que uso P77, y no se ajusta despues de
ver los resultados.

Este experimento NO decide por si solo nada sobre superar a V03. Su unico PRODUCTO es
una estimacion honesta del efecto del embedding bajo un protocolo sin fuga, mas el
diagnostico de cobertura que decide si abrir la Fase 4.

## 8. Lo que P78 no hace

- No genera submission, no consulta el leaderboard, no lee test ni holdout.
- No lee el parquet completo. La unica fuente de texto es `src.data.read_train()` mas la
  particion `dev` que fija `holdout_v2`, y el unico OOF es
  `features/oof_v03_final.npz`. No se usan slices en P78, precisamente para no abrir la
  lectura del parquet.
- No reentrena V03, no reentrena ni modifica el Word2Vec congelado.
- No cambia la arquitectura ni los hiperparametros.
- No usa fine-tuning, BiGRU, atencion, stacking, kNN, ni ninguna dimension nueva.
- No edita ni borra P71 a P77.

## 9. Aserciones obligatorias

El script aborta con `SystemExit` si cualquiera falla:

1. `abs(f1m(y, pred_v03) - 0.7772114525165721) > 1e-9`.
2. Los ids del OOF de V03 coinciden con `tr.iloc[idev]["id"]`.
3. Cada fila esta en exactamente un pliegue de validacion.
4. `sha256` de `w2v_twitter_100d.npz` igual a `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`.
5. `sha_matriz_pad(matriz_W2V)` igual a `c1b774e52e0e1dc131311bddea967e2a72a96e29`.
6. Para cada pliegue y cada rama: `set(inner_val) <= set(ia)` y `set(inner_val)` vacio
   intersecado con `set(ib)`. Se registra el hash de los indices usados en cada rama.
7. El numero de evaluaciones sobre `ib` es exactamente 1 por (brazo, semilla, pliegue).
8. A y B comparten `init`, filas, batches y longitudes; sus matrices tienen la misma forma
   y hash distinto.
9. Toda probabilidad emitida es finita y ninguna fila queda sin prediccion.
10. El fichero de salida del script, el JSON de resultados y las figuras son ASCII puro,
    verificado byte a byte.

## 10. Salidas esperadas

- `experiments/p78_01_control_anidado.py`, script unico y completo.
- `reports/p78_control_anidado.json`, con los 4 brazos, las 4 comparaciones, deltas por
  pliegue, IC95, bootstrap, metricas de probabilidad, cobertura y los contadores de
  auditoria.
- `reports/p78_cobertura.json`, diagnostico de cobertura separado.
- `reports/p78_pares.csv`, OOF pareado en unidades tweet-etiqueta con las cuatro
  probabilidades, para que el resultado sea recomputable sin reentrenar.
- `figuras/p78/`, figuras de deltas por pliegue, cobertura y distribucion de
  probabilidades.
- Una entrada en `DECISIONS.md`, un bloque en `RUNLOG.md` y una actualizacion de
  `PROJECT_STATE.md` que reetiqueten P73 y P74 como exploratorios.
- Todo el texto nuevo en ASCII puro, sin tildes ni enie.
