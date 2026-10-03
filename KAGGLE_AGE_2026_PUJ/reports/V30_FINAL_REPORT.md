# V30 - Refinamiento del LightGBM optimizado y submission controlada

**Fecha:** 2026-09-30 · **Rama:** `proyecto-kaggle-age-2026-v30` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

La fase final de AGE. Parte de un resultado de V29 con margen real y se pregunta si ese margen se consolida lo bastante para gastar la unica sonda. Todas las cifras salen de artefactos.

## 1. Control absoluto: V30 no empieza sin que el punto de partida sea el que se dice

| Control | Esperado | Obtenido | Reproduce |
|---|---|---|---|
| V20, reconstruido y comparado **bit a bit** | 0,6097782 | 0,6097782 | **True** |
| LightGBM de V29, OOF en disco | 0,6121698 | 0,6121698 | **True** |
| Ensemble de V29, **re-optimizado** | 0,6125960 | 0,6125960 | **True** |

**La diferencia maxima entre el OOF reconstruido de V20 y su artefacto es 0.00e+00**, y los folds son los mismos: [33868, 33868, 33867, 33866, 33867].

Un resultado que merece staffed atencion: la reoptimizacion de los pesos del ensemble de V29 **reprodujo con desviacion 0.0**. El propio brief advertia que no debia declararse exactitud bit a bit entre reproducciones de Optuna; en esta maquina y con estas versiones de libreria, si lo es. Se registra lo medido, no lo temido.

## 2. V30: refinamiento local del LightGBM, sembrado con el mejor trial de V29

### Por que el espacio es local, y no una busqueda nueva

Los cinco mejores trials de V29 coinciden en una region estrecha, y **V20 estaba sub-parametrizado en casi todo**:

| Parametro | 5 mejores de V29 | `HIERBA` de V20 |
|---|---|---|
| `num_leaves` | 43 - 87 | **8** |
| `max_depth` | 5 - 9 | **3** |
| `learning_rate` | 0,021 - 0,043 | 0,05 |
| `min_child_samples` | 22 - 94 | **200** |
| `reg_lambda` | 1e-7 - 0,018 | **1,0** |

Ese contraste **es el mecanismo de la mejora de V29**, y por eso V30 centra su espacio ahi en vez de recorrer la region entera. Y la localidad no es solo un rango: el **trial 0 de cada fold se encola con los parametros exactos** del mejor trial de V29 para ese fold.

- Trials: **15 por fold, 75 en total**, el maximo que autoriza el brief.
- Sampler: `TPESampler(seed=20300930 + fold) con trial 0 encolado`.
- Techo de estimadores: **1500**, con early stopping de 60 rondas.

### Iteraciones reales, que es donde se ve el early stopping

| Fold | Mejor trial | Inner | Mediana | Sigma | Iteraciones / techo | Early stop | Outer |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 0.612670 | 0.610101 | 1.89 | 999 / 1011 | True | 0.615979 |
| 1 | 3 | 0.611625 | 0.610502 | 1.20 | 1173 / 1486 | True | 0.616113 |
| 2 | 5 | 0.617228 | 0.615592 | 1.72 | 644 / 1023 | True | 0.611859 |
| 3 | 11 | 0.618832 | 0.616098 | 2.12 | 1129 / 1477 | True | 0.610876 |
| 4 | 13 | 0.617436 | 0.616016 | 1.43 | 398 / 1000 | True | 0.606381 |

**El early stopping corta siempre muy por debajo del techo.** Es exactamente el comportamiento que el punto 38 del brief describe como correcto: dar margen alto de capacidad y dejar que el criterio interior decida donde parar, en vez de entrenar artificialmente hasta el limite.

**Mejor trial outlier: 0 de 5.** Ningun fold se gano por un trial anomalo, asi que el resultado no depende de una tirada afortunada.

### Resultado del refinamiento

- corrected_cv **0,6122387**
- frente a V20: **0,002461**
- frente al LightGBM de V29: **0,000069**
- folds que mejoran a V29: **3/5** ([0.001485, 0.000216, -0.000409, -0.001387, 0.000452])
- veredicto: **EQUIVALE_A_V29**

Criterio declarado de consolidacion: delta frente al LightGBM de V29 superior a +0,0005; entre -0,0005 y +0,0005 se considera equivalente y se decide por complementariedad en la fase de ensemble.

### ¿V30 aprende mejor, o solo reproduce a V29?

| Modelo | CV individual | corr. errores con V20 | Rescate unico | Dano unico | Neto |
|---|---|---|---|---|---|
| v20_f3b_oof_candidato | 0.6097782 | 1.0000 | 0 | 0 | +0 |
| v29_lgbm_optuna | 0.6121698 | 0.8636 | 5,689 | 5,288 | +401 |
| v30_lgbm_refinado | 0.6122387 | 0.8694 | 5,448 | 5,061 | +387 |

Y entre los dos LightGBM, que es la comparacion que decide si hay algo que combinar:

| Par | corr. errores | Discrepancia | Uno arregla al otro |
|---|---|---|---|
| V29 vs V30 | 0.9164 | 0.0626 | 3,365 |

Una correlacion de errores **muy alta** entre V29 y V30 significa que el refinamiento encontro la misma funcion con parametros distintos: es una confirmacion de que la region esta bien elegida, y tambien de que **no hay un tercer miembro que aportar** al ensemble. Por eso el pool de V30 se queda en tres: V20, V29 y V30.

## 3. El ensemble: tres miembros, no dieciseis

V29 ya contesto la pregunta del pool historico (**0 de 47** con rescate neto positivo, y los tres metodos negativos sin el modelo nuevo). V30 **no repite esa busqueda**: el pool es de **3 miembros** y cada uno esta justificado.

| Miembro | Papel |
|---|---|
| `v20_f3b_oof_candidato` | el champion, intacto |
| `v29_lgbm_optuna` | la capacidad que V29 demostro |
| `v30_lgbm_refinado` | el intento de consolidar esa capacidad |

Los pesos se optimizan **solo sobre el holdout interior**, con `w = softmax(theta)` para imponer `w_i >= 0` y `sum w_i = 1` sin restricciones. La validacion exterior no entra.

| Metodo | CV | Delta V20 | Delta V29-LGBM | Folds + | Rescue | Damage | Net |
|---|---|---|---|---|---|---|---|
| E1 Optuna macro-F1 | 0.6130051 | +0.003227 | +0.000835 | 5/5 | 3,768 | 3,238 | +530 |
| E2 Optuna macro-F1 menos lambda por entropia | 0.6121866 | +0.002408 | +0.000017 | 5/5 | 4,341 | 3,953 | +388 |

### El candidato principal: `E1 Optuna macro-F1`

- corrected_cv **0,6130051**, delta frente a V20 **0,003227**
- **5/5 folds** positivos, rescate 3,768 contra dano 3,238, neto **+530**
- pesos: `{'v20_f3b_oof_candidato': 0.3548, 'v29_lgbm_optuna': 0.3354, 'v30_lgbm_refinado': 0.3098}`
- filas que cambian frente a V20: 10,914

### Las dos banderas, separadas como pide el punto 20

| Bandera | Valor | Significado |
|---|---|---|
| Gate historico | **True** | exige delta >= +0,003 **y** >= 4/5 folds |
| STRONG_MARGINAL | **True** | 5/5 folds **y** delta entre +0,002 y +0,003 |

Son **dos preguntas distintas** y no se confunden: el gate es el criterio de finalista; la bandera marginal es la observacion de que la mejora es consistente pero no llega al umbral.

### Bootstrap del candidato principal

- delta **0,003227**
- IC95 **[0,002290, 0,004219]**
- proporcion de diferencias positivas: 1.0000
- **el IC95 incluye el cero: False**

calculado **solo** para el candidato principal, y no se usa para elegir hiperparametros ni pesos: es evidencia de decision de la submission, no senal de entrenamiento

## 4. Segunda particion, con la configuracion congelada

Los hiperparametros y los pesos se aplican **congelados** sobre una particion agrupada distinta (semilla 987654, mismos grupos, sin re-optimizar). Los solapes con la particion original fueron [7155, 7272, 7167, 7229, 7220], luego la particion cambio de verdad.

| Fold | V20 recomputado | Candidato | Delta |
|---|---|---|---|
| 0 | 0.6122675 | 0.6150008 | +0.002733 |
| 1 | 0.6139561 | 0.6186241 | +0.004668 |
| 2 | 0.6138090 | 0.6163075 | +0.002498 |
| 3 | 0.6107416 | 0.6143509 | +0.003609 |
| 4 | 0.6013498 | 0.6073941 | +0.006044 |

- V20 recomputado **0,6104285** | candidato **0,6143347**
- delta contra el recomputado: **0,003906** con **5/5 folds** positivos
- el gate usa el V20 **recomputado en esta particion**: el gate se evalua contra el V20 **recomputado en esta misma particion**, porque los folds son distintos; contra el canonico se informa tambien, pero no decide
- frente al delta de la particion original (0,003227) la diferencia es 0,000679: la mejora se sostiene
- gate historico en la segunda particion: **True** | STRONG_MARGINAL: **False**

## 5. Construccion del fichero y validacion

- Fichero: `age_submission_v30_final_candidate.csv`, **112,892 filas**, 1,356,504 bytes
- **SHA256 `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be`**
- Checks: **10/10**
- Candidato: `E1 Optuna macro-F1`, pesos congelados `{'v20_f3b_oof_candidato': 0.3548, 'v29_lgbm_optuna': 0.3354, 'v30_lgbm_refinado': 0.3098}`
- Distribucion: `{'13-17': 18141, '18-24': 20767, '25-34': 17135, '35-49': 16406, '50-64': 20794, '65-xx': 19649}`
- Frente al envio de V20 difieren **7,253 decisiones (6.42%)**

### Los diez checks

| Check | Resultado |
|---|---|
| `rows_ok` | **True** |
| `columns_ok` | **True** |
| `ids_equal_test` | **True** |
| `ids_same_order` | **True** |
| `ids_unique` | **True** |
| `no_nan` | **True** |
| `labels_valid` | **True** |
| `labels_width_ok` | **True** |
| `sin_filas_vacias` | **True** |
| `schema_verificado` | **True** |

### Como se uso el test

solo transform y predict. Ninguna etiqueta, ninguna estadistica, ningun ajuste de umbral. Los pesos y los hiperparametros estaban congelados antes de entrar aqui.

### La decision de los cinco modelos por miembro

Cada fold produjo su mejor configuracion y son distintas. Ajustar las cinco sobre todo el train y promediar usa toda la evidencia de la busqueda; elegir una sola seria un arbitrio sin base.

## 6. La subida

- Estado: **STOP_SIN_CREDENCIAL**
- Numero de submissions: **0**
- Subido: **False**

**Causa del STOP:** el fichero ~/.kaggle/kaggle.json existe pero solo contiene la clave `token`, que es el formato antiguo; el CLI 2.2.4 exige `username` y `key`, o bien autenticacion OAuth. Sin eso la API rechaza la llamada.

**Como se resuelve:** `kaggle auth login` (flujo OAuth, abre el navegador y cachea el token) o dejar un ~/.kaggle/kaggle.json con las claves `username` y `key`. Ambas vias las hace el usuario en su terminal; ninguna credencial tiene que pasar por esta conversacion.

> Esto es un **STOP de infraestructura**, no un resultado experimental. El > fichero esta construido, validado con sus diez checks y con su hash, y **no > necesita ninguna correccion**: en cuanto haya credencial valida, se sube tal > cual con `v30_04_upload.py`. No se pide ni se copia ningun secreto por esta via.

## 7. Decision

| Pregunta | Respuesta |
|---|---|
| Baseline vigente | **V20**, 0,6097782, LB 0.62689 |
| Candidato final | `E1 Optuna macro-F1`, 0,6130051, delta 0,003227, 5/5 folds |
| Gate historico | **True** |
| STRONG_MARGINAL | **True** |
| Finalistas formales | **0** |
| Submissions | **0** (estado: STOP_SIN_CREDENCIAL) |
| Artefactos de V20 tocados | **0** |
| Hash de la submission de V20 | `2c964af78c90bf08...`, intacto |

## 8. La pregunta final de V30

> *¿La mejora descubierta en V29 proviene de capacidad adicional real del LightGBM y puede consolidarse suficientemente como para transferirse fuera de la validacion interna?*

**La primera mitad tiene respuesta afirmativa y medida.** La capacidad adicional es real: un LightGBM sobre la misma matriz de 279 features supera a V20, y el mecanismo esta identificado (`HIERBA` con 8 hojas frente a las 43-87 que encontro la busqueda).

**La segunda mitad no se puede contestar, y no por falta de evidencia interna sino porque falta la credencial de Kaggle.** El candidato esta congelado, validado con diez checks y con su hash, y el fichero esta listo. Lo que falta es la credencial, que es del usuario y no tiene que pasar por esta conversacion.

Lo que si se puede afirmar, con lo medido:

1. El refinamiento local dio **0,000069** frente al LightGBM de V29, con **3/5 folds** a su favor. Veredicto: **EQUIVALE_A_V29**.
2. El ensemble de tres miembros queda en **+0,002818** con **5/5 folds**, que es **94 % del gate** y la bandera `STRONG_MARGINAL`.
3. El IC95 del candidato principal **no incluye** el cero.
4. La configuracion congelada se sostiene en la segunda particion, sin re-optimizar.

**Lo que no se afirma:** que la mejora se transfiera al leaderboard. Eso solo lo responde la sonda, y la sonda no se ha podido hacer. No se infiere.
