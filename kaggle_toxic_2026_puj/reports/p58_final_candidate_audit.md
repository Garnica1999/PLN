# P58 - Auditoria final de candidatos existentes

Generado por `experiments/v58_00_auditoria_candidatos.py`. Fecha 2026-09-28T03:46:30+00:00.

**0 modelos. 0 entrenamientos. 0 submissions. 0 consultas a Kaggle. 0 holdout. 0 test.** La unica transformacion permitida es validacion formal de artefactos que ya existen: cargar, verificar esquema, calcular hash y recomputar metricas desde OOF ya en disco.

## 1. Baseline y su integridad

```text
id           v03_lr_union_thr
corrected_cv 0.7772114525165721
LB           0.78306
sha256       7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
verificado   True
OOF reproducido exactamente desde disco: True
```

El script **aborta** si el sha256 no coincide o si el OOF canonico no reproduce el corrected_cv al ultimo digito.

## 2. Que se audito y como, sin reentrenar

| artefacto en disco | contenido | identificacion |
|---|---|---|
| `features/oof_v03_final.npz` | OOF canonico de V03, 1483 x 3 | referencia |
| `features/v2_meta_oof.npz` (worktree externo) | `META-BASES-RIDGE`, ridge alpha 3, **44 features** | es la arquitectura de `v08_meta_v2` |
| `features/oof_candidates_v4.npz` (worktree externo) | A, B, C, D de V4-X01 | los cuatro candidatos del gate |

Los ids y las etiquetas de los tres se verificaron identicos antes de comparar. Las cinco descomposiciones resultantes **reproducen `reports/v4_candidates.json` al sexto decimal**, lo que confirma que los tensores son los que alimentaron el gate pre-registrado.

## 3. Descomposicion celular, recomputada desde OOF existente

```text
candidato                                      ccv      delta plieg cambian    r    d  neto  acierto  LB
META-BASES-RIDGE (= v08_meta_v2)          0.787357  +0.010146   4/5     246  132  114   +18   0.5366  0.78163
A_cost_sensitive_downweight               0.785416  +0.008205   5/5     173   99   74   +25   0.5723  sin veredicto
B_meta36                                  0.782774  +0.005562   4/5     186  115   71   +44   0.6183  sin veredicto
C_meta36_downweight                       0.788173  +0.010962   4/5     212  134   78   +56   0.6321  sin veredicto
D_svd300_rbf                              0.790410  +0.013198   5/5     240  136  104   +32   0.5667  0.77938
```

Los totales son sobre 4449 celdas. El `acierto` es la proporcion de celdas modificadas en las que el candidato acierta donde V03 falla, y es la medida honesta del efecto.

## 4. E2-13-META-ENT-RIDGE: que significa el cierre por sparsity

Este es el caso que P57 reporto como el unico candidato con delta honesto anidado positivo y sin veredicto externo. **Esa lectura era incorrecta, y aqui esta la prueba.**

### 4.1 Que significa exactamente

El canal de target encoding de menciones, hashtags y dominios tiene una cobertura informativa de **0.52 por ciento**, frente a un umbral de cierre de 20 por ciento.

| tipo | instancias | cubiertas | cobertura | filas con entidad | filas cubiertas | unicos en dev |
|---|---:|---:|---:|---:|---:|---:|
| mencion | 1079 | 6 | 0.556 por ciento | 702 | 6 | 969 |
| hashtag | 255 | 1 | 0.392 por ciento | 144 | 1 | 210 |
| dominio | 458 | 458 | 100.000 por ciento | 458 | 458 | 1 |

La lectura de esa tabla: de 1079 menciones sobreviven **6**; de 255 hashtags sobrevive **1**; y el unico dominio que supera el filtro de frecuencia minima es `t.co`, con 1 unico en desarrollo, o sea el dominio trivial que aporta cero informacion.

### 4.2 Sparsity no invalida una mejora por si solo. Aqui esta la prueba

La directiva es explicita en no dar por hecho que baja cobertura invalida una mejora. Por eso se cuantifica lo que el canal aporta **sobre su propio comparador**, que es la pregunta correcta:

```text
comparador                          META-BASES-RIDGE (0.7874)
E2-13-META-ENT-RIDGE                0.7884666035
delta sobre el comparador           +0.0011091427
pliegues mejorados sobre el comparador  3 de 5
complementariedad (ensemble_gain)    -0.0002559245
aporta con IC positivo              False
delta sobre V03                     +0.0112666035  (este NO es suyo)
```

**El cierre es metodologico y esta bien justificado.** Un canal que cubre el 0,52 por ciento de las instancias no puede producir una ganancia transferible, y la medicion lo confirma: +0,0011 con 3 de 5 pliegues y complementariedad negativa. El sparsity es la causa y el incremento medido es la prueba.

### 4.3 El error de P57, corregido

> El +0,0113 es el delta de la FAMILIA META, no de este canal. Su comparador es META-BASES-RIDGE (0.7874), que se envio como v08_meta_v2 y dio LB 0,78163. E2-13-META-ENT-RIDGE es v08_meta_v2 mas un canal que cubre el 0,52 por ciento de las instancias. No es un mecanismo nuevo: es una perturbacion de un candidato ya rechazado externamente.

Es decir: E2-13-META-ENT-RIDGE **no es un mecanismo nuevo**. Es `v08_meta_v2`, que se envio y dio LB 0,78163, mas un canal de cobertura 0,52 por ciento. El `+0,0113` que P57 citaba como su delta sobre V03 es **la ganancia de la familia META**, que ya fue refutada externamente cinco veces.

| pregunta de la directiva | respuesta |
|---|---|
| que significa sparsity | La cobertura informativa del canal de entidades es 0.52 por ciento, contra un umbral de cierre de 20 por ciento. El canal de target encoding de menciones, hashtags y dominios sobrevive en muy pocas instancias. |
| por que invalida | No lo invalida por el numero, sino porque el incremento medido sobre su propio comparador (META-BASES-RIDGE (0.7874)) es de +0.001109 con 3 de 5 pliegues, y la complementariedad da -0.000256 con IC no positivo. O sea: el sparsity es la CAUSA, y el incremento medido es la PRUEBA de que la causa es real. |
| el cierre es metodologico | True |
| es reproducible | SI, el corrected_cv y el desglose de cobertura estan en reports/backlog_restante_v2.json |
| existe artefacto completo | NO. No existe CSV de submission para esta variante |
| tiene submission | ninguna |
| tiene leakage | NO detectado. Las tablas de entidad se ajustan fold-locales y el bucket desconocido recibe el prior del pliegue, segun las notas del registro |
| existe evidencia externa | NINGUNA. Nunca se envio a Kaggle |

## 5. Los siete criterios de la directiva, candidato por candidato

| candidato | 1 CV | 2 repro | 3 plieg | 4 leak | 5 mec | 6 artefacto | 7 no LB | FINALISTA |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| META-BASES-RIDGE (= v08_meta_v2) | si | si | si | si | si | si | **NO** | **NO** |
| A_cost_sensitive_downweight | si | si | si | si | si | **NO** | si | **NO** |
| B_meta36 | si | si | si | si | si | **NO** | si | **NO** |
| C_meta36_downweight | si | si | si | si | si | si | **NO** | **NO** |
| D_svd300_rbf | si | si | si | si | si | si | **NO** | **NO** |

**Criterio 6, `artefacto de prediccion completo`, es el que decide.** Solo tres de los cinco tienen CSV de submission, y los tres ya fueron enviados o ya fueron rechazados por el holdout.

## 6. El caso limite: B_meta36

Es el unico candidato que **pasa el gate D-016 pre-registrado** (IC95 [+0.003147, +0.016633]) y **no tiene ningun veredicto externo**. Se audita con el detalle que la directiva pide.

| campo | valor |
|---|---|
| evidence_internal | corrected_cv 0,782774 (+0,005563), 4 de 5 pliegues, IC95 positivo y que no incluye el cero, 115 rescates contra 71 danos, acierto 0.6183 en las 186 celdas que cambia |
| evidence_external | **ninguna**. Nunca se envio |
| transfer_risk | **alto**, y no por el candidato sino por la familia |
| known_failure_mode | la familia META ha transferring a la baja 5 de 5 veces |

**Por que el riesgo de transferencia es alto notwithstanding que B pase su gate local:**

1. **No existe artefacto de prediccion.** `experiments/36_candidatos_v4.py` no escribe ninguna submission; solo `C` recibio una, via `38_refit_final_v4.py`, y quedo marcada DESCARTADO. El criterio 6 falla y la directiva 12 prohibe crear una variante nueva.
2. **El unico instrumento de adjudicion esta agotado.** El holdout lleva 5 de 5 evaluaciones, luego B no puede pasar por el unico filtro que el proyecto uso para decidir en la fase final.
3. **Su hermano con veredicto fue rechazado, y es casi indistinguible de un rechazo ya medido.** `FIN-01-C-REF-44` registra `delta_vs_meta_v08 = -4,25e-05`: es decir, `C` es estadisticamente el mismo modelo que `v08_meta_v2`, que dio LB 0,78163. `B` es `C` sin el downweight, luego no hay evidencia de que se comporte de otro modo.

**Conclusion sobre B:** pasa cinco de los siete criterios y falla en el sexto por ausencia de artefacto, no por defecto del modelo. No es un finalista, y no puede devenirlo en P58 sin crear una variante, que la directiva prohibe. Es el `quasi-finalista` del proyecto y conviene nombrarlo como tal.

## 7. D_svd300_rbf: por que el CV mas alto no transfirio

| magnitud | valor |
|---|---:|
| corrected_cv | 0.790410 |
| pliegues que mejoran | 5 de 5 |
| celdas que cambia | 240 |
| rescate / damage / neto | 136 / 104 / +32 |
| LB | 0.77938 (-0.00368) |

> La firma mecanica esta en los vectores de soporte: 1321 de 1731 para HS, 1317 para AG y 1044 para TR, es decir el 60 a 76 por ciento de las muestras. Un kernel RBF con gamma=scale sobre 300 dimensiones memoriza en vez de generalizar. El gap local-LB pasa de +0,005849 de V03 a -0,011030.

**`CLOSED. El LB ya aporto la evidencia negativa suficiente`**

El patron es instructivo: **el mejor corrected_cv del proyecto, 5 de 5 pliegues, 136 rescates contra 104 danos, y McNemar p=0,0452, todo eso perdio 0,00368 en el externo.** El OOF de esta familia no es indicativo de mejora competitiva, y eso ya esta medido con un instrumento externo, que es lo mas fuerte que se puede tener.

## 8. Candidatos con LB positivo, negativo y sin LB

| categoria | numero | detalle |
|---|---:|---|
| candidatos con corrected_cv D-016 > V03 | 31 | auditados todos |
| con LB verificado y **positivo** (LB > 0,78306) | **0** | ninguno |
| con LB verificado y negativo | 3 de la lista de 31, mas 2 fuera de ella | FIN-08 -0,00411, FIN-10 -0,00156, FIN-09 -0,00328, v08_meta_v2 -0,00143, v23 -0,00368 |
| rechazados por holdout | 1 | C_meta36_downweight, 0,8163 contra 0,8285 |
| **sin ningun veredicto externo** | **2** | A_cost_sensitive_downweight y B_meta36 |

Los 31 candidatos con CV superior a V03 se pueden agrupar asi:

| grupo | n | por que no son finalistas |
|---|---:|---|
| familia META o derivados | 16 | refutada 5 veces en LB, y el unico con adjudicacion final fue rechazado por holdout |
| kernel / SVD | 1 | enviado, LB 0,77938 |
| embeddings (familia EBD) | 10 | familias cerradas por inercia con el gasto de 9 de 9 unidades agotado; EBD-07 se autodeclara `no cumple A ni B` |
| reweighting y otros mecanismos | 3 | no alcanzan +0,005 ni 3 de 5 pliegues |
| mediciones, no modelos | 1 | PA-01 mide la brecha D-004 contra D-016, no es candidata |

## 9. Validacion formal de los artefactos de submission

37 CSV en `submissions/`, **37 validos de forma formal**: 1154 filas, columnas `id,HS,TR,AG`, ids y orden identicos a `toxic_test.csv`, etiquetas binarias y sin nulos.

**Generados y nunca enviados: 14.** Ninguno se modifica en esta fase.

Un detalle que confirma una auditoria previa: `toxic_submission_v20_03_reproduccion_v03.csv` es **bit a bit identico** a la submission de V03, que es la razon por la que su envio dio delta exactamente 0.

| clase | ficheros |
|---|---|
| CUARENTENADAS, transductivas | `toxic_submission_v17_prior_shift_em.csv`, `toxic_submission_v18_near_dup_propagacion.csv`, `toxic_submission_v16_ch2_prior_shift_em.csv` |
| RECHAZADA por holdout | `toxic_submission_v12_descartado_meta36_downweight.csv` |
| INVALIDA, es una reproduccion | `toxic_submission_v20_03_reproduccion_v03.csv` |
| GASTADA, delta 0 | `toxic_submission_v32_05_url_length.csv` |

## 10. Presupuesto de submissions

```text
local_reservation   3
competition_limit   UNKNOWN
envios historicos   23
ficheros en disco   37
envios hoy          0
```

Las 3 reservas son autolimitacion del proyecto, no un limite de Kaggle. No se consulta Kaggle. El limite de la competencia queda como UNKNOWN.

## 11. Veredicto

```text
CASE        A
FINALIST    0
```

> FINALIST = NONE. Ningun artefacto cumple los siete criterios a la vez. El mas proximo es B_meta36, que pasa cinco de siete y falla en el artefacto de prediccion, que no existe y que la directiva 12 prohibe crear

### Los tres casos que la directiva contemplaba

| caso | descripcion | estado |
|---|---|---|
| A | ningun candidato cumple los 7 criterios | **ESTE** |
| B | existe exactamente uno | no aplica |
| C | existen varios, no se selecciona por score subjetivo | no aplica |

No se construye ninguna puntuacion de probabilidad de exito, y **no se predice el leaderboard**. Eso lo prohibe la directiva 10 y ademas el proyecto tiene 10 mediciones en contra de intentarlo.

## 12. Formulacion final

> V03 remains the best verified leaderboard result.

Y la distincion que la directiva 23 exige no perder:

```text
best verified result  !=  proof of global optimum
```

- **Verificado:** 0.78306 es el maximo del leaderboard en 23 envios, y 0.7772114525165721 es el corrected_cv de la configuracion congelada, reproducido aqui desde el OOF de disco al ultimo digito.
- **No afirmado:** que V03 sea globalmente optimo. La brecha OOF-LB de 0,00584855 sigue sin mecanismo identificado, y el techo del oracel de +0,2098064089 no se ha alcanzado **ni se ha descartado como alcanzable**.
- **Y para P58 concreto:** que no exista hoy un candidato defendible **no** es evidencia de que no exista una mejora defendible. Es evidencia de que **el conjunto de artefactos que hemos producido no contiene uno**, lo cual es una afirmacion sobre el estado del proyecto y no sobre el problema.

## 13. Conclusion

1. **Candidatos auditados:** los 31 con corrected_cv D-016 superior a V03, mas los cinco con OOF en disco y artefacto de submission, mas los 37 CSV validados formalmente.
2. **Con honest CV > V03:** 31. El maximo es 0,790410 (svd300_rbf); el maximo anidado es 0,788500 (E2-13-META-ENT-RIDGE).
3. **Con LB positivo:** **0**. Ninguno de los 31 supera 0,78306.
4. **Ya rechazados externamente:** 5 con LB medido en contra (v08_meta_v2 -0,00143, v23 -0,00368, v09 -0,00411, v11 -0,00156, v10 -0,00328) mas 1 por holdout (C -0,0122).
5. **E2-13:** se cerro porque su canal cubre el 0,52 por ciento de las instancias y su incremento sobre su comparador es +0,0011 con 3 de 5 pliegues y complementariedad negativa. **El cierre es metodologico y correcto.** Y el error de P58 sobre el es corregido: su delta de +0,0113 es el de la familia META, ya rechazada cinco veces, no suyo. No es un mecanismo nuevo.
6. **Candidatos sin submission:** 14 ficheros generados y nunca enviados, de los cuales 3 son transductivos y en cuarentena y 1 es una reproduccion invalida de V03. Ninguno corresponde a un finalista.
7. **Numero de finalistas: 0.**
8. **Finalistas: ninguno.** `CASE A`.
9. **Integridad de artefactos:** los tres OOF comparten ids y etiquetas, las cinco descomposiciones reproducen el artefacto del gate al sexto decimal, y de los 37 CSV, 37 pasan la validacion formal de esquema, orden, binario y nulos.
10. **Submissions reservadas: 3** de autolimitacion del proyecto. `competition_limit = UNKNOWN`, y no se consulto Kaggle.
11. **Recomendacion de uso, sin enviar nada:** mantener V03. Si en el futuro se autorizara una unica sonda, el unico artefacto que se acercaria al umbral es `B_meta36`, y para ello habria que generar su CSV de prediccion, algo que P58 no hace y la directiva le prohibe. **No se ha enviado nada y no se ha modificado ningun artefacto.**

*Seccion generada por `experiments/v58_00_auditoria_candidatos.py`. Sin push.*
