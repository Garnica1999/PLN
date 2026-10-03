# Informe de la fase v10: calibracion, distribucion y gobierno

Worktree: `/home/carlos/Git/PLN-v10/kaggle_toxic_2026_puj`.
Rama: `proyecto-kaggle-toxic-2026-v10`, desde `23aac93`.
Pre-registro: `tasks/plan_v10.md`, con un addendum de protocolo fechado antes de la via A.
Estado: `tasks/todo_v10.md`. Sin commit, por indicacion expresa.

Experimentos de la fase: **V10-00 a V10-10**, once filas nuevas en `experiments.csv`, que pasa de
207 a **218 filas** con 218 claves unicas.

## 1. La conclusion primero

**Ninguna candidata de la fase v10 se envia. V03 sigue siendo la submission recomendada**, con
0.777211 local y 0.78306 en el externo, y las cinco condiciones de adopcion del plan no las
cumple ninguna de las 218 lineas medidas.

La razon de fondo se midio en esta fase y no se podia conocer antes: **la metrica corregida tiene
una banda de ruido de mas de 0.017 entre particiones equivalentes**, con desviacion tipica de
0.0053. Ese numero se obtaining por una via completamente distinta del ruido de 0.01524 que V8-02
midio en el externo, y coincide con el. Dos instrumentos, misma magnitud.

Consecuencia directa: **el objetivo de superar 0.80 esta a 1.1 veces el ruido del leaderboard**, y
ninguna linea medida mueve la metrica una fraccion de esa cantidad. La mayor ganancia anidada de
toda la fase es de **+0.000148**, y ninguna supera el criterio de 4 de 5 particiones.

## 2. El hallazgo que reorganizo el protocolo

El gate de la via A estaba escrito, en el plan original, como "mejora anidada de al menos +0.002 en
un solo calculo". V10-02 midio que eso es **inoperante**: con una banda de 0.017, un resultado de
+0.002 es indistinguible de elegir una particion afortunada, que es exactamente el mecanismo por el
que D-067 concluyo que la mejora local de las lineas anteriores era el promediado y no la tecnica.

Por eso, **antes de ejecutar la via A** se escribio un addendum fechado que exige tres cifras para
toda candidata: delta medio sobre cinco particiones, intervalo pareado al 95 por ciento, y
**recuento de particiones que mejoran**, con adopcion solo si se cumplen las tres a la vez. Los
gates por sub-linea no se relajaron, solo se dijo con que instrumento se miden.

El addendum esta escrito en `tasks/plan_v10.md`, seccion 7, y es lo que explica que en toda la fase
el criterio decisivo sea **cuantas repeticiones mejoran** y no el delta.

## 3. Veredicto de cada sub-via

| sub-via | experimento | veredicto | cifra que lo decide |
|---|---|---|---|
| P0 verificacion | V10-00 | correcto | `corrected_cv` 0.7772114525165721 exacto |
| P0b holdout2 | V10-01 | **gate NO superado** | dev2 a 0.763436, desviacion -0.013776 |
| diagnostico del ruido | V10-02 | cerrado con hallazgo de gobierno | banda de particion de 0.017184 |
| arnes de evaluacion | V10-03 | validado en dos sentidos | nulo 0/5, oraculo 5/5 |
| P1-1 calibracion | V10-04 | **NO ACCIONABLE** | ECE mejor un 29.94 %, F1 en +0.000148 |
| P1-2 regla F1 optima | V10-05 | cerrada | -0.003921 con prior simulado, 1 de 5 |
| P1-3 umbrales estables | V10-06 | cerrada | gate 1 no superado |
| P2 shift de features | V10-07 | **VIA CERRADA** | AUC 0.487717, por debajo del azar |
| P3 integridad | V10-08 | cerrada | nada enviable, 0 de 3 evaluaciones gastadas |
| correlacion | V10-09 | cerrada en el umbral | -0.030626 con 0 de 5 |
| regla de conservacion | V10-10 | cerrada, con control | aleatoria -0.028887, brecha 0.005 |

## 4. Los cuatro resultados que mas dicen

### 4.1 El discriminador de shift esta por debajo del azar

V10-07 da un AUC de **0.487717** separando entrenamiento de prueba, con los cinco pliegues entre
0.462749 y 0.525974. El valor esta **por debajo de 0.5**: el discriminador no solo no separa, que va
algo por detras del azar. No hay shift de covariables. El unico desplazamiento medido entre las dos
poblaciones es el de prevalencia de factor 0.94 que midio V9-02, y V10-05 ya mostro que adaptarle
la capa de decision empeora. Diagnostico unicamente, sin entrenar con la prueba.

### 4.2 La regla F1 optima hace lo que se derivo, y por eso falla

V10-05 es el experimento mas instructivo de la fase, porque **el mecanismo funciono exactamente
como la teoria predijo** y aun asi perdio. Con el shift de prevalencia la regla bajo sus umbrales
entre 0.0029 y 0.0073, que es justo la adaptacion automatica que V03 no puede hacer. El problema es
la premisa: los umbrales congelados de V03, 0.44, 0.51 y 0.46, son ya **mas altos** que el optimo
plug-in con prior natural, 0.3841, 0.4093 y 0.3789. V03 ya esta por encima del optimo esperado, y la
regla se aleja de el al bajar mas.

La leccion generaliza: **los umbrales elegidos empiricamente ganan a la regla optima en esperada**,
porque la metrica de la competencia premia un punto empirico de la superficie y no el maximo de la
expectativa del clasificador.

### 4.3 La correccion: +0.0036 no era el efecto de optimizar umbrales

La fase v6 atribuyo a la reoptimizacion de umbrales un inflado de **+0.0036**, y esa cita se
repetia en la documentacion de V10-04 y V10-06. Es incorrecta. El informe de la fase v7 ya lo habia
demostrado: el control de tres niveles con umbrales congelados da **-0.0081** solo por ajustar
sobre cinco sextos de las filas, y al comparar contra un brazo que comparte conjunto de ajuste, el
efecto aislado del umbral queda en **-0.0019**. El +0.0036 era un error de comparacion entre brazos
con distinto conjunto de ajuste.

La medicion propia de V10-06, reestimar los umbrales por pliegue, da **+0.000131**. Las tres cifras
miden cosas distintas y **las tres coinciden en lo substancial: el efecto del umbral es pequeno**.
Lo que no cabe es citar +0.0036 como si fuera el efecto del umbral. Correccion registrada en D-072.

### 4.4 La correlacion: la conclusion de la v8 era correcta y su prueba no la sostenia

La fase v8 cerro la seleccion por correlacion diciendo que reducir 20.870 rasgos a 251 por magnitud
y redundancia dejaba la metrica en 0,594, y que la linea quedaba **descartada de raiz**. Ese
veredicto era correcto, pero la prueba no lo sostenia: **colapsaba el 99 por ciento de las
caracteristicas**, lo que no permite distinguir quitar una de cada par de quitar casi toda la senal,
y usaba un criterio de **magnitud**, no de redundancia.

Lo que si se midio, y nunca se habia medido:

- **Las caracteristicas estan poco correlacionadas.** La correlacion maxima por caracteristica tiene
  mediana **0.1704** y solo 1.203 pares por encima de 0.90. La redundancia alta es rara y localizada.
- **Casi todas aportan poco por si solas.** AUC mediano de 0.5133, 0.5198 y 0.5147, y solo entre
  el 0.29 y el 0.80 por ciento fuera de la banda [0.45, 0.55]. Las mejores llegan a 0.67, 0.76 y
  0.68. **El modelo lo construye combinando miles de rasgos debiles**, y por eso quitar unos pocos
  duele mas de lo que parece.
- **Eliminar una de cada par empeora en los tres umbrales, con 0 de 5 particiones**: -0.030626,
  -0.027661 y -0.027129. El dano es desproporcionado: el 1.38 por ciento de las caracteristicas
  cuesta 0.0306 de F1.

Ese desproporcion abrio una duda legitima: si el problema fuera la regla de conservacion, que se
quedaba con la caracteristica mas informativa **en solitario**, entonces la contraprueba seria
quedarse con la mas **debil**, y el control decisivo seria **al azar**.

**El control desconfirma esa hipotesis.** Las cuatro reglas dan -0.031252, -0.031879, -0.033885 y
**-0.028887**. La aleatoria es la mejor, y la brecha entre la mejor y la peor es de 0.005, menos que
la banda de ruido de 0.0053. **La regla de conservacion es irrelevante: lo que cuesta 0.03 es
cuantas caracteristicas se quitan, no cuales.**

Consecuencia: esas caracteristicas correlacionadas **llevan senal real y no sustituble**, y en la
estructura de correlacion **no hay nada que explotar**. La eliminacion por correlacion queda
descartada de forma solida, y esta vez con el control que faltaba.

## 5. Lo que la fase NO hizo

- **No se envio ninguna submission.** Las tres que habia pre-registradas siguen sin enviarse: v17
  y v18 no las respalda ningun gate de esta fase, y v19 no se genero porque ninguna candidata llego
  a la mesa de finales.
- **No se entreno con la prueba** en ningun momento. El discriminador de V10-07 es solo diagnostico.
- **No se uso el holdout v2**, que sigue agotado con 5 de 5.
- **No se toco V03** ni ninguno de sus artefactos. La referencia se verifica exacta en cada entrada.
- **P4 si se ejecuto, al final**, y dio el resultado mas instructivo de la fase sobre el
  mecanismo. El BiGRU afinado da **0.711639** frente a **0.693994** del congelado, o sea
  **+0.017645**, la mayor mejora entre dos variantes de la misma arquitectura que ha producido el
  proyecto. Pero el control con una **sola** tasa de aprendizaje da 0.699086, solo **+0.005092**:
  de los +0.017645, **+0.012553 los aporta la tasa de aprendizaje diferencial** y no el hecho de
  mover el embebido. Sigue a -0.065572 de V03, de modo que H1 queda cerrada con dos variantes y la
  via de representacion secuencial se cierra. Ver seccion 10.
- **No se abrio ninguna familia cerrada** por la evidencia de la fase. Se cerraron dos sub-vias mas.

## 6. La regla de adopcion, aplicada

Las cinco condiciones del plan, y como queda cada una para la mejor candidata de la fase:

| condicion | mejor candidata de la fase | se cumple |
|---|---|---|
| mejora anidada de al menos +0.002 | +0.000148 (V10-04) | no |
| simulacion de shift coherente | si, pero con delta negativo | no |
| holdout2 aprobado en evaluacion unica | no llego ninguna candidata | no |
| reproducibilidad exacta | si, 0.0 de diferencia | si |
| revision adversarial sin contraindicios | si | si |

Dos de cinco, y las dos que fallan son las determinantes. **V03 se mantiene.**

## 7. Lo que queda abierto y por que

La ambiguedad de fondo sigue sin resolverse, y es deliberado: **que V03 sea el maximo de 18 sondas
puede ser que sea genuinamente el mejor o el maximo de ruido de una serie de comparaciones**. Con
este tamano de conjunto no se puede distinguir, y la fase v10 no ha anadido el dato que lo
resolveria.

Lo que si ha anadido es un dato que reencuadra el problema: **el proyecto opera por debajo de su
propio suelo de medicion**. No es que las ideas sean malas; es que sus efectos son mas pequenos
que la precision con la que se pueden medir. Por eso el camino que queda no es buscar otra tecnica,
sino **cambiar el instrumento**: mas datos etiquetados, o un objetivo que no sea la F1 column-wise
y sufra menos del ruido.

## 10. P4 y el valor del control

El resultado de P4 merece su propia seccion, porque es el caso donde **el control fue mas
informativo que el experimento principal**.

La pregunta era si afinar los embeddings propios, en vez de congelarlos, hacia que el modelo
secuencial compitiera con V03. La respuesta es que **afinar funciona**: la variante afinada da
0.711639 frente a 0.693994 de la congelada. Y el mecanismo esta identificado, porque se anadio una
tercera variante que el plan no pedia, la misma red afinada pero con una sola tasa de aprendizaje
para todo:

| variante | F1 | delta sobre la congelada | mejor epoca por pliegue |
|---|---|---|---|
| congelado | 0.693994 | - | 12, 10, 10, 8, 13 |
| **afinado**, LR 1e-4 y 1e-3 | **0.711639** | **+0.017645** | 12, 11, 10, 8, 12 |
| afinando todo, LR 1e-3 | 0.699086 | **+0.005092** | 7, 7, 7, 7, 8 |

**De los +0.017645 del afinamiento, +0.012553 los aporta la tasa de aprendizaje diferencial y no
el hecho de mover el embebido.** Sin ese control, la conclusion habria sido que afinar aporta
+0.0176, cuando la version ingenua aporta +0.0051: casi cuatro veces menos.

Las mejores epocas explican el mecanismo sin ambiguedad. La variante de tasa unica para **siempre
entre 7 y 8**, muy por debajo del tope de 14, porque una tasa alta destruye la representacion
preentrenada antes de que la red pueda aprovechar. Congelado y afinado llegan a 8-13 y aprovechan
el entrenamiento completo.

El control interno del protocolo tambien esta: el congelado reproduce V9-04 en **0.693994** frente
a su 0.6943, con tope de epocas y semillas ligeramente distintos.

**La decision de sobreentrenamiento** fue **early stopping estricto**, y la fijo el pre-registro
de `tasks/plan_v10.md` antes de ejecutar nada. Con la salvedad anotada: la mejor epoca se elige por
la perdida de validacion, que es la misma senal que se usa para las curvas de aprendizaje. Para
estimar un techo eso no es fuga y la metrica de la competencia se evalua una sola vez al final; pero
si alguien quisiera reutilizar este protocolo para elegir una variante, si lo seria.

**H1 queda cerrada con dos variantes, congelada y afinada**, a -0.065572 de V03, sin sonda. La via
de representacion secuencial no se abre con mas capas, mas unidades o mas epocas, porque el criterio
de cierre estaba escrito antes de ejecutar y ya se cumplio.

## 8. Trazabilidad

- Fichas: `V10-00` a `V10-10` en `experiments.csv`, 218 filas en total.
- Decisiones: `D-068` a `D-076` en `DECISIONS.md`, 76 decisiones en total.
- Informes: `reports/v10_00_verificacion`, `v10_01_holdout2`, `v10_02_ruido_particion`,
  `v10_04_calibracion_fuerte`, `v10_05_regla_f1_optima`, `v10_06_umbrales_estables`,
  `v10_07_shift_features`, `v10_08_confirmacion_dev2`, `v10_09_correlacion_eliminacion`,
  `v10_10_regla_conservacion`, cada uno en `.md` y `.json`, mas dos `.csv` de datos.
- Artefactos congelados: `config/holdout2.json` y `data/processed/holdout2_ids.csv`, con `sha256`
  verificados en V10-08.
- Pre-registro: `tasks/plan_v10.md` con el addendum de protocolo, y `tasks/todo_v10.md`.
- Sin commit, segun indicacion del usuario.

## 9. Defectos propios corregidos durante la fase

Se dejan registrados porque un proyectowhose evidencia vale lo que vale su bitacora de errores:

- **V10-01**: `ETIQUETAS` indexada como lista con el nombre de la etiqueta, que rompia el guion.
- **V10-04**: claves de metricas inicializadas y nunca llenadas, que producian una media de lista
  vacia y un `nan` silencioso.
- **V10-05**: `pesos_para_prevalencia(y, 1.0)` asignaba **peso cero a los negativos**, la precision
  se iba a 1 y el punto fijo no convergia. Se diagnostico la trayectoria numerica **antes** de tocar
  la tolerancia, y la tolerancia no se toco.
- **V10-07**: matriz de ajuste con longitudes distintas, nombres de caracteristicas pedidos al
  estimador en vez de a los vectorizadores, y discriminador ajustado sobre filas que no eran las suyas.
- **V10-08**: la prueba de mejora era "delta mayor que cero", lo que daba por buena una candidata con
  +0.000131 y contradecia el propio veredicto del guion.
- **V10-09**: se indexaba filas donde se querian columnas, y se intentaba materializar una matriz de
  producto escalar de 4.86 GB. Ademas, el AUC se calculaba con un bucle de 74.000 ordenamientos por
  pliegue, y se vectorizo.
- **Caracteres**: ocho ideogramas CJK, dos cirilicos y un termino japones colados en los comentarios
  los detecto `tools/verificar_caracteres.py` en cada escritura. Uno de ellos, un cirilico, habia
  sobrevivido a una revision previa porque la herramienta solo detecta alfabetos no latinos.
- **Documentacion**: la cita erronea de +0.0036 en V10-04 y V10-06, corregida en D-072.
