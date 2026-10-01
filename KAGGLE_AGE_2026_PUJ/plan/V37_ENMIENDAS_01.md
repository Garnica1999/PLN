# V37 · ENMENDAS 01 (pre-ejecucion, declaradas)

Fecha: 2026-10-01. Estado: **emitidas antes** de congelar el pre-registro y antes de correr nada.
Modifican `plan/V37_DIAGNOSTICO_POS_PLAN.md`. No anulan su tesis: la confirman y la endurecen.

Origen: validacion independiente de la spec por `kaggle-specialist` (protocolo, gates, veredictos) y
`nlp-specialist` (premisas POS/UPOS, unidad de muestreo, referencias). Cada punto abajo marca quien
loDetecto y si bloquea la ejecucion.

---

## E1 · BLOQUEANTE · nulo de `A_perm` mal fijado (kaggle-specialist P2, confirmado empiricamente)

**Defecto.** La spec fijaba `|A_perm - A0| <= 0,005` con `A0 = 0,0485421`. Medido: **`A_perm = 0,06575`**,
es decir `|A_perm - A0| = 0,0072`, **fuera** de la banda. Sobre un pipeline limpio el criterio
disparaba `[STOP] fuga suspected` (G1c). El mismo criterio fragil estaba replicado en cinco sitios
(linea 480, 492, 658, 825, 832).

**Causa.** Permutar las etiquetas de `train_k` conserva el multisej y por tanto los priors, pero
`theta[c,.]` de cada clase es un subconjunto aleatorio. Las clases con menos tokens reciben mas
suavizado de Laplace, se acercan a la uniforme y reciben **menor** verosimilitud para una fila
tipica; la clase mayoritaria, con menos suavizado, gana mas veces. El permutado **degenera hacia la
clase mayoritaria**, no hacia el azar uniforme: predijo la clase 0 en 159.653 de 169.336 filas.

**Enmienda.** `A_perm` tiene un unico criterio, definido en una sola constante:
- **Criterio (una sola via):** `A_perm <= 0,176667` (azar uniforme 0,166667 + 0,010). Es **unilatero**:
  un control permutado que **supere** el azar es indicio de fuga; un control que rinda por debajo es
  admisible y se reporta.
- **Se elimina** el criterio de banda alrededor de `A0` en los cinco sitios.
- Se **reporta** el valor dentro de la banda de expectativa declarada `[A0, 0,166667]`, sin
  pass/fail en el extremo inferior.
- Se mantiene, y ahora es el criterio de inferencia real: el **IC95 pareado de `A1 - A_perm` excluye
  cero**.

## E2 · CORRECCION · `A2` **no** es identico a `A1` (kaggle-specialist P1, refutado con medicion)

El auditor afirmo que `A2 - A1 == 0` por algebra. **Es falso con priors**, y lo midio el
orquestador: `A1 = 0,1720532`, `A2 = 0,2052270`, `A2 - A1 = +0,0331738`, y **48.440 de 169.336 filas
cambian de prediccion**. La invariancia se cumple solo si no hay prior (`log p(c|i) = n_i * L(c)`);
como A1 y A2 usan `log pi_c` sin escalar, la escala `n_i` cambia el balance prior/verosimilitud fila a
fila. (El especialista NLP Sinfue en este punto: dio la invariancia para el caso **sin priors**.)

**Enmienda.** A2 se conserva y se **corrige su enunciado**: no cuantifica "cuanto de A1 es longitud"
(afirmacion falsa) sino que es el **brazo de sensibilidad a la longitud**: permite que en tweets
largos decida la evidencia POS y en cortos el prior. La diferencia `A2 - A1` se interpreta como
cuanto aporta la longitud, y se declara que es exactamente `n_i` escalando el termino de
verosimilidad.

## E3 · `A_perm_val`: control del lado de la EVALUACION (kaggle-specialist P3)

`A_perm` con etiquetas de train permutadas calibra el lado del **ajuste**; es ciego a una
desalineacion entre `X`/`y` en el lado de la **evaluucion** (con el permutado, el modelo casi
predeciria una sola clase y el control pasaria).

**Enmienda.** Se anade el brazo **`A_perm_val`**: se ajusta `A1` con las etiquetas reales de `train_k`
y se puntua contra una copia de `y_true` con las etiquetas **permutadas a nivel de cluster**
(semilla 20261010). Criterio: `A_perm_val <= 0,176667`. Si lo supera, `[STOP]` por desalineacion.

## E4 · FPR conjunta de V1..V4 declarada (kaggle-specialist P3b)

Cuatro veredictos de Fase B en AND sobre la misma rejilla 18x6, sin calibrar su tasa de falso
positivo conjunta. **Enmienda:** reportar cuantas etiquetas y celdas sobreviven a los floors de V1..V4
bajo una permutacion de etiquetas. Si el nulo ya sobrevive los floors, los floors estan mal calibrados
y se dice en el informe.

## E5 · Las 104 filas sin tokens, regla declarada (kaggle-specialist P7)

`x = n_tokens/n_tokens = 0/0` esta indefinido (verificado: **104 filas** con `pos_seq` vacio).
**Enmienda:** regla unica y declarada: `x = 0` para esas filas, luego `pred = argmax_c pi_c`; se
cuentan y se reportan en el JSON. No se excluyen filas: se Incluyen con la regla escrita.

## E6 · Convenciones de metrica y comparacion like-for-like (kaggle-specialist P4)

**Enmienda:** A1 se reporta en **dos convenciones** (pooled sobre las 169.336 filas, y media de
macro-F1 por fold). Toda comparacion con P2, P1a y el campeon se hace **like-for-like** y se declara
la convencion de cada numero. `A1 - P2` y `A1 - P1a` se declaran **estimaciones puntuales con el IC
de cada lado, sin claim de significancia**, porque proceden de estimadores y convenciones distintas.

## E7 · V5 degrada de gate de novedad a sanity floor (kaggle-specialist P5, P6)

El floor `A1 >= 0,20` estaba justificado contra el suelo estricto (0,0485), ignorando que el azar
uniforme es 0,1667, y usaba un brazo **rechazado** (P2) como vara de novedad.

**Enmienda:** V5 pasa a **sanity floor**: `A1 > 0,176667` (azar + 0,010, el mismo margen que el
criterio de fuga) para que "supera el azar" sea una afirmacion con margen, no una moneda al aire. La
**novedad no se decide con V5**: se decide con el argumento de las cinco razones de la seccion 6.2. Se
reporta la posicion de A1 frente a P1a (0,3997), P2 (0,2751) y campeon (0,6159732) como tabla, sin
umbral.

## E8 · La rama PASS deja de ser inerte (kaggle-specialist P9, P10)

El arbol de decision converge en "conservar el campeon" en las dos ramas, luego ningun resultado
posible cambia una decision y los gates son decorativos.

**Enmienda:** la rama PASS **persiste las probabilidades OOF de A1 indexadas por fila** en
`artifacts/v37/a1_prob_oof.npy` (con `ids.npy` de referencia y su SHA256 en el JSON), y el
pre-registro declara **que prueba concreta ejecutaria una spec posterior** si el patron resultara
potente. Con eso PASS tiene consecuencia y el arbol informational deja de ser inerte. El archivo se
escribe **tambien** cuando el veredicto es cierre, porque es evidencia reutilizable y no una
apertura al campeon.

## E9 · Sonda exploratoria previa, declarada (honestidad metodologica)

Para adjudicar la disputa P1/P2 de la auditoria el orquestador ejecuto una sonda con **los modelos
exactos de la seccion 5.2** (sin ajustar nada): `A1 = 0,1720532`, `A2 = 0,2052270`,
`A_perm = 0,06575`. No se toco ningun parametro y no hay paso de seleccion, luego no introduce
sesgo de ajuste, pero **es una medicion previa a la congelacion** y por eso se declara.

**Enmienda:** el pre-registro incluye `sonda_previa` con esos tres valores y la regla: la corrida
congelada debe **reproducirlos con tolerancia 1e-6**; si no reproduce, `[STOP]` y se investiga la
divergencia antes de leer ningun veredicto. Esto convierte la sonda en un test de reproducibilidad.

## E10 · `SPACE` entra en la estandarizacion por longitud (nlp-specialist)

`SPACE` no esta en el Universal POS tagset (viene de `token.is_space`), ocurre 32.865 veces en 25.232
filas (14,9 % del corpus), y su cuota **depende de la clase**: 1,956 % de los tokens en clase 0 frente
a 0,911 % en clase 3 (2,1x); filas con `SPACE` 19,35 % vs 12,09 %. Ademas esta confundido con la
longitud por construccion (longitud media 13,19 -> 18,06 tokens, r = 0,224 con la edad). La spec lo
trata como fila 18 mas una linea de sensibilidad, y su afirmacion de cuota "desinflada ~1,3 %" es un
promedio pooled que esconde el spread 2,1x.

**Enmienda:** `SPACE` se reporta **dentro de la estandarizacion A1.2** (quintiles de `n_tokens`), no
solo como fila mas; la cuota por clase se publica por quintil y no solo pooled; y se declara como
limite que es el unico "tag" confundido con longitud por construccion.

## E11 · La unidad del error estandar es el tweet, no el token (nlp-specialist)

La spec (linea 238) trata 2.590.877 tokens como iid. La unidad de muestreo es el tweet. Con el
estimador de razon (delta method) el SE correcto es **~3x menor** (factores 0,28-0,41 en
PRON/DET/ADP/NOUN/VERB), lo que **cambia veredictos**: `INTJ` pasa de z = 1,09 (n.s.) a z = 13,2;
`NUM` de 1,57 a 7,51; `SPACE` de 4,28 a 23,3.

**Enmienda:** (1) el error analitico se calcula con la unidad tweet y se etiqueta como **secundario**;
(2) **la inferencia la gobierna el bootstrap por grupos de duplicados** (B3), que es el instrumento
correcto; (3) todo veredicto "no significativo" del error analitico token-iid se **re-deriva** y se
reporta el nuevo valor. La spec era conservadora pero equivocada: se corrige en vez de mantener el
error.

## E12 · Declarar que `p[c,t]` es ratio de sumas ponderado por token (nlp-specialist)

`p[c,t] = n_tokens[c,t] / n_tokens[c]` **no** es la media de proporciones por tweet: es una razon de
sumas, ponderada por token, y por tanto confunde preferencia POS con longitud (y la longitud esta
correlacionada con la edad, E10). **Enmienda:** se declara como eleccion consciente con su sesgo, y se
publica **en paralelo** la version de media de proporciones por tweet, de modo que el
document-level mixture problem quede量化 y visible en lugar de oculto.

## E13 · B2 (md vs lg) degrada a sanity check (nlp-specialist)

Los parquets congelados contienen **solo UPOS** en `pos_seq`; el tag especifico del modelo nunca se
congelo. Ambos anotadores proyectan sobre el **mismo inventario de 18** por construccion, luego la
comparacion no puede discriminar el modelo. Ademas no son "modelos independientes": mismo pipeline y
mismo tokenizer. La coincidencia de secuencia fila a fila es solo **16,3 %**, y el desajuste de
presencia (~6,0 %) es casi constante entre clases.

**Enmienda:** B2 pasa de "comparacion entre anotadores" a **sanity check de la proyeccion universal**;
se **elimina** laformula "modelos independientes"; se publica la coincidencia de secuencia (16,3 %)
como limitacion declarada del conjunto de datos, no como resultado.

## E14 · Correcciones bibliograficas (nlp-specialist, con verificacion web)

Las cuatro referencias son **reales**; tres cifras estaban mal atribuidas:
- **Flekova y Gurevych (2013)**: 0,53 EN / 0,57 ES / mayoritario 0,33 corresponden al sistema
  **solo-edad de 3 grupos**, no al conjunto de 6 clases (que llega a 0,29 EN / 0,38 ES con mayoritario
  0,17). Las cifras de "236.600 / 75.900 documentos" son de PAN-2012 y **no aparecen** en el paper.
- **Cruz, Haro y Ortega (2013)**: el 0,62 de edad es el **accuracy del ensemble apilado completo**
  (BoW + POS + estilisticos + JRip), **no** del bloque POS. Usarlo como "los n-gramas POS dan 0,62" es
  una mala atribucion.
- **Johannsen, Hovy y Sogaard (2015)**: el paper reporta genero sobre **clases lexicass BabelNet**
  (adverbios/adjetivos), no sobre treelets POS. Se retira la afirmacion de "tres treelets atomicos para
  genero" y el texto corrupto que la acompañaba.

**Enmienda:** reescribir la fila affected con la atribucion correcta y **reencuadrar las cuatro
referencias como prior art con metricas de otra tarea**, nunca como "otros con mejores numeros": 3
grupos y accuracy contra 6 clases y macro-F1, mayoritario 0,33 contra 0,0485.

## E15 · Se cierra tambien la pregunta de "familia no medida" (nlp-specialist)

Restaban dos familias legales derivables de los artefactos congelados y **no** medidas como ratios:
subtipos UD de `dep_seq` (22 ratios) y morfologia completa (23 ratios: PronType, Case, VerbForm,
Degree). Sonda del especialista: `dep = 0,2147`, `morph = 0,2064`, ambas en la **misma banda 0,19-0,24**
que las seis familias ya cerradas, muy por debajo de P2 (0,2751), y con fusion siempre <= 0.
Ademas `SPACE` aislado da **0,04854 = exactamente la prediccion constante de la mayoritaria**: su
aporte es renormalizacion, no senal propia.

**Enmienda:** queda registrado en el informe que **no existe ninguna familia legal no medida con
potencial**, y que por tanto la conclusion "conservar el campeon" no es una decision de austeridad sino
el cierre de un espacio ya agotado. No se programan mas experimentos de rasgos POS.

---

## Resultado de la validacion: la tesis de la spec se mantiene

Las dos auditorias independientes **confirman** el veredicto previsto y **no** habilitan ninguna
corrida predictiva:

- `kaggle-specialist`: veredicto operativo **correcto** conservar el campeon por defecto; objecion de
  instrumentacion (rama PASS inerte) **resuelta por E8**; bloqueantes P1/P2 **resueltos por E2 y E1**.
- `nlp-specialist`: **VALIDA CON RESERVAS**; ninguna familia legal no medida con potencial (E15); el
  antecedente mas proximo a la fase A (Modaresi et al. 2016) es **negativo**.

El plan que Carlos aprobo ya contemplaba esta bifurcacion: si la auditoria de novedad dava (i) o (ii),
el paso (d) **no** se ejecuta. La auditoria dio **(ii) VARIANTE DISFRAZADA**. Por tanto la decision por
defecto es conservar el campeon, y V37 se ejecuta **solo** como diagnostico.