# Pre-registro: validacion confirmatoria del plug-in por clase

**Fecha:** 2026-10-11
**Commit del pre-registro:** `e1ad973740b873397b59e015f8386c4b143f69cf`
**Baseline congelado:** V30 = 0.6134471
**Pesos congelados:** `[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`

> Este documento se commitea **antes** de ejecutar la segunda particion. Fija el protocolo, los pesos, los controles y el criterio de exito. Nada posterior puede cambiarlos.

---

## Estado del pre-registro

Este archivo se ejecuta y se commitea **antes** de mirar el resultado de la segunda particion. Fija
el protocolo, los pesos, los controles y el criterio de exito. Nada de lo que venga despues puede
cambiarlos.

## 1. Procedencia exacta de los pesos

Los pesos congelados son:

```text
w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]
```

**Como se derivaron, con formula:**

Por cada fold `k` de la particion **canonica** (la de V20/V30, `StratifiedGroupKFold` sobre
`dup_clusters`, semilla 42) se resolvio

```text
w_k = argmax_w  macro_F1( y[idx],  argmax_c( w_c * p_c[idx] ) )     con  idx = { i : fold(i) != k }
```

por **ascenso por coordenadas en log-espacio**: para cada clase se prueba una rejilla de 29
multiplicadores `exp(linspace(-0.7, 0.7, 29))` dejando las demas fijas, iterando hasta que no
mejora (maximo 8 pasadas). La clase 0 se ancla en 1.0 porque multiplicar **todos** los pesos por una
constante no cambia ningun `argmax`: solo hay 5 grados de libertad reales.

Y los pesos congelados son **la media aritmetica de los cinco**:

```text
w = (1/5) * suma_{k=0..4} w_k
```

## El problema metodologico que se declara

La union de los cinco `idx` cubre **las 169.336 filas**. Por tanto, cuando estos pesos se apliquen
a la segunda particion —que reparte las **mismas** 169.336 filas de otra forma— esas filas **si
fueron usadas para ajustar los pesos**. Es decir: la evaluacion con pesos congelados **no es una
evaluacion fuera de muestra para los pesos**.

Hay un dato que confirma que el efecto es real y no despreciable: en la particion canonica, los
pesos congelados (media) dieron **+0,0025437**, mientras que los anidados dieron **+0,0020172**.
La version congelada puntua **mas alto**, que es exactamente la firma de haber visto los datos.

Por eso este pre-registro mide **dos numeros**, y los dos se reportan:

| Medida | Que es | Papel |
|---|---|---|
| `congelado` | los pesos congelados aplicados a la segunda particion | **el que pide el brief**: responde si el candidato mejora bajo otro reparto |
| `nested` | pesos **re-derivados** dentro de la segunda particion (cada fold en los otros cuatro) | **el test limpio del mecanismo**: si el procedimiento reproduce, no depende de la particion |
| `nested_con_congelados` | pesos re-derivados en la segunda particion, tabulados contra los congelados | §7 del brief: estabilidad del mecanismo frente a tuning dependiente de la particion |

El resultado confirmatorio que decide es `congelado`, con su caveat declarado. El `nested` es el
diagnostico que permite interpretarlo: si `nested` tambien es positivo, el mecanismo aparece solo;
si `nested` es positivo y `congelado` no, la diferencia es de pesos. Si los dos son negativos, la
hipotesis cae.

## 2. Auditoria de la afirmacion teorica

Conviene separar lo que la literatura sostiene de lo que es una adaptacion propia.

### TEORIA, con su ambito exacto

- **Lipton, Elkan y Naryanaswamy, ICML 2014** (*Optimal Thresholding of Classifiers to Maximize F1
  Measure*) demuestra, para clasificacion **binaria**, que la regla optima para F1 es un **umbral**
  sobre el score, y que si las probabilidades estan bien calibradas el umbral optimo es `F*/2`.
  **Ambito: binario.**
- **Dembczynski et al.** (arXiv 1402.1892) extiende las reglas plug-in al caso **multietiqueta**.
  **Ambito: multietiqueta, donde cada etiqueta es una decision binaria independiente.**
- **arXiv 2511.21794** propone sustituir `argmax` por un umbral multidimensional en el simplex para
  **multiclase**, y observa que es una regla de inferencia que no requiere reentrenar.

### LO QUE NO ES TEORIA, y se declara como adaptacion

- **No existe**, en lo que se ha revisado, un teorema que diga que `argmax_c(w_c * p_c)` es
  **optimo** para macro-F1 **multiclase con una sola etiqueta por fila**. El caso binario no lo
  implica, y el multietiqueta tiene una estructura distinta porque las etiquetas son decisiones
  independientes, mientras que aqui son mutuamente excluyentes.
- La forma concreta que se usa **—pesos multiplicativos por clase, elemento a elemento— es una
  adaptacion propia**. Es el analogo multiclase natural y es una heuristica estandar, pero:
  1. **no esta probada como optima**;
  2. **es una familia restringida**: solo puede reescalar cada clase, no expresar cualquier region
     de decision;
  3. **su unico aval es empirico**: se optimiza sobre validacion y se mide. No se apoya en una
     garantia.

### La justificacion que si se sostiene, y es mas debil que un teorema

Dos hechos medidos, ninguno teorico:

1. `argmax` **maximiza accuracy**, no macro-F1. Cuando el reparto predicho se desvia del real hacia
   las clases faciles —y se desvia, esta medido: 0,146 a 0,187 predicho frente a 0,160 a 0,170
   real— macro-F1 tiene margen que `argmax` no aprovecha.
2. La familia `argmax_c(w_c * p_c)` **contiene** a `argmax` (con todos los pesos a 1), asi que
   cualquier mejora sobre `argmax` es por definicion una mejora dentro de la familia, y su tamano se
   mide en validacion.

**Formulacion honesta**: no se afirma que la regla sea optima. Se afirma que (a) `argmax` no lo es
para macro-F1, (b) esta familia contiene a `argmax` y permite favorecer clases infra-predichas, y
(c) en la particion canonica la mejora medida es `+0,0020172` con controles favorables. Lo que la
segunda particion tiene que decir es si (c) se reproduce bajo otro reparto.

## 3. Protocolo congelado

| Elemento | Valor |
|---|---|
| Particion alternativa | `StratifiedGroupKFold` 5 sobre los **mismos** `dup_clusters`, **semilla 987654** (la de V30-05, para no elegir semilla) |
| Reentrenamiento | V20 y los dos LightGBM de V30 **desde cero** sobre los folds nuevos, con las **configuraciones congeladas** de V30 (sin reoptimizar) |
| Mezcla | los **pesos medios del manifest de V30**: `v20_f3b 0,3548 · v29_lgbm 0,3354 · v30_lgbm 0,3098` |
| Regla de decision | `argmax_c(w_c * p_c)` con `w` **congelado** |
| Controles | `uniforme` (todos a 1), `permutado` (los mismos pesos con las clases en orden barajado), `aleatorio` (pesos log-normales con la misma dispersion) |
| Criterio de exito | `delta >= +0,0015` **y** `>= 4/5 folds positivos` **y** `real > permutado` **y** `real > aleatorio` |
| Bootstrap | solo si confirma: pareado, 2.000 remuestreos, unidad la fila, semilla 20261010 (no se cambia para mejorar el IC) |
| Alta confianza | umbral 0,60, el historico del proyecto |

**Semilla del bootstrap**: 20261010, la misma del diagnostico canonico. **No se elige semilla.**

## 4. Lo que NO se hace

No se reajustan los pesos. No se busca otra rejilla. No se elige otra semilla. No se ejecuta una
tercera particion. No se usa el test —ni siquiera para contar filas. No se usa el holdout. No se
consulta el leaderboard. No se genera submission. No se proyecta ningun score.

## 5. Criterio de decision pre-registrado

| Resultado | Clasificacion |
|---|---|
| `congelado` cruza el criterio completo | **PROMISING_CONFIRMED** (no FINALIST: falta la segunda particion *independiente* y no se declara adoptado) |
| `congelado` positivo pero bajo el gate, con `nested` positivo | **MARGINAL**: el mecanismo aparece, la magnitud no basta |
| `congelado` positivo y `nested` negativo | **UNSTABLE**: el resultado depende de los pesos concretos |
| `congelado` negativo | **UNSTABLE** si el signo cambia respecto de la canonica; **MARGINAL** si es positivo pero menor |
| `congelado` negativo con controles favorables a los controles | **CLOSED** |
