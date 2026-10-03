# V12-03: entrenamiento por pares sobre TF-IDF disperso (H3)

Generado por `experiments/v12_03_entrenamiento_por_pares.py` (V12-03 (H3, entrenamiento por pares sobre TF-IDF disperso)).

## 1. La pregunta

> Una funcion de entrenamiento por pares sobre TF-IDF disperso puede explotar la estructura
> relativa entre HS, TR y AG que el objetivo binario independiente no esta aprovechando?

V03 ajusta **tres clasificadores binarios independientes**, y su objetivo es la suma de
tres entropias cruzadas que no se hablan entre si. Eso deja sin usar una informacion que el
propio conjunto de entrenamiento contiene: **HS es una condicion necesaria de TR y de AG**.
Es una relacion de **orden** entre las tres etiquetas, y un objetivo que las trata por
separado no puede expressarla.

## 2. Que es por pares aqui, y que mecanismos NO es

La auditoria de brecha describio H3 como pares de **ejemplos**. La pregunta de la fase es mas
especifica: pares de **etiquetas** dentro de cada ejemplo. Se implementa la segunda, que es
la que ataca la estructura relativa entre las tres etiquetas, y se deja constancia de la
diferencia porque son mecanismos distintos.

El termino, con margen fijo de 1,0 en unidades de logit:

- si `HS=1`, TR y AG estan definidas y HS es cierta, luego se espera `z_h >= z_t` y
  `z_h >= z_a`, con penalizacion `max(0, m - (z_h - z_t))` y `max(0, m - (z_h - z_a))`;
- si `HS=0`, TR y AG no estan definidas y se espera lo contrario, con penalizacion
  `max(0, m - (z_t - z_h))` y `max(0, m - (z_a - z_h))`.

**No es la cadena cerrada E2-01**, que anade el score de la etiqueta previa como feature:
aqui los tres clasificadores se acoplan **en la funcion objetivo**. **No es el post-proceso
duro**, que usa HS predicho como llave en la prediccion: aqui **no hay llave**, la
prediccion sigue siendo la de V03 con los mismos umbrales congelados, y lo unico que cambia
es la funcion que se minimiza. Ese es justo el defecto que H2 le atribuyo al post-proceso.

## 3. La derivada, verificada antes de usar ningun resultado

El termino acopla tres clasificadores y por eso exige un optimizador propio, igual que en
H2. **El vector de coeficientes va primero**, que es como lo invoca el optimizador, para que
la verificacion y el ajuste recorran el mismo camino de llamada.

| lambda | punto | coordenadas | residuo maximo | tolerancia | correcto |
| --- | --- | --- | --- | --- | --- |
| 0.0 | aleatorio | 12 | 1.557e-07 | 1e-06 | si |
| 0.0 | cerca de la solucion | 12 | 1.043e-07 | 1e-06 | si |
| 0.25 | aleatorio | 12 | 2.163e-07 | 1e-06 | si |
| 0.25 | cerca de la solucion | 12 | 1.152e-07 | 1e-06 | si |
| 1.0 | aleatorio | 12 | 3.627e-07 | 1e-06 | si |
| 1.0 | cerca de la solucion | 12 | 2.774e-07 | 1e-06 | si |

## 4. Resultados, y el control interno lambda=0

| brazo | metrica corregida | delta vs A | HS F1 | TR F1 | AG F1 | celdas que cambian | deteccion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lambda=0.0 | 0.777265 | +0.000053 | 0.7623 | 0.8153 | 0.7542 | 3 | 0.6667 |
| lambda=0.25 | 0.774742 | -0.002469 | 0.7524 | 0.8186 | 0.7532 | 118 | 0.4915 |
| lambda=1.0 | 0.769274 | -0.007938 | 0.7435 | 0.8134 | 0.7509 | 199 | 0.4523 |

**El control interno `lambda=0` da +0.000053** con
3 celdas de diferencia. Con el termino por pares a peso cero
el objetivo es la suma de las tres entropias cruzadas, que es exactamente V03, asi que
este control valida a la vez el objetivo, el optimizador, el pipeline y los pesos de clase.

### Delta por pliegue

| brazo | pliegue 1 | pliegue 2 | pliegue 3 | pliegue 4 | pliegue 5 | a favor |
| --- | --- | --- | --- | --- | --- | --- |
| lambda=0.0 | +0.000000 | +0.001492 | +0.000000 | -0.002281 | +0.000986 | 2/5 |
| lambda=0.25 | -0.009131 | -0.002267 | +0.001458 | +0.000322 | -0.002888 | 2/5 |
| lambda=1.0 | -0.011271 | -0.015922 | -0.011989 | +0.006564 | -0.006954 | 1/5 |

## 5. El efecto sobre la dependencia HS hacia TR y AG

Es la medida que hace especifica a esta hipotesis, porque el mecanismo acopla las tres
etiquetas y deberia notarse justo en la relacion entre HS y las otras dos.

| brazo | F1 TR donde esta definida | F1 AG donde esta definida | F1 HS global | FP de TR en zona no definida | FP de AG en zona no definida |
| --- | --- | --- | --- | --- | --- |
| V03 (A) | 0.8941 | 0.8328 | 0.7617 | 78 | 104 |
| lambda=0.0 | 0.8927 | 0.8328 | 0.7623 | 77 | 104 |
| lambda=0.25 | 0.9024 | 0.8414 | 0.7524 | 84 | 121 |
| lambda=1.0 | 0.9091 | 0.8473 | 0.7435 | 97 | 130 |

Los dos ultimos numeros son los que el oraculo de H2 corrigio: 78 falsos positivos de TR y 104 de AG. El oraculo cambiaba 196 celdas, que son esos 78 mas 104 mas 14 positivos reales de AG que el oraculo tambien eliminaba por estar en la zona no definida. Un mecanismo que
atacase la dependencia deberia reducirlos **sin** a costa de las F1 donde las etiquetas si
estan definidas, porque corregir la estructura no es recortar positivos.

## 6. Gates

| brazo | delta | pliegues a favor | IC95 | excluye cero | deteccion | exploratorio | candidato |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lambda=0.0 | +0.000053 | 2/5 | [-0.001094, +0.001182] | no | 0.6667 > 0.5166 | no | no |
| lambda=0.25 | -0.002469 | 2/5 | [-0.008159, +0.003328] | no | 0.4915 > 0.5166 | no | no |
| lambda=1.0 | -0.007938 | 1/5 | [-0.015542, +0.000206] | no | 0.4523 > 0.5166 | no | no |

## 7. La curva de lambda, diagnostico del cierre

**El maximo de esta curva no es adoptable**, porque se ha calculado mirando los pliegues
exteriores. Se reporta solo como diagnostico.

| lambda | metrica corregida | delta frente a V03 |
| --- | --- | --- |
| 0.00 | 0.777265 | +0.000053 |
| 0.10 | 0.777187 | -0.000024 |
| 0.25 | 0.774742 | -0.002469 |
| 0.50 | 0.771042 | -0.006169 |
| 1.00 | 0.769274 | -0.007938 |
| 2.00 | 0.766954 | -0.010258 |

- Maximo: **lambda = 0.0** con **+0.000053**.
- Recorrido de la curva: **0.010311**.
- El control `lambda=0` da **+0.000053**.

## 8. Salud del modelo

- Iteraciones maximas de L-BFGS-B: **91** de 2000.
- Evaluaciones de funcion: **104**.
- Gradiente verificado: **True**.

## 9. Veredicto

**NINGUN BRAZO SUPERA EL GATE EXPLORATORIO.**

El mejor valor de lambda da **-0.002469**, y eso es el **0.0 por ciento** del techo de +0.046271
que H2 midio con la condicion de HS verdadera.

Con las tres hipotesis cerradas, la lectura es de **rendimientos decrecientes en el
objetivo de entrenamiento**: H1, que era regularizacion, +0.002738 sostenido por un
solo pliegue; H2, que era la perdida, +0.001616 con intervalo incluyendo el cero; y H3,
que es el acoplamiento por pares, -0.002469. Tres mecanismos distintos sobre la misma
funcion, ninguno cerca del techo, y el techo sigue intacto.

Y la conclusion que hay que escribir sin sobregeneralizar, porque es la parte importante
de este cierre: **H2 y H3 no demuestran que la funcion de perdida no sea el problema en
sentido absoluto**. Demuestran algo mas preciso y mas util: **las variaciones de
objetivo evaluadas hasta ahora tienen un impacto pequeno comparado con el techo que
introduce el error de la condicion de HS**. Un solo mecanismo positivo, el GCE, no
autoriza a descartar la familia entera; lo que autoriza es a decir que el margen
recuperable por esa via es pequeno frente a otro problema medido.

Fecha: 2026-09-26T23:37:10+00:00. Duracion: 93.8 s.
