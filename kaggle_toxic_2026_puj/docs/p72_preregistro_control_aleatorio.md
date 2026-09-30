# Pre-registro del control de aleatoriedad

Escrito y commiteado **antes** de ejecutar la corrida. Cualquiera puede comprobar el orden con
`git log`.

## 1. Que se pregunta

Antes de invertir en un lector secuencial (TextCNN) hay que separar dos hipotesis:

```text
H1  el Word2Vec NO contiene suficiente senal util
H2  el Word2Vec SI contiene senal, pero mean pooling + LogisticRegression no la recupera
```

El control compara el mismo downstream con la matriz de embeddings sustituida por una aleatoria.

## 2. Las dos ramas

```text
A   Word2Vec entrenado + mean pooling + LogisticRegression por etiqueta
B   vectores ALEATORIOS congelados + MISMO mean pooling + MISMA LogisticRegression
```

Identico en las dos ramas: vocabulario, dimension `d = 100`, normalizacion, particiones, semilla
del clasificador, umbrales congelados `0.44/0.51/0.46` mapeados por NOMBRE de etiqueta, y regla
de evaluacion.

## 3. La correccion de vocabulario que exige el revisor de compliance

```text
INCORRECTO  "los vectores aleatorios no vienen de ningun dato"
CORRECTO    "las DIRECCIONES no provienen de datos; el SOPORTE (el vocabulario) y las
             MAGNITUDES por token si vienen del corpus externo"
```

La rama aleatoria hereda del corpus externo la pertenencia al vocabulario, la norma de cada
token y, a traves del criterio de poda, la frecuencia. Se declara asi. No viola nada: la
autorizacion escrita del profesor cubre el uso de datos externos para entrenar el modelo de
vectorizacion, y el vocabulario es parte del andamiaje del vectorizador.

**Condicion obligatoria asociada**: el pipeline de vocabulario y normas **no consulta el dataset
de la competencia de ninguna forma**. Ni conteos, ni interseccion de vocabularios, ni filtrado
por tokens presentes en train.

## 4. El esquema de aleatorizacion, fijado ahora

```text
E_rand[j] = ||E_train[j]|| * u_j        u_j uniforme sobre la esfera unitaria en R^100
si ||E_train[j]|| == 0 entonces E_rand[j] = 0
```

Se aleatoriza **solo la direccion**. La norma por token se conserva exacta, lo que iguala todos
los momentos de escala de primer orden, incluido el acoplamiento norma-frecuencia del token.

Se ejecuta **un unico esquema**. No se prueban permutaciones de coordenadas ni ruido gaussiano:
elegir el esquema despues de ver cual da la nula mas debil seria seleccion.

## 5. K = 10 tiradas, fijadas ahora

```text
semillas de las tiradas   20261020, 20261021, 20261022, 20261023, 20261024,
                          20261025, 20261026, 20261027, 20261028, 20261029
```

**Las diez se reportan individualmente**, no solo sus resumenes. El resumen oculta
multimodalidad y permite recortes silenciosos.

**No es una grid.** En una grid hay un argmax sobre configuraciones y se descarta el resto. Aqui
no hay eleccion: el objeto reportado es la distribucion nula completa y la regla de decision usa
su **maximo**, que es un estadistico de orden conservador **en contra** de la hipotesis
favorable. No se amplia K ni se reduce despues de ver los numeros. Cualquier cambio se declara
como desviacion y se reporta el conjunto completo.

## 6. Reproduccion del brazo entrenado, sin busqueda

El brazo entrenado se recomputa **en la misma corrida**, con las semillas ya fijadas, **una sola
vez**. Su desviacion respecto del valor historico `0.594729` se **reporta como parte del
resultado**, no se corrige iterando.

```text
tolerancia declarada para considerar que reproduce   1e-6
si no reproduce, se reporta la desviacion, NO se reintenta
```

## 7. La regla de decision, con su referencia definida

La objeccion de compliance es que la banda `0,005-0,015` no tenia referencia. Se define ahora:

```text
delta = macro_F1(brazo A) - max(macro_F1 de las 10 tiradas de B)
```

Se usa el **maximo** de la nula, no la media, porque es lo conservador: exige que el entrenado
bata a la mejor de diez tiradas aleatorias, no a la tipica.

```text
|delta| < 0,005                              EMPATE
0,005 <= |delta| < 0,015                     SENAL DEBIL
|delta| >= 0,015 con A > max(B)              EFECTO CLARO
   y signo consistente en >= 4 de 5 pliegues
```

**El estadistico por pliegues, definido exactamente**: en cada pliegue se compara el macro-F1 de
A contra el **maximo entre las 10 tiradas** del macro-F1 de B en ese mismo pliegue. Cuenta como
positivo si A es mayor. Se cuentan los cinco pliegues, todos.

El agregado es el **macro-F1 de la metrica de competencia**, no la mejor celda etiqueta x
pliegue.

## 8. Anclas obligatorias

Para poder leer la nula en su escala, se reportan ademas:

```text
piso trivial all-positivo por etiqueta,  F1 = 2p/(1+p) con p la prevalencia
predicciones uniformes aleatorias
area bajo la curva precision-recall por etiqueta, que es threshold-free
```

El PR-AUC importa porque comparar F1 bajo umbrales congelados puede ocultar diferencias de
ranking. Si A gana en PR-AUC y empata en F1, la conclusion correcta es que hay informacion de
ranking que el umbral oculta.

## 9. Bootstrap

**Agrupado por near-duplicados**, nunca por filas. Los grupos se construyen sin etiquetas y son
los mismos en las dos ramas.

Se reporta la **semianchura** y se compara con las bandas `0,005` y `0,015`. Si la semianchura
supera `0,015`, la banda de "efecto claro" seria inalcanzable por construccion y hay que decirlo,
para no presentar como hallazgo lo que es limite de resolucion.

## 10. El parrafo de alcance, obligatorio en cualquier redaccion

> El control mide la contribucion marginal de las **direcciones aprendidas** bajo una unica
> lectura fija: mismo vocabulario, `d = 100`, mismo mean pooling, misma regresion logistica,
> mismos umbrales congelados. Las dos ramas pertenecen a la **misma familia funcional**  - un
> clasificador lineal sobre la bolsa de tokens normalizada, con coeficientes en un subespacio de
> dimension menor o igual que 100 -  y lo unico que cambia es **cual** subespacio se usa. Un empate
> acota la conclusion a esta lectura; **no prueba ni descarta** senal dependiente de orden,
> composicion o no linealidad.

Nota de asimetria, que va a favor de la honestidad del control: como el clasificador reajusta sus
coeficientes, un subespacio aleatorio de 100 dimensiones puede capturar bastante senal de
frecuencia o palabra suelta, luego **la nula puede ser fuerte y el control puede fallar en
detectar una contribucion real**. En consecuencia, un empate es evidencia **debil** y una victoria
clara del entrenado es evidencia **fuerte**. Se dice asi.

## 11. Que NO se puede afirmar con este control

```text
"Word2Vec no sirve" ni "los embeddings no aportan", como afirmaciones generales
"no hay senal semantica en los tuits"
"los vectores aleatorios son tan buenos como los entrenados", como conclusion global
"el modelo secuencial funcionara" o "no funcionara"
salvedades post-hoc de las bandas ("casi 0,005, cuenta como senal debil")
"los vectores aleatorios no usan datos": falso, el vocabulario y las normas son del corpus
cualquier afirmacion sobre test o leaderboard derivada del control
```

## 12. Condiciones de ejecucion

```text
el control corre solo sobre los pliegues de desarrollo
el test no se lee para el control
ningun brazo nulo se envia al leaderboard
vocabulario y normas se construyen sin tocar train ni test, y se declara el criterio
el correo de autorizacion del profesor se archiva como evidencia
atribucion CC BY-SA 4.0 en el reporte
los vectores derivados quedan fuera del artefacto de submission
```
