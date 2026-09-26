# V15 · B — Poda de variables redundantes

## 1. Qué se podó y con qué criterio

De **312 variables** de entrada quedan **197**. El criterio se aplicó in-fold en V15a: un representante por cluster de |rho| >= 0,90, elegido como el de mayor eta cuadrado **dentro del fold**; se conservan todas las variables que no están en ningún cluster; y se eliminan por definición las columnas constantes. La estructura de pares es **idéntica en los cinco folds** (Jaccard 1,0000), así que la poda no es una decisión que dependa del fold.

## 2. Resultado

| Modelo | macro-F1 | accuracy | Delta | Folds positivos |
|---|---|---|---|---|
| V12 (279 variables, sin podar) | 0.6005 | 0.6007 | - | - |
| V12 podado (197 variables) | 0.5965 | 0.5967 | -0.0040 | 0/5 |

Folds podado: [0.5973, 0.5978, 0.6002, 0.5976, 0.5894] · folds V12: [0.6007, 0.605, 0.6032, 0.6004, 0.5928]

## 3. Segunda partición (4 folds)

| Fold | V12 | Podado | Diferencia |
|---|---|---|---|
| 1 | 0.6013 | 0.5944 | -0.0069 |
| 2 | 0.5998 | 0.5954 | -0.0044 |
| 3 | 0.6039 | 0.5968 | -0.0070 |
| 4 | 0.6048 | 0.5971 | -0.0077 |

**Positivos: 0/4** · IC95 del bootstrap pareado [-0.0054, -0.0027]

## 4. Doble métrica (§36)

| Frontera | V12 | Podado |
|---|---|---|
| 18-24|25-34 | 0.4963 | 0.4967 |
| 25-34|35-49 | 0.4974 | 0.4928 |

Complementariedad: rescate 6,093 · daño 6,772 · **neto -679**

## 5. Gate D-042

| Criterio | Valor | Cumple |
|---|---|---|
| efecto minimo | delta -0.0040 >= 0.001 | **no** |
| folds positivos | 0/5 >= 3 | **no** |
| significacion | IC95 [-0.0054, -0.0027] | **no** |
| segunda particion | 0/4 >= 3 | **no** |
| fronteras | 1/2 mejoran | **no** |

## 6. Lectura

**La poda empeora a V12 de forma fiable, no por ruido.** El delta es de -0,0040 con 0 de 5 folds
positivos, la segunda particion da 0 de 4 con una media de -0,0065, y el intervalo de confianza del
bootstrap pareado es [-0,0054; -0,0027]: **no contiene el cero y entero por debajo**. Las tres
medidas apuntan en el mismo sentido, asi que la diferencia es real.

**Lo que significa.** Las variables casi identicas no sobraban. La intuicion de partida —que una
variable duplicada o casi duplicada es ruido que se puede quitar— es correcta para la
*interpretacion* de un modelo con coeficientes, y equivocada para la *prediccion* de un conjunto de
modelos. La lectura que encaja con todo lo demas es la del conjunto de estimadores debiles: dos
variables con correlacion 0,95 no son la misma variable, son dos lecturas ligeramente distintas del
mismo fenomeno, y promediarlas reduce la varianza igual que reduce el ruido cualquier promedio.
Quitar 115 variables elimina una parte de ese promedio, y el promedio era lo que mas ajudaba.

**Que se sigue de aqui.** La estructura de redundancia es ahora un hecho medido y estable (Jaccard
de pares 1,0000 entre folds), asi que sirve para **interpretabilidad** y para reducir el tamano del
modelo si someday hace falta, pero no como via de mejora de la metrica. La via que queda abierta es
la arquitectura, que es donde apuntan todos los diagnosticos de V13, V14 y V15: se mide en V16.

> Nota de edicion (2026-09-26): el parrafo de esta seccion se reescribio despues de generar el
> informe para corregir dos defectos de redaccion del texto de la plantilla (unas palabras en
> ingles que se colaron) y una lectura que afirmaba que las diferencias estaban dentro del ruido de
> la estimacion, lo cual contradecía al intervalo de confianza. **No se modifico ninguna cifra**: el
> script que produce el informe no se ha vuelto a ejecutar ni se ha editado, de modo que los numeros
> publicados son los que dio la ejecucion original.
