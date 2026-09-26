# Track A: ponderacion por tamano de grupo (V6-03)

Generado por `experiments/v6_03_ponderacion_grupo.py` (V6-03 (A0-A3)).

## 1. Que se prueba y por que no es una repeticion

El gap analysis (V6-00) confirmo por busqueda en el codigo que la ponderacion por
tamano de grupo **nunca se ha probado**. Lo que si se probo fue ponderar por
**dureza** (V4-B03, -0.0101) y un downweight de frontera (D-027), que son fuentes de
varianza distintas: la dureza es una propiedad del texto, el tamano de grupo es una
propiedad de como se construyo el corpus.

La pregunta es si V03, ajustado con una fila por tuit, deja que los grupos repetidos
dominen la frontera de decision, y si devolver a cada grupo la influencia de un
unico tuit mejora algo.

## 2. Distribucion de los grupos en el desarrollo

- **1415 grupos distintos** en 1483 filas.
- tamaño mediano 1, maximo 7.
- 1378 filas en grupos de tamano 1.
- 105 filas en grupos de tamano mayor que 1 (7.1 % del desarrollo).

**Esta distribucion acota el track antes de mirar los resultados.** Los cuatro
esquemas solo pueden cambiar el peso de las 105 filas en grupos
repetidos, un 7.1 % del desarrollo: el
92.9 % restante ya tiene `g = 1` y por tanto peso 1 en
los cuatro esquemas, idéntico al control. La hipótesis era plausible, pero su
alcance máximo lo fija la estructura del corpus, no el modelo.

## 3. Los cuatro esquemas y el control

| variante | esquema | corrected_cv | delta | IC95 | HS | TR | AG |
|---|---|---|---|---|---|---|---|
| A0-control | w = 1 para todos (control: debe reproducir V03 exactamente) | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 |
| A1-sqrt | w = 1 / sqrt(g): el grupo grande pierde peso sin desaparecer | 0.777768 | +0.0006 | [-0.0029, +0.0040] | 0.7638 | 0.8153 | 0.7542 |
| A2-inversa | w = 1 / g: la version agresiva, el grupo grande casi no pesa | 0.777415 | +0.0002 | [-0.0045, +0.0051] | 0.7674 | 0.8153 | 0.7496 |
| A3-tope | w = min(1 / g, 0.25): limita el dano de A2 en grupos enormes | 0.762334 | -0.0149 | [-0.0240, -0.0057] | 0.7532 | 0.7939 | 0.7399 |

**El control A0 es obligatorio**: con `w = 1` para todos y el mismo codigo, el
resultado debe ser exactamente 0.777211. Si no lo fuera, cualquier delta posterior
seria inatribuible. Es el mismo principio que se aplico en el control B0 del track B.

## 4. El patron es monotonico en la agresividad, y eso tiene lectura

| esquema | agresividad | delta |
|---|---|---|
| A1 `1/sqrt(g)` | suave | +0.0006 |
| A2 `1/g` | media | +0.0002 |
| A3 `min(1/g, 0.25)` | fuerte | -0.0149 |

Los tres esquemas se ordenan **exactamente** segun cuanto penalizan a los grupos
repetidos, y el resultado empeora al hacerlo. Como el tope de A3 es mas agresivo que
`1/g` para todo `g >= 2`, la progresion no es un accidente de los parametros.

La lectura es que **un tuit repetido no es ruido de recoleccion, es señal**: que
el mismo texto ofensivo aparezca varias veces lo hace mas representativo de la
distribucion, no menos. Bajar su peso hasta la unidad basica confunde repeticion
con azar, y por eso A3 se lleva por delante las tres F1 a la vez.

## 5. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| A0-control | no (+0.0000) | no | 0/5 | no supera |
| A1-sqrt | no (+0.0006) | no | 3/5 | no supera |
| A2-inversa | no (+0.0002) | no | 2/5 | no supera |
| A3-tope | no (-0.0149) | no | 1/5 | no supera |

## 6. Nota sobre el calculo de los pesos

Los pesos se cuentan **dentro de cada pliegue**, sobre los grupos que aparecen solo
en su tren. Calcularlos una vez sobre todo el desarrollo haria que cada pliegue
arrastrase informacion de las filas con las que despues se valida, que es leakage
aunque el peso sea un entero. Una fila de validacion cuyo grupo no aparezca en el
tren recibe el peso mediano, que es el valor neutro.

Fecha: 2026-09-26T01:29:42+00:00. Duracion: 23.7 s.
