# P2: ponderacion por razon logaritmica de recuentos (V8-04)

Generado por `experiments/v8_04_log_count_ratio.py` (V8-04 (P2)).

## 1. Una premisa propia que era falsa

La auditoria de brecha de esta fase (V8-00) decia que el componente que el proyecto
no habia probado era **la razon logaritmica por etiqueta**. Al leer el codigo de
E2-15 se comprueba que ya estaba hecho: su bucle itera sobre las tres etiquetas y
calcula un ratio independiente para cada una. **La premisa era falsa, y se corrige
antes de ejecutar nada.** Lo que de verdad no se habia probado es otro: multiplicar el
ratio sobre la TF-IDF de V03, y con la constante 2.0 en vez de la 1.0.

Por que la combinacion no es redundante: la TF-IDF normaliza cada caracteristica por su
prevalencia **en el corpus**, y el ratio la normaliza por su prevalencia **en cada
clase**. Son dos ejes distintos, y multiplicarlos es una pregunta con sentido.

## 2. Que midio E2-15, y con que constantes

Las cuatro variantes de E2-15 usaron **conteos** y la **constante 1.0**:

| variante de E2-15 | diferencia frente a V03 |
|---|---|
| conteos de palabra(1,2) con ratio | -0.0234 |
| conteos de caracter(3,5) con ratio | -0.0392 |
| conteos de la union con ratio | -0.0160 |
| conteos de la union con ratio y maquina de margen | -0.0343 |

Ese es el precedente que hereda esta linea, y es adverso. La unica pieza que faltaba
era con la representacion y las constantes de V03, y eso es lo que se mide aqui.

## 3. Control de codigo

El OOF canonico de V03 da 0.777211 sobre las mismas filas, con los
identificadores en el mismo orden. El ratio se calcula **dentro del tren de cada
pliegue y por etiqueta**, de modo que ninguna estadistica toca la validacion.

## 4. Las tres variantes

| variante | nueva | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro |
|---|---|---|---|---|---|---|---|---|
| nb_tfidf | si | 0.718716 | -0.058496 | [-0.07611, -0.04061] | 0/5 | 597 | 0.3266 frente a 0.5161 | no pasa |
| nb_tfidf_suavizado | si | 0.781957 | +0.004745 | [-0.00708, +0.01714] | 3/5 | 303 | 0.5281 frente a 0.5317 | no pasa |
| nb_counts_c2 | control | 0.702184 | -0.075027 | [-0.09423, -0.05561] | 0/5 | 715 | 0.3217 frente a 0.5134 | no pasa |

| variante | HS | TR | AG |
|---|---|---|---|
| V03 (referencia) | 0.7617 | 0.8157 | 0.7542 |
| nb_tfidf | 0.7265 | 0.7416 | 0.6881 |
| nb_tfidf_suavizado | 0.7699 | 0.8071 | 0.7688 |
| nb_counts_c2 | 0.7089 | 0.7223 | 0.6754 |

## 6. Que hace el ratio dentro del modelo

El ratio no es un multiplicador inerte: en el primer pliegue cambia 108 de 891 celdas, un 12.1 %, al compararlo con el mismo modelo sin
ponderar. Es decir, el mecanismo actua, y lo que se mide es si actua para bien.

| pliegue | etiqueta | caracteristicas | ratio medio | desviacion | rango |
|---|---|---|---|---|---|
| 1 | HS | 20441 | -0.6154 | 3.4109 | [-8.39, +9.85] |
| 1 | TR | 20441 | -2.1103 | 3.5953 | [-9.69, +10.22] |
| 1 | AG | 20441 | -0.9671 | 3.4716 | [-9.00, +9.98] |
| 2 | HS | 21044 | -0.5752 | 3.4084 | [-9.01, +9.77] |
| 2 | TR | 21044 | -2.1263 | 3.5935 | [-9.40, +10.15] |
| 2 | AG | 21044 | -0.9357 | 3.4651 | [-8.92, +9.91] |
| 3 | HS | 20802 | -0.5888 | 3.4329 | [-8.49, +9.90] |
| 3 | TR | 20802 | -2.1277 | 3.5999 | [-9.45, +10.28] |
| 3 | AG | 20802 | -0.9636 | 3.4614 | [-8.49, +10.03] |
| 4 | HS | 21038 | -0.6301 | 3.4133 | [-8.53, +9.82] |
| 4 | TR | 21038 | -2.1616 | 3.5753 | [-9.60, +10.20] |
| 4 | AG | 21038 | -1.0168 | 3.4563 | [-8.51, +9.96] |
| 5 | HS | 20982 | -0.6331 | 3.3767 | [-8.54, +9.92] |
| 5 | TR | 20982 | -2.1089 | 3.5919 | [-9.51, +10.29] |
| 5 | AG | 20982 | -1.0077 | 3.4456 | [-8.44, +10.05] |

## 7. El hallazgo que domina a todos los demas: el suavizado lo cambia todo

Las dos variantes nuevas difieren **solo** en el suavizado de Laplace, de 1e-6 a
1e-3, y sus resultados no tienen nada que ver:

- con suavizado 1e-06: -0.058496
- con suavizado 0.001: +0.004745
- **salto de +0.063241 por un factor de mil en un solo parametro**

La tabla de arriba explica por que. Con la TF-IDF, las caracteristicas raras ya
tienen valores muy pequenos, de modo que dividir por una probabilidad de clase
casi nula **explota**: el ratio alcanza un rango de mas de veinte unidades de
ancho y una desviacion de mas de tres. Multiplicar la matriz por eso destruye la
escala relativa de las caracteristicas. El suavizado deja de ser un detalle
numerico y pasa a ser **la parte que sostiene el metodo entero**.

Y hay una segunda lectura, en la direccion contraria. Que una sola linea pase de
-0.058496 a +0.004745 cambiando un parametro que
nadie ha caracterizado significa que **el resultado positivo local no es una
senal: es un punto de una superficie que no se ha explorado**. Por eso no se barre
suavizado para buscar un valor mejor. Seria exactamente el rescate por
hiperparametro que las reglas del proyecto prohiben desde la fase v6, y ademas
V8-02 acaba de medir que casi nada de lo que se ajusta aqui llega al externo.

### El control dice algo que las cuatro variantes de E2-15 no decian

El control canonico con la constante 2.0 da -0.075027, frente
a los -0.0160 que E2-15 midio con la constante 1.0. **Subir la constante de 1.0 a
2.0 con el ratio empeora la linea casi cinco veces.** Eso explica algo que las
cifras anteriores no permitian separar: la constante y el ratio no se pueden subir
a la vez. El ratio ya introduce mucho ruido en la escala de las caracteristicas, y
subir la constante amplifica ese ruido en vez de reducirlo. El control no estaba
en E2-15 y sin el no se podria atribuir la diferencia a la representacion.

## 8. Decision

**No se genera submission**: no superan el gate de mejora: nb_tfidf, nb_tfidf_suavizado, nb_counts_c2.

El filtro de detectabilidad es el que manda, y conviene recordar por que. Ganar en
local por debajo del minimo de acierto entre las celdas modificadas no es ganar: es
cambiar la prediccion sin la senal que la respalde, que es exactamente lo que el
externo no puede distinguir del azar.

Fecha: 2026-09-26T03:16:13+00:00. Duracion: 8.8 s.
