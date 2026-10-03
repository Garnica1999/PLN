# Cuanto vale una unidad de movimiento en el leaderboard (V8-02)

Generado por `experiments/v8_02_resolucion_liderboard.py` (V8-02 (analisis)).
No entrena nada y no gasta ninguna sonda: usa las doce ya pagadas, una de ellas la propia P1.

## 1. Por que hace falta este analisis

La sonda P1 dio 0.77470 frente a los 0.78306 de V03, una diferencia de
-0.00836. Ese numero, aislado, no significa nada. Hay que saber cuanto
vale una unidad de movimiento en ese leaderboard, y eso se responde con aritmetica
y con las sondas que ya se pagaron.

## 2. El ruido del leaderboard

El conjunto de prueba tiene 1154 filas. La cota binomial del error de una
proporcion, con p igual a 0.78, es **0.01219**. Para la F1, que no es una
proporcion, el error tipico es alrededor de un cuarto mayor, y la banda que usa el
propio registro del proyecto es de **0.01524**.

## 3. La sensibilidad: cuanto mueve la metrica un acierto neto

La F1 de una etiqueta es `2 TP / (P + Q)`, donde P son los positivos reales y Q los
predichos. Con P y Q fijos **la F1 es lineal en los verdaderos positivos**, y un
acierto neto, que ademas descuenta un falso positivo y uno falso negativo, la mueve en
`2 / (P + Q)`. La metrica primaria es la media de tres etiquetas, asi que se divide
entre tres.

| etiqueta | sensibilidad de la metrica primaria por acierto neto |
|---|---|
| HS | 6.61e-04 |
| TR | 9.63e-04 |
| AG | 7.56e-04 |
| **media** | **7.93e-04** |

Eso significa que el ruido de 0.01524 equivale a unos **19.2
aciertos netos**. Para mover el leaderboard en 0.003 haria falta una senal de unos
4 aciertos netos a favor, y el ruido se come 19.

## 4. Las once sondas en unidades de aciertos netos

| sonda | externo | diferencia | celdas que cambian | aciertos netos | dentro del ruido |
|---|---|---|---|---|---|
| v08_meta_v2 | 0.78163 | -0.00143 | 199 | -1.8 | si |
| v11_candidate_c | 0.78150 | -0.00156 | 142 | -2.0 | si |
| v02_lr_union | 0.77993 | -0.00313 | 98 | -3.9 | si |
| v10_candidate_b | 0.77978 | -0.00328 | 99 | -4.1 | si |
| v09_candidate_a | 0.77895 | -0.00411 | 128 | -5.2 | si |
| v05_optuna | 0.77602 | -0.00704 | 125 | -8.9 | si |
| v04_ens04_mediana | 0.77510 | -0.00796 | 98 | -10.0 | si |
| v08_p1_fold_ensemble | 0.77470 | -0.00836 | 100 | -10.5 | si |
| v06_robustez_c1 | 0.77227 | -0.01079 | 68 | -13.6 | si |
| v07_robustez_mindf5 | 0.77034 | -0.01272 | 108 | -16.0 | si |
| v01_baseline_tfidf_lr | 0.75677 | -0.02629 | 231 | -33.2 | **no** |

**10 de 11 sondas quedan dentro del ruido.** La excepcion se comenta en el
apartado siguiente, porque es la que mas informa.

## 5. La premisa que hay que revisar

El proyecto venia sosteniendo, con la decision del freno de transferencia, que
**el unico movimiento positivo del leaderboard vino de la capa de decision**, el paso
de v02 a v03, que dio +0.0031. Al pasar esa cifra a la misma unidad que el ruido:

- la diferencia entre v02 y v03 equivale a **+3.9 aciertos netos a favor
  de v03** en 98 celdas que cambian, frente a un ruido de
  **19.2**.
- es decir, una senal unas 5 veces mas
  pequena que el ruido que la rodea.

**La conclusion cambia y hay que decirlo sin rodeos: ese movimiento positivo tambien
fue ruido.** No es evidencia de que la capa de decision funcione. Sigue siendo cierto
que V03 no tiene evidencia en contra, pero deja de ser cierto que tenga evidencia a
favor. La distincion importa, porque una referencia que se sostiene por inercia y una
que se sostiene por evidencia son cosas distintas.

## 5 bis. La unica sonda que si se distingue, y que hay que leer con calma

De las once sondas reales, **1 se distingue de V03**:

- **v01_baseline_tfidf_lr**, con -0.02629, o sea -33.2 aciertos netos, en 231 celdas.

Y hay que leerlo con calma, porque **no significa que cambiar la representacion sea
la salida**: esa sonda es la que mas se aleja de la referencia, y se aleja para
peor. Es el modelo de solo palabra con constante 1.0, sin la union de palabra y
caracter y sin los umbrales por etiqueta, o sea el mas pobre de la familia. La
afirmacion defendible es mas estrecha y es esta: **de once maniobras distintas, la
unica con suficiente senal es la que mas se aleja del mejor, y en la direccion
equivocada**. No hay ninguna maniobra con senal a favor.

Conviene corregir tambien lo que se podria sobreleer: cambiar la representacion
**tampoco** produce senal por si solo. La reoptimizacion completa de Optuna, que
cambio el rango de ngramas, el minimo de apariciones y la constante, quedo dentro
del ruido, igual que la sola palabra y caracteres con minimo 5. **Ninguna de las tres
maniobras que el proyecto ha explorado con mas esfuerzo produce una diferencia que
este leaderboard pueda resolver**: ni la capa de decision, ni los hiperparametros,
ni la representacion.

## 6. Que hacer con la sonda P1

P1 cambio 100 celdas y dio -10.5 aciertos netos, o
sea -10.5 por cada cien celdas que cambian.
Ese numero es **indistinguible de una moneda al aire**: las cien celdas que se movieron
lo hicieron porque el promedio desplaza los casos limite, no porque el promedio acierte
mas. La senal medida es un doceavo del ruido.

Por tanto la linea se cierra, pero **no como refutacion fuerte**. Se cierra porque no
aporta, y el techo de la sonda estaba declarado antes de envarla.

## 7. El filtro que gobierna el resto de la fase

El limite esta en el leaderboard, no en el modelo. El externo tiene 1154 filas, de modo
que **solo resuelve diferencias de unos 19 aciertos netos**, por muy
separable que sea la diferencia en el desarrollo. Eso se traduce en un criterio
cuantitativo y accionable, y no en un numero de celdas arbitrario.

Si una candidata cambia C celdas de la referencia, su senal en el externo es
`C por (2p - 1)`, donde p es la fraccion de esas celdas que ella acierta y la
referencia falla. Para que el externo pueda resolverla hace falta que `2p - 1` sea mayor
que 19.2 dividido por C. Traducido:

| celdas que cambia | tasa de acierto minima entre esas celdas |
|---|---|
| 50 | 0.692 |
| 100 | 0.596 |
| 200 | 0.548 |
| 400 | 0.524 |
| 800 | 0.512 |

En otras palabras: **con cien celdas modificadas, mas de un 60 % de ellas tienen que
ser aciertos netos para que el leaderboard pueda verlo**. Ese numero es estimable en el
desarrollo antes de enviar nada, y por tanto **es un filtro previo y gratis**, no un
criterio a posteriori.

Las tres reglas que se adoptan para el resto de la fase:

1. **Ninguna candidata cuya tasa de acierto entre las celdas modificadas no alcance el
   minimo de la tabla se envia.** No es que sea mala: es que el instrumento no puede
   distinguirla, y gastar una sonda en eso es gastar presupuesto en ruido.
2. **Cambiar muchas celdas no basta por si mismo.** Trescientas celdas movidas en
   direccion desordenada siguen siendo ruido grande, no senal. La coherencia se mide con
   el gate de pliegues que ya existe.
3. **El objetivo declarado hay que revisarlo a la luz de esto.** Superar 0.80 esta a
   +0.01694 de la referencia, o sea 1.1 veces
   el ruido y 21 aciertos netos de ventaja
   limpia, **obtenidos sobre once sondas de las que diez son indistinguibles entre si**.
   No es imposible, pero no es algo que se alcance empujando la capa de decision, que es
   donde se ha estado buscando durante varias fases.

## 8. Que implica para la ordenacion restante

1. **P1 se cierra sin variantes.** No se toca umbral, ni pesos, ni semilla. El
   promedio no se rescata, y el motivo no es que fuera malo sino que no era medible.
2. **Las lineas que solo mueven la capa de decision bajan de prioridad**, porque se
   mueven en la banda de cien a doscientas celdas, donde el minimo de acierto exigido
   supera el 60 %. Eso alcanza a P5, P6 y P7.
3. **P3, la estabilidad de features, sube a la primera posicion**, porque reduce la
   dimension y eso puede cambiar muchas celdas de golpe, que es la unica forma de
   acercarse al minimo de acierto exigido.

Fecha: 2026-09-26T03:05:20+00:00. Duracion: 0.0 s.
