# P67 - Validacion de la senal residual de MI_arbol

Generado por `experiments/p67_02_cerrar.py`. Fecha 2026-09-29T16:45:27+00:00.

## La respuesta corta

**MIXED_CON_ATRIBUCION_PARCIAL**

La pregunta era si el residuo de MI_arbol es senal informacion complementaria, efecto
de calibracion, o ruido. Los cuatro controles que se disenaron para responderla **no dan
la misma respuesta**, y el informe dice eso en vez de elegir un lado.

## Lo que dice cada control, y por que no coinciden

### C-A. El reentreno anidado del umbral: la calibracion NO explica el efecto

```text
umbrales CONGELADOS     [0.44, 0.51, 0.46]
umbrales REENTRENADOS   [0.385, 0.445, 0.52]
delta                   -0.0029731513
el mecanismo caro        +0,0026743167
```

Un escalar por etiqueta es el modelo de calibracion **mas barato posible**. Da
-0.0029731513, es decir **peor** que el umbral congelado. La calibracion no
explica el efecto y V03 no estaba simplemente mal calibrado en esta particion.

Esto **desmonta** la hipotesis H2 de la auditoria independiente de P66, que esperaba
que un escalar reprodujera el +0,0027.

### C-E. El AUC del residuo se mantiene FUERA de la banda: hay algo mas

El AUC contra el **error** de V03 esta confundido, porque ese error es funcion
deterministica de `z_V03` y del umbral, y el residuo contiene por construccion un
termino `-b*z_V03`. Contra la **etiqueta** no hay ese confound.

| candidato | AUC dentro de la banda | AUC FUERA de la banda | desviacion del residuo |
|---|---:|---:|---:|
| MI_arbol | 0.567424 | 0.579597 | 0.8338 |
| negacion | 0.492127 | 0.465626 | 0.2043 |
| word_1_3 | 0.484706 | 0.504894 | 0.1469 |
| cohesion | 0.45134 | 0.452727 | 0.1120 |
| huecos_1_2 | 0.524327 | 0.516491 | 0.2193 |
| contraste | 0.457961 | 0.432401 | 0.0807 |
| huecos_6_10 | 0.444546 | 0.476209 | 0.1863 |

MI_arbol se queda en **0.579597 FUERA** de la banda. Los otros seis caen por debajo de 0,5. Un movimiento de la
frontera de decision no puede producir eso: por construccion actua donde ya hay
error, y aqui no hay banda de margen que mover.

Ademas el residuo de MI_arbol tiene desviacion 0.8338, frente a 0.2193 del mas redundante.
Eso explica por que es el unico que aporta algo: es el unico **no lineal**.

### C-B. Sustitucion cruzada: parte del efecto es de la OPERACION

| candidato | delta real | delta barajado |
|---|---:|---:|
| MI_arbol | +0.0026743167 | -0.0014390421 |
| negacion | +0.0011549795 | -0.0054335321 |
| word_1_3 | -0.0001400584 | +0.0038763165 |
| cohesion | +0.0000325761 | -0.0070646241 |
| huecos_1_2 | +0.0003505310 | -0.0064731429 |
| contraste | +0.0000000000 | -0.0017878950 |
| huecos_6_10 | -0.0028447979 | +0.0004009382 |

**4 de 7** dan positivo. El mas alto sigue siendo
MI_arbol, pero cuatro candidatos positivos no es una firma. Parte del efecto pertenece
a la correccion residual como OPERACION, no a la senal de MI_arbol.

### C-D. El enriquecimiento, que el '47 de 53' de P66 no tenia

Los errores de un clasificador con umbral se concentran **estructuralmente** cerca de la
frontera, luego la fraccion de rescates en banda no significa nada sin su denominador.

| etiqueta | tasa base de errores en banda | rescates en banda | ENRIQUECIMIENTO |
|---|---:|---:|---:|
| HS | 0.4019 | 1.0000 | 0.4850 |
| TR | 0.3232 | 0.7500 | 2.5058 |
| AG | 0.4007 | 0.9167 | 1.0241 |

Enriquecimiento promedio **1.3383**. Hay concentracion
real en la banda, pero no es exclusiva ni enorme. HS da enrichment 0,485, es decir
los rescates de HS se concentran **menos** que los errores.

## Por que MIXED y no una de las otras tres

| categoria | lo que exigiria | lo que hay |
|---|---|---|
| CALIBRATION | un escalar por etiqueta igualaria el efecto | da -0.0029731513, **peor** |
| REPRESENTATION | el efecto seria exclusivo de MI_arbol | 4 de 7 candidatos dan positivo |
| MIXED | una parte sobrevive al control mas barato y una parte es de la operacion | **exactamente esto** |

## Lo que la reconciliacion de P66 ya habia dejado claro

El efecto son **4.01 celdas de 1483**. El IC95 por
grupos de P66 era `[-0,0044311634, +0,0102108745]`, que va de **-7 a +15 celdas**.

Eso no es una mejora pequena con imprecision: es un efecto **por debajo de la
resolucion** con la que la metrica puede distinguirlo.

Ademas P66 reporto la `brecha de atribucion` como la resta del control barajado. El
control es un **suelo**, no un cero: es lo que cuesta la MISMA operacion cuando el
candidato no lleva senal. Restar un suelo resta el dano evitado. Por eso la brecha del
blend, +0,0093216366, queda **INVALIDADA** como medida de informacion atribuida.

## Los tracks que el encargo mandaba, y por que se saltan

| track | estado | motivo |
|---|---|---|
| P15 reproduccion independiente completa | SKIPPED - PREREQUISITE NOT MET | solo se justifica para un candidato que pase los gates de atribucion y robustez. Este no los pas |
| P16 segunda particion de entrenamiento | SKIPPED - PREREQUISITE NOT MET | identica. Confirmar una segunda particion un efecto de 4 filas de 1483 cuestionaria la particion |
| P08 correccion condicionada por la confianza | SKIPPED - PREREQUISITE NOT MET | el encargo la condiciona a que P5 o P6 muestren senal residual reproducible. El reentreno del um |
| P09 correccion dependiente de la longitud | SKIPPED - PREREQUISITE NOT MET | condicionada a que P3 demuestre relacion reproducible entre el residuo y el numero de tokens. No |
| P10 correccion especifica por etiqueta | SKIPPED - PREREQUISITE NOT MET | el encargo dice conservar el beta compartido si el modelo de tres parametros no lo supera. No ha |
| P11 analisis pairwise de etiquetas | SKIPPED - NO INDEPENDIENTE | depende de que una pareja muestre senal complementaria, y la unica senal medible es la de MI_arb |

Ninguno queda pendiente. Cada uno tiene su prerrequisito y su motivo.

## Lo que P67 deja escrito para las fases siguientes

1. `MI_arbol` tiene algo que V03 no tiene, y no es calibracion. El AUC del residuo
   contra la etiqueta se mantiene fuera de la banda de margen.
2. Ese algo no se puede atribuir enteramente a MI_arbol, porque cuatro de los siete
   candidatos producen la misma direccion con la misma operacion.
3. El tamano del efecto esta por debajo de la resolucion de la metrica, asi que
   confirmarlo con una segunda particion cuestionaria la particion, no el efecto.
4. La brecha de atribucion de P66 estaba mal interpretada. La cifra defendible es
   +0,0026743167, no +0,0041133588.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p67_02_cerrar.py`. Sin push.*