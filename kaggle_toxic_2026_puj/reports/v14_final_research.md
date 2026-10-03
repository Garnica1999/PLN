# V14: auditoria de entrenamiento partially-label para TR y AG

Generado por `experiments/v14_01_informe_final.py` (V14-01 (informe final de la auditoria partially-label)).

## 1. La conclusion, y por que no es la que se esperaba

El enunciado de V14 advierte que la perdida enmascarada **no debe confundirse con H2-M**,
y propose P1 precisamente para no repetirlo. La auditoria demuestra lo contrario, y con la
mayor evidencia posible:

**La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero.**

| comprobacion | valor |
| --- | --- |
| peor diferencia de coeficientes | 0.000e+00 |
| peor diferencia de probabilidades | 0.000e+00 |
| delta de H2-M reconstruido | -0.116581 |
| delta de la enmascarada de P1 | -0.116581 |
| delta de H2-M registrado en v12 | -0.116581 |
| diferencia entre los dos deltas | 0.000e+00 |

Las cuatro cifras coinciden. Una formulacion que se presenta como nueva y que reproduce
exactamente el fallo de una familia cerrada no es una familia nueva.

## 2. Por que, en una linea

**Una fila con peso cero en la perdida no aporta gradiente.** Luego el vector de
coeficientes lo determinan unicamente las filas con peso uno, que son exactamente las
filas que H2-M conserva. Enmascarar y eliminar dejan el mismo subconjunto de filas
aportando senal, y por eso producen el mismo estimador.

Lo unico que puede distinguirlas es como se reescala el termino de perdida. Eso no es un
mecanismo de aprendizaje: es una constante de escala, y pertenece a la familia de
calibracion y umbrales, que ya estaba cerrada.

## 3. La correccion propia, que es la parte instructiva de la fase

**Mi primera verificacion de la equivalencia uso una sola pliegue y concluyo que la identidad era exacta. Al extenderla a los cinco pliegues aparecieron diferencias de orden 4.5e-01 en los coeficientes**

**Causa: La causa era mia: en la reconstruccion de H2-M pasaba sample_weight nulo, es decir sin pesos de clase, cuando el brazo M real usa class_weight balanced sobre la submuestra de HS=1. Estaba comparando H2-M contra un ajuste sin ponderar, que no es H2-M, y por eso mi H2-M reconstruido daba -0.132999 en lugar de -0.116581**

**Como se detecto: La comprobacion de que el delta reconstruido coincidiera con el registrado en v12. Sin esa comprobacion, la conclusion erronea habria pasado, porque un numero pequeno no delata un modelo mal puesto**

**Leccion: Un veredicto de equivalencia no se sostiene en una sola pliegue. Y reproducir un resultado historico conocido es la comprobacion que detecta un error de montaje.**

## 4. Las siete respuestas que el enunciado exige

### 1. Las etiquetas TR y AG de las filas HS=0 deben modelarse como negativos ordinarios o como observaciones parcialmente definidas?

**NO SE PUEDE RESPONDER CON EVIDENCIA, y la fase lo declara en vez de resolverlo por inercia. La razon es que la unica formulacion de la alternativa que puede implementarse con los mecanismos permitidos no es una alternativa: es H2-M. La comparacion que el enunciado quiere establecer, entre negativos ordinarios y observaciones parcialmente definidas, no se ha establecido en ningun experimento, porque las dos ramas disponibles son el mismo estimador**

*Evidencia*: Perdida enmascarada contra H2-M: 0.000e+00 en coeficientes, 0.000e+00 en probabilidades y 0.000e+00 en el delta del macro-F1

*Estado*: ABIERTA, y no por resultado negativo sino por identidad de estimador

### 2. Existe evidencia experimental de que cambiar esa semantica de entrenamiento mejora el macro-F1?

**NO. No se ha ejecutado ningun experimento, porque P0 cumplio la condicion de parada numero 1. Y la unica formulacion implementable produce -0.116581, es decir una perdida de tres puntos porcentuales, no una mejora**

*Evidencia*: 0 de 4 experimentos principales y 0 de 2 de reserva. Delta unico medido: -0.116581

*Estado*: CERRADA como via de mejora, y no como via de conocimiento

### 3. Es realmente distinto de H2-M?

**NO, y esta es la pregunta central de la fase. Enmascarar la perdida y eliminar las filas son la misma operacion sobre el estimador, porque una fila con peso cero no aporta gradiente. No es una cuestion de notacion ni de convencion: es que el subconjunto de filas que aporta senal es identico**

*Evidencia*: Diferencia maxima de coeficientes 0.000e+00 en los cinco pliegues y las dos etiquetas. Diferencia maxima de probabilidades 0.000e+00. El delta reproducido coincide con el registrado en v12 hasta el ultimo decimal

*Estado*: RESUELTA, en negativo y con prueba numerica

### 4. Que fraccion del +0.046271 del oraculo se recupera?

**NINGUNA, y el signo es negativo: -0.116581 es -252 por ciento del techo. La fraccion no es pequena, es del lado equivocado**

*Evidencia*: -0.116581 contra un techo de +0.046271, es decir -2,52 veces el techo disponible

*Estado*: CERRADA

### 5. La mejora afecta a los falsos positivos de TR y AG en HS=0?

**NO SE PUEDE MEDIR, porque ningun mecanismo llego a evaluarse. Lo que si consta es que el techo de +0.046271 consiste exactamente en 196 celdas de este tipo, 78 de TR y 118 de AG, y que H2-M no las toca: su efecto es un cambio de escala de la salida, que es el modo de fallo que v12 documento**

*Evidencia*: El techo borra 196 celdas en las 835 filas con HS verdadero cero. H2-M falla por desplazar la escala, y con umbrales congelados eso se traduce en un delta de -0.116581

*Estado*: NO MEDIDA, por no haber mecanismo superviviente

### 6. Existe alguna candidata que supere el gate?

**NO. Ninguna, porque no se ejecuto ningun experimento y el unico resultado medido es negativo**

*Evidencia*: Mejor delta -0.116581 frente a un gate exploratorio de +0.002 y un gate de candidato de +0.003. 0 candidatas

*Estado*: CERRADA

### 7. Si no, queda alguna familia experimental genuinamente no cerrada?

**DENTRO de los mecanismos permitidos, NO. Los cuatro mecanismos que la fase propone como nuevos por su terminologia, partial-label, positive-unlabeled, selective loss y masked BCE, tienen CERO menciones historicas una vez descontada la autorreferencia de la propia consulta. Pero su equivalente operativo, que es enmascarar la perdida de clasificacion, esta cerrado como H2-M. Quedaria abierta una via que aportara una senal distinta de cero y sin etiqueta a las filas HS=0, y esa via cae fuera de la jerarquia: exige datos externos, una representacion nueva o informacion transductiva, los tres prohibidos**

*Evidencia*: Barrido de los doce mecanismos del enunciado: cuatro con cero menciones historicas y cuatro con mas de cien, ninguno de los cuales condiciona la senal a HS. Sonda estructural: trece llamadas con sample_weight, ninguna definida por la definicion de la etiqueta

*Estado*: CERRADA dentro del conjunto permitido

## 5. Lo que esta fase NO cierra, dicho con precision

**No se afirma** que las etiquetas de TR y AG en el regimen HS=0 deban seguir siendo
negativos ordinarios. Esa conclusion seria comoda y estaria mal sostenida, porque la
alternativa **no ha fallado: ha resultado ser el mismo estimador**. No hay ninguna prueba
en contra de la semantica de observacion parcialmente definida, del mismo modo que no hay
ninguna a favor.

Lo que si queda establecido es una restriccion de lo que es posible construir: **cualquier
mecanismo que se implemente enmascarando la perdida de clasificacion sobre las filas HS=0 va
a reproducir H2-M**, con su cambio de escala y su -0.116581. Para apartarse de ahi hace
falta una senal distinta de cero en esas filas, y obtenerla exigiria datos externos, una
representacion nueva o informacion transductiva, **los tres prohibidos por la jerarquia de
esta fase**.

## 6. La formulacion correcta

> El techo del oraculo procede de conocer la condicion semantica HS al evaluar TR y AG, pero mejorar HS por si sola no produce beneficio, y ya esta medido con delta exactamente cero. Las familias historicas de routing, factorizacion, powerset, ranking y meta-combinacion estan cerradas. La unica cuestion residual era si el regimen HS=0 puede modelarse como etiqueta parcialmente definida durante el entrenamiento sin reproducir el fracaso de H2-M, y la respuesta es que enmascarar la perdida no es una forma alternativa de hacerlo: es el mismo estimador

Y en particular, lo que v13 ya habia medido y esta fase no ha movido: **no hay que mejorar
HS**. La correccion de la etiqueta de HS produce delta exactamente cero en TR y AG, porque
el meta produce una probabilidad para HS y no altera lo que el sistema decide sobre las
otras dos. El techo de +0.046271 no es un techo de HS.

## 7. Estado final

- Experimentos: **1**, que es la auditoria. **0 ejecuciones** de los 4 principales y de
  los 2 de reserva.
- Candidatas adoptables: **0**. Marginales: **0**. Rechazadas: **1**, la enmascarada de P1.
- Fraccion del techo del oraculo recuperada: -252.0 por ciento, de signo negativo.
- Submissions: **0**. Evaluaciones de holdout2: **0** de 3. Segunda particion: **no
  ejecutada**. Test: **no utilizado**.
- V03: **congelada e intacta**, con identidad verificada.

| restriccion | cumple | evidencia |
| --- | --- | --- |
| Auditoria de novedad ejecutada antes de cualquier experimento | si | V14-00 es el unico experimento y no entrena nada |
| Matriz de mecanismos obligatoria entregada | si | diez mecanismos con formula, diferencia, resultado y estado |
| Los doce mecanismos del enunciado barridos | si | con conteo con y sin autorreferencia de la propia consulta |
| Comparacion algebraica con H2-M realizada | si | diferencia 0.000e+00 en coeficientes, probabilidades y macro-F1 |
| Test no utilizado | si | solo se carga el desarrollo de 1483 filas |
| Holdout2 no utilizado | si | no se ha cargado en ningun punto |
| Sin submissions | si | 0 ficheros de submission generados |
| Sin leaderboard | si | ninguna decision usa la puntuacion externa |
| Segunda particion no ejecutada | si | no hay candidata que la justifique |
| V03 intacta | si | control de identidad con diferencia exactamente cero |
| Sin nuevo vectorizador ni representacion | si | solo word(1,2) y char_wb(3,5) de V03 |
| Sin datos externos ni informacion transductiva | si | ninguna fuente ajena al corpus de desarrollo |
| Sin embeddings nuevos | si | no se ha construido ninguna representacion |
| Presupuesto respetado | si | 0 de 4 principales y 0 de 2 de reserva |
| Ningun resultado editado a posteriori | si | el unico script se escribio antes de ver sus resultados |


Fecha: 2026-09-27T00:10:38+00:00. Duracion: 0.0 s.
