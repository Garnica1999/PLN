# V12: shift, calibracion y regla de decision. Informe final de fase

Generado por `experiments/v12_05_informe_final.py` (V12-05 (informe final de fase y tabla maestra)).
Commit: `b7d252b`.

Referencia: **V03 = 0.7772114525165721**, leaderboard 0.78306. V03 permanece congelada.
**5 experimentos, 0 submissions, 0 evaluaciones de holdout2 gastadas, 0 candidatas.**

## 1. La pregunta y la respuesta, en dos mitades que no hay que mezclar

> Queda alguna fuente de senal explotable por modelos permitidos que no se haya probado
> adecuadamente, o V03 constituye efectivamente el limite practico?

**En el eje del objetivo de entrenamiento, si quedaba, y la v12 lo ha cerrado.** La
auditoria de brecha, V12-00, encontro tres huecos genuinamente no probados, y los tres
estaban en ese eje: no existia ninguna clase de perdida propia en 127 modulos, ninguna
ocurrencia de entrenamiento por pares, y ninguna regularizacion por grupo de caracteristicas
dentro de la etiqueta. Los tres se ejecutaron con gates identicos y los tres fallaron.

**En el conjunto del proyecto, hay un techo medido de +0.046271 que sigue sin capturar.** Y
la v12 aporta algo mas valioso que el negativo: **localizo donde esta ese techo.**

## 2. La tabla maestra

| ID | via | hipotesis | mecanismo | decision |
| --- | --- | --- | --- | --- |
| V12-00 | P0 auditoria de brecha | Que mecanismos se han probado de verdad, y cuales siguen sin probarse | sonda estructural sobre el arbol sintactico de 127 modulos, no busqued | CLOSED |
| V12-01 | H1 regularizacion por etiqueta y grupo | La regularizacion distinta por etiqueta y por grupo de caracteristicas | escalar el bloque de caracter, que multiplica su penalizacion L2 efect | CLOSED |
| V12-02 | H2 perdida propia y estructura de etiquetas | Una perdida propia, o restringir el entrenamiento a donde la etiqueta  | entropia cruzada generalizada L_q, y entrenamiento del condicional sol | MARGINAL |
| V12-03 | H3 entrenamiento por pares | Un objetivo por pares sobre TF-IDF puede explotar la estructura relati | tres clasificadores entrenados jointemente con un termino de margen po | CLOSED |
| V12-04 | auditoria de V6-B | La formulacion de condicionamiento suave sobre la probabilidad de HS e | no aplica: lectura estructural del codigo de V6-02 | CLOSED |

## 3. Las cuatro hipotesis, y que capturaron del techo

| hipotesis | mejor delta | pliegues a favor | deteccion | fraccion del techo | decision |
| --- | --- | --- | --- | --- | --- |
| H1 | +0.002738 | 3/5 | 0.5196 | 5.9 por ciento | CLOSED |
| H2a | +0.001616 | 4/5 | 0.6667 | 3.5 por ciento | MARGINAL |
| H2b | -0.116581 | 0/5 | 0.1420 | 0.0 por ciento | CLOSED |
| H3 | -0.002469 | 2/5 | 0.4915 | 0.0 por ciento | CLOSED |

El techo de referencia es **+0.046271**, la mascara con la etiqueta verdadera de
HS, que no es adoptable porque usa la verdad y solo acota.

## 4. La conclusion que la fase deja establecida

**El techo condicionado por la etiqueta verdadera de HS demuestra que existe senal
estructural abundante**, del orden de +0.046, que es veintitres veces el gate exploratorio y
trescientas veces la mejor ganancia anidada de toda la historia del proyecto, que era de
+0.000148.

**El cuello de botella no es la ausencia de capacidad predictiva para TR y AG.** El
rendimiento de TR donde la etiqueta esta definida es de 0,8941, muy alto. La dificultad es
**inferir la condicion**, es decir saber, para cada fila, si HS es cero.

Y las tres vias para superar esa condicion fallan, y fallan por razones distintas y todas
documentadas:

1. **La llave dura**, el post-proceso HS=0 luego TR=0 y AG=0, da
   **-0.004187**. Falla porque su llave es una prediccion que se equivoca en
   143 filas de 1483, de modo que la mascara se aplica a filas donde TR si esta definida.
2. **La llave suave**, que es V6-B y ya estaba cerrada, cubre solo las 20 filas donde HS se
   predice bien, y no puede tocar las 58 restantes.
3. **Acoplar la relacion en el objetivo**, que es H3, mejora la discriminacion local y
   empeora los falsos positivos justo en la zona que hay que corregir, porque un objetivo de
   orden no restringe el nivel absoluto y la metrica se evalua con umbrales fijos.

## 5. La correccion conceptual que hay que conservar

Conviene decirlo con precision, porque es facil sobregeneralizar aqui.

**H2 y H3 no demuestran que la funcion de perdida no sea el problema en sentido absoluto.**
Un unico mecanismo positivo, el GCE de H2, **no autoriza a descartar la familia entera**.

Lo que si autorizan a decir es algo mas preciso y mas util: **las variaciones de objetivo
evaluadas hasta ahora tienen un impacto pequeno comparado con el techo que introduce el
error de la condicion de HS**. La mejor de ellas, el GCE con q=0,2, da +0.001616 frente a un
techo de +0.046271, es decir captura el 3,5 por ciento. Y la mas estruturada, el objetivo por
pares de H3, empeora.

La distincion importa porque las dos afirmaciones llevan a decisiones distintas. La primera
cerraria una familia de forma definitiva. La segunda solo establece que **el margen
recuperable por esa via es pequeno frente a otro problema ya medido**, y deja la familia
abierta para quien quiera explorarla con un mecanismo distinto.

## 6. Rendimientos decrecientes en el objetivo de entrenamiento

Tres mecanismos distintos sobre la misma funcion, en orden de sofisticacion:

- **H1**, regularizacion por etiqueta y grupo: **+0.002738**, pero sostenido por un solo
  pliegue de cinco, y el maximo de su curva cae sobre el valor neutro con +0.000006.
- **H2a**, perdida propia tolerante al ruido: **+0.001616**, con 4 de 5 pliegues y
  deteccion 0,6667, pero el intervalo pareado incluye el cero.
- **H3**, objetivo por pares que acopla las tres etiquetas: **-0.002469**, con deteccion
  0,4915, es decir **peores que el azar**.

Ninguno llega al gate de +0.002 de forma defendible, y el techo de +0.046271 sigue intacto.
Eso es la senal de **rendimientos decrecientes en el objetivo de entrenamiento**: no es que
la primera idea fuera mala y las siguientes peores, es que el eje entero tiene poco margen
recuperable frente a lo que queda disponible en otro sitio.

## 7. Que queda abierto, y por que no es de esta fase

**Queda abierto un solo problema, y la v12 lo deja escrito sin abrirlo: conseguir que la
condicion de HS se estime mejor.** Todo lo demas depende de eso, porque con HS verdadera la
senal esta ahi, con HS predicha no llega, y con HS mal predicha no sale

No es una hipotesis de esta fase porque no es una hipotesis sobre el objetivo, la
representacion, la capa de decision ni el ensemble: es una hipotesis sobre **la calidad de
una condicion**, y el proyecto ya ha cerrado reiteradamente las familias que podrian
atacarla, desde la regularizacion conjunta entre etiquetas, V8-05, hasta las cadenas, E2-01,
y la factorizacion condicional, V6-02.

## 8. El valor de la auditoria de brecha, que es methodologico

**V12-00 encontro una asimetria que reencuadro la fase**: 17 de las 24 familias de mecanismo
ya cerradas estaban en el eje de la representacion y del clasificador, y los tres huecos
que quedaban estaban todos en el objetivo de entrenamiento. El proyecto habia sido
exhaustivo en un eje y no habia tocado nunca la funcion que se minimiza. Esa asimetria es
lo que justificaba abrir la v12, y el hallazgo se sostiene a la vista de los tres negativos.

Y dos decisiones de instrumento que rinden mas que sus resultados:

1. **La sonda es estructural, no textual.** La auditoria se autovalido con un positivo
   conocido: el unico fichero del proyecto donde aparece la palabra contrastive es el
   propio script de la auditoria, que la nombra al describir la familia ausente. Un
   escaner textual concluiria que el mecanismo si se ha probado, con la maxima confianza, por
   coincidencia literal. El indice estructural devuelve cero, que es lo correcto.
2. **Todo objetivo a medida verifica su derivada antes de mirar un solo resultado.** En H3 eso
   detecto cuatro fallos propios que habrian producido cifras plausibles y falsas: la
   penalizacion L2 ausente del valor, el recorte del margen rompiendo la regla de la cadena,
   el vertice de la bisagra, y la busqueda lineal demasiado corta. Un gradiente mal
   implementado no da un fallo, da un numero.

## 9. Lo que esta fase NO ha hecho

- **No ha entrenado nada antes de cerrar la auditoria de brecha**, como exigia el
  enunciado.
- **No ha reabierto ninguna familia cerrada.** Donde el veredicto es CLOSED_VALID la
  lectura es que no se ejecuta, no que se ejecuta con mas cuidado.
- **No ha generado ni recomendado ninguna submission.** V03 sigue siendo la submission
  disponible.
- **No ha usado el texto de la prueba, ni holdout2, ni el leaderboard.** El uso
  transductivo del test sigue prohibido, como en v11.
- **No ha ejecutado la segunda particion**, porque ningun candidato llego a la mesa de
  finales y gastar la en un negativo seria trabajar para nada.

## 10. Estado final

| elemento | estado |
| --- | --- |
| V03 | congelada, 0.7772114525165721, LB 0.78306, huella sha256 intacta |
| submissions generadas en la v12 | 0 de 26 historicas, ninguna nueva |
| evaluaciones de holdout2 gastadas | 0 de 3 |
| segunda particion | no ejecutada, no hay candidata |
| candidatas | 0 |
| uso del texto de la prueba | ninguno, y sigue prohibido |
| leading board | no consultado |


Fecha: 2026-09-26T23:33:05+00:00. Duracion: 0.0 s.
