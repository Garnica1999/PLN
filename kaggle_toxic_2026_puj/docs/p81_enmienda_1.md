# Enmienda 1 al pre-registro de P81

Fecha: 2026-09-30
Estado: adoptada antes de escribir una linea de codigo
Documento que modifica: docs/p81_preregistro_cierre_huecos.md, sha256
55e6900cce49bd7e2ee98b00e0c315081c04718edb18df4abd49f484755faabc, commit 86a9140
Motivo: al implementar el bloque 1 aparecio una tension real entre tres clausulas del
pre-registro. Se resuelve aqui, y queda escrito antes de ejecutar.

## 1. La tension

La especificacion exige que el test de A5 y A6 sea DISCRIMINANTE, con una diferencia
minima entre los dos brazos y STOP con EL_TEST_NO_DISCRIMINA_Y_NO_CONCLUYE_NADA si cae
en banda muerta. Exige ademas un learning rate FIJO declarado, y prohibe reportar M1
mayor que 0.99 sin investigarlo.

El problema: el termino negativo del objetivo con SUMA pesa K veces mas que con MEDIA.
Con el mismo learning rate fijo, el brazo SUMA da un paso K veces mas fuerte. Eso
significa que comparar a un solo learning rate no aísla el objetivo: mezcla "que
objetivo es mejor" con "cual se entreno mas fuerte". Es exactamente el tipo de
confund que este proyecto lleva dos fases intentando cerrar.

Medicion exploratoria del implementador, tres semillas, 1500 pasos:
  lr 0.05   MEDIA 0.0056   SUMA 0.0000   diferencia menos 0.0056   banda muerta
  lr 0.20   MEDIA 0.1389   SUMA 0.3167   diferencia mas 0.1778    pasa, M1 lejos de 1
  lr 0.50   MEDIA 0.3167   SUMA 1.0000   diferencia mas 0.6833    M1 BETONAL

## 2. Por que se descartan las tres salidas obvias

La primera, normalizar el learning rate por K, no funciona. No existe un learning rate
unico que iguale los dos terminos a la vez: el positivo pesa 1 en los dos objetivos, y
solo el negativo cambia en K. Subir el learning rate de un brazo para compensar el
negativo hunde su propio termino positivo. Esas dos correcciones son
contradictorias y no existe punto de equilibrio.

La segunda, fijar un learning rate en la banda intermedia, no exige desviacion pero es
una comparacion CONFOUNDED: la diferencia observada en lr igual a 0.20 puede deberse
enteramente a que el brazo SUMA recibio K veces mas de entrenamiento. Cerraria el hueco
de forma aparente y no real.

La tercera, escribirlo tal cual, lleva a STOP y deja el hueco 1 abierto.

## 3. La decision

Se sustituye el learning rate fijo por una REJILLA declarada por adelantado, recorrida
por los DOS brazos con los mismos pasos, los mismos datos y el mismo orden de lotes. Se
publica la curva completa de M1 frente a learning rate para los dos brazos.

El primario pasa a ser el MEJOR de la rejilla de cada objetivo, y la diferencia entre
ambos es la afirmacion. La curva entera se publica en el JSON y en el markdown.

Por que esto aísla el objetivo: cada objetivo se evalua en SU mejor punto alcanzable
dentro de una rejilla declarada de antemano, no en un punto arbitrario. La objecion de
"gano porque se entreno mas fuerte" se responde con la curva: si el brazo SUMA ganara
solo en el extremo alto de la rejilla, la curva lo muestra y el primario deja de ser
una lectura de un punto suelto.

## 4. Rejilla y presupuesto, fijados aqui

Learning rate: 0.02, 0.05, 0.10, 0.20, 0.35, 0.50.
Pasos: 1500 fijos.
DimENSION: 32. K: 10. optimizador: SGD puro, sin momentum, sin Adam.
Semillas: 3, las mismas para los dos brazos en cada punto de la rejilla.
Presupuesto: menos de 60 segundos en CPU por punto de la rejilla.

Si algun punto de la rejilla produce M1 mayor que 0.99, ese punto se publica con el
numero y se marca como ZONA BETONAL, y no se usa para el primario sin repetirlo con mas
semillas. No se borra ni se oculta: se declara.

## 5. Lo que esta enmienda NO cambia

- El anclaje documental del bloque 1 no cambia. word2vec.c sha256
  df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4, 715 lineas. Verificado
  de forma independiente: en TrainModel el bucle de negative sampling corre de d igual
  a cero hasta negative, y en cada d hace neu1e mas igual a g por syn1neg, luego cada
  negativo acumula su gradiente con el mismo peso y sin dividir por K. Ese anclaje ya
  cierra la afirmacion "no es SGNS" con independencia de lo que diga el test.
- El limite permanente del punto 1c no cambia: el anclaje a una implementacion externa
  sigue SIN HACER, y sigue autorizando afirmar sobre el objetivo y no sobre la calidad
  relativa del artefacto frente a una libreria.
- Los bloques 2 a 5, de relleno contra OOV, de reejecucion, de identificadores y de
  scheduler, no cambian.
- El STOP por banda muerta no cambia: si la diferencia entre los dos mejores de la
  rejilla cae por debajo del minimo, el script para igual con
  EL_TEST_NO_DISCRIMINA_Y_NO_CONCLUYE_NADA.
- El artefacto congelado, V03, los umbrales, el holdout, el test, el leaderboard y la
  submission siguen sin tocarse.

## 6. Lo que esta enmienda cuesta

Se pierde la simplicidad de un unico punto. A cambio se gana que la afirmacion no
dependa de un learning rate elegido despues de ver el resultado, que es el modo de
fallo que este proyecto ya ha pagado dos veces.

V03 remains the best verified leaderboard result.