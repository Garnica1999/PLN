# Enmienda 1 al pre-registro de P82

Fecha: 2026-09-30
Estado: adoptada antes de ejecutar, y antes de que exista una sola cifra medida
Documento que modifica: docs/p82_preregistro_rnn_atencion.md, sha256
ba74ae71a3b54a273c7f0ac5d55e4d4fd7d1ede641f5fbd022c2ae6377a62061, commit e59548f
Motivo: una aritmetica del pre-registro es incorrecta para la arquitectura que el propio
pre-registro especifica. Se corrige aqui.

## 1. El error

La seccion 5.2 razona sobre el coste parametrico de la capa de atencion aditiva y escribe:

  una matriz W_h de 64 por 64, una matriz W_s de 64 por 64 y un vector v de 64, es decir
  8256 parametros anadidos

La suma 64 por 64 mas 64 por 64 mas 64 es exactamente 8256. La aritmetica de esa frase es
correcta. Lo que no es correcto es que ese numero corresponda a la arquitectura de P82.

Los cinco brazos de la seccion 5.1 son BIDIRECCIONALES con nucleo GRU o LSTM y oculto 64
POR DIRECCION. La salida de una celda bidireccional concatena las dos direcciones, luego
el estado que entra en la atencion tiene ancho 128, no 64. Con ancho 128:

  W_h de 128 por 64 = 8192
  W_s de 128 por 64 = 8192
  vector v de 64   =   64
  total anadido    = 16448

El numero pre-registrado, 8256, es el de una celda UNIDIRECCIONAL de ancho 64. Es decir:
el coste real de la atencion es el doble de lo que el pre-registro declara, y la
conclusion economica que sostiene la eleccion sigue siendo correcta pero por un margen
menor del que se creia.

## 2. Por que la conclusion no cambia

La razon de fondo de la eleccion no era el numero, era que con 1483 filas de desarrollo
cualquier capa gruesa seria un numero sobre el presupuesto de datos, no una decision
informada. Eso sigue siendo cierto con 16448, y sigue siendo mas cierto que con 8256.

La comparacion con las alternativas tampoco cambia. El producto escalar con multiples
cabezas proyecta a claves y valores y necesita al menos cuatro cabezas para tener sentido;
un transformer de encoder abriria dos preguntas en un solo experimento. Ninguna de las
dos esta cerca de 16448 parametros anadidos.

Ademas el valor de 16448 sigue siendo pequeno al lado de la propia red: con salida
bidireccional de ancho 128 y una capa densa de 128 a 3, el bloque recurrente aporta del
orden de 99000 parametros en B1. La atencion es una fraccion, no el grueso.

## 3. Que se cambia

Solo el numero. En la seccion 5.2, 8256 pasa a 16448, y la frase que lo sostiene pasa a
decir que el estado que entra en la atencion tiene ancho 128 porque la celda es
bidireccional con 64 por direccion.

Y se anade la obligacion de que el script PUBLIQUE las dos cifras, la pre-registrada y la
real medida contando parametros del modelo, y que si difieren lo declare como
discrepancia en vez de ajustar una de las dos. Publicar la discrepancia es preferible a
retocarla: es la diferencia entre un informe que se puede auditar y uno que se ha
alineado con lo que se queria leer.

## 4. Que NO se cambia

- Los cinco brazos, que siguen siendo bidireccionales. No hay ninguna propuesta de
  volver a unidireccional: el ancho 128 es una consecuencia de la bidireccionalidad, y la
  bidireccionalidad es la decision de Carlos.
- El nucleo aditivo tipo Bahdanau, y sus tres parametros v, W_h y W_s.
- La longitud de secuencia, que se queda en 64 con su razon ya escrita.
- La tabla de limite de deteccion minimo, verificada por busqueda de raiz con el t no
  central, con potencia 0.800000 exacta, y que reproduce el intervalo publicado de P78
  hasta la ultima cifra. Esa tabla no se toca.
- El protocolo de early stopping anidado, el clip de gradiente, las celdas separadas de
  relleno y de OOV, y los controles.
- El limite de deteccion declarado y la lectura exploratoria y no confirmatoria.

## 5. Nota de trazabilidad

Se/commit: e59548f es el commit del pre-registro con la aritmetica erronea. No se
reescribe. Esta enmienda es un commit nuevo, y el numero corregido vive aqui, con la
razon por la que estaba mal.

La discrepancy la detecto la implementacion al contar los parametros reales del modelo que
construye, que son 16448, y al contrastarlos con los 8256 que el pre-registro declaraba.
Fue, por tanto, una deteccion por medicion y no por revision, y conviene decirlo: una
revision de coherencia interna del pre-registro no la habria visto, porque la frase
aritmeticamente es correcta.

V03 remains the best verified leaderboard result.