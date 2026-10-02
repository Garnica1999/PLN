# P83: AVISO DE NO EJECUTABLE

Fecha: 2026-10-02
Estado: **CONGELADO, NO EJECUTABLE**
Script: experiments/p83_01_bayesiano.py, sha256
74389ba7812c922d94d8219899c18b4960d3f918afce4affebc41471f6bf3506, 7.614 lineas
Pre-registro: docs/p83_preregistro_bayesiano.md, sha256
7f376fa644c8ee4c696d8f47d6fe79ace7560d96af52235c5c14a20cd9d9ad36, commiteado en e59548f

ACTUALIZACION 2026-10-02, tras un ciclo de correccion. El ciclo avanzo mas de lo que
se suponia, y el estado que queda es este. Se conserva el aviso original mas abajo, que
describe el estado de partida.

## 0. Que avanzo el ultimo ciclo, y donde se detiene ahora

El modo completo **llego por primera vez a entrenar**. La correccion de A3 y la del
desempaquetado de la tupla de reescalado lo-permitieron, y la corrida ejecuta el brazo M1
completo, con las veinticinco combinaciones y la rejilla de alpha_0 de A6, en unos dos
minutos de GPU. Ahi ya no hay codigo muerto: el tiempo de reloj se gasta en calculate.

Cuatro de los diez bloqueantes quedan CERRADOS y verificados:

0. A3 binariza con los umbrales congelados de V03 antes de llamar a metricas_multietiqueta,
   y declara el alias de f1_media_columnas a corrected_cv de forma explicita. Verificado
   sobre el artefacto real: 0.7772114525165721, identico al valor guardado.
1. X_ev_e tomaba el PRIMER elemento de la tupla de reescalar_columnas, que es la matriz
   de ENTRENAMIENTO, y no el segundo, que es la de evaluacion. Con 836 filas de
   inner_train contra 297 de outer_validation. La misma línea reutilizada para la
   calibracion tenia el error identico.
2. decidir_alpha tiene ahora su llamada REAL, con filas igual a inner_train, y el
   ContextoDatos lleva proba_v03 para que la medicion interna tenga con que mezclarse.
3. La comprobacion de A14 mide POR COMBINACION. Antes comparaba las filas de calibracion
   contra la UNION de los ib de los cinco pliegues, que no son las filas de evaluacion de
   ninguna combinacion, y el solapamiento que salia, del 57 por ciento, era un emparejamiento
   mal hecho y no una fuga. Cada llamada registra ahora su ib.

Y dos fallos mas que aparecieron al ver el camino real correr, y que no estaban en el
diagnostico de los revisores:

- El contador de registros del instrumento usaba subscript, listar instrumento por el
  nombre del canal, sobre una clase que tiene atributos. Y despues, cuando la corrida paso
  a usar la instantanea, que es un diccionario, que era la otra mitad del mismo problema.
- El numero de parametros efectivos se indexaba por pliegue y semilla, con lo que al final
  solo quedaba la ultima combinacion, y A6 comparaba contra una etiqueta que no existia
  como clave. Ahora se desglosa por etiqueta, que es lo que el pre-registro pide.

DONDE SE DETIENE AHORA. A14, con tres comprobaciones falsas:

  el_alpha_no_usa_filas_del_calibrador
  las_filas_internas_y_externas_no_se_solapan
  la_comprobacion_sabe_fallar_por_mutacion

La causa esta localizada y es la misma de siempre: A14 calcula dos intersecciones sobre
los ATRIBUTOS VIVOS del instrumento, que tras veinticinco llamadas solo guardan la ultima
combinacion, luego comparan una combinacion contra otra. Lo que hace falta es calcularlas
REGISTRO CONTRA REGISTRO, que es el unico invariante que la regla 3.5 protege. Ese cambio
se empezó a escribir y NO se aplico.

LO QUE QUEDA DE LOS DIEZ, con el estado real:

  0  A3 binariza y declara el alias                    CERRADO, verificado
  1  X_ev_e tomaba la tupla equivocada                  CERRADO
  2  decidir_alpha sin llamador real                    CERRADO
  3  A14 con la comparacion por combinacion             A MEDIAS: falta el registro
  4  margenes y presupuesto como codigo muerto          SIN CONECTAR
  5  siete afirmaciones que pasan en vacio              SIN HACER
  6  A16                                                SIN HACER
  7  el break de A8                                     SIN HACER
  8  AFIRMACIONES_SIN_STOP, cuatro en vez de siete       SIN HACER
  9  comprobacion AST de funciones desconectadas        SIN HACER

Y una correccion a una afirmacion mia, porque importa. Durante este ciclo diagnostique que
la particion interna estaba INVERTIDA, con el 71 por ciento de las filas en el calibrador
en vez de en el entrenamiento. Era FALSO: mi comprobacion uso 0.29, que no es el valor del
script, que es 0.7. Con el valor real, inner_train recibe el 71 por ciento, inner_cal el
29, y son disjuntas. La particion es correcta. Un diagnostico hecho con un dato inventado
es un diagnostico que no vale, y hay que retirarlo aunque cueste.

## 1. Que dice este aviso

Este script **NO se puede ejecutar**. No es una afirmacion provisional: el modo completo
nunca ha llegado a evaluar una sola de sus veinte afirmaciones, y el modo de humo esta
roto.

Si estas leyendo esto dentro de seis meses, o dentro de seis dias, no lo ejecutes esperando
un resultado. El pre-registro de 871 lineas y las veinte afirmaciones de A0 a A19 siguen
siendo validos y son lo que hay que preservar. El codigo no esta a la altura.

## 2. Los dos hechos que lo demuestran

El modo COMPLETO, ejecutado con el fichero tal como estaba, falla a los 5,981 segundos, en
la linea 6674, antes de evaluar ninguna afirmacion:

    ValueError: Classification metrics can't handle a mix of binary and continuous targets
    metricas_multietiqueta(verdad_canonica, probabilidad_canonica)

La funcion recibe la matriz de probabilidades donde espera una prediccion BINARIA. Ese
fallo se ha corregido, con la binarizacion por los umbrales congelados de V03, y
verificado: binarizar proba con los umbrales reproduce pred exactamente y da
f1_media_columnas igual a 0.7772114525165721, que es el corrected_cv guardado.

El modo HUMO se detiene en el caso 46 de 194, con la linea 5580, por una firma de contador
que cambio y no se actualizo:

    TypeError: contar_fugas_calibrador() missing 1 required positional argument: 'instrumento'

Ese caso se ha corregido despues y no se ha vuelto a ejecutar el humo entero.

## 3. Que NO se ha hecho, y no es poco

El script tiene 7.502 lineas. Escritas, probadas por casos de humo, y **sin conectar al
camino real**. Ese es el fallo de clase, y se repite por tercera vez en este proyecto.

- El Grupo 0, la binarizacion de A3, esta HECHO y VERIFICADO.
- Del Grupo 1, la sobreescritura de X_ev_e esta HECHA, y queda por hacer la llamada real a
  decidir_alpha desde el sitio que mide las correcciones internas del alpha.
- Del Grupo 2, la instrumentacion del calibrador esta escrita, con su instantanea y con
  cada llamada registrando su propio conjunto permitido. NO se ha ejecutado.
- Los GRUPOS 3 y 4, Enteros, no han empezado. Son: conectar reproducir_liston,
  comparar_contra_el_liston y comprobar_presupuesto, que son CODIGO MUERTO; las siete
  afirmaciones que pasan en vacio porque fabrican sus propios numeros; A16; el break de A8;
  las siete afirmaciones sin STOP; y la comprobacion por AST de funciones desconectadas.

## 4. Los diez bloqueantes, con su localizacion

1. A3, linea 6674. metricas_multietiqueta recibe probabilidades donde espera binario.
   CORREGIDO y verificado.
2. X_ev_e, linea 4298. Se sobreescribia la matriz de evaluacion con la de entrenamiento,
   que tiene len de inner_train filas. CORREGIDO, sin ejecutar.
3. decidir_alpha, linea 2555. Su unico llamador estaba en un caso de humo. ESCRITO, sin
   ejecutar.
4. A14, cuatro fallos independientes, todos con marca de STOP, que se refutaba por
   construccion:
   a) ejecutar_brazo_m2, linea 4389, pisaba fijar_permitidos con el ultimo pliegue, y
      contar_fugas_ajuste_sobre_el_oov comparaba todas las llamadas contra ese unico
      conjunto. ESCRITO, sin ejecutar.
   b) La linea 6856 pasaba filas_internas, la union de outer_train, donde A14 espera
      filas_outer_validation. Como inner_cal es subconjunto de outer_train, la
      comprobacion de no solapamiento daba False SIEMPRE. ESCRITO, sin ejecutar.
   c) n_ajustes daba 100, que son cinco pliegues por cinco semillas por cuatro corridas de
      M1, contra los 25 que esperaba la comprobacion. ESCRITO, sin ejecutar.
   d) contar_fugas_alpha, sin declaracion de filas, caia en una rama que devolvia un numero
      que NO era cero, con lo que un ajuste sin declarar pasaba por limpio. ESCRITO, sin
      ejecutar.
5. reproducir_liston, comparar_contra_el_liston y comprobar_presupuesto no se llamaban
   nunca desde el modo completo. Los margenes de 0.005 en log-loss, 0.0015 en Brier y
   0.010 en ECE, y el presupuesto de 1.500 segundos, eran CODIGO MUERTO. La logica de no
   inferencia estaba bien escrita, pero no llegaba a aplicarse. SIN HACER.
6. Siete afirmaciones pasaban en vacio, cada una fabricando sus propios numeros:
   A10 con control_completo igual a 1.0 literal; A11 con la regla de 1.5 aplicada a un
   intervalo de ceros constantes; A12 con ceros escritos a mano; A13 con una fabrica que
   inventaba corrected_cv de 0.70, 0.701 y 0.702, y rescate, dano, neto, techo y piso de
   0, 1 y 2; A14 con deteccion_de_la_mutacion en True literal; A17 con un diccionario de
   dimensiones inventado, de cien columnas; y A19 con un F1 de 0.60 y un intervalo fijo.
   SIN HACER.
7. A16 fallaba siempre en el modo completo: recibia umbral con diccionario vacio, y
   por_alpha_0 con cuatro claves, las tres mas __texto__, contra una comprobacion de que
   sean tres. SIN HACER.
8. A8 medi un solo brazo, por un break en la linea 6794, y A8 exige los dos. SIN HACER.
9. AFIRMACIONES_SIN_STOP declaraba cuatro, A8, A12, A13 y A15. El pre-registro declara
   SIETE, A8, A9, A12, A13, A15, A16 y A17, en la tabla de la seccion 6 y en el criterio de
   parada de la seccion 13. Manda el pre-registro. La linea 484 del propio pre-registro dice
   cuatro, lo que contradice su tabla. SIN HACER.
10. Falta la comprobacion por AST que detecta funciones definidas y probadas SIN NINGUN
    LLAMADOR FUERA DE LOS CASOS DE HUMO. Es la defensa permanente contra el fallo de clase
    de este proyecto, y no existe. SIN HACER.

## 5. Lo que si se encontro por el camino, y vale la pena conservarlo

Un bug real de M2, encontrado porque el revisor leyo el codigo y no porque un test lo
detectara. contar_fugas y el intervalo de M2 no usaban los log-odds por variable, que se
calculaban y se descartaban. El intervalo salia centrado en 0.625 constante, que es la odds
del prior, mientras el punto M2 real iba de 0.5085 a 0.6717. Un sesgo sistematico de 0.116.
Corregido, y verificado por convergencia Monte Carlo: el desvio centro contra punto baja de
1.08e-4 con 4.000 muestras a 4.2e-5 con 20.000.

Y una correccion a una cifra que yo mismo di y que no se sostiene. Medi el conteo de
parametros efectivos del ARD con hiper-prior Gamma y reported 829, 652 y 595 sobre HS.
Recorriendo las quince combinaciones de cinco pliegues por tres etiquetas, con y sin
reescalado L2, el maximo medido es 500.2, y ninguno se acerca a 829. La DIRECCION si se
confirma en los quince casos, que es lo que importa: mas parametros efectivos con alpha_0
mas pequeno. Los numeros no. El caso de humo a6_medido_no_reproduce_los_declarados declara
esto en vez de escribir cifras que no salen.

## 6. Lo que hay que hacer con esto

Antes de volver a tocar el codigo, el orden queFlorida dos revisores dao y que respeta las
dependencias:

- Arreglar la firma del contador, linea 5580, y ejecutar el humo entero.
- Ejecutar el MODO COMPLETO antes de tocar nada mas. Con eso se sabe si el camino real
  arranca, y si arranca, cuanto de los diez bloqueantes era de conexion y cuanto es de
  diseno. Ese dato decide si P83 se salva o se cierra.
- Si arranca, seguir por los grupos 3 y 4 en el orden de dependencias.
- Y anadir la comprobacion del punto 10, que es la defensa permanente. Sin ella, el mismo
  fallo vuelve en el siguiente experimento.

Y una pregunta que P83 no responde y que sigue abierta: con 1.483 filas y celdas rarismas,
el papel esperado del bayesiano era miembro diverso, no campeon. P82 ya contesto en
negativo la parte de la complementariedad, con los tres netos negativos y con el brazo
detrayendo mas de lo que rescata. Queda por medir solo la calibracion, que es la pregunta
que no depende de V03.

## 7. Lo que este experimento NO puede decir

Nada. No se ha ejecutado ninguna vez, no hay resultados, y no hay nada que citar. La
formulacion que sigue vigente y que hay que conservar en cualquier documento que se
escriba sobre P83:

V03 remains the best verified leaderboard result.

Y no se puede afirmar que el bayesiano, ni Word2Vec, ni la RNN, ni la atencion sean mejores
que V03. Lo que se ha medido hasta ahora, con el limite de deteccion de cada diseno
declarado, es que la RNN esta a 0.1375 por debajo de V03 en el metodo local, y a 0.1016
por debajo en el leaderboard, donde dio 0.68150.