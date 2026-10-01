# Enmienda 2 al pre-registro de P82: eleccion de dispositivo de calculo

Fecha: 2026-10-01
Estado: adoptada antes de ejecutar el modo completo
Documentos que modifica: docs/p82_preregistro_rnn_atencion.md, sha256
ba74ae71a3b54a273c7f0ac5d55e4d4fd7d1ede641f5fbd022c2ae6377a62061, y
docs/p82_enmienda_1.md, sha256
53c8555d1a5ef0fcebb06c4a6f061acf56e568d95242e907387f0af1b03420d8
Motivo: el primer intento de ejecucion en modo completo resulto por CPU cuando el
proyecto tiene GPU disponible, y consumio 66 minutos sin cerrar el primer bloque de
mediciones. Se mide el coste de cada dispositivo y se declara la eleccion.

## 1. Lo que paso

La primera ejecucion del modo completo arranco sin que nada en el script eligiera
dispositivo. El fichero no contenia ninguna mencion a GPU, ni a ROCm, ni a cuda, y no
tenia ni una sola llamada a to con un argumento de dispositivo. Todo fue a CPU. Con eso:

  66 minutos de reloj, 30 nucleos ocupados, 531 MB de memoria residente, y cinco
  afirmaciones escritas, que son A0, A1, A2, A3 y A8. Las cinco son baratas. El bloque de
  entrenamiento no habia empezado.
  La GPU, en paralelo, estaba al 7 por ciento de uso y 13 por ciento de memoria.

El proceso se paro con su PID, y no dejo informes ni figuras a medias. Siete ficheros
parciales de A0 a A3 y A8 quedaron en reports/p82_parciales y son evidencia del intento.

La causa no fue solo una omision del encargo de ejecucion. El script no podia usar la GPU
porque no tenia forma de hacerlo. Eso se corrige en el script, no en el encargo.

## 2. Medicion del coste de cada dispositivo

Micro-benchmark con la clase real del script y la geometria real de P82: 1483 filas,
secuencia de 64, lote de 64, oculto de 64, optimizador Adam, clip de gradiente, tres
epocas por ajuste, mejor de tres repeticiones:

  CPU con 16 hilos   2.43 s por ajuste   que son   13.5 min para 1000 ajustes
  CPU con 30 hilos   3.22 s por ajuste   que son   17.9 min para 1000 ajustes
  GPU con ROCm       1.24 s por ajuste   que son    6.9 min para 1000 ajustes

Dos conclusiones, y la segunda es la que mas sorprende.

La primera, la GPU gana por un factor de dos, no por dos o tres ordenes de magnitud. La
razon es que el trabajo por paso es pequeno: un lote de 64 filas por 64 posiciones, con una
celda de oculto 64, deja el acelerador infrautilizado y el coste de lanzar los kernels
compite con el calculo. Es un experimento pequeno para una RX 9070 XT de 16 GB. El factor
dos es el que hay, y se declara el que hay.

La segunda, mas hilos NO es mas rapido, y el caso de 30 hilos es el peor de los tres. Con
16 hilos se tarda 2.43 s y con 30 se tardan 3.22 s, un 32 por ciento mas. Es
sobrecarga de planificacion: en un bucle con volumen de tensores pequenos, mas hilos
pelean por el mismo trabajo. La primera ejecucion ocupaba 30 nucleos, es decir, corria en
el peor de los tres puntos. Poner el numero de hilos a 16 por si solo habria recortado el
tiempo aproximadamente a la mitad sin tocar nada mas.

## 3. La decision

Se usa la GPU cuando este disponible, con caida a CPU cuando no lo este. Y se fijan 16
hilos, en ambos casos, por la medicion anterior.

El pre-registro no decia nada sobre el dispositivo, luego elegirlo no es una violacion de
un texto vinculante. Se declara igualmente, porque un informe que no dice sobre que
dispositivo se corrio es un informe que no se puede auditar.

## 4. Lo que esto cambia en los resultados, y no se puede esconder

Cambiar de CPU a GPU cambia el consumo del generador de numeros aleatorios, porque CPU y
CUDA usan generadores distintos. Afecta a la inicializacion de los pesos y al dropout. Por
tanto:

  Los valores absolutos de los cinco brazos, y las deltas pareadas, son los de GPU.
  No son los que habria dado una corrida en CPU, y no deben compararse con ninguna.

Lo que SI se conserva, y es lo que sostiene el diseno:

  El pareamiento DENTRO de la corrida. Los cinco brazos se ejecutan en el mismo
  dispositivo, luego comparten el mismo consumo de generador y las deltas pareadas
  siguen siendo validas.
  A9, que exige que el hash de la secuencia de lotes coincida entre brazos, sigue
  teniendo sentido: el generador de permutaciones se mantiene en CPU precisamente
  para que ese hash sea estable y auditable, aunque los datos esten en GPU.

No se usa precision mixta. En un diseno cuyo nucleo es el pareamiento entre brazos, meter
una via de calculo que cambia la precision introducing variabilidad numerica seria
justo el tipo de cosa que este proyecto ha pagado dos veces. Si alguien quiere usarla mas
adelante, que se registre como desviacion y se mida su efecto.

## 5. Lo que hay que publicar

El informe tiene que declarar, sin que se le pregunte:

  El dispositivo usado, con su nombre. GPU con su modelo, o CPU.
  El numero de hilos fijados.
  Que no se uso precision mixta.
  Que la medicion del coste por dispositivo es la de la seccion 2, con sus numeros, para
  que un lector pueda juzgar si la eleccion estaba justificada.

Y una comprobacion en tiempo de ejecucion, no una suposicion: que el modelo este
efectivamente en el dispositivo que se dice, y que la afirmacion falle si no lo esta.
Un informe que dice que se uso la GPU y un modelo que se quedo en la CPU es peor que un
informe que no dice nada.

## 6. Que NO se cambia

- La arquitectura, que sigue siendo bidireccional en los cinco brazos.
- El protocolo de early stopping anidado, el clip de gradiente, las celdas separadas de
  relleno y de OOV, y los controles.
- La tabla de limite de deteccion minimo, verificada con potencia 0.800000 exacta.
- La lectura declarada: EXPLORATORIO y no confirmatorio, y un resultado negativo NO se
  lee como que la atencion no aporta.
- El presupuesto de reloj de la seccion 12, con su hito a los 40 minutos.

## 7. Trazabilidad

El commit del pre-registro y el de la enmienda 1 no se reescriben. Esta es la segunda
enmienda, y las tres Together dejan el protocolo de P82 completo y auditable.

V03 remains the best verified leaderboard result.