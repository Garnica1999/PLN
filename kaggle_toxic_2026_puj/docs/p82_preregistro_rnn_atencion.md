# P82: pre-registro de RNN con capas de atencion

Fecha de escritura: 2026-09-30
Proyecto: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj
Rama: proyecto-kaggle-toxic-2026-p67
Script previsto: experiments/p82_01_rnn_atencion.py
Salidas previstas: reports/p82_rnn_atencion.json y reports/p82_rnn_atencion.md
Estado: PRE-REGISTRADO. Este texto se hashea y se commitea ANTES de escribir una linea de
codigo. Sin ese commit no hay experimento, igual que en P80 y en P81.

## 0. Decision de Carlos, y su advertencia tecnica al lado

El plan maestro, bitacora de las Fases 6 a 8, registra la decision de Carlos sobre P82:

> P82, RNN con capas de atencion. DECISION: ABRIR, y abrir directamente con atencion, sin
> pasar antes por una reestimacion limpia del BiGRU. ESTA DECISION SOBRESCRIBE MI
> RECOMENDACION y la de kaggle-specialist.

Este pre-registro NO discute esa decision. La ejecuta con el protocolo mas limpio que el
proyecto sabe construir, y escribe por adelantado el limite que la decision no elimina: la
potencia del diseno. La decision de Carlos es abrir la via. La de este documento es hacer
que lo que salga de esa via no se pueda sobreinterpretar.

Los tres dictamenes que contraindicaban la decision, y que aqui quedan citados para que
exista constancia de que se Known, no se ocultan:

- kaggle-specialist, Fase 5: "La via RNN con atencion no se abre. Con desviacion entre
  semillas de cerca de 0.02 frente a un gap de 0.117, el diseno no puede medir la magnitud
  que necesitaria."
- dl-specialist, Fase 5: los bucles de v9_04 y v10_11 no tienen clip_grad_norm_.
- nlp-specialist, Fase 5: en v9_04 linea 84 el OOV cae en indice.get(token, 0) y
  padding_idx es tambien 0, luego relleno y OOV comparten celda.

Lo que este pre-registro hace con esas tres objeciones:

- la primera se convierte en la seccion 3, con el calculo explicito y no con una frase;
- la segunda se convierte en la afirmacion A6, STOP si;
- la tercera se convierte en las afirmaciones A4 y A5, STOP si.

## 1. La pregunta, y lo que este diseno NO puede responder

Pregunta, una sola:

> Dentro de una MISMA corrida, con early stopping estrictamente anidado, con celda de
> relleno y celda de OOV separadas, con el relleno aplicado ANTES de la red recurrente, y
> con clip de gradiente, la capa de atencion aditiva sobre las salidas de una RNN bidireccional
> mejora la metrica primaria frente a la MISMA RNN sin atencion, en la misma corrida?

Lo que este diseno NO puede responder, escrito por adelantado y con numeros en 3:

- Si la mejora, si la hay, es menor que el limite de deteccion del diseno, este
  experimento NO LA VE. Con los numeros de 3, el limite es del orden de 0.021 de
  f1_media_columnas, es decir alrededor de una sexta parte del gap con V03.
- Por lo tanto un resultado NEGATIVO no se lee como que la atencion no aporta. Se lee como
  que el efecto, si existe, es menor que lo que este diseno puede resolver. La afirmacion
  A12 ata esa frase a la regla aritmetica, y es STOP si la frase no esta.

## 2. Alcance

### 2.1 Que se lee

- Solo lectura:
  - experiments/v9_04_dl_secuencial_fasttext.py: MAX_LEN (:47), secuencias (:78-85), BiGRU
    (:88-107), construccion del vocabulario e indice (:170-175). Se lee para saber que se
    esta arreglando, y NO se importa su logica de celda 0.
  - experiments/p75_01_complementariedad_textcnn.py: epoca_estrella (:102-160) y
    entrenar_fijo (:163-211). Son el mecanismo de early stopping anidado ya verificado con
    180 llamadas reales en P78, y P82 los reutiliza con la instrumentacion de A5.
  - src/data.py read_train, que se llama, no se edita.
  - src/splits.py, para reproducir la particion sin abrir el holdout.
  - src/evaluation.py:64-123, f1_media_columnas. Se llama, no se edita.
  - modelos/twitter_domain/w2v_twitter_100d.npz, SOLO lectura, sha256
    302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a.
  - reports/p78_addendum.md, reports/p78_control_anidado.json y reports/p81_cierre_huecos.json,
    SOLO para los numeros de referencia y para el metodo de intervalo.

### 2.2 Que NO se lee, y que NO se hace

- test, cualquier leaderboard, cualquier submission. Ni una vez.
- El holdout. Ni su texto, ni sus etiquetas, ni su fichero de identificadores.
- Ningun script de P71 a P81. No se editan, no se borran, no se corrigen.
- No se reentrena, no se modifica, no se sobrescribe el artefacto Word2Vec 100d. Los
  embeddings van CONGELADOS en P82, y la razon se escribe en 4.1.
- No se reentrena V03, no se tocan sus umbrales congelados 0.44, 0.51, 0.46.
- No se genera submission, no se consulta el leaderboard.
- No se elige umbral, tolerancia, margen ni criterio despues de ver un numero. Todos estan
  en este documento y estan hasheados antes de ejecutar.

### 2.3 El limite permanente que P81 declaro, y que P82 hereda

P81 dejo escrito que el anclaje a una implementacion externa de SGNS queda SIN HACER, de
forma permanente con el entorno actual, porque gensim no esta instalado, no esta en
requirements.txt y con Python 3.14.7 probablemente no compile. Ese limite no se cierra en
P82. En consecuencia:

- P82 NO autoriza afirmar que el artefacto Word2Vec sea mejor o peor que el de word2vec o
  el de gensim.
- P82 NO reentrena el trainer, luego P82 no dice nada nuevo sobre H1, H2, H3 ni H4.
- La consecuencia metodologica que si importa: todo lo que P82 afirme sobre la atencion se
  apoya en un artefacto cuyo objetivo de entrenamiento esta verificado como NO publicado
  (P80 H1, P81 bloque 1) y cuya calidad relativa esta SIN anclar externamente. Si la atencion
 Ayudase o no, ese limite persiste. Se escribe aqui para que P82 no se lea despues como si
  hubiera improvements la representacion.

## 3. EL LIMITE DE DETECCION, POR DELANTE Y CON NUMEROS

Esta seccion es la parte mas importante del documento. Va antes de la lista de
afirmaciones porque condiciona como se puede leer todo lo demas.

### 3.1 De donde salen los tres numeros

| Simbolo | Valor | Origen |
|---|---|---|
| sd_pareada | 0.021588 | P78, P1, desviacion tipica entre las tres semillas del delta pareado. Es la UNICA desviacion de un delta pareado medida en este proyecto. |
| gap | 0.117450 | reports/p78_addendum.md seccion 8, distancia del mejor brazo de P78 a V03 en f1_media_columnas. |
| ancla | +0.024893 | P78, P1, estimacion puntual del mismo delta pareado. |

Comprobacion de que el ancla es la correcta: con sd igual a 0.021588, n igual a 3 y t de
Student con 2 grados de libertad, el intervalo al 95 por ciento del punto +0.024893 es
(-0.028735, +0.078521), y el valor publicado por P78 es (-0.028736, +0.078522). La
diferencia es de la ultima cifra. El ancla reproduce el intervalo publicado, luego la
aritmetica de 3.2 es la misma que uso P78 y no una convencion nueva.

### 3.2 La tabla, calculada con t no central, dos lados, alpha 0.05

MDE es el delta minimo detectable al 80 por ciento de potencia. semiancho es la mitad del
intervalo al 95 por ciento.

**La formula es la PAREADA, y la razon se escribe porque no es la unica posible.** El
estimador de P82 es el delta por SEMILLA entre dos brazos emparejados, luego la desviacion
relevante es la de los deltas, y su error tipico es sd partido por la raiz de n. La formula
de dos muestras independientes, que lleva sd por la raiz de dos sobre n, es la que se usa
cuando los dos brazos se entrenan con datos distintos, y aqui no es el caso. Usarla
INFRAVALORA el MDE y, en consecuencia, INFRANUMERA el numero de semillas necesario: con la
formula de dos muestras el numero de semillas para resolver un delta de 0.010 sale de 39 a
58, es decir casi un 50 por ciento mas de entrenamiento del que haria falta, y el error se
paga en horas de GPU, no en decimales.

La tabla se calculo por tres vias independientes, que tienen que coincidir:

1. La t no central, con la potencia como P(T mayor que el critico) mas P(T menor que el
   critico negativo), evaluada con el cuantil critico, no con una aproximacion de normales.
2. Integracion numerica de la misma potencia sobre la distribucion de la variable chi
   cuadrado del denominador del cociente, que es la definicion de la potencia del t no
   central y no una aproximacion de ella.
3. Monte Carlo, con cuatro millones de tiradas del par normal y chi cuadrado, contrastando
   el mismo cociente.

Las tres coinciden en la tabla de potencias hasta la cuarta cifra decimal. Y ADEMAS la
aritmetica reproduce el intervalo publicado por P78 para P1, que es (-0.028736, +0.078522),
con su ultima cifra, como se comprueba en 3.1. ESA reproduccion es la prueba de que la
aritmetica es correcta, y por eso se comprueba y no se afirma: un calculo de potencia que no
reproduce un intervalo ya publicado no tiene por que ser correcto, por muy bien escrita que
este la formula.

| n | MDE (potencia 0.80) | semiancho IC95 | potencia si el delta real es el ancla | MDE / gap |
|---|---|---|---|---|
| 3 | 0.070464 | 0.053628 | 0.2179 | 0.60 |
| 5 | 0.036311 | 0.026805 | 0.5000 | 0.31 |
| 8 | 0.024956 | 0.018048 | 0.7980 | 0.21 |
| 10 | 0.021502 | 0.015443 | 0.8993 | 0.18 |
| 12 | 0.019184 | 0.013716 | 0.9522 | 0.16 |
| 20 | 0.014258 | 0.010103 | 0.9983 | 0.12 |
| 30 | 0.011425 | 0.008061 | 1.0000 | 0.10 |
| 50 | 0.008726 | 0.006135 | 1.0000 | 0.07 |

### 3.3 Que dice la tabla, en cinco frases

1. **El gap NO es el problema.** Para resolver un efecto del tamano del gap completo,
   0.117450, basta n igual a 1. El diseno tiene potencia de sobra para detectar efectos
   GRANDES. Por eso el gap de 0.117 no es la objecion.
2. **El problema son los efectos PEQUENOS, que son los esperables.** A diez semillas el
   limite de deteccion es 0.021502, es decir el 18 por ciento del gap. Cualquier efecto de
   la atencion por debajo de 0.021 es indistinguible del ruido de reentrenamiento con este
   diseno. La atencion aporta, si aporta, poco: es una capa mas sobre 1483 filas.
3. **Tres semillas habrian sido un error, y el numero lo dice.** A tres semillas la
   potencia contra el propio ancla de P78 es 0.2179, y el semiancho del intervalo es
   0.053628, que es casi la mitad del gap. Con tres semillas, mas de tres de cada cuatro
   corridas Habrian dorado exactamente cero, y un cero asi no se distingue de un efecto
   real pequeno. Por eso el minimo es diez, y no nueve ni ocho.
4. **Diez semillas es un SUELO, no un confort.** A diez semillas la potencia contra el
   ancla es 0.8993, es decir queda un 10 por ciento de probabilidad de no detectar un
   efecto que existe y es del tamano del ancla. Y esa cifra asume que la desviacion real
   es la medida en P78 con un TextCNN, que es una familia de arquitectura MAS estable que
   una BiGRU con atencion. Si la desviacion real fuera el doble, el limite de deteccion
   pasaria a 0.043003, la potencia caeria a 0.3708 y el cociente MDE/gap a 0.37. Si fuera el
   triple, el limite seria 0.064505 y la potencia 0.1930.
   Tabla de sensibilidad, declarada antes de ejecutar:

   | factor sobre 0.021588 | sd supuesta | MDE a n igual a 10 | potencia en el ancla | MDE / gap |
   |---|---|---|---|---|
   | 1.0 | 0.021588 | 0.021502 | 0.8993 | 0.18 |
   | 1.5 | 0.032382 | 0.032253 | 0.5825 | 0.27 |
   | 2.0 | 0.043176 | 0.043003 | 0.3708 | 0.37 |
   | 3.0 | 0.064764 | 0.064505 | 0.1930 | 0.55 |

   Por eso la afirmacion A10 obliga a recalcular y republicar esta tabla con la desviacion
   MEDIDA en la propia corrida, y no con la planificada. La planificada es un techo
   optimista declarado, no un resultado.
5. **Que resolver la mitad del limite costaria 39 semillas, y la cuarta parte costaria
   149.** El numero de semillas necesario para tener potencia 0.80 contra un delta de
   0.010 es 39, y contra un delta de 0.005 es 149. Con 39 semillas el tiempo de GPU seria
   del orden de cuatro horas; con 149, del orden de quince horas. Carlos debe saber
   que el diseno mas honesto posible, un diseno que resolviera efectos de 0.010, no cabe en
   el presupuesto de este proyecto, y que por eso 0.021 es el suelo de lo que aqui se puede
   afirmar.

### 3.35 Nota de trazabilidad de la aritmetica

Se escribe para que conste lo que ocurrio, no para carregar culpa a nadie.

El calculo inicial de esta tabla se hizo con la formula de dos muestras independientes, con
sd partido por la raiz de dos sobre n. Con esa formula el numero de semillas necesario para
resolver un delta de 0.010 salia de 58 en lugar de 39, y el MDE de n igual a 10 salia de
0.024006 en lugar de 0.021502. Se detecto porque esa tabla NO reproducia el intervalo
publicado por P78 para P1, y esa comprobacion es la prueba que 3.2 declara obligatoria.

Lo que se corrige es el METODO de calculo, no el resultado del experimento. P82 no ha
ejecutado nada: no hay ningun numero de P82 que cambiar, y ningun numero de P78 cambia. La
tabla de potencias no cambio, porque ya estaba calculada con la t no central y con la formula
pareada. Lo que cambio es la tabla de MDE, que estaba calculada con una aproximacion de
manual, t(0.975, n-1) mas t(0.80, n-1) por sd sobre la raiz de n, y esa aproximacion se
desvia de la exacta mas cuanto menor es n, 0.066847 frente a 0.070464 en n igual a 3, y
converge a la exacta cuando n crece, 0.008727 frente a 0.008726 en n igual a 50.

La leccion, y es la que importa mas alla de este documento: una formula de potencia que no
se contrasta contra un intervalo ya publicado es una formula que nadie ha comprobado. El
intervalo de P78 estaba disponible, se pudo usar como prueba antes de escribir nada, y
tampoco se uso hasta que la discrepancia salto. La regla que sale de aqui es que toda tabla
de potencia de este proyecto se verifica reproduciendo un intervalo publicado, y si no lo
reproduce, la tabla esta mal antes de que los numeros seanfelices.

### 3.4 La regla de lectura, fijada antes de ejecutar

Sea D el delta pareado del primario, definido en 5.3, y sea IC su intervalo t al 95 por
ciento sobre las 10 semillas, con t de Student y 9 grados de libertad.

- Si el extremo inferior de IC es mayor que 0: el efecto es detectable, se publica como
  AFIRMATIVO DENTRO DE LA NATURALEZA EXPLORATORIA, y se declara explicitamente que el
  intervalo no cubre el gap de 0.117450, luego el efecto sigue sin alcanzar a V03.
- Si IC contiene el 0: se publica como NO CONCLUYENTE, y con estas palabras, que la
  afirmacion A12 exige encontrar literales en el informe:
  UN_RESULTADO_NEGATIVO_AQUI_NO_AFIRMA_QUE_LA_ATENCION_NO_APORTA_AFIRMA_QUE_EL_EFECTO_ES
  INFERIOR_AL_LIMITE_DE_DETECCION_DE_ESTE_DISENO
  y el limite de deteccion con su numero recien medido.
- Si el extremo superior de IC es menor que 0: se publica como NEGATIVO CONTRA LA
  ATENCION, que es el unico caso en el que si se puede leer como que la atencion no aporta
  en ESTE diseno y con ESTA arquitectura. Se escribe que la conclusion es condicional a la
  arquitectura y al protocolo, no general.
- En los tres casos se publica ademas el cociente entre el semiancho de IC y el gap, que
  es la medida de cuanto del gap este diseno es capaz de ni siquiera mirar.

En ningun caso se escribe que la atencion mejora o no mejora la clasificacion del problema,
solo lo que este diseno resuelve.

## 4. Protocolo, y lo que arregla frente a v9_04 y v10_11

### 4.1 Embeddings congelados, y por que

La matriz es modelos/twitter_domain/w2v_twitter_100d.npz, 100 dimensiones, CONGELADA.

La razon de congelar, y no de reentrenar o afinar, se escribe porque es una decision de
diseno y es cara de revertir:

- P80 y P81 confirmaron que el objetivo del trainer NO es el SGNS publicado, porque
  promedia un termino que la referencia suma, con ratio 1/K exacto y residuo 4.2e-17.
- P81 no pudo anclar contra una implementacion externa de SGNS, y declaro ese limite como
  permanente.
- Reentrenar o afinar en P82 seria multiplicar dos cosas no verificadas: una atencion
  nueva y un artefacto reentrenado con un objetivo que no es el de la referencia. Si P82
  saliera bien, no se sabria cual de las dos cosas aporto, y la lectura seria imposible.

Consecuencia que se declara: los resultados de P82 NO son comparables con v9_04
(0.694343, FastText 200d) ni con v10_11 (0.711639) ni con DL-03 (0.736367). Cambian la
dimension del embedding, el tratamiento del relleno y del OOV, el early stopping y el
clip. El unico control valido de P82 es INTERNO, y por eso 5.1 obliga a que el primario
compare contra un brazo de la misma corrida y no contra un numero de otro experimento.

### 4.2 Celda de relleno y celda de OOV, separadas, y el OOV ENTRA

El defecto, con su linea: experiments/v9_04_dl_secuencial_fasttext.py:172 construye
indice con i mas 1, luego el id 0 no se asigna a ninguna palabra; :84 escribe
matriz[fila, columna] igual a indice.get(token, 0), luego un token fuera de vocabulario cae
en el id 0; :94 pone padding_idx igual a 0, luego el relleno tambien es el id 0; :102 arma
mascara igual a entrada distinto de 0, luego la mascara borra relleno Y OOV a la vez; y el
pooling de :105 a :107 promedia solo sobre lo que la mascara deja. P80 y P81 lo remedian
por diagnostico. P82 lo REMEDIA POR CONSTRUCCION.

Diseno, fijado aqui:

| Elemento | Valor |
|---|---|
| id_relleno | 0 |
| id_oov | len(vocabulario) mas 1 |
| filas de la matriz | len(vocabulario) mas 2 |
| fila id_relleno | identicamente cero, y se publica el residuo, que tiene que ser 0.0 exacto |
| fila id_oov | vector CONGELADO DECLARADO, no nulo, con sus componentes publicados |
| mascara de la atencion | softmax sobre las posiciones NO de relleno, y el OOV ENTRA en el promedio |
| mascara del pooling en los brazos SIN atencion | identica a la de la atencion: excluye relleno, INCLUYE OOV |

La ultima fila es la decision sustantiva, y es la que corrige el defecto de fondo. En
v9_04 el OOV no solo compartia celda con el relleno: ademas se TIRABA en el pooling, luego
la red nunca podia usar un token fuera de vocabulario aunque tuviera celda propia. En P82
el OOV tiene celda propia, tiene vector propio, y ademas entra en el promedio. Eso es un
cambio de protocolo, no un retoque, y por eso los brazos de P82 no son comparables con
ningun BiGRU anterior.

Por que importa mas de lo que parece, con los numeros de P81 y no con adjetivos:

- proporcion de posiciones OOV: 0.04124873567093729
- proporcion de posiciones de relleno: 0.7309191672285907
- proporcion de OOV sobre los tokens que llegan a la matriz: 0.153295
- tipos distintos fuera de vocabulario: 2008
- MAX_LEN: 64, y longitud mediana declarada de 16 palabras en la discusion de diseno

El 15.3295 por ciento de OOV sobre los tokens que llegan a la matriz NO es un detalle de
redondeo: es casi uno de cada seis tokens que el modelo ve. Y el relleno es del 73 por
ciento de las posiciones. En una secuencia de 64 posiciones con 16 palabras utiles, hay
como 48 posiciones de relleno y, de las 16 utiles, del orden de 2 o 3 serian OOV.

MAX_LEN se queda en 64, y la razon se escribe: bajarlo mejoraria el rendimiento probable y
 ROMPERIA la comparabilidad con las cifras de cobertura medidas por P81 con el tokenizador
real de v9_04. P82 no cambia la longitud de secuencia para no tener que explicar despues
que comparo dos cosas distintas. La fraccion de filas truncadas se mide y se publica en A8.
El efecto del relleno sobre la representacion se mide reutilizando la condicion C1 de P81 y
se publica como cifra de contexto, NO como resultado de P82.

### 4.3 El relleno se aplica ANTES de la red recurrente

P80 y P81 confirmaron H6: en v9_04 y v10_11 la mascara de relleno se aplica DESPUES de la
GRU, luego los pasos de relleno pueden contaminar los estados validos en la direccion
inversa.

En P82 se usa packed sequence, con pack_padded_sequence y pad_packed_sequence, sobre las
longitudes reales, con enforce_sorted igual a False. La afirmacion A3 lo verifica de forma
ejecutable y no por inspeccion: el estado oculto en el primer paso de una secuencia con
relleno inicial tiene que ser IDENTICO al de la misma secuencia sin el relleno, con
tolerancia 1e-6 en norma relativa. Si no coincide, la red esta leyendo el relleno y A3
falla.

### 4.4 Early stopping estrictamente anidado, y la razon por la que es STOP

P80 y P81 confirmaron H5: los BiGRU de v9_04 y v10_11 eligen el checkpoint con la perdida
de validacion del MISMO pliegue externo que despues puntuan. Esa es exactamente la fuga que
condeno P73 y P74, y la que llevo a reetiquetar +0.0316259 y +0.02427 como exploratorios.

Protocolo por pliegue externo k, con indices ia para outer_train e ib para
outer_validation:

1. Split interno: StratifiedGroupKFold(3, shuffle igual a True, random_state igual a
   20261400 mas 100 multiplicado por k) sobre grupos[ia] e y[ia], que produce tres
   particiones internas (m_in j, m_cal j) para j igual a 0, 1 y 2. El MISMO split interno se
   usa para los CINCO brazos del pliegue k, de modo que los brazos se comparan sobre filas
   internas identicas.
2. Busqueda de E estrella: para cada brazo y cada semilla, se entrena sobre ia[m_in j] y se
   hace early stopping contra ia[m_cal j], para los tres j. E estrella es la MEDIANA de los
   tres valores. Ninguna fila de ib participa en ningun paso de este punto.
3. Reajuste: se entrena sobre ia COMPLETO durante E estrella epocas EXACTAS, sin early
   stopping y sin scheduler, y se predice ib UNA SOLA VEZ.
4. ib se evalua exactamente una vez por (brazo, semilla, pliegue). Un contador explicito lo
   lleva y el script aborta con SystemExit si se supera. Es la afirmacion A20.

El mecanismo es el de experiments/p75_01_complementarien_tectextcnn.py, epoca_estrella
(:102-160) y entrenar_fijo (:163-211), que P78 verifico capturando 180 llamadas reales a
partir_interna. P82 no reescribe ese mecanismo: lo reutiliza CON INSTRUMENTACION.

La instrumentacion que hace falta, y que es A5: se parchea la funcion que calcula la perdida
de validacion para que registre el conjunto de filas que ve. Se exige que la UNION de todas
las filas vistas por las llamadas que DECIDEN epoca sea un subconjunto de ia, y que ib no
aparezca en ninguna llamada, ni una vez. Ademas, por mutacion, se comprueba que la
comprobacion SABE FALLAR: se ejecuta una vez con un fallo inyectado de decision de epoca
sobre ib y la comprobacion tiene que detectinglo. Un chequeo que no puede fallar no prueba
nada, y esa es la leccion mas cara de la serie P81.

### 4.5 clip_grad_norm_ en todos los bucles

dl-specialist senalo que los bucles de v9_04 y v10_11 no tienen clip_grad_norm_, contra la
regla de clipping en redes recurrentes. No estaba en H1 a H7 y se adopta como limite
adicional.

En P82 se aplica torch.nn.utils.clip_grad_norm_ sobre TODOS los parametros con gradiente,
con umbral DECLARADO igual a 1.0, en los tres bucles: la busqueda interna j, el
reentrenamiento de E estrella y, si lo hubiera, cualquier evaluacion. Se publica la NORMA
DEL GRADIENTE ANTES DEL CLIP en cada epoca, no despues, y su distribucion por brazo, por
pliegue y por semilla. Es la afirmacion A6, STOP si.

Se publica porque una red recurrente con atencion y solo 1483 filas es exactamente el
entorno donde el gradiente hace el ridculo, y sin la serie de normas no se puede saber si
el clip esta haciendo algo o si solo se esta ejecutando.

## 5. Brazos, controles y comparaciones

### 5.1 Los cinco brazos, y por que estos y no otros

| Id | Brazo | Nucleo | Capa de atencion | Embedding | Papel |
|---|---|---|---|---|---|
| B1 | BiGRU sin atencion | GRU bidireccional, oculto 64 | NO | W2V 100d congelado | CONTROL INTERNO PRIMARIO |
| B2 | BiGRU con atencion | GRU bidireccional, oculto 64 | SI, aditiva | W2V 100d congelado | BRAZO PRIMARIO |
| B3 | BiLSTM sin atencion | LSTM bidireccional, oculto 64 | NO | W2V 100d congelado | CONTROL INTERNO SECUNDARIO |
| B4 | BiLSTM con atencion | LSTM bidireccional, oculto 64 | SI, aditiva | W2V 100d congelado | BRAZO SECUNDARIO |
| B5 | BiGRU con atencion, matriz PERMUTADA | GRU bidireccional, oculto 64 | SI, aditiva | multiconjunto de filas de W2V, permutado | CONTROL DE GEOMETRIA |

Por que la atencion ADITIVA y no la de producto escalar ni un transformer:

- La atencion aditiva es la referencia de la literatura de clasificacion de texto corto con
  celdas recurrentes, y es la que hace el experimento comparativo mas barato de siempre: una
  matriz W_h de 64 por 64, una matriz W_s de 64 por 64 y un vector v de 64, es decir 8256
  parametros anadidos. Con 1483 filas, cualquier cosa mas gruesa seria un numero sobre el
  presupuesto de datos, no una decision informada.
- La atencion de producto escalar con multiples cabezas exige proyectar a claves y valores, multiplica los
  parametros y necesita al menos 4 cabezas para tener sentido, y no hay 1483 filas que lo
  justifiquen.
- Un transformer de encoder no cabe: este proyecto ya tiene una linea de red recurrente, y
  abrir un transformer seria abrir dos preguntas en un solo experimento.

Por que BiGRU y BiLSTM, y no solo uno:

- El proyecto tiene numeros de las dos familias, y P80 y P81 confirmaron que NINGUNO es
  interpretable bajo el protocolo actual, por H5, H6 y H7. P82 es la primera oportunidad de
  medir las dos en el mismo protocolo limpio.
- El coste de B3 y B4 es informational, no decia: si la atencion tuviera el mismo signo en
  los dos nucleos, seria la senal mas fuerte posible; si tuviera signos opuestos, seria la
  prueba mas fuerte posible de que era ruido. Ese es el resultado mas informativo que puede
  salir de P82 y sale casi gratis.

Por que B5 es indispensable:

- Es el control que nlp-specialist pidio en la Fase 5 y que NUNCA se ha hecho: permutar la
  asignacion token a vector CONSERVANDO EL MULTICONJUNTO. P78 ya uso una matriz aleatoria
  pareada, que responde a una pregunta mas debil, si una matriz con la misma forma y las
  mismas normas aporta senal. B5 responde a la pregunta fuerte: si ESA geometria, la que las
  77657 filas del artefacto contienen, esta asignada a ESTOS tokens.
- Sin B5, un resultado positivo de B2 es compatible con que lo que funciona es una
  perturbacion pequena y aleatoria de la entrada, no la semantica. P78 ya mostro que el
  efecto del embedding es de +0.0249 con desviacion de 0.0216, es decir fragile. B5 es la
  forma de no repetir esa fragilidad sin notarla.

### 5.2 La capa de atencion, especificada al detalle

Para que @ml-coder no tenga que decidir nada:

- Mecanismo: atencion aditiva tipo Bahdanau sobre la secuencia de salidas de la RNN
  bidireccional.
- Consulta: el ULTIMO estado oculto valido, proyectado. No un vector entrenable
  adicional, para no anadir un parametro que no aporta senal con 1483 filas.
- Puntaje por paso t: v transpuesto por tanh de W_h multiplicado por h_t mas W_s multiplicado
  por la consulta. v, W_h y W_s son los tres unicos parametros anadidos, mas los sesgos.
- Salida de puntuacion: softmax sobre las posiciones NO de relleno SOLAMENTE. La longitud
  real de cada fila se pasa explicitamente y las posiciones de relleno reciben menos
  infinito antes del softmax, de modo que su peso es exactamente cero y no un numero
  pequeno. Es A4.
- Contexto: suma ponderada de los h_t sobre las posiciones con peso distinto de cero.
- Clasificador: el contexto concatenado con el ultimo estado oculto valido, luego una capa
  lineal y sigmide. Se declara la concatenacion porque, sin ella, la atencion puede
  degradar a la informacion del ultimo paso y el efecto medido seria de la red y no de la
  atencion.
- Normalizacion: se usa dropout de 0.5 en la entrada de la capa recurrente y de 0.3 antes
  del clasificador, y capa de dropout entre las dos medias de la RNN. Son los valores que
  ya usan v9_04 y v10_11, y se declaran para que un cambio no se lea como efecto de la
  atencion.
- Optimizador: AdamW con weight decay 1e-4, learning rate 1e-3, con reduccion en plateau
  sobre la perdida de validacion INTERNA, factor 0.5, paciencia 3, y minimo 1e-5.
- Epocas maximas 30, paciencia de early stopping 5 sobre la perdida de validacion interna,
  con minimo de mejora 1e-4. El valor minimo de mejora se declara porque sin el una mejora
  de 1e-7 cuenta como mejora y la epoca nunca se estabiliza.
- Lote 64, que cabe con holgura en los 16304 MB de VRAM de la RX 9070 XT.
- Longitud maxima 64, ver 4.2.

### 5.3 Las comparaciones, con su jerarquia declarada

Todas son deltas PAREADOS POR SEMILLA sobre el OOF, calculados con la misma semilla en los
dos brazos, de modo que comparten inicializacion, orden de lotes, splits internos y
E estrella salvo por lo que el brazo cambie.

| Id | Comparacion | Naturaleza | Papel |
|---|---|---|---|
| P1 | B2 menos B1 | PRIMARIA | efecto de la atencion sobre GRU |
| P2 | B4 menos B3 | SECUNDARIA | el mismo efecto sobre LSTM |
| P3 | B2 menos B5 | SECUNDARIA | la geometria del artefacto aporta, o el efecto es de perturbacion |
| P4 | B1 menos B5 | DIAGNOSTICA | cuanto de B1 es geometria y cuanto es estructura de RNN |
| P5 | B1 menos B3 | DIAGNOSTICA | GRU contra LSTM sin atencion, en la misma corrida |

P1 es la UNICA que se lee con la regla de 3.4. P2 se lee con la MISMA regla, y se publica
aunque contradiga a P1. P3, P4 y P5 se publican como DESCRIPTIVAS, sin lectura inferencial
y sin jerarquizacion, exactamente como hizo P78 con P2, P3 y P4.

Y una regla que se escribe porque P78 la cumplio y conviene mantener: si P1 y P2 tienen
signos opuestos, ese hecho se publica como el resultado principal de P82, porque es la
evidencia mas fuerte disponible de que el efecto de la atencion no es real. No se elige el
brazo que sale bien.

## 6. Semillas

Diez semillas, 20261401 a 20261410, INCLUIDAS en el diseno por adelantado y no ajustadas
despues. Es el minimo que fija la seccion 3.3.

Emparejamiento, exigido por A9: para la misma condicion de entrada, la misma semilla produce
el mismo init de pesos, el mismo generador de permutaciones de lote y por tanto el mismo
orden de mini-lotes, el mismo reparto de filas, los mismos splits internos, y el mismo E
estrella salvo por la perdida de validacion interna que es lo unico que el brazo cambia.

Por que diez y no mas, escrito antes de correr: mas de diez NO se autoriza en P82 por
presupuesto, y la consecuencia es que el limite de deteccion se queda en 0.021502, que es
la cifra de 3.2 y no una mejora. Si Carlos quisiera resolver 0.010, hacen falta 39 semillas
y se declara en 3.3 que eso no cabe. Diez es el maximo affordable, no el optimo, y el
informe tiene que decirlo con esas palabras: DIEZ_SEMILLAS_ES_EL_MAXIMO_QUE_CABE_NO_EL
OPTIMO_ESTADISTICO.

## 7. Las veintitres afirmaciones, indice

| Id | Bloque | Que hace | Naturaleza | STOP |
|---|---|---|---|---|
| A0 | 0 | sha256 del congelado, tamano y mtime antes y despues | CONFIRMAR | si |
| A1 | 0 | solo se leen las fuentes declaradas, contadores publicados | CONFIRMAR | si |
| A2 | 0 | los tres umbrales de V03 intactos y no reoptimizados | CONFIRMAR | si |
| A3 | 1 | el relleno se aplica ANTES de la RNN, verificado por estado | CONFIRMAR | si |
| A4 | 1 | relleno y OOV en celdas distintas, y el OOV ENTRA | CONFIRMAR | si |
| A5 | 1 | early stopping estrictamente anidado, verificado por filas | CONFIRMAR | si |
| A6 | 1 | clip_grad_norm_ en todos los bucles, norma pre-clip publicada | CONFIRMAR | si |
| A7 | 1 | determinismo, dos pasadas misma semilla, variacion cero | CONFIRMAR | si |
| A8 | 1 | cifras de cobertura medidas y publicadas, y MAX_LEN | CONFIRMAR | no |
| A9 | 2 | al menos 10 semillas, pareadas, conteo publicado | CONFIRMAR | si |
| A10 | 2 | tabla de limite de deteccion recalculada con la sd medida | CONFIRMAR | si |
| A11 | 2 | el intervalo que decide es un t pareado de semillas | CONFIRMAR | si |
| A12 | 2 | la lectura EXPLORATORIA esta atada a la regla de 3.4 | CONFIRMAR | si |
| A13 | 3 | existe brazo sin atencion de la MISMA corrida y es el control | CONFIRMAR | si |
| A14 | 3 | la permutacion conserva el multiconjunto exacto | CONFIRMAR | si |
| A15 | 3 | el control permutado usa el mismo protocolo y las 10 semillas | CONFIRMAR | si |
| A16 | 3 | las arquitecturas se reportan por separado, sin agregado | CONFIRMAR | no |
| A17 | 4 | interseccion y rescue damage net contra V03 sobre OOF | CONFIRMAR | no |
| A18 | 4 | la formulacion literal esta escrita y no sustituida | CONFIRMAR | si |
| A19 | 4 | presupuesto de reloj respetado y tiempo real publicado | CONFIRMAR | si |
| A20 | 4 | ib se evalua una vez por brazo, semilla y pliegue | CONFIRMAR | si |
| A21 | 4 | los 10 deltas por semilla del primario, con su signo | CONFIRMAR | no |
| A22 | 4 | cada comprobacion de A3 a A7 y A14 sabe FALLAR, por mutacion | CONFIRMAR | si |

Veintitres afirmaciones. Veinte y una de CONFIRMAR con STOP y dos de CONFIRMAR sin STOP.
Ninguna es REFUTAR, ninguna es NO APLICABLE POR CONSTRUCCION y ninguna es NO_EVALUABLE,
y las dos ultimas estan prohibidas como salida comoda: si algo no se puede evaluar, se dice
que no se pudo evaluar, se marca NO_EVALUABLE con su razon, y no se convierte en un
CONFIRMADO ni se omite.

## 8. Bloque 0: integridad y fronteras

**A0. Integridad del artefacto congelado.**

- Fichero: modelos/twitter_domain/w2v_twitter_100d.npz
- sha256 declarado: 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a
- Se comprueba ANTES de la primera lectura y DESPUES de la ultima. Se registran tambien
  size y mtime_ns. El fichero se abre solo en modo rb y en streaming para el hash.
- Tolerancia: coincidencia exacta de los tres, antes y despues, y entre si.
- Exito: los tres coinciden.
- Fallo: cualquiera difiere. STOP INMEDIATO con SystemExit y aviso a Carlos. No se registra
  ningun otro numero.
- STOP: si.

**A1. Solo las fuentes declaradas.**

- Que se verifica: un contador de filas leidas por fuente, con el pool esperado de 1483
  filas de desarrollo y CERO lecturas de test y CERO del holdout, y que coincide con el
  conteo de filas que devuelve read_train menos las que devuelve indices_holdout, sin
  abrir el holdout. Se publica el contador por fuente.
- Tolerancia: enteros exactos. Filas de test leidas igual a 0, filas de holdout leidas igual
  a 0.
- Exito: 1483 filas de desarrollo, 0 de test, 0 de holdout.
- Fallo: cualquier lectura de test o del holdout. STOP.
- STOP: si.

**A2. Los umbrales de V03, intactos.**

- Que se verifica: que los tres umbrales usados son 0.44, 0.51 y 0.46, leidos del
  artefacto congelado de V03 y no escritos a mano en el script; que el numero de veces que
  el script ajusta un umbral sobre cualquier conjunto es 0; y que V03 no se reentrena.
- Tolerancia: igualdad exacta de los tres, y contador de reoptimizacion igual a 0.
- Exito: los tres.
- Fallo: cualquiera distinto, o cualquier reoptimizacion. Un umbral reoptimizado sobre el
  OOF es exactamente la fuga que condemned P73 y P74, y en V03 seria peor, porque V03 es el
  campeon y su numero es la referencia de todas las comparaciones.
- STOP: si.

## 9. Bloque 1: el protocolo limpio

**A3. El relleno se aplica ANTES de la red recurrente.**

- Naturaleza: CONFIRMAR.
- Que se verifica, por ejecucion y no por inspeccion: el estado oculto en el primer paso de
  una secuencia con relleno inicial es IDENTICO al de la misma secuencia sin el relleno, con
  tolerancia 1e-6 en norma relativa. Ademas, que la funcion de empaquetado usada es
  pack_padded_sequence y no una mascara posterior, verificado por la firma del callable y
  por la presencia de longitudes reales en la llamada.
- Tolerancia: 1e-6 en norma relativa del vector de estado. Banda muerta 1e-9.
- Exito: la diferencia esta por debajo de 1e-6 en los dos nucleos, GRU y LSTM.
- Fallo: la diferencia es mayor. Entonces la RNN esta leyendo el relleno en la direccion
  inversa, que es H6, y los brazos de P82 estarian contaminados por el mismo defecto que se
  dice corregir. STOP.
- STOP: si.

**A4. Relleno y OOV en celdas distintas, y el OOV entra en el promedio.**

- Naturaleza: CONFIRMAR.
- Que se verifica: id_relleno igual a 0; id_oov igual a len(vocabulario) mas 1;
  id_relleno distinto de id_oov; id_oov mayor que 0; forma de la matriz
  (len(vocabulario) mas 2, 100); residuo de la fila id_relleno exactamente 0.0; la fila
  id_oov distinta de la fila id_relleno y con norma mayor que 0, y sus componentes
  publicados; los pesos de atencion en las posiciones de relleno exactamente 0.0, con
  tolerancia 0 y no 1e-9, porque se restan menos infinito antes del softmax; y el peso
  total de las posiciones OOV en el contexto mayor que 0 en al menos el 99 por ciento de las
  filas que contienen algun OOV.
- Tolerancia: enteros exactos para los ids; 0.0 exacto para el residuo de la fila de
  relleno y para los pesos de relleno; 0 para el peso total de OOV.
- Exito: los siete.
- Fallo: cualquiera. Si el peso total de OOV es 0, el OOV se sigue tirando y el bloque no ha
  corregido el defecto, solo lo ha decorado. STOP.
- STOP: si.

**A5. Early stopping estrictamente anidado.**

- Naturaleza: CONFIRMAR. Es la afirmacion mas importante del bloque, porque H5 la confirmo
  como la fuga que condemned P73 y P74.
- Que se verifica, por instrumentacion y no por confianza: se parchea la funcion que
  calcula la perdida de validacion y se registra el conjunto de filas que cada llamada
  recibe. Se exige que (i) la union de las filas vistas por las llamadas que DECIDEN epoca
  sea un subconjunto estricto de ia; (ii) ib no aparezca en ninguna llamada de decision de
  epoca, ni una vez; (iii) ib se evalue exactamente una vez por (brazo, semilla, pliegue),
  con contador explicito; (iv) el numero de llamadas de decision de epoca sea 3 por (brazo,
  semilla, pliegue), una por cada split interno, y no mas; y (v) E estrella sea la mediana
  de los tres valores de epoca, y no el maximo ni el ultimo.
- Tolerancia: enteros exactos para los contadores. Identidad de conjuntos para la
  pertenencia de filas.
- Exito: los cinco.
- Fallo: cualquiera. Si ib aparece en una llamada de decision de epoca, el numero de
  cualquier brazo de P82 esta optimisticamente sesgado y hay que abortar antes de
  publicarlo. STOP.
- STOP: si.

**A6. clip_grad_norm_ en todos los bucles.**

- Naturaleza: CONFIRMAR.
- Que se verifica: que el numero de llamadas a torch.nn.utils.clip_grad_norm_ es igual al
  numero de pasos de optimizacion de TODOS los bucles, con umbral 1.0 y sobre el conjunto
  completo de parametros con gradiente; y que la serie de la norma del gradiente ANTES del
  clip esta medida y publicada por epoca, por brazo, por pliegue y por semilla, con al menos
  el maximo y el percentil 95.
- Tolerancia: igualdad exacta entre el numero de llamadas y el numero de pasos. Umbral
  1.0 exacto.
- Exito: las dos, y la serie publicada con al menos cinco elementos no nulos por brazo.
- Fallo: una llamada de menos que un paso, es decir un bucle sin clip, o la serie vacia. La
  segunda condicion es la que mas importa: un clip ejecutado y una serie vacia es un clip
  que no se puede auditar. STOP.
- STOP: si.

**A7. Determinismo.**

- Naturaleza: CONFIRMAR.
- Que se verifica: dos pasadas con la MISMA semilla producen la misma perdida de validacion
  interna y la misma epoca elegida, con variacion exactamente cero. Y lo mismo para la
  prediccion sobre ib, con variacion exactamente cero en la matriz de probabilidades.
- Tolerancia: variacion exactamente CERO. Cero, no 1e-12. Cualquier variacion no nula entre
  dos pasadas con la misma semilla significa que el generador no esta fijado, y entonces
  las cifras de A5, A8 y A10 no son reproducibles. STOP.
- Exito: variacion cero en los tres.
- Fallo: cualquier variacion no nula. STOP.
- STOP: si.

**A8. Cifras de cobertura, medidas y publicadas.**

- Naturaleza: CONFIRMAR.
- Que se mide, con el tokenizador REAL de P82 y no con el de v9_04: proporcion de posiciones
  OOV; proporcion de posiciones de relleno; proporcion de OOV sobre los tokens que llegan a
  la matriz; numero de tipos distintos fuera de vocabulario; longitud mediana y media de
  tokens por fila; MAX_LEN; fraccion de filas truncadas por MAX_LEN; y numero de filas con
  cero tokens cubiertos.
- Por que hay que medirlo y no arrastrarlo: el 15.3295 por ciento de OOV sobre los tokens
  que llegan a la matriz y el 73.0919 por ciento de relleno vienen de P81, medidos con
  MAX_LEN igual a 64 y el tokenizador de v9_04. P82 usa celdas separadas y un tokenizador
  propio, luego las cifras pueden cambiar y hay que recalcularlas. Se publica ademas una
  linea explicita que dice si cada una de las dos cifras queda REPRODUCIDA o NO REPRODUCIDA,
  y con que codigo. La longitud mediana de 16 palabras viene de la discusion de diseno y se
  trata como un valor a verificar, no como un valor asumido.
- Exito: las ocho cifras medidas y publicadas, y la linea de reproduccion de las dos cifras
  de referencia.
- Fallo: no se puede medir, o la reproduccion se queda sin declarar.
- STOP: no. Es medicion de contexto, no prueba.

## 10. Bloque 2: el limite de deteccion, medido y no supuesto

**A9. Al menos diez semillas, pareadas, y el conteo publicado.**

- Naturaleza: CONFIRMAR.
- Que se verifica: el numero de semillas usadas es mayor o igual que 10; el conjunto es
  exactamente 20261401 a 20261410; las mismas diez semillas se usaron en los CINCO brazos de
  cada pliegue; y para cada par comparado, el hash del init de pesos y del orden de lotes es
  identico entre los dos brazos.
- Tolerancia: enteros exactos. El numero de semillas se publica, y si fuera menor que 10 la
  afirmacion FALLA.
- Exito: los tres.
- Fallo: menos de 10 semillas, o un par cuyos hashes no coinciden. Con menos de 10, por lo
  demostrado en 3.3, NO se hace ninguna lectura inferencial y el informe no publica ningun
  intervalo como si decidiera. STOP.
- STOP: si.

**A10. El limite de deteccion, recalculado con la desviacion medida.**

- Naturaleza: CONFIRMAR.
- Que se verifica: el informe publica la desviacion tipica ENTRE SEMILLAS del delta pareado
  P1, medida en la propia corrida; con esa desviacion, el numero de semillas, t de Student
  con n menos 1 grados de libertad y potencia objetivo 0.80, recalcula el MDE; y publica
  las CUATRO filas de la tabla de sensibilidad de 3.3, con el factor 1.0, 1.5, 2.0 y 3.0
  sobre la desviacion medida. Se publica ademas el cociente MRE dividido por el gap, con el
  gap igual a 0.117450 leido del informe de P78 y NO escrito a mano.
- Tolerancia: las formulas se aplican tal cual, sin ajustar ningun parametro. La tabla se
  publica completa, con las cuatro filas.
- Exito: desviacion medida, MDE recalculado, tabla de cuatro filas publicada, cociente
  publicado.
- Fallo: falta cualquiera de las cuatro, o el MDE se publica con la desviacion planificada en
  lugar de la medida. Eso seria publicar un numero que no se midio, que es el fallo que
  condemned P79 sobre el scheduler y P80 sobre la matriz de salida. STOP.
- STOP: si.

**A11. El intervalo que decide es un t pareado de semillas.**

- Naturaleza: CONFIRMAR.
- Que se verifica: el intervalo que decide en la regla de 3.4 se calcula sobre los n valores
  del delta POR SEMILLA, con t de Student y n menos 1 grados de libertad; y NO es un bootstrap
  de muestras.
- Por que es STOP, y es la leccion mas cara de P78, enunciada: el bootstrap pareado por
  clusters de P78 remuestrea tweets y promedia sobre las redes ya entrenadas, luego
  propaga CERO variabilidad de reentrenamiento. Su intervalo era (-0.028736, +0.078522)
  segun el bootstrap, y el t sobre las tres semillas daba (-0.028736, +0.078522) con el cero
  dentro. El bootstrap excluia el cero y el t no. Elegir el intervalo equivocado ahi era
  exactamente el modo de fallo que produjo el titular de P78. En P82, si se usara bootstrap,
  se repetiria el fallo aunque el bootstrap saliera a favor.
- Tolerancia: el informe declara explicitamente el metodo, y se verifica por recalculo
  independiente a partir de la serie de deltas por semilla, que se publica.
- Exito: el metodo es el t pareado de semillas, y el recalculo independiente coincide con el
  publicado dentro de 1e-9.
- Fallo: se usa bootstrap, o el recalculo no coincide. STOP.
- STOP: si.

**A12. La lectura EXPLORATORIA esta atada a la regla aritmetica.**

- Naturaleza: CONFIRMAR.
- Que se verifica: el informe contiene la cadena literal
  UN_RESULTADO_NEGATIVO_AQUI_NO_AFIRMA_QUE_LA_ATENCION_NO_APORTA_AFIRMA_QUE_EL_EFECTO_ES
  INFERIOR_AL_LIMITE_DE_DETECCION_DE_ESTE_DISENO ; y la cadena literal
  DIEZ_SEMILLAS_ES_EL_MAXIMO_QUE_CABE_NO_EL_OPTIMO_ESTADISTICO ; y un campo
  naturaleza_del_resultado con el valor EXPLORATORIO ; y que el valor EXPLORATORIO aparece
  SIEMPRE, tanto si el intervalo excluye el cero como si lo contiene, porque el limite de
  deteccion de 3.3 no desaparece en ninguno de los dos casos.
- Tolerancia: coincidencia exacta de cadenas, cero margen.
- Exito: las tres.
- Fallo: cualquiera. En particular, si el intervalo excluye el cero y el informe publica un
  AFIRMATIVO sin la palabra EXPLORATORIO, el titular estara por encima de la medida, que es
  el fallo que A19 de P81 trampa y que aqui se vuelve a trampar. STOP.
- STOP: si.

## 11. Bloque 3: brazos y controles

**A13. Existe un brazo SIN atencion de la MISMA corrida, y es el control.**

- Naturaleza: CONFIRMAR.
- Que se verifica: en el JSON existe la clave comparacion_primaria, y su campo
  con_tra_manode tiene el valor B1; el campo corrida tiene el valor P82; y el campo
  experimento_contraste tiene el valor NINGUNO, es decir, que el brazo de contraste NO es un
  numero de v9_04 ni de v10_11 ni de DL-03 ni de P75.
- Por que es STOP: comparar la atencion contra un numero de otro experimento mediria el
  cambio de protocolo entero, que es enorme, en lugar del efecto de la atencion. P80 y P81
  confirmaron H5, H6 y H7, luego los BiGRU anteriores no son la referencia de nada. La
  atencion tiene que medirse CONTRA SI MISMA.
- Tolerancia: igualdad exacta de cadenas.
- Exito: los tres.
- Fallo: cualquiera. STOP.
- STOP: si.

**A14. La permutacion conserva el multiconjunto exacto.**

- Naturaleza: CONFIRMAR.
- Que se verifica: la matriz permutada de B5 tiene la misma forma que la original; el
  MULTICONJUNTO de sus filas es IGUAL al multiconjunto de las filas originales, verificado
  por dos vias independientes: la suma de los hashes sha256 de cada fila, sumados con
  aritmetica modular, y la suma de las trazas de las filas; y la permutacion aplicada dos
  veces con la misma semilla da la misma matriz, con variacion exactamente cero.
- Por que dos vias: una sola suma de trazas podria coincidir por azar con muchisima menos
  frecuencia de la que se cree, y la reproducibilidad de la fila 0 de A4 ya enseno que una
  comprobacion que parece solida tiene que tener una segunda.
- Tolerancia: las dos sumas coinciden EXACTAMENTE con las de la matriz original, y la doble
  permutacion da variacion cero.
- Exito: las tres.
- Fallo: cualquiera. Si el multiconjunto no se conserva, B5 no es el control de P78 con
  forma exacta, es otro experimento, y no se puede atribuir su resultado a la geometria.
  STOP.
- STOP: si.

**A15. El control permutado usa el mismo protocolo y las mismas diez semillas.**

- Naturaleza: CONFIRMAR.
- Que se verifica: B5 se entrena con el MISMO protocolo de 4.4, con los MISMOS splits
  internos, con las MISMAS diez semillas, con el MISMO umbral de clip, y con el MISMO numero
  de evaluaciones de ib; y su resultado se publica COMPLETO, con su media, su desviacion,
  sus diez deltas por semilla y su intervalo, sea cual sea. Se exige explicitamente que el
  informe contenga el resultado de B5 aunque B5 gane, porque un control que solo se publica
  cuando sale favorable es un control que no controla.
- Tolerancia: coincidencia exacta de los identificadores de semilla y de split, y presencia
  de las seis cifras de B5.
- Exito: las dos.
- Fallo: cualquiera. STOP.
- STOP: si.

**A16. Las arquitecturas se reportan por separado.**

- Naturaleza: CONFIRMAR.
- Que se verifica: el informe publica una fila por brazo, con su media, su desviacion entre
  semillas y su intervalo, y NO publica ninguna cifra agregada de los cinco brazos. Se
  publica tambien el conteo de parametros entrenables por brazo, y el numero de parametros
  anadidos por la capa de atencion, que se espera igual a 8256 mas los sesgos.
- Tolerancia: cinco filas, cero filas agregadas. Conteo de parametros publicado.
- Exito: las dos.
- Fallo: aparece una cifra agregada, o falta el conteo de parametros.
- STOP: no. Es una condicion de redaccion del informe, no una prueba.

## 12. Bloque 4: lectura, redaccion y presupuesto

**A17. Interseccion y rescue damage net contra V03 sobre el OOF pareado.**

- Naturaleza: CONFIRMAR.
- Que se mide: sobre las 4449 unidades tweet-etiqueta del OOF, para cada brazo y frente a
  V03 con sus umbrales congelados, la matriz de interseccion, el rescate, el dano, el neto,
  las tasas de rescate y de dano, y el AUC de la probabilidad del propio brazo como
  predictor del error de V03, por etiqueta. Todo DESCRIPTIVO.
- Se declara, y se escribe en el informe: estas cifras NO son comparables con
  corrected_cv ni con ningun macro-F1. Rescue damage net es un recuento de celdas, no una
  mejora de metrica. Es exactamente la advertencia que P75 publico y que aqui se repite
  porque P82 la necesita igual.
- Tolerancia: enteros exactos para los recuentos, seis decimales para los AUC.
- Exito: los tres por cada uno de los cinco brazos, publicados aunque sean malos.
- Fallo: falta alguno.
- STOP: no.

**A18. La formulacion literal esta escrita y no sustituida.**

- Naturaleza: CONFIRMAR.
- Que se verifica: el informe contiene la cadena literal exacta
  V03 remains the best verified leaderboard result.
  y NO contiene ninguna variante de ella; y el informe contiene una cadena explicita de que
  Word2Vec no se afirma mejor que V03.
- Tolerancia: coincidencia exacta. Cero margen.
- Exito: las dos.
- Fallo: cualquiera. STOP.
- STOP: si.

**A19. El presupuesto de reloj se respeta y se publica el tiempo real.**

- Naturaleza: CONFIRMAR.
- Presupuesto declarado: 90 minutos de reloj para el proceso completo, con un hito de
  decision obligatorio a los 40 minutos, descrito en 13.
- Que se verifica: el tiempo real de reloj del proceso se publica, y se publica que tramo se
  completo y que tramo, si lo hay, se dejo sin ejecutar por presupuesto.
- Tolerancia: el tiempo real se publica con 6 decimales y con la etapa a la que se atribuyo.
- Exito: el tiempo real esta publicado, y si se supero el presupuesto esta escrito que tramo
  quedo sin ejecutar y por que.
- Fallo: no se publica el tiempo real, o se supero el presupuesto y no se declara que parte
  quedo sin ejecutar. Publicar el resultado de un tramo sin decir que el otro no se ejecuto
  es el modo de fallo de los informes parciales que P81 sufrio tres veces. STOP.
- STOP: si.

**A20. ib se evalua una vez por brazo, semilla y pliegue.**

- Naturaleza: CONFIRMAR.
- Que se verifica: un contador de evaluaciones de outer_validation, con valor exacto igual a
  5 pliegues por 5 brazos por 10 semillas, es decir 250. Y que el script aborta con
  SystemExit si en algun momento el contador supera lo que corresponde al trabajo hecho.
- Tolerancia: entero exacto 250, y cero abortos.
- Exito: 250, y el numero de abortos igual a 0.
- Fallo: cualquier valor distinto, o cualquier aborto.
- STOP: si.

**A21. Los diez deltas por semilla del primario, con su signo.**

- Naturaleza: CONFIRMAR.
- Que se verifica: se publica la serie de los diez valores del delta pareado de P1, uno por
  semilla, con su signo, y la media, y la desviacion, y el intervalo. Se publica tambien el
  conteo de signos positivos, porque en P78 ese recuento fue el que revelo que la tercera
  semilla estaba a +0.0009 y no a +0.0428.
- Tolerancia: seis decimales, los diez valores presentes.
- Exito: los diez valores, mas el recuento de signos.
- Fallo: falta alguno, o el informe publica solo la media.
- STOP: no. Es una condicion de redaccion, y es la que evita que P78 vuelva a pasar.

**A22. Cada comprobacion sabe FALLAR.**

- Naturaleza: CONFIRMAR.
- Que se verifica, por MUTACION DELIBERADA y registrada: se ejecutan al menos seis corridas
  de humo, cada una con un fallo inyectado en un punto distinto, y la afirmacion
  correspondiente TIENE que detectarlo. Los seis fallos inyectados son: (i) decision de
  epoca sobre ib en lugar de sobre el split interno, que tiene que hacer fallar A5; (ii)
  aplicacion de la mascara DESPUES de la RNN, que tiene que hacer fallar A3; (iii) omision de
  clip_grad_norm_ en uno de los bucles, que tiene que hacer fallar A6; (iv) permutacion que
  no conserva el multiconjunto, por ejemplo una permutacion de filas MAS una sustitucion,
  que tiene que hacer fallar A14; (v) ocho semillas en lugar de diez, que tiene que hacer
  fallar A9; (vi) intervalo por bootstrap de muestras en lugar de t pareado, que tiene que
  hacer fallar A11.
- Por que es STOP y no una cortesia: la bitacora de P81 dice que CINCO fallos de la serie
  fueron de contrato entre capas y que NINGUNO estaba cubierto por los checks de humo, que
  llegaban a 126. Un caso de humo que no puede fallar no prueba nada. Esta afirmacion es la
  que convierte los 126 casos de humo en evidencia.
- Tolerancia: los seis fallos inyectados se detectan. Cero tolerancias.
- Exito: seis de seis detectados.
- Fallo: alguno no se detecta. Si una comprobacion no puede fallar, esa comprobacion no
  existe, y las afirmaciones que dependen de ella no son evidencia. STOP.
- STOP: si.

## 13. Presupuesto y tiempo, ANTES de que Carlos autorice

### 13.1 El numero de ajustes

Cinco brazos, cinco pliegues externos, diez semillas. Cada combinacion exige cuatro
ajustes: tres busquedas internas de E estrella y un reajuste sobre outer_train completo.

    5 brazos por 5 pliegues por 10 semillas por 4 ajustes = 1000 ajustes

Mas los ajustes de las seis corridas de mutacion de A22, que son de un solo par y un solo
pliegue, y mas la verificacion de protocolo, que no entrena. Estimacion de los ajustes de
verificacion y de mutacion: 40 ajustes mas.

### 13.2 El coste de un ajuste, y de donde sale la estimacion

ESTIMACION DE PLANIFICACION. NO ES UNA MEDIDA. No se ha ejecutado nada de P82, y el numero
que sigue se deriva de conteos, no de un cronometro.

Origen de los conteos:

- 1483 filas de desarrollo, 5 pliegues, luego outer_train de unas 1186 filas.
- El split interno de 3 deja unas 790 filas de entrenamiento interno.
- MAX_LEN igual a 64, lote 64, luego unas 13 lotes por epoca.
- Epocas maximas 30, con paciencia 5 y minimo de mejora 1e-4, luego un promedio esperado
  de entre 10 y 14 epocas antes de estabilizar, es decir del orden de 160 pasos por ajuste.
- Un paso de una BiGRU de oculto 64 con secuencia 64 y lote 64 es una operacion pequena.
  El limite de la estimacion no es la matematica, son los 64 lanzamientos de kernel por paso
  que hace una celda recurrente en PyTorch. Estimacion: de 4 a 8 milisegundos por paso, mas
  de 1 a 2 segundos de coste fijo por ajuste, que es preparar el lote, construir el
  optimizador y evaluar el split interno.
- Con eso, un ajuste esta entre 2 y 4 segundos.

### 13.3 El resultado de la estimacion, y la respuesta a la pregunta de la hora

    1000 ajustes por 3 segundos en el centro = 3000 segundos = 50 minutos
    1000 ajustes por 2 segundos en el extremo bajo = 33 minutos
    1000 ajustes por 4 segundos en el extremo alto = 67 minutos
    mas 40 ajustes de verificacion, mas inicio de proceso y escritura de informes

ESTIMACION TOTAL: de 40 a 80 minutos de reloj, con 50 minutos como estimacion central.

Y aqui esta la respuesta, explicita, para que Carlos la lea antes de autorizar:

    SI, SUPERA UNA HORA. La estimacion central de 50 minutos y el extremo alto de 80
    minutos convierten la hora en el valor central, no en el peor caso. Se declara por
    adelantado y no se maquilla.

### 13.4 El fraccionamiento en dos tramos, con hito de decision

Para que Carlos pueda autorizar por tramos y no todo o nada:

- TRAMO 1, autorizado al aprobar este pre-registro: A0 a A12, mas B1, B2 y B5 en los cinco
  pliegues y las diez semillas, mas las seis mutaciones de A22. Son tres brazos, luego
  3 por 5 por 10 por 4 = 600 ajustes mas 40, y la estimacion es de 25 a 45 minutos. Es el
  par primario y el control de geometria, es decir, lo que responde a la pregunta de 1.
- TRAMO 2, que requiere un segundo visto bueno en el hito de los 40 minutos: B3 y B4, que
  son 2 por 5 por 10 por 4 = 400 ajustes, con estimacion de 20 a 35 minutos. Es
  informativo, y el proyecto puede entregar el tramo 1 completo sin el tramo 2 y sin que
  falte nada para responder la pregunta de 1.

- HITO DE DECISION, a los 40 minutos de reloj: el script imprime el tiempo real y el numero
  de ajustes completados. Si el tiempo real proyecta superar los 90 minutos, el tramo 2 se
  salta y se publica el tramo 1 con la declaracion TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO.
- A19 obliga a que el tiempo real se publique en cualquier caso, exista el tramo 2 o no.

### 13.5 Lo que NO se hace para ahorrar tiempo, y por que no se recorta en otra parte

- No se recortan las diez semillas. Es el unico parametro que sostiene 3.3, y recortarlo
  devuelve el experimento al error que P78 cometio con tres.
- No se recorta el clip. Es A6 y es STOP.
- No se recorta el empaquetado. Es A3 y es STOP.
- No se recorta el control permutado. Es B5, y sin el el resultado de B2 no es
  interpretable.
- No se recorta el anidamiento. Es A5 y es STOP.

## 14. Criterio de parada

Se para, con SystemExit, sin registrar ningun otro numero, si:

- A0 falla: el sha256, el size o el mtime_ns del congelado difieren. STOP INMEDIATO y
  aviso a Carlos.
- A1 falla: se ha leido test o el holdout.
- A2 falla: un umbral de V03 ha sido reoptimizado.
- A3 falla: la RNN lee el relleno, es decir, H6 sigue vivo.
- A4 falla: los pesos de relleno no son exactamente cero, o el peso total de OOV es cero, y
  el bloque no ha separado nada.
- A5 falla: una fila de outer_validation participa en una decision de epoca, o ib se
  evalua mas de una vez por combinacion. Es la fuga que condemned P73 y P74.
- A6 falla: hay un bucle sin clip, o la serie de normas pre-clip esta vacia.
- A7 falla: dos pasadas con la misma semilla difieren.
- A9 falla: hay menos de diez semillas, o un par no esta pareado.
- A10 falla: el limite de deteccion se publica con la desviacion planificada y no con la
  medida.
- A11 falla: el intervalo que decide no es un t pareado de semillas.
- A12 falla: la lectura exploratoria no esta atada a la regla, o el resultado se publica como
  confirmatorio.
- A13 falla: el primario no se mide contra un brazo de la misma corrida.
- A14 falla: la permutacion no conserva el multiconjunto.
- A15 falla: el control permutado no comparte protocolo o no se publica completo.
- A18 falla: la formulacion literal falta o esta sustituida.
- A19 falla: no se publica el tiempo real, o se supero el presupuesto sin declarar que tramo
  quedo sin ejecutar.
- A20 falla: ib se evalua un numero distinto de 250 veces.
- A22 falla: alguna de las seis comprobaciones no puede fallar.
- Aparece un numero sospechosamente perfecto, por ejemplo una f1_media_columnas mayor que
  0.99. Se investiga leakage antes de reportar nada.
- Se viola cualquier frontera de 2.2.

Criterio de parada de presupuesto: 90 minutos de reloj, con el hito de decision de 13.4 a
los 40. Si se supera, se para y se publica lo que haya con su estado real, nombrando el
tramo que quedo sin ejecutar.

## 15. Resumen de naturalezas

| Naturaleza | Cuantas | Ids |
|---|---|---|
| CONFIRMAR | 23 | A0 a A22 |
| REFUTAR | 0 | ninguna |
| NO APLICABLE POR CONSTRUCCION | 0 | ninguna |
| NO_EVALUABLE | 0 | ninguna |

Ninguna afirmacion es NO APLICABLE POR CONSTRUCCION y ninguna es NO_EVALUABLE, y ambas
cosas estan prohibidas como salida comoda. Si una afirmacion no llega a ejecutarse por
fallo PROPIO del script, se marca ERROR con su tipo y su traza resumida, la corrida
CONTINUA con la siguiente, y ERROR no es un CONFIRMADO ni un REFUTADO ni un NO_EVALUABLE
por el dato. Lo unico que aborta de verdad es un STOP.

## 16. Lo que P82 no hace

- No genera submission, no consulta el leaderboard, no lee test.
- No lee el holdout ni su texto ni sus etiquetas ni su fichero de identificadores.
- No reentrena, no modifica, no sobrescribe el artefacto Word2Vec 100d, y no entrena con el
  ningun vector que no venga de ese artefacto, salvo el control permutado de A14.
- No reentrena V03, no toca sus umbrales congelados 0.44, 0.51 y 0.46.
- No modifica ni borra ningun script de P71 a P81, ni sus informes, ni sus JSON.
- No ancla el objetivo del trainer contra una implementacion externa de SGNS. Ese limite
  sigue abierto y es permanente.
- No afirma que Word2Vec sea mejor que V03, ni que una RNN lo supere, ni que la atencion
  aporte o no aporte. Afirma, como maximo, lo que 3.4 permite.
- No elige umbral, tolerancia, margen ni criterio despues de ver un numero.

## 17. Formulacion

Toda conclusion que P82 escriba lleva esta frase, y solo esta frase sobre el estado del
proyecto:

    V03 remains the best verified leaderboard result.

No se afirma jamas que Word2Vec sea mejor que V03. No se afirma jamas que la atencion
aporte. Si el intervalo excluye el cero, se afirma que el efecto es DETECTABLE POR ENCIMA
DEL RUIDO DE REENTRENAMIENTO en esta arquitectura y con este protocolo, y que sigue sin
alcanzar el gap de 0.117450. Si el intervalo contiene el cero, se afirma que el efecto es
INFERIOR AL LIMITE DE DETECCION DE ESTE DISENO, que es 0.021502 mas lo que dira la
desviacion realmente medida, y no se afirma nada mas.

    V03 remains the best verified leaderboard result.
