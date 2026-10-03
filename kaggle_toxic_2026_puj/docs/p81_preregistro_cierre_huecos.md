# P81: pre-registro de cierre de los cinco huecos que dejo P80

Fecha de escritura: 2026-09-30
Proyecto: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj
Rama: proyecto-kaggle-toxic-2026-p67
Script previsto: experiments/p81_01_cierre_huecos.py
Salidas previstas: reports/p81_cierre_huecos.json y reports/p81_cierre_huecos.md
Estado: PRE-REGISTRADO. Este texto se hashea y se commitea ANTES de escribir una linea
de codigo. Sin ese commit no hay experimento, igual que en P80.

## 0. Proposito

P80 salio con los siete hallazgos H1 a H7 CONFIRMADO y cero REFUTADO. Los siete son
solidos en su parte estructural. Pero cuatro dictamenes de especialistas dejaron cinco
huecos reales, y esos huecos no son cosmeticos: son huecos de EVIDENCIA.

Este pre-registro cierra exactamente esos cinco huecos y nada mas. No anade hallazgos
nuevos, no reabre H3, H4, H5 ni H7, no vuelve a medir lo que P80 ya midio bien, y no
reformula ninguna conclusion de P79.

Los cinco huecos, enunciados como preguntas que P81 responde con una asercion y no con
un adjetivo:

| # | Hueco | Quien lo senalo | Pregunta que P81 responde |
|---|---|---|---|
| 1 | H1 se concluyo contra una reimplementacion propia | ml-reviewer | el objetivo del trainer es o no el SGNS de Mikolov et al. 2013, medido contra un ancla independiente de nuestra propia formula |
| 2 | relleno y OOV comparten la celda 0 | nlp-specialist | cuanto pesa cada uno de los dos canales, por separado |
| 3 | el campo coincide en true no es replicacion | ml-reviewer | que clase de evidencia es en realidad la comparacion de P80 con P79 |
| 4 | A17 y A18 demuestran por tamano, no por identidad | ml-reviewer | si las 248 filas del pool son LITERALMENTE las del holdout, y si los tres informes de la fase DL comparten las mismas filas |
| 5 | el impacto del desfase del scheduler no esta cualificado | ml-reviewer | cuanto afecta de verdad el desfase de una epoca, con el numero |

Ademas hay un limite que P81 NO puede cerrar y que hay que escribir ahora, antes de
correr nada, para que nadie lo lea despues como si estuviera cerrado. Va en 3.4.

## 1. Alcance

### 1.1 Que se lee

Solo lectura, y solo estos origenes:

- experiments/p71_04_entrenar_word2vec.py: la clase SkipGram (:182-216), el RE_TOKEN y el
  lote. Es el codigo cuyo termino negativo se promedia.
- experiments/p71_03_preparar_corpus.py: la seccion de tokenizacion (:50-105) y
  escribir_corpus_general (:131-149). NOTA DE PROCEDENCIA: la instruccion de preparacion
  hablaba de una funcion construir_vocabulario en esas lineas. No existe ninguna funcion
  con ese nombre en ese fichero, ni en ningun otro fichero de experiments/. Las lineas
  131 a 149 son el principio de escribir_corpus_general, que escribe el corpus general y
  NO construye ningun vocabulario. El vocabulario que de verdad usa el clasificador BiGRU
  se construye en experiments/v9_04_dl_secuencial_fasttext.py:170-175. P81 cita el sitio
  real y deja constancia de la correccion.
- El patron de token del proyecto se registra como la cadena ASCII
  [#@]?\w+(?:['\u2019]\w+)* , leida del fichero y reemitida con escapes, nunca con el
  caracter U+2019 escrito en el informe. Procede de reports/p79_auditoria_sgns.json, clave
  tokenizador_del_corpus.
- experiments/p72_01_*.py: los bucles de plan y de entrenamiento del scheduler, leidos
  por AST igual que en P80, sin heuristica de buscar hacia atras un encabezado.
- experiments/v9_04_dl_secuencial_fasttext.py: MAX_LEN (:47), secuencias (:78-85), BiGRU
  (:88-107), y la construccion de vocabulario e indice (:170-175).
- experiments/v10_11_bigru_finetuning.py: MAX_LEN (:99), BiGRUAfinable (:117-171) y la
  llamada v9_04.secuencias (:323), que hace que el defecto de la celda 0 sea el MISMO
  codigo en los dos scripts.
- experiments/11_dl_desde_cero.py: read_train (:446), el StratifiedGroupKFold y su
  random_state (:484-491), y el bucle de CV (:1362).
- src/splits.py: holdout_v2 (:130-150) y pliegues_corregidos (:153 en adelante).
- src/data.py: read_train, que se llama, no se edita.
- reports/p79_auditoria_sgns.json, claves tokenizador_del_corpus, punto_1_perdidas y
  punto_2_gradientes, SOLO para contrastar la convencion de referencia.
- reports/p80_auditoria_w2v.json y reports/p80_auditoria_w2v.md, SOLO lectura.
- reports/dl_desde_cero_resultados.json: n_validacion de DL-01, DL-02 y DL-03.
- reports/p78_cobertura.json: SOLO las tasas de cobertura ya publicadas.
- config/holdout_v2.json, y SOLO los campos n_tweets_dev, n_tweets_holdout y hash_ids_csv,
  leidos con expresion regular sobre el texto del fichero, SIN json.loads, igual que hizo
  P80, para que ninguna lista del documento llegue a existir como objeto en memoria.
- Fuente externa, consultada UNA vez al escribir este pre-registro y ya transcrita aqui,
  de modo que la ejecucion NO necesita red: el codigo C original de word2vec en
  https://raw.githubusercontent.com/tmikolov/word2vec/master/word2vec.c , sha256
  df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4 , 715 lineas.
  Es el repositorio exportado automaticamente desde el proyecto original de Google Code.
  El fragmento pertinente va transcrito con su numero de linea en 3.1.

### 1.2 Que NO se lee

- test, cualquier leaderboard, cualquier submission. Ni una vez.
- El holdout. Ni su texto, ni sus etiquetas, ni su fichero de identificadores. La
  particion se reproduce con src.splits.holdout_v2, que es un determinista
  StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42) sobre train y devuelve
  indices POSICIONALES. Con eso basta para la identidad de conjuntos que pide el hueco 4,
  y no hace falta abrir el holdout ni un solo byte suyo.
- Ningun .npz de modelos/ salvo el congelado, y el congelado solo en lectura.
- Ningun .pt de modelos/.
- Ningun script de P71 a P80. No se editan, no se borran, no se corrigen. La correccion
  de los nombres de campo va en la salida de P81, nunca dentro de los informes viejos.

### 1.3 El artefacto congelado: solo lectura, sha256 antes y despues

Fichero: modelos/twitter_domain/w2v_twitter_100d.npz

- sha256 declarado: 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a
- Se comprueba ANTES de la primera lectura y DESPUES de la ultima, y se registran
  tambien size y mtime_ns.
- El fichero se abre solo en modo rb y en streaming para el hash. Nunca se escribe, nunca
  se renombra, nunca se copia.
- Exito: los tres valores coinciden antes y despues, y entre si.
- Fallo: cualquiera de los tres difiere. STOP INMEDIATO con SystemExit y aviso a Carlos.
  No se registra ningun otro numero.
- Naturaleza: CONFIRMAR. Es la asercion A0 y es la unica que puede abortar antes de
  cualquier medicion.

### 1.4 Reconciliacion con la frase del plan maestro

El plan maestro, en la decision de Carlos sobre P81, dice Sin GPU y sin entrenar nada.
P81 si entrena, pero SOLO dos modelos diminutos sobre un corpus SINTETICO PLANTADO de
65 palabras y 400 documentos, con dimension 32, en CPU, con un presupuesto declarado de
menos de 60 segundos por par de brazos. No toca el artefacto congelado, ni ningun corpus
del proyecto, ni ningun dato del proyecto, ni la GPU. La reconciliacion se declara aqui
para que no haya duda de que es intencionada y no un desliz: P81 no reentrena nada que
exista.

## 2. Las diecinueve afirmaciones, indice

| Id | Hueco | Que hace | Naturaleza | STOP |
|---|---|---|---|---|
| A0 | todos | integridad del congelado, sha256 antes y despues | CONFIRMAR | si |
| A1 | 1 | transcripcion literal de la ecuacion (4) | CONFIRMAR | no |
| A2 | 1 | el codigo C original no divide por K | CONFIRMAR | no |
| A3 | 1 | la convencion de P81 coincide con la de P79 | CONFIRMAR | si |
| A4 | 1 | el corpus plantado cumple sus invariantes | CONFIRMAR | si |
| A5 | 1 | el test DISCRIMINA | CONFIRMAR | si |
| A6 | 1 | gana el objetivo de la referencia | CONFIRMAR | si |
| A7 | 1 | robustez en 10 semillas y control negativo | CONFIRMAR | si |
| A8 | 2 | tasa de OOV y de relleno con el tokenizador real | CONFIRMAR | no |
| A9 | 2 | celdas separadas, ids distintos y mayores que 0 | CONFIRMAR | si |
| A10 | 2 | los tres canales medidos por separado | CONFIRMAR | no |
| A11 | 2 | el canal OOV es al menos tan grande como el de relleno | CONFIRMAR | no |
| A12 | 2 | reproducibilidad y valor por condicion y por clase | CONFIRMAR | si |
| A13 | 3 | el bloque se llama reejecucion y no replicacion | CONFIRMAR | si |
| A14 | 3 | criterio de replicacion genuinea, ejecutable | CONFIRMAR | no |
| A15 | 4 | conjuntos de identificadores de los cinco pliegues | CONFIRMAR | si |
| A16 | 4 | las 248 filas del residuo, por identificador | CONFIRMAR | si |
| A17 | 4 | los tres informes de la fase DL, por identificador | CONFIRMAR | no |
| A18 | 5 | learning rate paso a paso y clasificacion del impacto | CONFIRMAR | no |
| A19 | 5 | la declaracion de impacto esta atada a la medida | CONFIRMAR | si |

Veinte filas contando A0: diecinueve afirmaciones de contenido mas la de integridad.

## 3. Bloque 1: anclar H1 contra una referencia independiente

### 3.1 Ancla documental

**A1. La ecuacion (4) de Mikolov et al. 2013 se transcribe literalmente.**

- Naturaleza: CONFIRMAR.
- Referencia: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean,
  Distributed Representations of Words and Phrases and their Compositionality,
  arXiv:1310.4546v1 [cs.CL], 16 de octubre de 2013. DOI 10.48550/arXiv.1310.4546.
  Seccion 2.2, Negative Sampling, ecuacion numerada (4).
- Transcripcion, en notacion ASCII:

      log sigma( v'_{w_O}^T v_{w_I} )  +  sum_{i=1}^{k} E_{w_i ~ P_n(w)} [ log sigma( - v'_{w_i}^T v_{w_I} ) ]

- Que se verifica: que la transcripcion coincide con la fuente; que el indice de la suma
  corre de i igual a 1 hasta k, luego hay k terminos; que la expectativa es SOLO sobre
  la distribucion de ruido; y que NO aparece ningun factor 1/k en ningun sitio de la
  ecuacion. Esto ultimo es lo que importa: no es que la referencia divida, es que la
  referencia SUMA, y por eso el codigo de p71_04, que promedia, no es la misma funcion.
- Tolerancia: coincidencia de texto, cero margen. Si se pide verificacion en linea, se
  intenta contra la fuente ya verificada y se registra el campo verificacion_en_linea con
  el valor DISPONIBLE o NO_DISPONIBLE. Sin red, la transcripcion de este pre-registro, que
  esta hasheado y commiteado, es la fuente, y eso se declara.
- Exito: la transcripcion esta en el informe con seccion, numero de ecuacion, arXiv id y
  DOI, y las cuatro comprobaciones de forma dan verdadero.
- Fallo: el informe no la contiene, o una comprobacion de forma falla.
- STOP: no.

**A2. El codigo C original no divide por el numero de negativos.**

- Naturaleza: CONFIRMAR.
- Fichero: word2vec.c, sha256
  df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4 , 715 lineas.
  Fragmento, rama skip-gram, numeracion del fichero:

      520:         // NEGATIVE SAMPLING
      521:         if (negative > 0) for (d = 0; d < negative + 1; d++) {
      522:           if (d == 0) {
      523:             target = word;
      524:             label = 1;
      525:           } else {
      526:             next_random = next_random * (unsigned long long)25214903917 + 11;
      527:             target = table[(next_random >> 16) % table_size];
      528:             if (target == 0) target = next_random % (vocab_size - 1) + 1;
      529:             if (target == word) continue;
      530:             label = 0;
      531:           }
      532:           l2 = target * layer1_size;
      533:           f = 0;
      534:           for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
      535:           if (f > MAX_EXP) g = (label - 1) * alpha;
      536:           else if (f < -MAX_EXP) g = (label - 0) * alpha;
      537:           else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
      538:           for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
      539:           for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
      540:         }

- Que se verifica: (i) el sha256 del fichero recuperado coincide con el declarado en 1.1;
  (ii) el fragmento de las lineas 520 a 540 coincide caracter a caracter con el de este
  pre-registro; (iii) en ese rango la cadena / negative aparece CERO veces; (iv) el
  gradiente se acumula una vez por cada d del bucle, luego los negative negativos aportan
  negative veces, no una vez.
- Tolerancia: (iii) es un entero y debe ser exactamente 0. (i) y (ii) son coincidencia
  exacta, cero margen.
- Exito: las cuatro comprobaciones pasan.
- Fallo: cualquiera falla. En particular, si / negative apareciera una vez, la referencia
  de codigo seria otra y habria que rehacer el bloque 1 entero.
- STOP: no. Un fallo aqui se reporta como REFUTADO y detiene el bloque 1; no se intenta
  arreglar la referencia.

Con A1 y A2 la convencion deja de ser una transcripcion nuestra: pone un nombre publico,
una seccion, un numero de ecuacion y unas lineas de C verificables. Eso es un ancla, no
una opinion.

### 3.2 Lo que A1 y A2 NO hacen

A1 y A2 por si solos NO cierran el hueco de ml-reviewer. Ensenan que el objetivo
publicado y el codigo C no promedian. No engarian que nuestro codigo se parezca al
publicado. Para eso hace falta 3.3, que es lo unico que de verdad discrimina.

### 3.3 Ancla ejecutable y discriminante: corpus sintetico con estructura plantada

Diseno fijado aqui, completo, para que @ml-coder no tenga que decidir nada.

**Corpus plantado.**
- 6 familias de 10 palabras cada una, mas 5 palabras neutras que no pertenecen a ninguna
  familia. Total en vocabulario: 65.
- Familias con nombres ASCII sinteticos. Ningun token real del proyecto, ningun texto
  real. Semilla fija 20261201 para la generacion.
- Cada documento sortea UNA familia de forma uniforme y emite una permutacion aleatoria
  de las 10 palabras de esa familia, intercalando palabras neutras con probabilidad
  declarada p_neutra.
- 400 documentos. ventana igual a 3, subsampling desactivado, K igual a 10 negativos, que
  es el K del proyecto.
- Invariante plantada y verificada: dos palabras de familias DISTINTAS nunca aparecen en
  la misma ventana. Se comprueba contando los pares centro, contexto generados y exigiendo
  pares_entre_familias_distintas igual a 0, exacto.
- La respuesta correcta es conocida por construccion: para la palabra w, los 9 vecinos
  correctos son las otras 9 palabras de su familia.

**Los dos brazos.**
- Brazo MEDIA: el objetivo del codigo de p71_04:214-216, con logsigmoid(...).mean()
  sobre el tensor de negativos de forma (lote, K).
- Brazo SUMA: el objetivo de la ecuacion (4), con sum(dim=2) en el mismo punto.
- IDENTICOS EN TODO LO DEMAS, y verificado mecanicamente, no declarado: mismo sha256 de
  los pesos iniciales, mismo sha256 de los tensores de negativos en CADA paso, mismo sha256
  de la secuencia de lotes, mismo numero de pasos, mismo learning rate, mismo numero de
  negativos por ejemplo, misma matriz de ruido. Si uno de esos cuatro hash difiere entre
  los dos brazos, A5 falla.
- Optimizador: SGD puro, con learning rate fijo declarado. NO se usa Adam, y la razon se
  escribe: Adam normaliza por parametro y en buena parte se comeria el factor 1/K, de
  modo que la comparacion mediria el optimizador y no el objetivo.
- Numero de pasos FIJO para los dos brazos. Sin early stopping y sin scheduler, y por que:
  aqui no se compara convergencia, se compara el objetivo. Si cada brazo parara cuando su
  perdida deja de bajar, cada uno tendria su propio criterio y la comparacion mediria eso.
- Dimension 32. CPU. Presupuesto declarado: menos de 60 segundos los dos brazos con las
  10 semillas. Sin GPU. Sin datos del proyecto. Sin el artefacto congelado.

**Metricas, tres, y las tres se publican.**
- M1 precision_en_familia: fraccion de consultas cuyas n_vecinos igual a 5 vecinos mas
  proximos, excluyendo la propia consulta y las neutras, estan todos en la misma
  familia. Medida sobre las 60 palabras de familia.
- M2 separacion: similitud coseno media DENTRO de familia MENOS similitud coseno media
  ENTRE familias. Se publica con su signo.
- M3 nivel_de_azar: valor esperado de M1 bajo una asignacion aleatoria de familias con
  los mismos tamanos, por 2000 permutaciones con semilla fija 20261202.

**A4. El corpus plantado cumple sus invariantes.**
- Naturaleza: CONFIRMAR.
- Tolerancia: enteros exactos. pares_entre_familias_distintas igual a 0,
  pares_intra_familia mayor que 0, num_documentos igual a 400, palabras_por_familia igual
  a 10 en las 6 familias, palabras_neutras igual a 5, y M3 calculado con las 2000
  permutaciones.
- Exito: los cinco exactos.
- Fallo: cualquiera distinto. Si pares_entre_familias_distintas es mayor que 0, la
  estructura no esta plantada y todo el bloque pierde sentido.
- STOP: si. Sin corpus limpio no hay prueba.

**A5. El test DISCRIMINA. Esta es la asercion de la que depende el bloque entero.**
- Naturaleza: CONFIRMAR.
- Umbrales fijados aqui, antes de ejecutar, y no se negocian despues:
  margen_de_no_discriminacion igual a 0.005 absoluto sobre M1, promediado en las 10
  semillas; margen_de_discriminacion igual a 0.020 absoluto sobre M1, promediado en las
  10 semillas.
- Regla: si la diferencia absoluta entre M1_suma y M1_media es menor o igual que
  0.005, el test NO DISCRIMINA. STOP, y el informe lo dice con esas palabras:
  EL_TEST_NO_DISCRIMINA_Y_NO_CONCLUYE_NADA.
- Exito: la diferencia absoluta es mayor o igual que 0.020 Y los cuatro hash de identidad
  coinciden entre los dos brazos.
- Fallo: diferencia en la banda muerta, o cualquier hash distinto.
- STOP: si. Es el STOP que se pidio explicitamente: si los dos objetivos dieran el mismo
  numero, el test no diria nada y hay que parar en vez de publicar un titular.

**A6. Gana el objetivo de la referencia.**
- Naturaleza: CONFIRMAR. Direccion fijada: la SUMA, que es la de la ecuacion (4).
- Hipotesis: el objetivo que recupera mejor la estructura plantada es el de la referencia.
- Tolerancia: M1_suma menos M1_media mayor o igual que 0.020, Y M2_suma mayor que 0, Y
  M2_media menor que M2_suma.
- Exito: las tres.
- Fallo: la MEDIA gana por encima del margen. Entonces se publica como REFUTADO, con
  STOP, y NO se reinterpreta el resultado ni se cambia el umbral. Un REFUTADO aqui es
  informacion, no un problema que se arregla ajustando el criterio.
- STOP: si.

**A7. Robustez en semillas y control negativo.**
- Naturaleza: CONFIRMAR.
- Parte 1: 10 semillas pareadas, de 20261301 a 20261310, misma semilla para los dos
  brazos. La SUMA tiene que ganar en al menos 9 de 10, y la diferencia pareada promediada
  tiene que cumplir el margen de A6. Se publica el conteo de victorias, las 10
  diferencias pareadas con su signo, y la prueba de los signos a dos lados.
- Parte 2, control negativo: se entrena un tercer brazo, SUMA_ESTRUCTURA_ROTA, con las
  familias barajadas de forma independiente del texto, con semilla fija, de modo que la
  estructura plantada ya no esta en los datos. Exigencia: su M1 queda a menos de 0.020
  del nivel de azar M3. Si el brazo de control tambien recupera estructura, la metrica no
  mide estructura y hay que decirlo.
- Tolerancia: enteros para el conteo de victorias (9 de 10), 0.020 para la distancia al
  azar, 0.020 para el margen pareado.
- Exito: parte 1 y parte 2.
- Fallo: cualquier parte.
- STOP: si.

**A3. La convencion de P81 coincide con la de P79, y eso se declara como lo que es.**
- Naturaleza: CONFIRMAR.
- Que se afirma: que la referencia con SUMA que P81 usa es la MISMA convencion que
  reports/p79_auditoria_sgns.json, punto_1_perdidas, donde la loss_referencia con K igual
  a 10 vale 13.00537948118 y el termino_negativo_referencia vale -10.901938677039, frente a
  un termino_negativo_codigo de -1.090193867704 con ratio 0.1. Es decir, la convencion
  que P80 uso como referencia nace en P79, y las dos nacieron del mismo autor.
- Tolerancia: con los K evaluados, el ratio_del_termino_negativo de P79 tiene que coincidir
  con 1/K dentro de 1e-9, y el termino positivo tiene que ser identico entre codigo y
  referencia dentro de 1e-9.
- Exito: ambas, Y el informe lleva escrito el texto
  ESTA_CONVENCION_ES_COHERENTE_CON_P79_Y_SIGUE_SIENDO_UNA_REIMPLEMENTACION_PROPIA .
- Fallo: cualquiera. Si la convencion de P81 no fuera la de P79, el hilo de P79, P80 y
  P81 estaria roto y hay que decirlo antes de sacar conclusiones.
- STOP: si. Es barato y evita que la cadena entera se lea como si fuera evidencia externa.
- Lo que A3 NO hace: no convierte la convencion en externa. Solo registra de donde viene.

### 3.4 El limite permanente: el anclaje a una implementacion externa queda SIN HACER

Enunciado, para que quede escrito ANTES de ejecutar y no se lea despues como un olvido:

> El anclaje a una implementacion externa de SGNS queda SIN HACER.

Razones, verificadas hoy:

- gensim NO esta instalado en /home/carlos/venv.
- gensim NO aparece en requirements.txt.
- El interprete es Python 3.14.7, version en la que es razonable que gensim no compile
  por sus dependencias.
- Instalar una dependencia nueva requiere autorizacion de Carlos y no se asume.

Consecuencias, escritas para que no se recorten al citar:

- La evidencia de P81 sobre H1 es de dos tipos: un ancla DOCUMENTAL, que es una cita
  verificable, y un ancla EJECUTABLE, que es un experimento discriminante con estructura
  plantada. Ninguno de los dos es una corrida de una implementacion de terceros.
- En consecuencia, la formulacion que P81 autoriza es: el objetivo del trainer NO es el
  SGNS publicado, y el motivo es que promedia un termino que la referencia suma. La
  formulacion que P81 NO autoriza es: el resultado del trainer es peor que el de word2vec
  o que el de gensim, porque eso seria una comparacion con una implementacion que este
  proyecto no ha ejecutado.
- Este limite es PERMANENTE dentro de este proyecto con el entorno actual. No se cierra
  en P81, ni en P82, ni en P83. Se cerraria solo con una autorizacion de Carlos para
  instalar una dependencia, o con una reimplementacion externa independiente escrita por
  alguien que no sea este equipo, que es una decision de proyecto y no una decision de
  pre-registro.
- El limite no invalida A1 a A7. Acota que conclusion se permite: la evidencia sostiene una afirmacion
  sobre el OBJETIVO, no una afirmacion sobre la CALIDAD RELATIVA del artefacto frente a
  una libreria.

## 4. Bloque 2: relleno y OOV separados

### 4.1 El defecto, con sus lineas

El defecto existe en los dos scripts y es el mismo codigo:

- experiments/v9_04_dl_secuencial_fasttext.py:172 :
  indice = {palabra: i + 1 for i, palabra in enumerate(vocabulario)} , luego el id 0
  nunca se asigna a ninguna palabra.
- :173 : matriz_embedding = np.zeros((len(vocabulario) + 1, 200), dtype=np.float32) , luego
  la fila 0 es un vector de ceros.
- :84 : matriz[fila, columna] = indice.get(token, 0) , luego un token fuera de vocabulario
  cae en el id 0.
- :94 : padding_idx = 0 , luego el relleno tambien es el id 0.
- :102 : mascara = (entrada != 0) , luego la mascara borra relleno Y OOV a la vez, y
  ademas el pooling de :105-107 promedia solo sobre lo que la mascara deja.
- experiments/v10_11_bigru_finetuning.py:143 y :168 tienen el mismo padding_idx y la misma
  mascara, y :323 llama a v9_04.secuencias, luego hereda el identificador.

Consecuencia: relleno y OOV son la misma celda, y el pooling enmascarado descarta
tambien los OOV. La sonda de A16 de P80 no puede separar que el relleno moleste de que el
OOV se tire, y con una tasa de OOV de orden 0.11 el segundo canal es al menos tan grande
como el primero. Ese numero exacto lo mide A8.

### 4.2 Diseno de la separacion, fijado aqui

Cuatro condiciones, sobre la clase BiGRU REAL extraida del fichero, sin entrenar, sin
datos del proyecto, con matriz sintetica y semilla fija, igual que hizo A16 de P80 pero
con las celdas separadas. En TODAS las condiciones la mascara es la unica que cambia
para el token no relleno, y en TODAS se publica el vector agregado y la distancia
relativa L2 frente a la condicion de referencia.

| Condicion | Relleno | OOV | Mascara | Que aisla |
|---|---|---|---|---|
| C0 compartida | id 0 | id 0 | entrada distinto de 0 | el codigo real, tal cual |
| C1 solo relleno | id 0 | ninguno en el lote, todos los tokens en vocabulario | entrada distinto de 0 | solo el relleno |
| C2 solo OOV | ninguno, todas las secuencias a MAX_LEN | id propio | entrada distinto de 0, que es lo que hace el codigo | solo el OOV, descartado |
| C2b solo OOV, correccion de mascara | ninguno | id propio | entrada distinto de id_relleno | solo el OOV, entra en el promedio |
| C3 ambos separados | id propio | id propio | entrada distinto de id_relleno | los dos canales a la vez |

Con celdas separadas, los ids exigidos son:
- id_relleno igual a 0.
- id_oov igual a len(vocabulario) mas 1, es decir, UNA fila mas de las que el codigo
  reserva. La matriz pasa de len(vocabulario) mas 1 filas a len(vocabulario) mas 2.
- id_relleno distinto de id_oov, e id_oov mayor que 0. Los dos ids por encima de 0
  excepto el de relleno, que es 0 por convencion, y esa excepcion se declara.
- La fila id_relleno se deja a cero, que es lo que hace el codigo, para que la medicion
  del cambio sea del cambio y no de un vector nuevo.
- La fila id_oov se deja tambien a cero en C2, para que lo unico que cambia sea la
  mascara. En C2b se rellena con un vector declarado, no nulo, para que se pueda
  distinguir el efecto de tirar el OOV del efecto de meter un vector nulo, y ese vector
  se publica.

Las cuatro distancias que se publican, una por condicion y una por clase, con v9_04
BiGRU y con v10_11 BiGRUAfinable por separado. Ninguna cifra agregada: P80 publico una
sola distancia y eso impedia separar canales.

Las cifras de contexto que se publican ademas: MAX_LEN igual a 64 en los dos scripts, y
el numero de pasos de relleno anadidos por fila observada.

**A8. Las tasas de OOV y de relleno, medidas con el tokenizador real.**
- Naturaleza: CONFIRMAR.
- Que se mide: sobre dev, con `secuencias` y el indice reales, la proporcion de
  posiciones OOV, la proporcion de posiciones de relleno, el numero de tipos distintos
  fuera de vocabulario, y el numero de filas con cero tokens cubiertos.
- Por que hay que medirlo: la cifra que se ha citado en la discusion de auditoria es
  11.2726 por ciento de OOV, mientras que reports/p78_cobertura.json publica 0.112658 con
  la variante normalizada y 0.313639 con la cruda, y ademas con MAXLEN igual a 60 mientras
  que v9_04 usa MAX_LEN igual a 64 y normalizar_emb_normal. Las tres cifras no pueden
  ser la misma magnitud y no se arrastra ninguna: P81 recalcula con el codigo de v9_04 y
  publica el numero.
- Tolerancia: se publica con 6 decimales. No hay cota de exito ni de fallo numerico; lo
  que falla es la medicion.
- Exito: las cuatro cifras medidas y publicadas, y una linea que dice explicitamente si
  la cifra de 11.2726 por ciento queda reproducida o no, y con que codigo.
- Fallo: no se puede medir, o la reproduccion se queda sin declarar.
- STOP: no. Es una medicion de contexto, no una prueba.

**A9. Las celdas quedan separadas de verdad.**
- Naturaleza: CONFIRMAR.
- Que se verifica: id_relleno igual a 0; id_oov igual a len(vocabulario) mas 1;
  id_relleno distinto de id_oov; id_oov mayor que 0; forma de la matriz
  (len(vocabulario) mas 2, 200); fila id_relleno identicamente cero; la mascara de cada
  condicion es la declarada en 4.2 y ninguna otra.
- Tolerancia: igualdad exacta de enteros, y residuo cero exacto en la fila de relleno.
- Exito: todo.
- Fallo: cualquiera. Si id_oov no es mayor que 0, las dos celdas siguen siendo la misma y
  el bloque 2 no ha hecho nada.
- STOP: si.

**A10. Los tres canales se miden por separado.**
- Naturaleza: CONFIRMAR.
- Que se verifica: existen y se publican las cuatro distancias de la tabla de 4.2, mas
  la distancia entre C2 y C2b, que es el efecto de la mascara sobre el OOV aislado del
  resto; y existe al menos una cifra distinta de cero por condicion y por clase.
- Tolerancia: 1e-9 como banda muerta. Una distancia menor o igual que 1e-9 se marca como
  SIN EFECTO MEDIBLE.
- Exito: las cinco magnitudes presentes, con su clase y su condicion.
- Fallo: falta alguna, o sale un NaN.
- STOP: no.

**A11. El canal OOV es al menos tan grande como el de relleno.**
- Naturaleza: CONFIRMAR. Es la afirmacion que nlp-specialist puso sobre la mesa, y se
  comprueba en vez de repetirse.
- Que se afirma: la distancia del canal OOV, que es la de C2b menos la de C2, es mayor o
  igual que la distancia del canal de relleno, que es la de C1.
- Tolerancia: diferencia mayor o igual que 0. Con la banda muerta de A10 declarada.
- Exito: se cumple.
- Fallo: NO se cumple. Entonces se publica que el canal de relleno pesa MAS que el de
  OOV, con los numeros, y se corrige el titular de la objecion. No se maquilla ni se
  reinterpreta.
- STOP: no. P81 puede seguir y su trabajo es precisamente decir cual de los dos pesa mas.

**A12. Reproducibilidad y granularidad.**
- Naturaleza: CONFIRMAR.
- Que se verifica: dos pasadas con la misma semilla dan cifras identicas en las cinco
  condiciones y en las dos clases, con variacion exactamente cero; y el informe publica
  una cifra POR condicion y POR clase, nunca una sola cifra agregada.
- Tolerancia: variacion exactamente cero. Cero, no 1e-12. Cualquier variacion no nula
  entre dos pasadas con la misma semilla es STOP, porque significa que el generador no
  esta fijado y que las cifras de A10 y A11 no son reproducibles.
- Exito: variacion cero y granularidad por condicion y por clase.
- Fallo: variacion no nula, o una sola cifra agregada.
- STOP: si.

## 5. Bloque 3: reejecucion no es replicacion

El problema, textual: en reports/p80_auditoria_w2v.json, afirmacion A8, el bloque
comparacion_con_p79 dice coincide igual a true. Eso es verdad y no es lo que parece. P80
reejecuto el mismo procedimiento, con el mismo recorte declarado de 2000 lineas, con el
mismo lote y con la misma semilla. El nombre del campo induce a leer una replicacion por
via independiente, y no lo es.

Lo que P81 NO hace: editar reports/p80_auditoria_w2v.json ni el informe de P80. Esos
ficheros son evidencia cerrada.

Lo que P81 hace: en SU PROPIA salida, el bloque se llama reejecion_no_replicacion, nunca
comparacion_con_p79, lleva el campo tipo_de_evidencia con el valor REEJECION, y publica
la lista de requisitos de una replicacion genuinea junto con el resultado de comprobar
cada uno para el par P79, P80.

**A13. El bloque se llama reejecion y lleva su tipo de evidencia.**
- Naturaleza: CONFIRMAR.
- Que se verifica: en reports/p81_cierre_huecos.json existe la clave
  reejecion_no_replicacion con el valor de su clasificacion; existe el campo
  tipo_de_evidencia con el valor REEJECION para el par P79, P80; y NO existe ninguna clave
  llamada coincide dentro de un bloque cuyo nombre contenga la cadena comparacion_con_p79.
- Tolerancia: exactitud de cadenas, cero margen.
- Exito: las tres.
- Fallo: cualquiera. Si reaparece un campo con ese nombre, el error que se queria cerrar
  se reproduce en un fichero nuevo.
- STOP: si.

**A14. El criterio de replicacion genuinea, escrito y comprobado.**
- Naturaleza: CONFIRMAR.
- Cinco requisitos. Cada uno se publica con su valor booleano para el par P79, P80:
  R1 implementacion independiente: el numero se obtiene por un camino de codigo que no
  es una reimportacion ni una reejecucion del procedimiento del otro informe.
  R2 datos independientes: recorte distinto, o corpus completo. El mismo recorte NO
  cumple R2.
  R3 estimulo independiente: semilla y lote distintos de los del otro informe.
  R4 extraccion independiente: el numero no se lee del informe del otro para contrastar,
  se recalcula desde el origen.
  R5 tolerancia declarada: la coincidencia se acepta dentro de una tolerancia escrita
  antes, no por igualdad de decimales.
- Regla de clasificacion, fijada aqui: si falla R1, R2 o R4, la evidencia es REEJECION.
  Solo es REPLICACION si pasan las cinco. Para el par P79, P80 fallan R1, R2 y R4, luego
  es REEJECION, y ese veredicto se escribe.
- Tolerancia: booleana, con la regla de clasificacion de arriba como unica.
- Exito: los cinco requisitos publicados, la regla aplicada y el veredicto REEJECION
  escrito para el par P79, P80.
- Fallo: falta algun requisito, o el veredicto sale distinto de la regla.
- STOP: no.

Lo que este bloque deja dicho, y que es lo que ml-reviewer pedia: reejecutar el mismo
procedimiento con el mismo recorte y el mismo lote es una verificacion de que el
procedimiento es DETERMINISTA. Es informacion real y util. No es una replicacion, y a
partir de ahora no se etiqueta como tal.

## 6. Bloque 4: por identificadores, no por tamano

El problema, textual: A17 prueba que el pool viene de read_train y que el tamano es 1483
mas 248. Pero la igualdad de tamanos NO distingue el holdout dentro de 248 filas de
cualquier otro origen de 248 filas. A18, con los mismos n_validacion, demuestra el mismo
recuento, no las mismas filas.

La via que lo cierra sin tocar el holdout: read_train devuelve el train completo; el
vocabulario del proyecto no es un fichero de ids. src.splits.holdout_v2 es un
determinista StratifiedGroupKFold(7, shuffle, random_state 42) sobre train y devuelve
indices posicionales, indices_dev de 1483 e indices_holdout de 248. Los ids del holdout
son los ids de esas filas posicionales. Por tanto la identidad de conjuntos se puede
comprobar sin leer el holdout.

**A15. Los cinco pliegues, por conjunto de identificadores.**
- Naturaleza: CONFIRMAR.
- Que se verifica: se recalcula la particion de CV de experiments/11_dl_desde_cero.py:491
  con su StratifiedGroupKFold(5, shuffle igual a True, random_state igual a 42) sobre el
  train completo, y para cada pliegue se publican el numero de filas, el sha256 de la
  lista de ids ordenada y separada por saltos de linea, y el primer y el ultimo id.
- Que se exige ademas: que los conteos reproduzcan exactamente 347, 346, 346, 346, 346;
  que la union de los cinco pliegues sea exactamente el train completo, con
  interseccion pairwise vacia; y que los cinco sha256 se publiquen.
- Tolerancia: enteros exactos para los conteos; igualdad exacta de conjuntos para la
  union y la interseccion; coincidencia exacta de los hash.
- Exito: las tres cosas.
- Fallo: los conteos no reproducen los publicados, o la union no es el train completo, o
  dos pliegues se solapan. En ese caso la cadena de A17 queda rota y hay que decirlo.
- STOP: si.

**A16. Las 248 filas del residuo, identificadas por id.**
- Naturaleza: CONFIRMAR.
- Que se verifica: el conjunto de ids del pool es el train completo; el conjunto de ids de
  desarrollo son los 1483 de indices_dev; el conjunto diferencia pool menos desarrollo
  tiene exactamente 248 elementos, es disjunto de desarrollo, y su sha256 se publica; y
  el conjunto de indices posicionales que devuelve src.splits.holdout_v2 como holdout es
  EXACTAMENTE ese conjunto diferencia, elemento a elemento. Esta ultima identidad de
  conjuntos es la que ml-reviewer pedia y la que la igualdad de tamanos no daba.
- Ademas, y esto es una toma de contacto sobre una fuente externa al calculo: se intenta
  reproducir hash_ids_csv de config/holdout_v2.json, que vale
  e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd . Se recalcula bajo
  tres convenciones declaradas por adelantado: ids ordenados separados por comas, ids
  ordenados separados por saltos de linea, y orden de fichero. Si alguna coincide, se
  publica la convencion que coincide. Si ninguna coincide, NO se fuerza: se publica que
  ninguna coincide, se publican los tres hash recalculados, y se declara como limite que
  el hash de config/holdout_v2.json no se pudo vincular a las 248 filas por esa via. La
  identidad de conjuntos de A16 se sostiene igual, porque no depende de ese hash.
- Tolerancia: enteros exactos para los tamanos; igualdad exacta de conjuntos para las
  intersacciones y para la identidad con indices_holdout; coincidencia exacta de hash
  solo si alguna de las tres convenciones coincide, y si no coincide no hay fallo sino un
  limite declarado.
- Exito: identidades de conjuntos correctas, hash del residuo publicado, y la linea sobre
  hash_ids_csv escrita de una de las dos formas, la del acierto o la del limite.
- Fallo: el residuo no tiene 248 elementos, o indices_holdout no es ese residuo. Eso
  refutaria A17, y entonces A17 se marca REFUTADO en el informe de P81, en documento
  nuevo, sin editar el de P80.
- STOP: si.

**A17. Los tres informes de la fase DL, por identificador y no por conteo.**
- Naturaleza: CONFIRMAR.
- Que se verifica: que los tres informes, DL-01, DL-02 y DL-03, comparten el MISMO
  conjunto de ids por pliegue, y no solo el mismo n_validacion.
- Via, declarada: los tres salen del MISMO script, experiments/11_dl_desde_cero.py, con el
  MISMO pool, el MISMO StratifiedGroupKFold y el MISMO random_state. Luego el conjunto de
  ids de cada pliegue es UNO solo, el que recalcula A15, y eso es lo que se afirma.
- Limite, declarado y obligatorio: los tres informes NO persisten los ids por pliegue. Por
  tanto la identidad a nivel de ids se establece por RECALCULO DETERMINISTA del camino de
  codigo compartido, no por lectura de ids persistidos. El informe tiene que decir
  exactamente eso, con el campo ids_por_pliegue_persistidos_en_los_informes igual a false.
- Tolerancia: igualdad exacta entre el conjunto recalculado y si mismo, y publicacion de
  los cinco sha256 por pliegue.
- Exito: el limite declarado esta escrito Y los cinco hash estan publicados.
- Fallo: se afirma identidad de ids sin declarar el limite, o el limite se declara pero no
  se publican los hash.
- STOP: no.

## 7. Bloque 5: cualificar el impacto del scheduler

El problema, textual: el desfase de una epoca es REAL y H2 lo confirma, pero su impacto
es despreciable, y el informe de P80 no lo dice. Las cifras que P80 ya publico y que P81
reutiliza como punto de partida son: total_lotes_plan igual a 1395, total_lotes_recorridos
igual a 1398, desviacion relativa igual a 0.002150537634408602, es decir 0.215053 por
ciento; y el learning rate con dos trayectorias, la que produce el codigo y la que
produciria el scheduler si hubiera usado el total real, con los siguientes pares tal como
P80 los nombro:

| Magnitud | campo en P80 | valor |
|---|---|---|
| total de pasos | pasos_totales | 1398 en las dos trayectorias |
| learning rate inicial | lr_inicial | 6.4 en las dos |
| learning rate minimo declarado | lr_minimo | 0.001 en las dos |
| ultimo learning rate | ultimo_real | 0.004577968526466236 en la real, 0.001 en la ideal |
| pasos en el piso del minimo | pasos_en_el_piso_del_minimo | 0 en la real, 3 en la ideal |
| fraccion en el piso | fraccion_en_el_piso | 0.0 en la real, 0.002145922746781116 en la ideal |
| media del learning rate | media_real | 3.202288984263234 en la real, 3.195424177396281 en la ideal |

De ahi salen los dos numeros que hay que cualificar: la diferencia de conteo de lotes es
0.215053 por ciento, y la diferencia entre las dos medias de learning rate es 0.006864806866953
en absoluto, es decir 0.0021437187276626846 en relativo, es decir 0.214372 por ciento.
Los tres porcentajes son practicamente el mismo numero, y eso ya dice bastante, pero la
afirmacion no se apoya en que se parezcan: se apoya en A18.

**A18. El learning rate, paso a paso, y clasificacion del impacto.**
- Naturaleza: CONFIRMAR.
- Que se mide: con el recorte declarado de 2000 lineas, 53243 tokens y lote 256, se
  recalcula el learning rate en CADA uno de los 1398 pasos, en las dos trayectorias: la
  que produce el codigo de p72_01, que usa el total del plan, y la contrafactual, que usa
  el total recorrido. Se publican la serie completa, el maximo de la diferencia absoluta,
  la media de cada trayectoria, el numero de pasos con diferencia relativa mayor que
  1e-6, el numero de pasos en el piso en cada una, y los diez primeros y los diez
  ultimos pares.
- Umbral de clasificacion, fijado aqui antes de ejecutar:
  umbral_de_impacto_despreciable igual a 0.005 en relativo sobre la media del learning
  rate. Regla: si la diferencia relativa de las medias es menor o igual que 0.005, el
  impacto se declara DESPRECIABLE. Si es mayor, se declara NO DESPRECIABLE y se explica.
- Honestidad del pre-registro, escrita de antemano: con las cifras que P80 ya publico, la
  diferencia relativa es 0.0021437187276626846, que cae por debajo de 0.005, luego el
  resultado probable de A18 es DESPRECIABLE. El valor de A18 no es la etiqueta, que ya se
  puede intuir, sino la SERIE PASO A PASO, que nadie ha publicado: P80 publico la media y
  los extremos, y la serie entera es lo que hace falta para saber si el desfase toca algo
  que no sea la media.
- Exito: la serie esta medida y publicada, y la clasificacion sale de aplicar el umbral.
- Fallo: la serie no se puede calcular, o la clasificacion no sale del umbral.
- STOP: no.

**A19. La declaracion de impacto esta atada a la medida.**
- Naturaleza: CONFIRMAR.
- Que se verifica: que el informe de P81 contiene una frase explicita sobre el impacto del
  desfase, y que esa frase dice DESPRECIABLE si y solo si la regla de A18 se cumple, y
  NO DESPRECIABLE si y solo si no se cumple. Se exige un campo
  declaracion_de_impacto_del_scheduler con el valor DESPRECIABLE o NO DESPRECIABLE, y se
  exige que ese valor sea igual al que dicta la regla de A18.
- Tolerancia: igualdad exacta de la etiqueta. No hay interpretacion.
- Exito: la etiqueta existe y coincide con la regla.
- Fallo: la etiqueta falta, o dice DESPRECIABLE cuando la regla dice NO DESPRECIABLE. En ese
  caso concreto, STOP, porque es exactamente el problema que este bloque cierra: un titular
  escrito por encima de la medida.
- STOP: si.

Lo que este bloque deja escrito, con o sin excepcion: el desfase de una epoca EXISTE y es
un defecto real de codigo, porque el plan usa range(EPOCAS) y el entrenamiento usa
enumerate(..., start=1). Lo que se cualifica es su consecuencia numerica, que es
despreciable. Las dos cosas se pueden decir a la vez y no se contradicen.

## 8. Resumen de naturalezas

| Naturaleza | Cuantas | Ids |
|---|---|---|
| CONFIRMAR | 20 | A0 a A19 |
| REFUTAR | 0 | ninguna |
| NO APLICABLE POR CONSTRUCCION | 0 | ninguna |
| NO_EVALUABLE | 0 | ninguna |

Ninguna afirmacion es NO APLICABLE POR CONSTRUCCION y ninguna es NO_EVALUABLE, y ambas
cosas estan prohibidas como salida comoda: si algo no se puede evaluar, se dice que no se
pudo evaluar y se marca NO_EVALUABLE con la razon, no se convierte en un CONFIRMADO ni se
omite. El limite externo de 3.4 NO es una afirmacion NO APLICABLE: es un limite
declarado, y esta escrito como limite.

## 9. Criterio de parada

Se para, con SystemExit, sin registrar ningun otro numero, si:

- El sha256, el size o el mtime_ns del artefacto congelado difieren antes o despues. A0.
- La asercion A5 falla: el test no discrimina. Se escribe EL_TEST_NO_DISCRIMINA_Y_NO_CONCLUYE_NADA
  y no se publica ningun titular sobre H1.
- La asercion A6 falla: gana el objetivo de la media. Se publica REFUTADO y no se
  reinterpreta.
- La asercion A4 falla: el corpus plantado no cumple sus invariantes. Sin estructura
  plantada no hay prueba.
- La asercion A7 falla: el efecto no se sostiene en semillas o el control negativo tambien
  recupera estructura.
- La asercion A9 falla: las celdas no quedan separadas, luego el bloque 2 no ha medido nada.
- La asercion A12 falla: variacion no nula entre dos pasadas con la misma semilla.
- La asercion A13 falla: reaparece un campo llamado coincide bajo un bloque llamado
  comparacion_con_p79.
- La asercion A15 o A16 falla: la cadena de identificadores no cierra, luego A17 de P80 se
  marca REFUTADO en documento nuevo.
- La asercion A19 falla: la declaracion de impacto no coincide con la regla de A18.
- Aparece un numero sospechosamente perfecto, por ejemplo una precision en familia mayor
  que 0.99. Se investiga antes de reportar nada.
- Se viola cualquier frontera de la seccion 1.

Criterio de parada de presupuesto: menos de 60 segundos de CPU por par de brazos y menos
de 5 minutos de reloj para el script completo. Si se supera, se para y se publica lo que
haya con su estado real.

## 10. Lo que P81 no hace

- No genera submission, no consulta el leaderboard, no lee test.
- No lee el holdout ni su fichero de identificadores.
- No reentrena, no modifica, no sobrescribe el artefacto Word2Vec 100d.
- No reentrena V03, no toca sus umbrales congelados 0.44, 0.51 y 0.46.
- No modifica ni borra ningun script de P71 a P80, ni sus informes, ni sus JSON.
- No instala gensim ni ninguna dependencia nueva.
- No usa la GPU.
- No afirma que Word2Vec sea mejor que V03, ni que una RNN o un modelo bayesiano lo
  superen, ni que el artefacto sea mejor o peor que el de una libreria.
- No anade hallazgos que no cierren los cinco huecos.
- No elige umbral, tolerancia ni criterio despues de ver un numero. Todas las tolerancias,
  margenes y umbrales estan arriba, y estan hasheados antes de ejecutar.

## 11. Formulacion

Toda conclusion que P81 escriba lleva esta frase, y solo esta frase sobre el estado del
proyecto:

    V03 remains the best verified leaderboard result.

No se afirma jamas que Word2Vec sea mejor que V03.

    V03 remains the best verified leaderboard result.
