# Pre-registro P80 - Auditoria del entrenador Word2Vec, sin repetir los errores de P79

Fecha de escritura: 2026-09-30
Rama: `proyecto-kaggle-toxic-2026-p67`
Script previsto: `experiments/p80_01_auditoria_w2v.py`
Informes previstos: `reports/p80_auditoria_w2v.json` y `reports/p80_auditoria_w2v.md`
Estado: PRE-REGISTRADO. Este texto se hashea y se commitea ANTES de escribir una linea de codigo.
sha256 de este fichero: se registra en el mensaje de commit y en `RUNLOG.md`. No se escribe
aqui, porque un fichero no puede contener su propio hash.

Formulacion obligatoria, literal, en el informe, en el markdown, en el log y en cualquier
mensaje de salida de P80:

    V03 remains the best verified leaderboard result.

Prohibido afirmar, insinuar o dejar entender que Word2Vec sea mejor que V03. Prohibido
afirmar que una RNN o un modelo bayesiano superen a V03 sin medicion out-of-fold limpia.

---

## 0. Por que existe P80, y no una extension de P79

P79 (`experiments/p79_01_auditoria_sgns.py`, `reports/p79_auditoria_sgns.json`) acierta en
lo importante: confirma que `SkipGram.forward` de P71-04 no es SGNS y que el gradiente se
desbloquea con el factor 1/K. Los tres hallazgos de H1 se sostienen. P79 falla, en cambio,
en su propio instrumental, y de ahi nacen tres afirmaciones que este pre-registro corrige:

1. `auditar_semillas_en_el_fuente` (`p79_01:561-625`) declara `semillas_desalineadas: false`.
   El codigo de P72-01 si usa dos rangos distintos, `for ep in range(m.EPOCAS)` en
   `p72_01:92` y `for ep, _ in enumerate(planes, start=1)` en `p72_01:99`. El fallo es doble:
   el encabezado del bucle se busca hacia arriba 15 lineas buscando un `for `, y la etiqueta
   "plan" se decide buscando la cadena `planes.append` en una ventana de 4 lineas. Con esa
   heuristica las dos apariciones caen en el mismo grupo, `entrenamiento`, y la union de
   {0,1,2,3,4} con {1,2,3,4,5} es {0,1,2,3,4,5}. El booleano sale False porque solo hay un
   grupo, no porque los rangos coincidan. El propio dato de P79 lo contradice: su lista
   `entrenamiento: [0, 1, 2, 3, 4, 5]` tiene 6 elementos para `EPOCAS = 5`.
2. La asercion `la_identidad_se_cumple` (`p79_01:361`) compara `pos_r + neg_c` contra
   `l_cod`, es decir contra `-(pos + neg_c)`. Falta el signo menos. Sale False en los cuatro
   valores de K, y el script no para: un indicador que siempre falla no informa de nada.
3. `evaluar_artefacto_congelado` (`p79_01:994-995`) asigna `E = S = W`, la misma matriz
   guardada, a la tabla de entrada y a la de salida. El artefacto solo guarda `W`, que es
   `modelo.entrada.weight` (`p72_01:124-127`). La tabla de salida no existe en ninguna parte
   del repositorio, luego las cifras absolutas de loss y de norma de gradiente de P79
   describen un par (E, S) que nunca existio durante el entrenamiento.

P80 no edita P79. P80 rehace el instrumental, declara sus limites y corrige las tres
afirmaciones en un documento nuevo.

---

## 1. Proposito

### 1.1 Que se audita

- Que la loss de `SkipGram` de `p71_04:182-216` es o no SGNS, y con que peso exacto.
- Que el gradiente que produce el codigo se desbloquea o no con el factor 1/K, componente a
  componente, con la descomposicion explicita y el residuo.
- Que el calculo de `total_lotes` del scheduler de `p72_01:91-96` y el recorrido real de
  `p72_01:98-120` usan el mismo conjunto de pares y el mismo numero de lotes.
- Que la matriz de salida del artefacto congelado es o no recuperable, y que consecuencias
  tiene para las cifras que P79 publico sobre ese artefacto.
- Que la eleccion de epoca de los BiGRU de `v9_04` y `v10_11` usa el mismo pliegue que
  despues se puntua.
- Que el padding de esos mismos BiGRU se enmascara antes o despues de la GRU, y cuanto
  pesa.
- Que DL-03, en `reports/dl_desde_cero_resultados.json`, entreno sobre el conjunto completo
  de etiquetas, holdout v2 incluido, y que eligio epoca con el pliegue externo.

### 1.2 Que NO se audita, por construccion

- NO se reentrena el Word2Vec, ni de 10 millones de tokens ni de un corpus sintetico con
  fines de conclusion. P80 hace unicamente producto punto a punto y lectura de fuentes. Las
  microrondas de Adam sobre tensores diminutos se permiten SOLO como sonda de
  discriminacion de una asercion, y su resultado no se reporta como medida del artefacto.
- NO se modifica, reentrena ni sobrescribe `modelos/twitter_domain/w2v_twitter_100d.npz`.
- NO se modifica V03, ni sus umbrales congelados 0.44 / 0.51 / 0.46, ni
  `experiments/baseline_v03_frozen/`.
- NO se genera submission, NO se consulta el leaderboard, NO se lee test, NO se lee el
  holdout, NO se lee `features/v2_structured_features.parquet`.
- NO se edita ni se borra P71, P72, P73, P74, P77, P78, P79, ni V03, ni los informes que
  ellos escribieron. La correccion se hace en documentos NUEVOS.
- NO se propose reentrenar el Word2Vec. P80 cuantifica. La decision de reentrenar es de otra
  fase, y es de Carlos.
- NO se responde a la pregunta de si conviene una LSTM, un GRU con atencion o un modelo
  bayesiano. Eso es P81 y P82, y solo se abren si P80 deja margen.

### 1.3 Lo que P80 NO usa como evidencia

`reports/p79_auditoria_sgns.json` se lee, si acaso, para contrastar y para listar que hay que
rehacer. Ninguna cifra de P79 se da por buena. En concreto, estas cifras de P79 NO se dan
por replicadas ni por refutadas de antemano, y P80 debe producir su propia medida o
declarar que no puede:

    ratio del termino negativo con K=10            0.1
    ratio de la norma del gradiente de entrada      0.08033795   (artefacto, E = S = W)
    loss del codigo sobre el artefacto              1.08412074
    loss de la referencia sobre el artefacto        9.78555861
    cota inferior y superior del ratio total        0.1 y 1.0
    desvio relativo del total de lotes             0.00215054   (recorte de 2000 lineas)
    lectura del plan frente al entrenamiento        "los dos bucles usan el mismo rango"

Las tres ultimas lineas se presumen ERRONEAS antes de ejecutar. Las demas se presumen
medibles, con el matiz de A5 y de A11 a A13.

---

## 2. Alcance

### 2.1 Ficheros que P80 lee

Solo lectura, todos de texto o de informe:

    experiments/p71_04_entrenar_word2vec.py        clase SkipGram, forward, init, pares_de_epoca
    experiments/p71_08_word2vec_corregido.py       contraste del mismo patron de scheduler
    experiments/p72_01_dimensiones.py              loops de plan y de entrenamiento, guardado del npz
    experiments/v9_04_dl_secuencial_fasttext.py    BiGRU, pooling, early stopping
    experiments/v10_11_bigru_finetuning.py         BiGRU afinable, pooling, early stopping
    experiments/11_dl_desde_cero.py                pool de CV y seleccion de epoca
    reports/p79_auditoria_sgns.json                contraste, no evidencia
    reports/twitter_word2vec_*.json                procedencia del congelado, solo metadatos
    reports/dl_desde_cero_resultados.json          n_validacion por pliegue, metricas OOF
    data/external/corpora/twitter_domain/corpus_twitter_tokens.txt   primeras 2000 lineas
    config/holdout_v2.json                         SOLO n_tweets_dev y n_tweets_holdout
    docs/p78_preregistro.md                        referencia de formato

De `config/holdout_v2.json` se leen EXCLUSIVAMENTE los dos recuentos de filas y el
`hash_ids_csv`. La lista de identificadores NO se lee, NO se deserializa y NO se cruza con
ninguna otra tabla. Si el codigo necesita mas de esos dos campos, para y avisa.

### 2.2 Ficheros que P80 NO lee

    data/raw/toxic_test.csv  y cualquier test
    cualquier particion del holdout v2, y sus etiquetas
    cualquier score de leaderboard
    submissions/
    features/v2_structured_features.parquet
    features/oof_v03_final.npz
    experiments/baseline_v03_frozen/   (no hace falta: V03 no se toca)

### 2.3 El artefacto congelado: solo lectura, con hash antes y despues

`modelos/twitter_domain/w2v_twitter_100d.npz`, sha256 declarado
`302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`.

Reglas obligatorias:

1. Se abre con `open(ruta, "rb")` o con `numpy.load(ruta, mmap_mode="r")`. Nunca se abre en
   modo escritura, nunca se copia dentro de `modelos/`, nunca se `savez` encima.
2. Antes de la PRIMERA lectura se registra sha256, `st_size` y `st_mtime_ns`. Si el sha256 no
   coincide con el declarado, STOP inmediato y aviso a Carlos. No se sigue.
3. Despues de la ULTIMA lectura se vuelven a registrar los tres valores. Si cualquiera de
   los tres cambia, STOP inmediato, y el informe se marca como invalido.
4. Ninguna matriz derivada se escribe en disco. Todo lo calculado vive en memoria y sale
   solo por el JSON.
5. P80 no abre ningun otro fichero de `modelos/`.

### 2.4 Restricciones de ejecucion

- Un experimento = un script nuevo y completo. `experiments/p80_01_auditoria_w2v.py`.
- venv `/home/carlos/venv/bin/python`. ROCm, GPU no es necesaria: P80 es CPU, y el presupuesto
  es de minutos.
- ASCII puro en todo lo generado, verificado con `tools/verificar_caracteres.py` MAS un
  scanner byte a byte propio que rechace cualquier byte mayor que 127. El verificador oficial
  no detecta Latin-1 Supplement.
- Sin emojis ni emotes en el codigo ni en los comentarios.

---

## 3. Convenciones numericas

- Todo el calculo en `float64`, con `torch` en doble. Si alguna vez hace falta `float32`, se
  declara y se sube la tolerancia.
- Los indices de `centro`, `ctx` y `neg` se repiten a proposito, como en word2vec real, donde
  un mismo centro aparece muchas veces. Toda comparacion de gradientes se hace sobre la
  matriz ACUMULADA, porque autograd entrega el gradiente de la matriz entera con las
  repeticiones ya sumadas. Indexar ese tensor por los indices volveria a multiplicar por el
  numero de repeticiones.
- Red de prueba para H1: dimension 5, vocabulario 7, tipos 6, semilla 20261140,
  `K` en {1, 5, 10, 20}. Se mantiene la de P79 para que el contraste sea directo, y porque
  una red diminuta hace imposible un fallo de forma.
- En H1 se ejecutan DOS implementaciones: la replica local y el `SkipGram.forward` REAL
  importado de P71-04, con los mismos pesos. La auditoria se hace sobre el codigo que se
  ejecuto, no sobre una reimplementacion.
- Tolerancias:

| Magnitud | Tolerancia | Motivo |
|---|---|---|
| fidelidad replica local frente a `SkipGram.forward` | 1e-12 absoluta | mismo codigo en doble, debe dar bits identicos |
| identidad de la loss con el signo correcto | 1e-12 absoluta | suma de tres terminos en doble |
| identidad de la loss con el signo INCORRECTO (control negativo) | 1e-6 minima de separacion | tiene que fallar, no rozarse |
| ratio del termino negativo contra 1/K | 1e-12 absoluta | es una razon analitica exacta |
| componentes de gradiente, relativo | 1e-6 relativa | comparacion contra autograd |
| residuo de la descomposicion de gradientes | 1e-9 absoluta | el analitico debe reproducir a autograd |
| sondas de padding | 1e-3 relativa (umbral de decision, no de igualdad) | ver A16 |
| recuentos de filas y de enteros | 0, igualdad exacta | no admiten tolerancia |

---

## 4. Mapa de afirmaciones: que se CONFIRMA, que se REFUTA, que NO APLICA

Las 19 afirmaciones de la seccion 5 se resumen aqui. "Naturaleza" es lo que la afirmacion
hace con el mundo, no lo que espera encontrar.

| Id | Hallazgo | Naturaleza | Veredicto esperado | STOP si falla |
|---|---|---|---|---|
| A1 | H1 la replica es fiel a `SkipGram.forward` | CONFIRMAR | CONFIRMADO | si |
| A2 | H1 el termino positivo es identico al de la referencia | CONFIRMAR | CONFIRMADO | si |
| A3 | H1 el termino negativo pesa exactamente 1/K | CONFIRMAR | CONFIRMADO | si |
| A4 | H1 el gradiente es la referencia con la repulsion en 1/K | CONFIRMAR | CONFIRMADO | si |
| A5 | H1 la razon de la NORMA total del gradiente | REFUTAR | cota de P79 REFUTADA | no |
| A6 | H2 procedimiento de lectura, sin heuristica | CONFIRMAR | CONFIRMADO | si |
| A7 | H2 los dos rangos de semillas son distintos | CONFIRMAR | CONFIRMADO | no |
| A8 | H2 cuantificacion del efecto en el learning rate | CONFIRMAR | CONFIRMADO | no |
| A9 | H3 identidad de la loss con el SIGNO correcto | CONFIRMAR | CONFIRMADO | si |
| A10 | H3 control negativo: la identidad mal puesta TIENE que fallar | CONFIRMAR | CONFIRMADO | si |
| A11 | H4 el npz solo guarda `W` y `palabras` | CONFIRMAR | CONFIRMADO | si |
| A12 | H4 la matriz de salida no se puede reconstruir | CONFIRMAR | CONFIRMADO | no |
| A13 | H4 las cifras absolutas sobre el artefacto no son evidencia | NO APLICABLE | NO APLICABLE POR CONSTRUCCION | no |
| A14 | H5 la epoca se elige con el pliegue que se puntua | CONFIRMAR | CONFIRMADO | no |
| A15 | H6 el enmascarado se aplica DESPUES de la GRU | CONFIRMAR | CONFIRMADO | si |
| A16 | H6 cuanto pesa la contaminacion por relleno | CONFIRMAR | sin veredicto fijado de antemano | no |
| A17 | H7 el pool de CV de DL-03 incluye el holdout v2 | CONFIRMAR | CONFIRMADO | si |
| A18 | H7 lo mismo en DL-01 y DL-02, y que el OOF de DL-01 no supera a V03 | CONFIRMAR | CONFIRMADO | no |
| A19 | H7 la epoca de DL tambien se elige con el pliegue externo | CONFIRMAR | CONFIRMADO | no |

Un hallazgo queda CLASIFICADO, no medido, en estos tres estados y solo en estos tres:
CONFIRMADO, REFUTADO, NO APLICABLE POR CONSTRUCCION. NO APLICABLE significa que la pregunta
no tiene respuesta posible con los datos que existen, y se usa cuando la afirmacion depende
de una matriz que no se guardo. Un hallazgo que no se puede resolver no se reporta como
"probablemente cierto".

---

## 5. Afirmaciones a verificar

Cada afirmacion lleva: que se afirma, con que tolerancia, que cuenta como exito, que cuenta
como fallo, y si su fallo para el script.

### A1. La replica local de la loss es fiel al codigo que se ejecuto

Afirmacion: la loss reimplementada en P80 coincide con `SkipGram.forward` de
`p71_04:199-216` cuando se ejecuta con los mismos pesos y los mismos indices.
Tolerancia: 1e-12 absoluta sobre el valor de la loss, en float64, para los cuatro K.
Exito: la diferencia maxima sobre los cuatro K es menor o igual que 1e-12.
Fallo: cualquier diferencia mayor, o que la clase importada no acepte la firma de llamada
usada. STOP: si. Sin fidelidad, todo lo demas de H1 seria una auditoria de una copia.

### A2. El termino positivo del codigo es identico al de la referencia

Afirmacion: `logsigmoid(v . u_pos).mean()` no depende de K ni de como se trate el termino
negativo, luego el componente positivo es identico en ambas formulas.
Tolerancia: 1e-12 absoluta.
Exito: diferencia menor o igual que 1e-12 en los cuatro K.
Fallo: diferencia mayor. STOP: si.

### A3. El termino negativo del codigo pesa EXACTAMENTE 1/K del de la referencia

Afirmacion: el codigo promedia el termino negativo sobre los ejes lote y K, de modo que
`neg_codigo = neg_referencia / K`, con `neg_referencia` definida con la SUMA sobre K. El
`1/K` es exacto, no aproximado: es una division por un entero, no un estimador.
Tolerancia: `abs(ratio_neg - 1.0 / K) <= 1e-12`, con `ratio_neg = neg_codigo / neg_referencia`.
Exito: se cumple en los cuatro K, con K en {1, 5, 10, 20}. En K = 1 el ratio vale 1 y la
asercion es trivialmente cierta, luego K = 1 NO cuenta como evidencia: la evidencia estan
K = 5, K = 10 y K = 20.
Fallo: cualquier desviacion mayor que 1e-12, o un ratio que no sea 1/K. STOP: si.
Ademas se exige citar la linea del fuente que lo produce, `p71_04:215`, con su texto literal
en el JSON, y comprobar que no contiene `.sum(dim=1)` antes del `.mean()`.

### A4. El gradiente del codigo es el de la referencia con la repulsion en 1/K

Afirmacion: con la descomposicion explicita
`dv = ( -sigma(-s_pos) u_pos + f SUM_k sigma(s_neg_bk) u_neg_bk ) / B`,
`du_pos = ( -sigma(-s_pos) v ) / B`, `du_neg = ( f sigma(s_neg_bk) v ) / B`, el componente
positivo tiene peso 1 y el negativo tiene peso `f`. Para el codigo `f = 1/K`; para la
referencia `f = 1`. Ademas, la suma de componentes reproduce a autograd con residuo cero.
Tolerancia: 1e-6 relativa en cada componente; 1e-9 absoluta en cada residuo de descomposicion.
Exito: los tres ratios pesan lo que deben, `ratio_positivo = 1` y `ratio_negativo = 1/K`, con
los residuos por debajo de 1e-9, en los cuatro K, comparado sobre la matriz acumulada con
`index_add_` y los mismos indices.
Fallo: cualquier ratio fuera de tolerancia, o cualquier residuo por encima de 1e-9, lo que
significaria que la forma cerrada esta mal. STOP: si.

### A5. La razon de la NORMA del gradiente TOTAL de entrada NO se afirma que sea 1/K

Afirmacion, y es una AFIRMACION NEGATIVA: P80 no asegura que
`||g_entrada_codigo|| / ||g_entrada_referencia||` valga `1/K`, ni que este dentro de
`[1/K, 1]`. P79 afirmo esa cota, `cota_inferior_del_ratio_total` y
`cota_superior_del_ratio_total`, sin demostrarla, y su propia medida sobre el artefacto la
incumple: 0.08033795 con K = 10, por debajo de 1/K = 0.1. La cota no se sostiene en general,
porque `g_codigo = g_pos + g_neg / K` y `g_referencia = g_pos + g_neg` no son colineales:
si `g_neg` se opone a `g_pos` con razon K, el numerador puede anularse.
Tolerancia: no aplica. Es una correccion de una asercion indebida, no una medicion.
Exito: el JSON de P80 contiene el ratio total como numero INFORMATIVO, con la etiqueta
`no_es_una_asercion`, y NO contiene ninguna cota sobre el. Ademas contiene una
`demostracion_de_la_cota` con un contraejemplo numerico explicito.
Fallo: que aparezca cualquier cota, o que el ratio se compare contra 1/K. STOP: no.

### A6. El scheduler se lee linea a linea, sin heuristica de busqueda de encabezado

Afirmacion: el desfase se establece leyendo el fichero COMPLETO con el modulo `ast` de la
biblioteca estandar, no con busqueda de cadenas. Requisitos de procedimiento, todos
obligatorios:

1. Leer el fichero entero y registrar su numero total de lineas y su sha256.
2. Localizar cada llamada `default_rng` recorriendo el AST con `ast.walk` y un mapa de
   padres. PROHIBIDO buscar la cadena `default_rng` en el texto.
3. Obtener el bucle contenedor leyendo el padre `ast.For` en el mapa de padres. PROHIBIDO
   buscar hacia atras una linea que empiece por `for ` en una ventana de 15 lineas.
4. Interpretar el iterador, y solo dos formas se admiten:
   - `range(N)`: los valores son `0 .. N - 1`, con `N` resuelto desde el modulo importado
     (por ejemplo `m.EPOCAS`).
   - `enumerate(x, start=k)`: los valores son `k .. k + len(x) - 1`, con `len(x)` tomado
     del objeto que se itera, no supuesto.
   Cualquier otra forma de iterador, o cualquier variable que no se pueda resolver, produce
   STOP y la ausencia de veredicto. No hay valor por omision y no hay suposicion.
5. El discriminador entre el bucle del plan y el de entrenamiento se obtiene del AST: un
   bucle es del plan si su cuerpo contiene una llamada `.append` sobre el nombre `planes`.
   PROHIBIDO decidirlo por la presencia de la cadena `planes.append` en una ventana de 4
   lineas alrededor de la llamada, que es lo que produjo el error de P79.
6. El informe guarda, para cada llamada, su linea, su bucle contenedor con linea y texto
   literal, el rango de valores resuelto, y el conjunto de enteros resultante.
Tolerancia: igualdad exacta de conjuntos de enteros. Cero margen.
Exito: el procedimiento se completa sin ninguna regla de proximidad y los dos rangos
quedan resueltos.
Fallo: hace falta la heuristica de P79, o el iterador no es de las dos formas admitidas.
STOP: si.

### A7. Los dos bucles usan rangos de semillas distintos

Afirmacion: en `p72_01`, el bucle del plan resuelve `ep` en `SEMILLA + {0, 1, 2, 3, 4}` y el
bucle de entrenamiento en `SEMILLA + {1, 2, 3, 4, 5}`. Los conjuntos son distintos: el plan
gasta la semilla `SEMILLA + 0` y la ultima `SEMILLA + 5`, y el entrenamiento nunca usa la
primera. Por tanto los dos generan conjuntos de pares distintos, y `total_lotes` se calibra
sobre un numero de lotes que no es el que se recorre.
Tolerancia: igualdad exacta de conjuntos. `conjunto_plan == conjunto_entrenamiento` es
FALLO, no exito.
Exito: los conjuntos son distintos, y la diferencia se reporta con las listas completas.
Fallo: son iguales, lo que refutaria H2 y, de paso, explicaria el booleano de P79.
STOP: no. Se registra REFUTADO y se documenta que P79 acertaba por el motivo equivocado.

### A8. Cuantificacion del efecto del desfase sobre el learning rate

Afirmacion, en dos partes, con dos alcances distintos:

Parte 1, EXENTA DE RECORTE, y es la que sostiene el hallazgo: la diferencia entre las dos
secuencias de lotes existe para cualquier corpus, porque las semillas son distintas y
`pares_de_epoca` depende de la semilla. Se comprueba ejecutando los dos bucles tal como
estan escritos, con el numero de filas del recorte, y se reporta `total_lotes_plan` frente a
`total_lotes_recorridos`, con la asercion `desvio_relativo != 0`.

Parte 2, CON RECORTE DECLARADO: el recuento se hace sobre las primeras 2000 lineas de
`corpus_twitter_tokens.txt`, con lote 256, para poder comparar con P79. Se exige declarar en
el JSON que los recuentos ABSOLUTOS de pares no son los de la corrida real, porque la
frecuencia de los tipos cambia al recortar, y que lo que se extrapola es la DESVIACION
relativa, no los recuentos.

Se reportan ademas: learning rate inicial, minimo, ultimo real, ultimo ideal, numero de
pasos en el piso del minimo, y pasos totales.
Tolerancia: `desvio_relativo != 0` con margen de 1e-9 en valor absoluto. Ademas, si el
recorte coincide con el de P79, el par de conteos debe ser identico al de P79 dentro de
1e-9, y si no coincide, la diferencia se explica antes de seguir.
Exito: desvio distinto de cero, con la parte 1 declarada sin recorte.
Fallo: desvio exactamente cero, lo que significaria que el recorte Borra el efecto, y habria
que medirlo sobre el corpus entero antes de afirmar nada. STOP: no.

### A9. La identidad de la loss se cumple con el SIGNO correcto

Afirmacion: la loss del codigo es `-(pos + neg_codigo)`, luego la identidad correcta es
`l_codigo + pos_codigo + neg_codigo = 0`. Equivalente, `l_codigo == -pos - neg_codigo`. Como
`pos_codigo == pos_referencia` por A2, la identidad se puede escribir con cualquiera de los
dos terminos positivos.
Tolerancia: 1e-12 absoluta sobre `l_codigo + pos + neg_codigo`, en los cuatro K.
Exito: se cumple en los cuatro K, con el signo menos puesto.
Fallo: no se cumple. STOP: si, INMEDIATO. Si la identidad con el signo correcto falla, el
resto del informe de P80 no se escribe: un objetivo cuya composicion no cuadra no se audita
por partes. Se escribe un JSON de parada con el residuo y nada mas.

### A10. Control negativo: la identidad con el signo MAL puesto tiene que fallar

Afirmacion: la prueba de A9 tiene poder discriminante. La identidad que uso P79,
`pos_referencia + neg_codigo == l_codigo`, es falsa salvo que `pos + neg == 0`, y por tanto
tiene que FALLAR.
Tolerancia: separacion minima de 1e-6 en valor absoluto, en los cuatro K.
Exito: la identidad incorrecta falla en los cuatro K, con separacion mayor o igual que 1e-6.
Fallo: la identidad incorrecta tambien "se cumple". Eso significaria que `pos + neg` es
cercano a cero, y entonces la prueba de A9 no distingue una formula de su negation. En ese
caso la afirmacion A9 se marca INVALIDA por falta de poder discriminante, no CONFIRMADA.
STOP: si.

### A11. El artefacto congelado solo guarda la tabla de entrada

Afirmacion: `w2v_twitter_100d.npz` contiene exactamente dos claves, `W` y `palabras`, y una
sola matriz. `W` es `modelo.entrada.weight`, segun `p72_01:124-127`. El log de `p72_01:129`
anuncia `parametros = vocabulario * dim * 2`, porque el modelo tiene dos matrices, lo que no
significa que las dos se guarden.
Tolerancia: igualdad exacta del conjunto de claves. Se exige `set(claves) == {"W", "palabras"}`.
Exito: exactamente esas dos claves, forma `(77657, 100)`, `float32`, todos los valores
finitos, vocabulario unico, 0 filas nulas.
Fallo: hay una tercera clave, o la forma no es la declarada. STOP: si. Si aparece una tercera
clave con una matriz, A12 cambia de veredicto y hay que decirlo en el informe.

### A12. La matriz de salida NO se puede reconstruir

Afirmacion: no existe en el repositorio ninguna fuente que contenga la tabla de salida del
entrenamiento que produjo el congelado. La busqueda es EXHAUSTIVA y NO heuristica: se
recorren todos los ficheros `.npz` de `modelos/` y `features/`, todas las claves de todos los
`.json` de `reports/`, y se busca cualquier arreglo de forma `(n_vocab, dim)` o `(77657, d)`
distinto de `W`. Se exige el recuento de ficheros inspeccionados y el de candidatos
descartados, para que el negativo sea un dato y no una frase.
Tolerancia: no aplica. Es una enumeracion con cobertura declarada.
Exito: cero fuentes candidatas. El veredicto de A12 es CONFIRMADO.
Fallo: aparece una fuente, y entonces A13 pasa a ser NO APLICABLE y el hallazgo H4 se
reescribe con la fuente nueva. STOP: no, pero hay que avisar antes de publicar el informe.

### A13. Las cifras absolutas de loss y de gradiente sobre el artefacto NO son evidencia

Naturaleza: **NO APLICABLE POR CONSTRUCCION**. No es una medicion que P80 haga, es una
afirmacion que P80 deja sin hacer.

Afirmacion: como el artefacto no guarda la tabla de salida, cualquier evaluacion que
necesite `u_pos` y `u_neg` tiene que inventar esa matriz. P79 puso `E = S = W`. Los numeros
que salieron de ahi, `loss_codigo = 1.08412074`, `loss_referencia = 9.78555861`,
`norma del gradiente de entrada` 0.02042704 y 0.25426385, describen un modelo ficticio en el
que la tabla de salida es igual a la de entrada. NO son evidencia del estado final del
entrenamiento, y no pueden usarse para afirmar nada sobre la calidad del artefacto: ni para
decir que esta bien, ni para decir que esta mal.
Tolerancia: no aplica.
Exito: el JSON de P80 contiene el bloque del artefacto con la clave
`evidencia_del_estado_final: false` y el motivo escrito; y, si se incluyen cifras, cada una
lleva el prefijo literal `condicional_a_S_igual_a_W` y ninguna se cita en un veredicto.
Ademas, la frase de P79 de que "el ratio entre objetivos no depende del subconjunto, porque
es una propiedad de la formula" queda sustituida por esta, que es la correcta:
`el_ratio_entre_objetivos_es_propiedad_de_la_formula_para_ese_par_E_S, y_el_par_usado_no_es_el_del_entrenamiento`.
Fallo: que P80 reporte esas cifras como medida del artefacto entrenado, o que repita la
frase sustituida. STOP: no, pero es motivo de rechazo en la revision de codigo.

Lo que SI es admisible sobre el artefacto, y por tanto lo unico que P80 puede afirmar: la
integridad descriptiva de `W` de A11, y el hecho de que el artefacto lo produjo `p72_01`
usando la clase `SkipGram` de `p71_04` sin tocarla, que se comprueba leyendo el diccionario
`cx` de `p72_01:156-158` y verificando que `cx["SkipGram"]` es `base.SkipGram`.

### A14. Los BiGRU eligen epoca con el mismo pliegue que despues puntuan

Afirmacion: en `v9_04` y en `v10_11` la perdida de validacion que decide cuando restaurar
pesos es la del pliegue externo, y las probabilidades que se reportan salen de ese mismo
pliegue con esos mismos pesos restaurados. No hay particion interna. Es la fuga que
condeno P73 y P74.
Tolerancia: no aplica. Es una comprobacion estructural sobre el fuente, con lineas exactas.
Procedimiento, obligatorio: se leen los ficheros enteros, se localizan por linea exacta
1. la linea que inicializa `mejor_perdida` e `mejor_estado`,
2. la linea que compara la perdida de validacion y sobrescribe `mejor_estado`,
3. la linea `load_state_dict(mejor_estado)`,
4. la linea que devuelve las probabilidades de validacion,
y se comprueba que la variable de validacion, la misma en las cuatro, es la que devuelve la
funcion de entrenamiento. Ademas se comprueba que la funcion de entrenamiento NO recibe ni
construye una particion interna: cero apariciones de un segundo conjunto de identificadores
de validacion.
Exito: las cuatro piezas existen, la variable es la misma, y no hay particion interna. El
veredicto de H5 es CONFIRMADO.
Fallo: hay particion interna, con lo cual H5 queda REFUTADO y los numeros de V9-04 y
V10-11 serian comparables. STOP: no.
Consecuencia obligatoria del veredicto, sea cual sea: los numeros 0.694343 de V9-04 y
0.711639 de V10-11 se marcan como NO COMPARABLES con el protocolo actual, con un sesgo
optimista de magnitud NO MEDIDA. Y se registra la cadena literal con la que esos informes se
describen a si mismos, `early stopping estricto`, con su linea, porque es una descripcion
incorrecta. No se editan esos ficheros: la correccion va en un documento nuevo.

### A15. El padding se enmascara DESPUES de la GRU

Afirmacion: en `v9_04` y en `v10_11` la mascara se construye a partir de los identificadores
y se multiplica por la SALIDA de la GRU. La recurrencia consume los pasos de relleno. Con
`padding_idx=0` el vector de relleno es el vector cero, luego el efecto no es de explosion
sino de arrastre: los pasos de relleno se acumulan en los estados y, en la direccion inversa, se
mezclan en las posiciones reales anteriores al relleno.
Tolerancia: no aplica. Es una comprobacion estructural.
Procedimiento, obligatorio, exhaustivo y no heuristico: se lee el fichero entero y se
comprueba 1. el numero de apariciones de la cadena `pack_padded_sequence` es EXACTAMENTE 0,
2. el orden de las operaciones es GRU y despues multiplicacion por la mascara, con las dos
lineas y su numero,
3. el `padding_idx` del embedding es 0.
Exito: las tres comprobaciones. H6 queda CONFIRMADO en su parte estructural.
Fallo: aparece `pack_padded_sequence`, y H6 queda REFUTADO. STOP: si, porque entonces el
resto de la lectura de esos dos ficheros cambia deInterpretacion.

### A16. Cuanto pesa la contaminacion por relleno, con umbral fijado antes

Afirmacion: el defecto estructural de A15 tiene un peso medible, y el peso se mide antes de
decir nada. Sonda controlada, sin entrenar y sin datos del proyecto: se importa la clase
`BiGRU` o `BiGRUAfinable` del propio fichero, se instancia con la misma dimension y con
semilla fija, se toma UNA secuencia de tokens con longitud conocida, y se evalua el
vector agregado en dos formas: 1. con el relleno hasta la longitud real, 2. con el relleno
hasta el maximo de la tanda. Se mide la distancia relativa L2 entre los dos vectores
agregados, con los mismos pesos y la misma mascara.
Tolerancia y umbral de decision, fijados aqui: diferencia relativa menor o igual que 1e-3 se
considera SIN EFECTO MEDIBLE; mayor que 1e-3 se considera EFECTO MEDIBLE.
Exito: la sonda se ejecuta y devuelve un numero, con los dos vectores y la distancia
registrados. Este es el UNICO punto del pre-registro donde el veredicto no esta fijado de
antemano, y por eso el umbral queda aqui, antes de ejecutar, y no se negocia despues.
Fallo: distancia no calculable, o variacion no nula entre dos corridas con la misma semilla.
STOP: no. Si sale SIN EFECTO MEDIBLE, H6 se cierra como `presente_en_el_codigo_sin_efecto_
medible_a_1e-3` y NO se afirma contaminacion de los resultados. Si sale EFECTO MEDIBLE, se
afirma la contaminacion y su magnitud queda acotada por la sonda, no por los BiGRU reales.

### A17. DL-03 entreno sobre el conjunto completo de etiquetas, holdout v2 incluido

Afirmacion: el pool de validacion cruzada de la fase DL son las 1731 filas etiquetadas
completas, no las 1483 del desarrollo. El holdout v2 de 248 filas estaba dentro del pool, y
por tanto cada pliegue de su OOF incluye filas que el protocolo actual reserva.
Tolerancia: CERO. Igualdad entera exacta.
Procedimiento, obligatorio, y con dos vias que deben concordar:
1. Via numerica, la que decide: en `reports/dl_desde_cero_resultados.json`, sumar los
   `n_validacion` de los cinco pliegues de DL-03. El resultado debe ser exactamente 1731, y
   se comprueba que 1731 = 1483 + 248 con los dos recuentos publicados en
   `config/holdout_v2.json`. Si la suma fuera 1483, H7 quedaria REFUTADO.
2. Via estructural, que apoya pero no decide: en `experiments/11_dl_desde_cero.py`, la
   llamada que construye el pool de CV no excluye el holdout, y el numero de filas que
   registra por pliegue es coherente con la via 1. Se exige citar la linea exacta del
   `read_train` y la del bucle de CV, con su texto literal.
Exito: la suma es 1731 y la identidad 1731 = 1483 + 248 se cumple exactamente. H7 queda
CONFIRMADO.
Fallo: la suma es 1483, o cualquier otro numero que no se descomponga como dev mas holdout.
STOP: si. Si la suma no cuadra, no se emite el informe de DL hasta entenderlo.
Nota de procedencia, que se registra sin corregir el informe viejo: el campo `rutas` del JSON
apunta a otro arbol de ficheros. El fichero de codigo SI existe aqui, luego la via 2 es
posible y no queda como indeterminada.

### A18. Lo mismo ocurre en DL-01 y DL-02, y su OOF no puede leerse como mejora sobre V03

Afirmacion, en dos partes.
Parte 1: los tres informes de la fase DL declaran los mismos `n_validacion` por pliegue, con
lo que el pool es el mismo y la contaminacion del holdout alcanza a los tres, no solo a
DL-03. H7 se extiende a DL-01 y DL-02 sin cambiar de naturaleza.
Parte 2, y es la que hay que dejar escrita: DL-01 declara
`f1_media_columnas = 0.7948542132994776` en OOF, que es SUPERIOR al `corrected_cv` de V03,
0.7772114525. Por A17 y por A19 ese numero esta contaminado por el holdout y por una
seleccion de epoca no anidada, y NO puede leerse como que un modelo supera a V03. La
comparacion no es valida porque los dos numeros no se calcularon sobre el mismo conjunto de
filas ni con el mismo protocolo.
Tolerancia: la parte 1 es igualdad exacta de los cinco `n_validacion` entre los tres
informes. La parte 2 es una identidad aritmetica: la diferencia se reporta con 12 decimales
y NO se etiqueta como delta, ni como mejora, ni como ventaja.
Exito: parte 1 verificada, y la parte 2 escrita con la advertencia completa en el informe y
en el markdown.
Fallo: los `n_validacion` difieren entre arquitecturas. STOP: no.

### A19. La epoca de la fase DL tambien se elige con el pliegue externo

Afirmacion: el entrenamiento de la fase DL restaura los pesos de la epoca con mejor
metrica de VALIDACION del propio pliegue que despues se puntua, sin particion interna. No
es early stopping anidado, aunque el informe lo describa como `early stopping estricto`.
Tolerancia: no aplica. Comprobacion estructural con lineas exactas.
Procedimiento: se localiza en `experiments/11_dl_desde_cero.py` la seccion que declara que
la mejor epoca es la de mejor metrica de validacion, la linea que restaura los pesos, y la
linea que devuelve la probabilidad de validacion; y se comprueba que las tres se refieren al
mismo pliegue. Se exige registrar la cadena literal `early stopping estricto` con su linea.
Exito: las tres piezas, mismo pliegue, sin particion interna. H7 queda CONFIRMADO en su
segunda mitad, y el sesgo de los tres OOF es optimista por dos causas independientes.
Fallo: existe particion interna, y H7 queda REFUTADO en esta parte. STOP: no.

---

## 6. Lo que P80 deja written para otros, sin ejecutarlo

P80 deja tres afirmaciones corregidas, en un documento NUEVO, sin tocar P79:

1. `semillas_desalineadas` de P79 es un artefacto de su heuristica de etiquetado. El
   veredicto real lo da A7.
2. `la_identidad_se_cumple: false` de P79 no era un defecto del codigo de P71-04, era un
   signo mal puesto en la propia prueba. El veredicto real lo da A9.
3. `punto_5_artefacto_congelado` de P79 no mide el artefacto entrenado. El veredicto real
   lo da A13, por construccion.

Ninguna de las tres correcciones cambia los hallazgos de H1 que P79 si acerto.

---

## 7. Criterio de parada

P80 para, sin escribir el informe completo, y avisa a Carlos, si:

1. El sha256 del artefacto congelado no coincide con el declarado AL EMPEZAR. No se sigue.
2. El sha256, el tamano o el `st_mtime_ns` del artefacto cambian ENTRE la primera y la ultima
   lectura. El informe se marca invalido.
3. Falla A1, A2, A3, A4, A9, A10, A11, A15 o A17, porque son las que hacen STOP. En A9 y A10
   no se emite ningun otro numero.
4. El iterador del scheduler no es de las dos formas admitidas de A6. Se emite un JSON de
   parada con el nodo del AST y no hay veredicto para H2.
5. A12 encuentra una fuente con la tabla de salida. Se para antes de publicar, porque
   entonces el hallazgo H4 cambia de contenido.
6. El scheduler necesita una particion de holdout para contar lotes. NO se pide. P80 cuenta
   sobre el corpus recortado, con el recorte declarado, y la parte 1 de A8 es la que no
   depende del recorte.
7. Aparece un numero sospechosamente perfecto, por ejemplo una perdida negativa o un ratio
   de gradiente mayor que 1. Se investiga replicacion antes de reportar.
8. El script intenta escribir fuera de `reports/`, `figuras/p80/` y el propio script. Cualquier
   otra escritura es un fallo de fronteras.
9. Se agota el presupuesto. Se entrega lo hecho, con su estado real, sin rellenar huecos.

P80 ademas ABORDA, sin excepcion, si su salida contiene un byte mayor que 127, verificado
byte a byte y no solo con el verificador oficial.

---

## 8. Salidas esperadas

- `experiments/p80_01_auditoria_w2v.py`, script unico y completo, con las funciones
  `auditar_semillas_por_ast`, `test1_loss`, `test2_gradientes`, `test3_scheduler`,
  `test4_identidad_con_signo`, `test5_artefacto`, `test6_bigru`, `test7_fase_dl` y un
  `main` que las ejecuta en orden y para en el primer STOP.
- `reports/p80_auditoria_w2v.json`, con una entrada por afirmacion, cada una con su
  tolerancia, su medida, su exito o su fallo, y su veredicto en uno de los tres estados
  permitidos.
- `reports/p80_auditoria_w2v.md`, con el veredicto por hallazgo, la correccion de las tres
  afirmaciones de P79, y la frase de formulacion literal.
- `figuras/p80/`, como maximo tres figuras: el ratio frente a 1/K en escala log, el learning
  rate real frente al ideal, y la sonda de relleno. Si una figura no aporta, no se genera.
- Una entrada en `DECISIONS.md`, un bloque en `RUNLOG.md` y la actualizacion de
  `PROJECT_STATE.md`.
- Todo el texto nuevo en ASCII puro.

---

## 9. Formula de cierre

Al terminar, y sea cual sea el resultado:

    V03 remains the best verified leaderboard result.

Si P80 confirma H1, H2, H4, H6 y H7, la conclusion sobre el artefacto congelado es que es
utilizable pero PESIMISTA, no invalido, y que los resultados de V9-04, V10-11 y de la fase DL
no son comparables con el protocolo actual. En ningun caso se afirma que Word2Vec sea mejor
que V03, y en ningun caso se propone reentrenar sin que lo decida Carlos.
