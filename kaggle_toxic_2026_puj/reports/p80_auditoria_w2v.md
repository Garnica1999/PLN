# P80: auditoria del entrenador Word2Vec, sin repetir los errores de P79

## Formulacion

    V03 remains the best verified leaderboard result.

En ningun caso se afirma que Word2Vec sea mejor que V03, ni que una RNN o un modelo bayesiano lo superen.

## Pre-registro

- fichero: `docs/p80_preregistro_auditoria_w2v.md`
- sha256 declarado: `440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f`
- sha256 recomputado: `440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f`
- coincide: True | commit `8726c9f`

## Veredicto por hallazgo

| Hallazgo | Estado | Afirmaciones |
|---|---|---|
| H1 | CONFIRMADO | A1, A2, A3, A4, A5 |
| H2 | CONFIRMADO | A6, A7, A8 |
| H3 | CONFIRMADO | A9, A10 |
| H4 | CONFIRMADO | A11, A12, A13 |
| H5 | CONFIRMADO | A14 |
| H6 | CONFIRMADO | A15, A16 |
| H7 | CONFIRMADO | A17, A18, A19 |

Un hallazgo queda CLASIFICADO, no medido, en cuatro estados y solo en cuatro: CONFIRMADO, REFUTADO, NO APLICABLE POR CONSTRUCCION y NO_EVALUABLE. NO APLICABLE significa que la pregunta no tiene respuesta posible con los datos que existen. NO_EVALUABLE significa que la comprobacion no llego a ejecutarse, que es distinto de que se ejecutara y saliera mal.

## Las diecinueve afirmaciones

| Id | Hallazgo | Naturaleza | Tolerancia | Veredicto |
|---|---|---|---|---|
| A1 | H1 | CONFIRMAR | 1e-12 | **CONFIRMADO** |
| A2 | H1 | CONFIRMAR | 1e-12 | **CONFIRMADO** |
| A3 | H1 | CONFIRMAR | 1e-12 | **CONFIRMADO** |
| A4 | H1 | CONFIRMAR | componentes: 1e-06 relativa; residuos: 1e-09 absoluta | **CONFIRMADO** |
| A5 | H1 | REFUTAR | no aplica: es una correccion de una asercion indebida, no una medicion | **CONFIRMADO** |
| A6 | H2 | CONFIRMAR | igualdad exacta; cero margen | **CONFIRMADO** |
| A7 | H2 | CONFIRMAR | igualdad exacta de conjuntos; que sean iguales es FALLO | **CONFIRMADO** |
| A8 | H2 | CONFIRMAR | desvio_relativo: |desvio| > 1e-09; conteo_de_pares: 0, igualdad exacta con P79 si el recorte coincide | **CONFIRMADO** |
| A9 | H3 | CONFIRMAR | 1e-12 | **CONFIRMADO** |
| A10 | H3 | CONFIRMAR | separacion minima 1e-06 en valor absoluto | **CONFIRMADO** |
| A11 | H4 | CONFIRMAR | igualdad exacta del conjunto de claves; cero margen | **CONFIRMADO** |
| A12 | H4 | CONFIRMAR | no aplica: es una enumeracion con cobertura declarada | **CONFIRMADO** |
| A13 | H4 | NO APLICABLE POR CONSTRUCCION | no aplica | **NO APLICABLE POR CONSTRUCCION** |
| A14 | H5 | CONFIRMAR | no aplica: comprobacion estructural con lineas exactas | **CONFIRMADO** |
| A15 | H6 | CONFIRMAR | no aplica: comprobacion estructural | **CONFIRMADO** |
| A16 | H6 | CONFIRMAR | umbral_de_decision: 0.001; regla: menor o igual que el umbral es SIN EFECTO MEDIBLE; mayor que el umbral es EFECTO MEDIBLE; fijado: en el pre-registro, antes de ejecutar, y no se negocia despues | **CONFIRMADO** |
| A17 | H7 | CONFIRMAR | cero, igualdad entera exacta | **CONFIRMADO** |
| A18 | H7 | CONFIRMAR | parte_1: igualdad exacta de los cinco n_validacion entre los tres informes; parte_2: identidad aritmetica, reportada con 12 decimales | **CONFIRMADO** |
| A19 | H7 | CONFIRMAR | no aplica: comprobacion estructural con lineas exactas | **CONFIRMADO** |

## H1: la loss no es SGNS, y el peso exacto es 1/K

- A3, desviacion maxima de 1/K en los K con evidencia: `4.163336342344337e-17`, con tolerancia `1e-12`. En K = 1 el ratio vale 1 y no cuenta como evidencia.
- A3, linea citada: `experiments/p71_04_entrenar_word2vec.py:215`
- A3, texto: `neg_l = torch.nn.functional.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).mean()`
- A4, maximo residuo de la descomposicion: `3.469446951953614e-17`, con tolerancia `1e-09`
- A4, los dos componentes se calculan en funciones separadas y cada una devuelve solo su termino. Si una sola devolviera el total, el componente negativo llevaria tambien el positivo, la suma lo contaria dos veces y la razon de los negativos no seria 1/K.
- A5, la razon de la norma total NO es una asercion: se publica con la etiqueta `no_es_una_asercion`, sin ninguna cota, y con dos contraejemplos numericos que la refutan

## H2 y H3: el desfase del scheduler y la identidad de la loss

- A7, conjunto del plan: `Lista(afirmaciones/A7/medida/conjunto_plan, elementos=5)`
- A7, conjunto del entrenamiento: `Lista(afirmaciones/A7/medida/conjunto_entrenamiento, elementos=5)`
- A8 parte 1, sin recorte: las semillas difieren en todas las epocas: `True`, y la primera semilla del plan no aparece en el entrenamiento: `True`
- A8 parte 2, con recorte declarado de 2000 lineas, 53243 tokens y lote 256: `total_lotes_plan` = `1395`, `total_lotes_recorridos` = `1398`, desviacion relativa `0.002150537634408602`
- A8, comparacion con P79: `True`. el par de conteos coincide con el de P79 sobre el mismo recorte, luego estos dos numeros quedan replicados. Lo que P80 corrige de P79 no es esta cuenta, sino el procedimiento que produjo la lista de semillas
- A9, residuo maximo de `l_codigo + pos + neg_codigo = 0`: `1.1102230246251565e-16`, con tolerancia `1e-12`
- A10, separacion minima de la identidad MAL puesta: `2.771907574551194`. Tiene que fallar, y falla.

## H4: el artefacto congelado

- A11, claves del npz: `Lista(afirmaciones/A11/medida/claves, elementos=2)`, y una sola matriz
- A12, ficheros `.npz` inspeccionados por cabecera: 19, con 412987 bytes leidos en total y sin descomprimir ningun array. Ficheros JSON de reports/: 320, con no disponible claves recorridas.
- A12, candidatos con forma de tabla de embeddings: `0`
- A13, `evidencia_del_estado_final = False`. La tabla de salida NO EXISTE y no se sustituye por `W`. En consecuencia, las cifras absolutas de P79 sobre el artefacto describen un par (E, S) que nunca existio, y P80 no publica ninguna cifra sobre el artefacto.
- A13, frase correcta: `el_ratio_entre_objetivos_es_propiedad_de_la_formula_para_ese_par_E_S, y_el_par_usado_no_es_el_del_entrenamiento`

## H5 y H6: los BiGRU

- A14, los numeros V9-04 0.694343, V10-11 0.711639 quedan marcados como NO COMPARABLES con el protocolo actual, con un sesgo optimista de magnitud NO MEDIDA
- A14, v9_04: los cuatro criterios pasan
- A14, v10_11: los cuatro criterios pasan
- A14 y A19, la cadena `early stopping estricto` con la que esos informes se describen a si mismos es una descripcion incorrecta: no hay particion interna, luego no es early stopping anidado. Los ficheros NO se editan; la correccion va aqui.
- A15, `pack_padded_sequence` aparece 0.0 veces en los dos ficheros, y la mascara se aplica DESPUES de la GRU
- A16, sonda de relleno: estado EJECUTADA, resultado EFECTO_MEDIBLE, con el umbral fijado en 0.001 antes de ejecutar

## H7: la fase DL

- A17, `n_validacion` por pliegue de DL-03: `Lista(afirmaciones/A17/medida/via_numerica_que_decide/n_validacion_por_pliegue_DL_03, elementos=5)`, suma `1731`, y `1483 + 248 = 1731`. El holdout v2 estaba dentro del pool.
- A17, el pool de CV se localiza por el hecho de repartir un objeto que viene de `read_train()`, y no tomando la ultima llamada `split`: un `linea.split()` de una cadena es indistinguible por forma de un `divisor.split(train, ...)`. Todas las candidatas quedan listadas en el JSON con su clasificacion.
- A17, el holdout se lee con expresion regular sobre el texto, sin deserializar el documento: `documento_deserializado = False`
- A18, los tres informes declaran los mismos `n_validacion`: `True`, luego la contaminacion alcanza a los tres
- A18, el OOF de DL-01 es `0.7948542132994776` y el `corrected_cv` de V03 es `0.7772114525165721`. La diferencia aritmetica es `0.017642760783` y NO se etiqueta como delta, mejora ni ventaja: los dos numeros no se calcularon sobre el mismo conjunto de filas ni con el mismo protocolo.
- A19, la epoca de la fase DL tambien se elige con el pliegue externo: el sesgo de los tres OOF es optimista por dos causas independientes. El campo `no_es_early_stopping_anidado` se calcula con los mismos cuatro criterios que A14, de modo que no puede contradecir al veredicto.

## Correccion de las tres afirmaciones de P79

### lectura de las semillas del scheduler: semillas_desalineadas = false, con una unica lista entrenamiento = [0, 1, 2, 3, 4, 5]

- por que estaba mal: su heuristica metio los dos bucles en el mismo grupo, y el booleano salia False no porque los rangos coincidieran, sino porque solo habia un grupo. Su propia lista tiene 6 elementos para EPOCAS = 5, y eso ya lo contradicia
- veredicto real: A7: CONFIRMADO
- lo que dice P80: los dos bucles usan rangos distintos, leidos por AST con mapa de padres, sin buscar hacia atras un encabezado ni etiquetar por una ventana de cadena

### la_identidad_se_cumple = false en los cuatro valores de K

- por que estaba mal: comparaba pos + neg_codigo contra l_codigo, que es -(pos + neg_codigo): faltaba el signo menos. Y el script no paraba, con lo que un indicador que siempre falla no informa de nada
- veredicto real: A9: CONFIRMADO
- lo que dice P80: la identidad se afirma con el signo correcto y con parada, y ademas se comprueba que la identidad mal puesta falla, para que la asercion tenga poder discriminante

### punto_5_artefacto_congelado, que mide loss y normas de gradiente sobre el artefacto

- por que estaba mal: asigno E = S = W, la misma matriz guardada, a la tabla de entrada y a la de salida. El artefacto solo guarda W
- veredicto real: A13: NO APLICABLE POR CONSTRUCCION
- lo que dice P80: no se mide el artefacto entrenado porque no se puede: se declara el limite y la afirmacion queda NO APLICABLE POR CONSTRUCCION, sin sustituir la tabla que falta por W

P80 no edita P79 ni su informe. La correccion va en un documento nuevo, y ninguna de las tres correcciones cambia los hallazgos de H1 que P79 si acerto.

## Limites declarados

- A12 exige recorrer todos los .npz de modelos/ y features/ y todos los .json de reports/, mientras que la seccion 2.3 punto 5 prohibe abrir otros ficheros de modelos/. P80 resuelve la tension por la via menos riesgosa: de cada .npz se leen solo las cabeceras .npy dentro del ZIP, sin descomprimir los arrays, y se declara el numero de bytes leidos
- la enumeracion de A12 no incluye los checkpoints .pt de models/ ni los ficheros de modelos/ que no son .npz: es un limite declarado del negativo, no un olvido
- P80 lee de config/holdout_v2.json exclusivamente n_tweets_dev, n_tweets_holdout y hash_ids_csv, y lo hace con expresion regular sobre el texto, sin deserializar el documento: la lista de identificadores no llega a existir como objeto
- la parte 1 de A8 establece el desfase de semillas con aritmetica entera, que es independiente del corpus; el valor numerico de la desviacion se mide con el recorte declarado y no se extrapola a la corrida real
- la convencion que clasifica nombres de validacion y la que detecta llamadas de reparto se publican en el informe, con sus expresiones regulares, para que sean auditables. La segunda ignora los nombres que empiezan por mayuscula, porque esos son constructores de clase y no repartidores
- el calculo del learning rate de A8 usa el LOTE de p72_01, que coincide con el recorte declarado, y el learning rate inicial como LR0_POR_EJEMPLO por LOTE

## Artefacto congelado: integridad

- sha256 declarado: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`
- antes de la primera lectura: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`
- despues de la ultima lectura: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`
- intacto en sha256, tamano y mtime_ns: True

## Conclusion


En ningun caso se afirma que Word2Vec sea mejor que V03, y en ningun caso se propone reentrenar sin que lo decida Carlos.

    V03 remains the best verified leaderboard result.

