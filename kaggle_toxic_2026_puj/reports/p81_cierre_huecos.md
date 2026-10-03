# P81: cierre de los cinco huecos que dejo P80

## Formulacion

    V03 remains the best verified leaderboard result.

En ningun caso se afirma que Word2Vec sea mejor que V03, ni que una RNN o un modelo bayesiano lo superen, ni que el artefacto sea mejor o peor que el de una libreria.

## Los dos documentos vinculantes

- pre-registro: `docs/p81_preregistro_cierre_huecos.md`
- sha256 declarado: `55e6900cce49bd7e2ee98b00e0c315081c04718edb18df4abd49f484755faabc`
- sha256 recomputado: `55e6900cce49bd7e2ee98b00e0c315081c04718edb18df4abd49f484755faabc`
- coincide: True | commit `86a9140`
- enmienda 1: `docs/p81_enmienda_1.md`
- sha256 declarado: `496223801147a414e74a53fb7cf407ae48ad4671c74e45daba4dce19f15801c6`
- sha256 recomputado: `496223801147a414e74a53fb7cf407ae48ad4671c74e45daba4dce19f15801c6`
- coincide: True | commit `c3dee26`

Que cambia la enmienda 1: sustituye el learning rate fijo del bloque 1 por una rejilla declarada: 0.02, 0.05, 0.10, 0.20, 0.35 y 0.50, con 1500 pasos, dimension 32, K 10, SGD puro sin momentum y tres semillas

## Veredicto de las veinte afirmaciones

| Id | Hueco | Naturaleza | STOP | Veredicto |
|---|---|---|---|---|
| A0 | todos | CONFIRMAR | True | **CONFIRMADO** |
| A1 | 1 | CONFIRMAR | False | **CONFIRMADO** |
| A2 | 1 | CONFIRMAR | False | **CONFIRMADO** |
| A3 | 1 | CONFIRMAR | True | **CONFIRMADO** |
| A4 | 1 | CONFIRMAR | True | **CONFIRMADO** |
| A5 | 1 | CONFIRMAR | True | **CONFIRMADO** |
| A6 | 1 | CONFIRMAR | True | **REFUTADO** |
| A7 | 1 | CONFIRMAR | True | **REFUTADO** |
| A8 | 2 | CONFIRMAR | False | **CONFIRMADO** |
| A9 | 2 | CONFIRMAR | True | **CONFIRMADO** |
| A10 | 2 | CONFIRMAR | False | **CONFIRMADO** |
| A11 | 2 | CONFIRMAR | False | **CONFIRMADO** |
| A12 | 2 | CONFIRMAR | True | **CONFIRMADO** |
| A13 | 3 | CONFIRMAR | True | **CONFIRMADO** |
| A14 | 3 | CONFIRMAR | False | **CONFIRMADO** |
| A15 | 4 | CONFIRMAR | True | **CONFIRMADO** |
| A16 | 4 | CONFIRMAR | True | **CONFIRMADO** |
| A17 | 4 | CONFIRMAR | False | **CONFIRMADO** |
| A18 | 5 | CONFIRMAR | False | **CONFIRMADO** |
| A19 | 5 | CONFIRMAR | True | **CONFIRMADO** |

El estado de un hallazgo es uno de cuatro y solo de cuatro: CONFIRMADO, REFUTADO, NO APLICABLE POR CONSTRUCCION y NO_EVALUABLE. NO_EVALUABLE significa que la comprobacion no llego a ejecutarse, que es distinto de que se ejecutara y saliera mal. Si algo no se puede evaluar se dice que no se pudo evaluar y se marca NO_EVALUABLE con la razon: no se convierte en un CONFIRMADO ni se omite.

## Bloque 1: el ancla ejecutable, y por que la curva no basta

**Titular del bloque 1: EL_TEST_NO_APOYA_A_LA_SUMA.** A5 afirma que el test discrimina por MAGNITUD, no por direccion. A6 mide la direccion y la direccion es CONTRARIA a la hipotesis: en el primario la MEDIA gana a la SUMA por 0.700000. El bloque 1 NO apoya al objetivo de la referencia, y se publica con esas palabras

- A5 verifica la veredicto CONFIRMADO y A6 verifica la veredicto REFUTADO. La direccion observada es favor_a_la_media y el titular apoya a la SUMA: False
- comprobacion de coherencia: un A5 CONFIRMADO con la direccion en contra no se publica como apoyo a la SUMA: A5 afirma magnitud, A6 afirma direccion

- A1, CONFIRMADO: la ecuacion (4) de la seccion 2.2, Negative Sampling de Mikolov et al. 2013, arXiv 1310.4546v1 [cs.CL], DOI 10.48550/arXiv.1310.4546, se transcribe literalmente y las cuatro comprobaciones de forma dan verdadero
- A2, CONFIRMADO: el fragmento anclado del codigo C original no contiene la cadena `/ negative` y acumula el gradiente una vez por cada negativo, luego los negativos aportan `negative` veces y no una
- A3, CONFIRMADO: ESTA_CONVENCION_ES_COHERENTE_CON_P79_Y_SIGUE_SIENDO_UNA_REIMPLEMENTACION_PROPIA

    ESTA_CONVENCION_ES_COHERENTE_CON_P79_Y_SIGUE_SIENDO_UNA_REIMPLEMENTACION_PROPIA
- A4, CONFIRMADO: el corpus plantado tiene 23004 pares y 0 pares entre familias distintas, que es el cero exacto que exige el pre-registro
- A5, CONFIRMADO: diferencia de M1 entre los dos mejores de la rejilla `-0.7000000000000001`, banda muerta 0.005, minimo 0.02
- A6, REFUTADO: REFUTADO: la direccion es CONTRARIA a la hipotesis. La MEDIA gana por 0.700000 en el primario, y el margen exigido es de 0.02 a favor de la SUMA. Esto no se reinterpreta y no se cambia el umbral: un REFUTADO aqui es informacion, no un problema que se arregla ajustando el criterio
- A7, REFUTADO: el efecto no se sostiene: ['el conteo de victorias por semilla', 'el control negativo tambien recupera estructura']

### La curva completa del learning rate

| learning rate | M1 MEDIA | M1 SUMA | diferencia | victoria de la SUMA | zona betonal |
|---|---|---|---|---|---|
| 0.02 | 0.0 | 0.0 | 0.0 | False | no |
| 0.05 | 0.0 | 0.0 | 0.0 | False | no |
| 0.1 | 0.005555555555555556 | 0.03888888888888889 | 0.03333333333333333 | True | no |
| 0.2 | 0.7388888888888889 | 1.0 | 0.26111111111111107 | True |  SUMA |
| 0.35 | 1.0 | 1.0 | 0.0 | False | MEDIA SUMA |
| 0.5 | 1.0 | 1.0 | 0.0 | False | MEDIA SUMA |

Puntos excluidos del primario, por brazo, con su numero y sin borrar nada:

- MEDIA: learning rates excluidos [0.35, 0.5]; mejor no betonal 0.7388888888888889 en learning rate 0.2
- SUMA: learning rates excluidos [0.2, 0.35, 0.5]; mejor no betonal 0.03888888888888889 en learning rate 0.1
- diferencia con el criterio por brazo: -0.7000000000000001

La lectura ALTERNATIVA por punto, que se calcula y se publica pero NO se usa:

- learning rates que se cairian enteros: [0.2, 0.35, 0.5]
- learning rates que sobrarian: [0.02, 0.05, 0.1]
- mejor MEDIA 0.005555555555555556 contra mejor SUMA 0.03888888888888889, diferencia 0.03333333333333333
- las dos lecturas coinciden en signo: False

Si las dos lecturas NO coinciden en signo, la eleccion del criterio deja de ser cosmetica y por eso se declara y se publica la alternativa con sus numeros.

La curva se publica COMPLETA aunque el primario salga a favor. Publicar solo el primario seria el modo de fallo que la enmienda 1 viene a evitar.

### El recuento de victorias por learning rate

- victorias de la SUMA: 2 de 6
- reparto: 2 a 4
- la curva es monotona: False, con 2 cambios de signo
- lectura del titular: ZONA DE NO CONCLUSION: el reparto de victorias por learning rate queda en 2 a 4 y eso es ruido con tres semillas. El primario NO debe leerse como confirmacion

El primario NO debe leerse como confirmacion si el reparto de victorias por learning rate queda en la banda muerta de 2 a 4, 3 a 3 o 4 a 2. Con tres semillas eso es ruido, no un hallazgo. El recuento se publica aunque salga a favor del primario.

### La zona betonal y como se excluye

- criterio aplicado: POR_BRAZO, umbral 0.99
- la zona betonal se excluye POR BRAZO y no por punto. Un punto de la rejilla son dos mediciones, una por objetivo: que la MEDIA sature en un learning rate es un hecho sobre el brazo MEDIA y no invalida la medicion de la SUMA en ese mismo learning rate. El criterio por punto obliga a comparar una cifra saturada con otra que no lo esta, y deja al primario sin ningun punto en cuanto un solo brazo satura. Se publica tambien la lectura por punto, con sus numeros, porque las dos lecturas no coinciden y elegir solo una seria ocultar la otra

### El limite permanente del bloque 1

El anclaje a una implementacion externa de SGNS queda SIN HACER. La evidencia de P81 sobre H1 es un ancla DOCUMENTAL, que es una cita verificable, y un ancla EJECUTABLE, que es un experimento discriminante con estructura plantada. Ninguno de los dos es una corrida de una implementacion de terceros. Lo que P81 autoriza es afirmar que el objetivo del trainer NO es el SGNS publicado, porque promedia un termino que la referencia suma. Lo que P81 NO autoriza es decir que el resultado del trainer es peor que el de word2vec o el de gensim.

## Bloque 2: relleno y OOV, celdas separadas

- A8, CONFIRMADO: 0.04124873567093729 de las posiciones son OOV y 0.7309191672285907 son de relleno, medido con v9_04.cargar_vec, v9_04.normalizar_emb_normal, v9_04.tokenizar y v9_04.secuencias, con v9_04.MAX_LEN
- A8, la cifra de 11.2726 por ciento de la discusion de auditoria NO queda reproducida, con la variante ninguna. Las tres cifras de cobertura que se han citado en la discusion NO se arrastran: se recalculan con el tokenizador real de v9_04
- A9, CONFIRMADO: id_relleno 0 e id_oov 3440, con forma de matriz [3441, 200] y residuo de la fila de relleno 0.0
- A10, CONFIRMADO: las 6 magnitudes estan presentes, con su clase y su condicion, y hay al menos una cifra distinta de cero por condicion y por clase
- A11, CONFIRMADO: el canal de OOV es al menos tan grande como el de relleno en las dos clases
- A12, CONFIRMADO: variacion maxima entre dos pasadas con la misma semilla 0.0, que tiene que ser cero exacto

### Las tres distancias por condicion y por clase

| clase | condicion | distancia relativa L2 | etiqueta |
|---|---|---|---|
| v9_04_BiGRU | C0 | 0.0920078456401825 | EFECTO MEDIBLE |
| v9_04_BiGRU | C1 | 0.0 | SIN EFECTO MEDIBLE |
| v9_04_BiGRU | C2 | 0.11460871249437332 | EFECTO MEDIBLE |
| v9_04_BiGRU | C2b | 0.11460871249437332 | EFECTO MEDIBLE |
| v9_04_BiGRU | C3 | 0.0 | SIN EFECTO MEDIBLE |
| v9_04_BiGRU | C2_menos_C2b | 0.0 | SIN EFECTO MEDIBLE |
| v10_11_BiGRUAfinable | C0 | 0.0920078456401825 | EFECTO MEDIBLE |
| v10_11_BiGRUAfinable | C1 | 0.0 | SIN EFECTO MEDIBLE |
| v10_11_BiGRUAfinable | C2 | 0.11460871249437332 | EFECTO MEDIBLE |
| v10_11_BiGRUAfinable | C2b | 0.11460871249437332 | EFECTO MEDIBLE |
| v10_11_BiGRUAfinable | C3 | 0.0 | SIN EFECTO MEDIBLE |
| v10_11_BiGRUAfinable | C2_menos_C2b | 0.0 | SIN EFECTO MEDIBLE |

Ninguna cifra agregada: P80 publico una sola distancia y eso impedia separar canales.

## Bloque 3: reejecion no es replicacion

- A13, CONFIRMADO: el bloque se llama reejecion_no_replicacion, lleva tipo_de_evidencia igual a REEJECION y no hay ninguna clave llamada coincide
- A14, CONFIRMADO: veredicto por la regla REEJECION para el par P79, P80

Los cinco requisitos de una replicacion genuinea, con su veredicto para el par P79, P80:

| Requisito | Texto | Cumple |
|---|---|---|
| R1 | implementacion independiente: el numero se obtiene por un camino de codigo que no es una reimportacion ni una reejecucion del procedimiento del otro informe | False |
| R2 | datos independientes: recorte distinto, o corpus completo. El mismo recorte NO cumple R2 | False |
| R3 | estimulo independiente: semilla y lote distintos de los del otro informe | False |
| R4 | extraccion independiente: el numero no se lee del informe del otro para contrastar, se recalcula desde el origen | False |
| R5 | tolerancia declarada: la coincidencia se acepta dentro de una tolerancia escrita antes, no por igualdad de decimales | True |

## Bloque 4: por identificadores, no por tamano

- A15, CONFIRMADO: conteos [347, 346, 346, 346, 346] frente a los publicados [347, 346, 346, 346, 346], union 1731 identificadores y solapamiento maximo [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- A16, CONFIRMADO: el residuo tiene exactamente 248 elementos, es disjunto del desarrollo y es EXACTAMENTE el conjunto de indices que holdout_v2 devuelve como holdout, elemento a elemento
- A16, el hash_ids_csv de config/holdout_v2.json NO se pudo vincular a las 248 filas por esa via: ninguna de las tres convenciones declaradas coincide. Se publican los tres hash recalculados y se declara como limite. La identidad de conjuntos de esta afirmacion se sostiene igual, porque no depende de ese hash
- A17, CONFIRMADO: el limite declarado esta escrito y los cinco sha256 por pliegue estan publicados

El limite de A17, declarado y obligatorio:

    LOS_TRES_INFORMES_DE_LA_FASE_DL_NO_PERSISTEN_LOS_IDS_POR_PLIEGUE_LA_IDENTIDAD_A_NIVEL_DE_IDS_SE_ESTABLECE_POR_RECALCULO_DETERMINISTA_DEL_CAMINO_DE_CODIGO_COMPARTIDO_Y_NO_POR_LECTURA_DE_IDS_PERSISTIDOS

Para A15 a A17 solo se tocan identificadores. El holdout no se abre, ni su texto, ni sus etiquetas, ni su fichero de identificadores.

### Los cinco sha256 por pliegue

| Pliegue | n filas | sha256 de los ids ordenados | primer id | ultimo id |
|---|---|---|---|---|
| 1 | 347 | `130fa7205f21d354b4cbd42e9801bcf5cce81cf7a8fae8735f5ad81a1144556a` | 20003 | 24989 |
| 2 | 346 | `a091c074e62aa76a2568dde673c97b3932bc2e89f098ad171bd39c8a2d0a2abc` | 20008 | 24999 |
| 3 | 346 | `77dc231255bb27339d4a5705cdc730b6b7e5bc6ee6e550c9b3d9fa9c083fcb14` | 20021 | 24996 |
| 4 | 346 | `a95a9d9482eb3c7550246b5fcb0c3ea4ebe2f0e5ced63c73ab7b86cbd42e23ff` | 20002 | 24992 |
| 5 | 346 | `1ee71732b9219aef9004352c938bcbad427da86afdf55d328a114d5e436d2ccb` | 20007 | 24981 |

## Bloque 5: el impacto del desfase del scheduler, paso a paso

- A18, CONFIRMADO: total de lotes del plan 1395 y recorrido 1398; media real 3.195424177396281 y contrafactual 3.202288984263234; diferencia relativa 0.0021437187276626846 frente al umbral 0.005, luego DESPRECIABLE
- A18, pasos con diferencia relativa mayor que 1e-6: 1397; pasos en el piso del minimo 3 en la real y 0 en la contrafactual
- A19, CONFIRMADO: el desfase de una epoca EXISTE y es un defecto real de codigo, porque el plan usa range(EPOCAS) y el entrenamiento usa enumerate(..., start=1). Lo que se cualifica es su consecuencia numerica, que se declara DESPRECIABLE. Las dos cosas se pueden decir a la vez y no se contradicen

declaracion_de_impacto_del_scheduler: DESPRECIABLE

La serie entera del learning rate, paso a paso, esta en el JSON, con los diez primeros y los diez ultimos pares. El valor de A18 no es la etiqueta, que ya se puede intuir, sino la serie completa: P80 publico la media y los extremos.

## Integridad del artefacto congelado

- sha256 declarado: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`
- antes de la primera lectura: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`
- despues de la ultima lectura: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`
- intacto: True
- A0, CONFIRMADO: intacto en sha256, tamano y mtime_ns antes y despues de la primera y la ultima lectura

## Robustez de la corrida: aislamiento y contrato

Cada afirmacion se ejecuta dentro de un guard. Si lanza, se registra el fallo con su tipo y su traza resumida, la afirmacion queda con veredicto ERROR y la corrida CONTINUA con la siguiente. Esto es una mejora de robustez DEL SCRIPT y no del dato: una afirmacion con ERROR lo esta por fallo propio, y ERROR no es un CONFIRMADO ni un REFUTADO.

- cuantas afirmaciones se aislaron con fallo en ESTA corrida: 0 de 20
- ERROR significa que la afirmacion no llego a ejecutarse por fallo PROPIO del script, y que la corrida continua con las demas. ERROR no es un CONFIRMADO, no es un REFUTADO y no es un NO_EVALUABLE por el dato
- escrituras parciales dentro del bloque 2: 5 de 5, una tras cada afirmacion del bloque
- JSON parcial, escrito al terminar cada bloque y tras cada afirmacion del bloque 2, y con un nombre PROPIO en cada escritura: `reports/p81_cierre_huecos.parcial_<corrida>_<motivo>.json`
- un fallo al escribir un parcial no tumba la corrida: degrada a AVISO en el log. Lo unico que aborta es una Parada con marca STOP.

El contrato de llamada se verifica con `inspect.signature`: cada parametro recibe el objeto que su nombre declara, y un parametro llamado `modulo_p71` tiene que recibir un modulo con `RE_TOKEN`. Esa comprobacion habria atrapado los tres fallos seguidos, que eran contratos asumidos y nunca verificados en la frontera.

- afirmaciones comprobadas: 20
- literales de brazo en afirmaciones: 0

## Lo que P81 no hizo


## Conclusion

Este informe cierra cinco huecos de EVIDENCIA y no anade hallazgos nuevos. Lo que queda declarado y sin cerrar es el anclaje a una implementacion externa de SGNS, que es permanente con el entorno actual.

    V03 remains the best verified leaderboard result.

