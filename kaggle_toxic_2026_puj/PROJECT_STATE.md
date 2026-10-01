<!-- P82-CERRADO 2026-10-01 -->
## P82: CERRADO Y PUBLICADO. 23 de 23 CONFIRMADO, y el resultado es EXPLORATORIO

```text
A0..A22 CONFIRMADO   23 CONFIRMADO   0 REFUTADO   0 ERROR   0 STOP   0 sin ejecutar
corrida buena 20261001T172549Z_p2427889, con los CINCO brazos
270 ajustes de red   5850.165344 s reales contra 5400 de presupuesto, excedencia AUTORIZADA
experiments/p82_01_rnn_atencion.py
  b19802dfa44b3b38ce6caf42829944e7320be82953f56ef60fb10a04c69328e6
reports/p82_rnn_atencion.json   94089 B
  3d0bc0d16a1ea8b427e0f2e0cfd1a765d962370477829ce9d11a2c81de7a9686
reports/p82_rnn_atencion.md      11887 B
  af808f8ecc4949bad2fbe81610b56e2300524f95267c19e368017d62e1cdcc7a
```

**Lo que sale, en una linea: la atencion es indistinguible de cero en los dos nucleos, y la
geometria del embedding se ve con diez de diez.**

### Los cinco brazos y las cinco comparaciones

| brazo | media | sd | | id | media | IC95 | signos + |
|---|---|---|---|---|---|---|---|
| B1 GRU sin atencion | 0.639659 | 0.011615 | | P1 B2-B1 | -0.002105 | [-0.012406, 0.008196] | 4/10 |
| B2 GRU con atencion | 0.637554 | 0.014905 | | P2 B4-B3 | 0.002031 | [-0.016628, 0.020690] | 7/10 |
| B3 LSTM sin atencion | 0.615163 | 0.012924 | | P3 B2-B5 | 0.076807 | [0.059823, 0.093791] | 10/10 |
| B4 LSTM con atencion | 0.617194 | 0.020434 | | P4 B1-B5 | 0.078912 | [0.067144, 0.090680] | 10/10 |
| B5 GRU con atencion, permutado | 0.560747 | 0.019137 | | P5 B1-B3 | 0.024496 | [0.013551, 0.035441] | 10/10 |

**La lectura.** El efecto de la atencion es indistinguible de cero en LOS DOS NUCLEOS y con **signos
opuestos**, -0.002105 en GRU y +0.002031 en LSTM, los dos muy por debajo del minimo detectable de
0.014342. El pre-registro decia que si la atencion tuviera el mismo signo en los dos nucleos seria
la senal mas fuerte posible, y que con signos opuestos seria la prueba mas fuerte de que era ruido.
**Salio lo segundo.**

**Y esto NO es que la atencion no aporte.** Es que este diseno, a diez semillas, no tiene potencia
para distinguir el efecto de cero. El limite de deteccion se calculo ANTES de correr, con potencia
0.8993 a diez semillas, y por eso el resultado es **EXPLORATORIO** y no confirmatorio. Un resultado
negativo **NO** se lee como que la atencion no aporta: se lee como que el efecto es inferior a lo que
este diseno resuelve. Que las veintitres afirmaciones pasen dice que el instrumento esta sano, no que
la pregunta tenga respuesta.

**Lo que si es solido:** la geometria del embedding importa, contra la matriz permutada 0.076807 con
atencion y 0.078912 sin ella, con el signo coherente en **diez de diez** y el intervalo entero por
encima de cero. Es la medicion que nlp-specialist pidio en la Fase 5 y que este proyecto nunca habia
hecho. Separa **existe geometria** de **esa geometria esta asignada a estos tokens**: la respuesta
es que si.

**Hallazgo no anticipado:** GRU supera a LSTM por **0.024496**, con diez de diez. No estaba en ninguna
hipotesis del pre-registro y se registra como tal. El mejor brazo se queda a **0.137552** de V03.

**Limite de deteccion.** Desviacion planificada 0.021588, medida 0.01440007631814541, MDE 0.014342,
gap leido del informe de P78 0.117450, no escrito a mano.

### Lo que P82 deja declarado

**Primera medicion con early stopping estrictamente anidado con particion interna**, luego sus numeros
**NO son comparables** con `v9_04` (0.694343) ni con `v10_11` (0.711639), que elegian el checkpoint
con el mismo pliegue externo que luego puntuaban, la fuga que condeno a P73 y P74.

**Limite permanente de P81 vigente:** sin anclaje a una implementacion externa de SGNS, porque gensim
no esta instalado y no compila con Python 3.14.7. Autoriza afirmar sobre el **OBJETIVO** y no sobre
la calidad relativa frente a una libreria.

**Enmienda 1:** el coste de la atencion es **16448** sin sesgos y 16576 con sesgos, no 8256, porque
8256 asumia una celda unidireccional de ancho 64 y los brazos son bidireccionales de salida 128.

**Enmienda 2:** dispositivo elegido **por medicion**, GPU AMD Radeon RX 9070 XT, 16 hilos, sin
precision mixta. 30 hilos es un 32 por ciento PEOR que 16. Los absolutos y los deltas son los de ese
dispositivo y no se comparan con los de otro.

**Tramo 2 autorizado por Carlos**, con excedencia del presupuesto registrada como autorizacion
explicita, no como unnoticed. Hito declarado 2400, observado 3762.75905. El presupuesto no se toco.

### Los ocho intentos, que se conservan

Ocho intentos = seis corridas en modo completo mas dos de humo. En `reports/p82_parciales/` hay
**cinco** subdirectorios con marca de corrida, con 143 ficheros, mas cinco ficheros sueltos en la raiz
del mismo directorio de la marca `20261001T051317Z_p1023604`, que es el intento en CPU: 148 ficheros.
El ultimo, `20261001T172549Z_p2427889`, es el bueno con los cinco brazos. El
`20261001T141035Z_p2054042` es el que **se detuvo en A19 con STOP**. El informe del tramo 1, marca
`20261001T080906Z_p1362206`, **solo tres brazos** B1, B2 y B5, 21 CONFIRMADO y 2 REFUTADO, **se
conserva y no se borra**: es una corrida previa que no publico informe con los cinco brazos.

**La leccion, para todo el proyecto:** un modo de humo que ejercita funciones con datos sinteticos NO
verifica el contrato entre capas, y un caso de humo que no puede fallar no prueba nada. En P82 lo
manifestaron dos constantes comparadas consigo mismas y cinco tuplas anidadas que hacian que TODO
saliera CONFIRMADO, y ninguna la cazo el humo. Y un hallazgo relevante puede hacer fallar una
comprobacion bien escrita si su criterio estaba mal definido: los netos negativos de A17 son el
hallazgo central, no un error de datos, luego **hay que mirar el criterio antes de culpar a los
datos**.

**Defecto residual apuntado y NO corregido**, porque este encargo es de registro: el bloque de
resumen del cronometro del informe dice `DENTRO_DE_PRESUPUESTO` cuando el tiempo real de 5850.165344
s **supera** el presupuesto de 5400, y su `nota_tramo2` se publica vacia. A19, que recalcula desde el
tiempo real, si detecta `EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA`, luego ninguna
afirmacion esta mal. Queda para una corrida posterior que lo arregle. Ademas el hito observado en el
informe es 3762.75905, y la Enmienda 2 habla de siete ficheros parciales donde en disco hay cinco.

```text
V03 remains the best verified leaderboard result.
```

V03 remains the best verified leaderboard result.
```

---

<!-- P81-CERRADO 2026-10-01 -->
## P81: CERRADO Y PUBLICADO. 20 de 20, 18 CONFIRMADO, 2 REFUTADO, 0 ERROR, salida 0

```text
A0..A5 CONFIRMADO   A6 REFUTADO   A7 REFUTADO   A8..A19 CONFIRMADO
18 CONFIRMADO   2 REFUTADO   0 ERROR   0 STOP   0 sin ejecutar
65.4 s con cronometro de subprocess, CODIGO DE SALIDA 0
sin traceback, sin STOP, sin ERROR
```

```text
experiments/p81_01_cierre_huecos.py
  1e52280f64635637429e141440cfe5efce773e91419bf9b9010b8e7d0c6de39f
  6412 lineas. El hash CAMBIO respecto al 4f7f8488ed9b2710 de la corrida anterior, y es lo
  esperado: ese delta es el arreglo de la serializacion
reports/p81_cierre_huecos.json  183426 B
  81a21033a83c8ffc560a5381cfed4d25f93e99c66ebc3976bc3777a65f381d4b
reports/p81_cierre_huecos.md     15752 B
  19e32b944db6d2927f4f58d1eb14505ccb2ab90fe9dc454dfe1062eabe7d30c4
21 parciales: 10 de la corrida 010045, 10 de esta 010636, y el de la corrida rota conservado
ningun temporal .tmp_p81
```

**ASCII byte a byte: los dos finales son ASCII PURO, 0 bytes por encima de 127.** Verificador oficial
`tools/verificar_caracteres.py`: **salida 0, 0 caracteres no permitidos**.
**NO_DISPONIBLE en el `.md`: 0**, y 0 en el `.json`. El `.md` tiene 219 lineas y **20 filas de tabla de
afirmaciones, todas pobladas**. El bug que llenaba el informe de huecos en silencio queda **descartado
por quinta vez**, ahora con el informe completo publicado.

El arreglo funciona: el informe publica **solo los NOMBRES** de los modulos, mas una tabla de que
modulos exige cada afirmacion.

```json
"contexto_de_modulos": {"p81_04": "p71_04", "p72_01": "p72_01", "v10_11": "v10_11", "v9_04": "v9_04"}
```

**1. El bloque 1 NO apoya a la suma.** A5 **CONFIRMADO** por magnitud, A6 **REFUTADO** por direccion,
titular **`EL_TEST_NO_APOYA_A_LA_SUMA`**, `apoya_a_la_suma` falso, `direccion` `favor_a_la_media`, y **la
MEDIA gana por 0.700000**. Reparto **2 a 4**, en zona de no conclusion, curva **no monotona con 2
cambios de signo**. Las dos lecturas, con umbral 0.99:

```text
por BRAZO, criterio declarado y el del primario
  MEDIA excluye 0.35, 0.50        mejor 0.7388888888888889 en lr 0.20
  SUMA  excluye 0.20, 0.35, 0.50  mejor 0.03888888888888889 en lr 0.10
  diferencia_por_brazo = -0.7000000000000001   GANA LA MEDIA por 0.700000   primario lr 0.1
por PUNTO, alternativa publicada, no usada: sobran 0.02, 0.05, 0.10
  MEDIA 0.005555555555555556  SUMA 0.03888888888888889
  diferencia_por_punto = +0.03333333333333333   GANA LA SUMA por 0.0333
las_dos_lecturas_coinciden_en_signo = False
```

**Las dos lecturas NO coinciden en signo**, y por eso el criterio esta escrito y la alternativa se
publica. Se publica el REFUTADO sin reinterpretar: tolerancia 0.02, banda muerta 0.005, rejilla,
criterio y signo intactos.

**2. La pata EJECUTABLE del anclaje de H1 no apoya la suma**, luego **la base unica que sostiene que
el objetivo no era SGNS es la pata DOCUMENTAL, el codigo C de word2vec**, sha256
`df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4`, 715 lineas. La reimplementacion
propia no aporta apoyo ejecutable.

**3. Limite permanente.** El anclaje a una implementacion externa queda **SIN HACER** porque gensim no
esta instalado y no compila con Python 3.14.7: autoriza afirmar sobre el **OBJETIVO**, no sobre la
calidad relativa del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor que V03.**

**4. El bloque 2, con el tokenizador real de `v9_04`.** Proporcion de posiciones **OOV 0.041249**, de
**relleno 0.730919**, **OOV sobre los tokens que llegan a la matriz 0.153295**, **2008 tipos fuera de
vocabulario**. Desglose: posiciones_totales 94912, tokens_que_entran 25539, tokens_cubiertos 21624,
tokens_oov 3915, posiciones_de_relleno 69373, vocabulario_usado 3439, n_filas_de_dev 1483, max_len 64,
filas_con_cero_tokens_cubiertos 1, fraccion_de_filas_sin_cubiertos 0.0006743088334457181.
**El 11.2726 por ciento de antes NO queda reproducido**, porque las tres cifras de la discusion usan
normalizaciones distintas y MAX_LEN 60 frente a MAX_LEN 64 y `normalizar_emb_normal` de `v9_04`.
**El reparto relleno contra OOV tiene magnitud: 69373 posiciones de relleno frente a 3915 tokens OOV,
sobre 94912 posiciones.** El 0.155020 del prototipo no reproduce ninguna de las cuatro.

**5. El bloque 5.** El desfase de una epoca del scheduler **es un defecto real de codigo**, plan
`range(EPOCAS)` frente a entrenamiento `enumerate(..., start=1)`, y **su consecuencia numerica es
DESPRECIABLE**: diferencia relativa de medias **0.002144** contra umbral **0.005**. A18 y A19 los dos
CONFIRMADO, con `total_lotes_plan 1395` y `total_lotes_recorridos 1398`.

**Salvedad de alcance que queda escrita.** A11 es valido como prueba de que la sonda separa las
celdas, pero **no** como medida del reparto: sus dos canales miden **exactamente 0.0** en las dos
clases, con `banda_muerta_de_la_diferencia: True`, luego "el OOV es al menos tan grande que el de
relleno" es una comparacion de **dos ceros**. **La magnitud del reparto es la de A8.**

**PATRON DE PROCESO, que es informacion cara.** Cinco fallos en la serie P81, todos de **contrato en la
frontera entre capas**, dos de calculo y el resto de cableado y persistencia, **ninguno cubierto por los
checks de humo** hasta que se reescribieron los casos de frontera:

```text
1  KeyError de clave literal en A6            calculo, corregido
2  AttributeError de RE_TOKEN en A18          cableado, corregido
3  tabla de origenes del chequeo, a medias    cableado, corregido
4  escritor incremental sobre su nombre final persistencia, corregido
5  contexto_de_modulos con modulos vivos      persistencia, corregido
```

**La leccion operativa: un modo de humo que ejercita funciones con datos sinteticos NO verifica el
contrato entre capas.** Los 126 checks pasaban enteros mientras el recorrido real reventaba. **Y hace
falta comprobar por mutacion que cada comprobacion discrimina**: un caso que nunca puede fallar no es
una prueba.

**Fronteras respetadas y comprobadas.**

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES y DESPUES
29478868 bytes, abierto en solo lectura, y A0 lo confirma tambien en mtime_ns
```

```text
V03 remains the best verified leaderboard result.
```

<!-- P81-CUARTA-CORRIDA 2026-10-01 -->
## P81 cuarta corrida: LAS VEINTE medidas, 18 CONFIRMADO y 2 REFUTADO, 0 ERROR. El informe final muere en un modulo

```text
20 de 20 afirmaciones medidas   0 sin ejecutar
18 CONFIRMADO   2 REFUTADO (A6 y A7)   0 ERROR   0 STOP
A0 CONFIRMADO: el congelado intacto en sha256, tamano y mtime_ns antes y despues
65.1 s con cronometro de subprocess, codigo de salida 1
```

```text
experiments/p81_01_cierre_huecos.py
  4f7f8488ed9b27106248d8035e3bb749318b0032788616e745421d2df09adb27
  6402 lineas, 126 de 126 en --humo con salida 0
10 parciales propios ESCRITOS, cada uno con marca de corrida y motivo, los 10 ASCII puros
reports/p81_cierre_huecos.parcial.json  el de la corrida rota, INTACTO y conservado
reports/p81_cierre_huecos.json           NO EXISTE
reports/p81_cierre_huecos.md             NO EXISTE
ningun temporal .tmp_p81, nada vaciado, nada renombrado
```

**El arreglo de la persistencia funciona.** Cada escritura parcial va a un fichero PROPIO, luego la
invariante "nunca sobrescribir un destino existente" es cierta para todos los destinos sin
excepciones, y un fallo de escritura degrada a AVISO con su causa en el log. Solo una Parada con marca
STOP aborta. **Los 34 `aviso de contrato` de la corrida anterior desaparecen**: la tabla de origenes
quedo construida.

**Fallo literal, sin arreglar.**

```text
  File ".../experiments/p81_01_cierre_huecos.py", line 6349, in main
    json.dumps(informe, ensure_ascii=True, indent=2)
TypeError: Object of type module is not JSON serializable
when serializing dict item 'p71_04'
when serializing dict item 'contexto_de_modulos'
```

La linea 3835 mete `contexto_de_modulos` en el cuerpo del informe, y ese diccionario contiene
**objetos modulo vivos**: `p71_04`, `p72_01`, `v10_11`, `v9_04`. Un modulo no es serializable. Los
parciales si serializan porque **su cuerpo no incluye ese campo**; el informe final si.
**Asimetria entre dos cuerpos del mismo informe.** El campo se anadio para auditar que modulos se
cargaron, lo cual exige el modulo y por eso mismo no se puede volcar: falta publicar los **nombres**.
Es la ultima sentencia efectiva de la corrida: las veinte mediciones y los diez parciales ya estaban
en disco. El guard de la linea 6344, que se niega a sobrescribir el informe, se respeto bien.

**Bloque 1, en disco, exactamente lo esperado.**

```text
por BRAZO (criterio declarado, el del primario)
  MEDIA excluye 0.35, 0.50        mejor 0.7388888888888889 en lr 0.20
  SUMA  excluye 0.20, 0.35, 0.50  mejor 0.03888888888888889 en lr 0.10
  diferencia_por_brazo = -0.7000000000000001   GANA LA MEDIA por 0.700000   primario lr 0.1
por PUNTO (publicada, no usada): sobran 0.02, 0.05, 0.10
  MEDIA 0.005555555555555556   SUMA 0.03888888888888889
  diferencia_por_punto = +0.03333333333333333   GANA LA SUMA por 0.0333
las_dos_lecturas_coinciden_en_signo = False
```

**Reparto de victorias: la SUMA 2 de 6, la MEDIA 4 de 6, "2 a 4".** Curva no monotona, 2 cambios de
signo, `recuento_en_banda_muerta: True`.
**Titular literal:** `EL_TEST_NO_APOYA_A_LA_SUMA`, `apoya_a_la_suma: False`,
`direccion: favor_a_la_media` -- "la diferencia entre los dos mejores de la rejilla es -0.700000, de
magnitud suficiente para discriminar pero con el SIGNO en contra de la SUMA. La MEDIA gana por
0.700000... el bloque 1 NO apoya al objetivo de la referencia", mas "ZONA DE NO CONCLUSION: el reparto
de victorias por learning rate queda en 2 a 4 y eso es ruido con tres semillas. El primario NO debe
leerse como confirmacion". **Sin ajustar nada**: tolerancia 0.02, banda muerta 0.005, rejilla, criterio
y signo intactos.

**Bloque 2: LAS CUATRO CIFRAS, que no habian sobrevivido a ninguna corrida.**

```text
proporcion_de_posiciones_oov        = 0.04124873567093729    4.124874 % de las posiciones
proporcion_de_posiciones_de_relleno = 0.7309191672285907     73.091917 % de las posiciones
oov_sobre_tokens                     = 15.329496 %            sobre los tokens que entran
numero_de_tipos_distintos_fuera_de_vocabulario = 2008
posiciones_totales 94912   tokens_que_entran 25539   tokens_cubiertos 21624   tokens_oov 3915
posiciones_de_relleno 69373   vocabulario_usado 3439   n_filas_de_dev 1483   max_len 64
filas_con_cero_tokens_cubiertos 1   fraccion_de_filas_sin_cubiertos 0.0006743088334457181
```

**Reparto relleno contra OOV CON MAGNITUD:** de 94912 posiciones, **69373 relleno = 73.091917 %**, y
**3915 tokens OOV = 4.124874 %**. Sobre los 25539 tokens que llegan a la matriz, el OOV es el
**15.329496 %**: una orden de magnitud mas pequeno que el relleno en posiciones.
**El 11.2726 % de la discusion NO queda reproducido**, `variante_que_la_reproduce: null`, porque las tres
cifras anteriores usan normalizaciones distintas y MAX_LEN 60 frente a MAX_LEN 64 y
`normalizar_emb_normal` de `v9_04`. **El 0.155020 del prototipo no reproduce ninguna de las cuatro**,
y ahora eso esta medido y publicado en vez de estimado.

**Salvedad sobre A11.** CONFIRMADO, pero sus magnitudes son `canal_de_relleno_c1 = 0.0` y
`canal_de_oov_c2b_menos_c2 = 0.0` en las dos clases, con distancia `0.0` y
`banda_muerta_de_la_diferencia: True`. **Los dos canales miden exactamente cero**, luego "el OOV es al
menos tan grande" es una comparacion de **dos ceros**: cierto y sin informacion. **La magnitud del
reparto es la de A8, no la de A11.**

**Bloque 5: A18 y A19 los dos CONFIRMADO.** A18: diferencia relativa de las medias 0.002143718728
contra umbral 0.005, luego **DESPRECIABLE**, con `total_lotes_plan 1395` y
`total_lotes_recorridos 1398` y el recorte declarado de las primeras 2000 lineas, 53243 tokens, con la
advertencia de que los recuentos absolutos no son los de la corrida real. A19: la etiqueta coincide con
la regla de A18, y dice explicitamente que el desfase de una epoca **existe y es un defecto real de
codigo** y que lo que se cualifica es su consecuencia numerica. **Ninguno quedo en ERROR**, y la
regla de que A19 emita ERROR si A18 no se ejecuto **no llego a dispararse** porque A18 si corrio.
**No hay ningun REFUTado que descartar por fallo aguas arriba**: A6 y A7 son del bloque 1.

**ASCII y verificador.** Los **diez** parciales nuevos: **0 bytes por encima de 127 cada uno**. El `.json`
y el `.md` finales no se pueden verificar porque no existen. Verificador oficial
`tools/verificar_caracteres.py` sobre el proyecto entero: **salida 0, 0 caracteres no permitidos**.
**NO_DISPONIBLE en el `.md`: no aplica, no se escribio.** En los diez parciales nuevos **0**, y **0**
en el parcial viejo. **El bug del informe lleno de huecos no se ha reproducido en cuatro corridas
seguidas.**

**Cinco fallos en la serie de P81, todos de contrato asumido y no verificado en la frontera**, solo dos
de calculo: el `KeyError` de clave literal, el modulo equivocado de A18, la tabla de origenes a medio
construir, el escritor incremental sobre su nombre definitivo, y ahora un objeto modulo dentro del
informe. **Ninguno lo cubren los 126 checks de humo.** El hueco real de este ultimo es que nadie
comprueba que **todo lo que se mete en el informe sea serializable**.

**CONCLUSION.** La pata **EJECUTABLE** del anclaje de H1 **no apoya la suma**: A5 discrimina por
magnitud, A6 mide la direccion y la encuentra contraria, la MEDIA gana por 0.700000, y el reparto 2 a
4 con dos cambios de signo deja el primario en zona de no conclusion con tres semillas. Luego **la base
que sostiene que el objetivo no era SGNS es la pata DOCUMENTAL, el codigo C de word2vec, y queda como
base unica.**

**Limite permanente.** El anclaje a una implementacion externa de SGNS queda **SIN HACER** porque gensim
no esta instalado y no compila con Python 3.14.7: autoriza afirmar sobre el **OBJETIVO**, no sobre la
calidad relativa del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor que V03.**

**Fronteras respetadas y comprobadas.**

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES y DESPUES
29478868 bytes, abierto en solo lectura, y A0 lo confirma tambien en mtime_ns
```

```text
V03 remains the best verified leaderboard result.
```

<!-- P81-TERCERA-CORRIDA 2026-10-01 -->
## P81 tercera corrida: 6 CONFIRMADO, 2 REFUTADO, 0 ERROR, se para en A8. El escritor incremental se come a si mismo

```text
A1..A5 CONFIRMADO   A6 REFUTADO   A7 REFUTADO   A8 CONFIRMADO
A0 y A9..A19 sin ejecutar
6 CONFIRMADO   2 REFUTADO   0 ERROR   0 STOP   12 sin ejecutar
64.4 s con cronometro de subprocess, codigo de salida 1. El presupuesto era de diez minutos.
```

```text
experiments/p81_01_cierre_huecos.py
  5b5db4b0f2cd514fec1cd667d898c808dd5a6d090f27ae4e9f15c7b7967e9100
  5973 lineas, 120 de 120 en --humo con salida 0, APROBADO por ml-reviewer
  nota: el briefing decia 5975 lineas. El sha256 coincide completo, luego es el fichero correcto
reports/p81_cierre_huecos.parcial.json   24196 B   ESCRIBE, 7 afirmaciones A1 a A7
reports/p81_cierre_huecos.json            NO EXISTE
reports/p81_cierre_huecos.md              NO EXISTE
ningun temporal .tmp_p81, nada vaciado, nada renombrado
```

**Ninguna afirmacion quedo en ERROR.** El guard de aislamiento por afirmacion nunca actuo,
`afirmaciones_aisladas_con_fallo` esta vacio. El fallo no ocurrio dentro de una afirmacion, ocurrio en
la capa de persistencia, entre A8 y la escritura que la iba a registrar. A8 **llego a CONFIRMADO** y
su veredicto esta en el log, pero **no quedo en disco**.

**Fallo literal, sin arreglar.**

```text
[P81-01] | JSON parcial escrito en reports/p81_cierre_huecos.parcial.json: 7 afirmaciones, 0 aisladas con fallo
[P81-01] |    A8: CONFIRMADO | las cuatro cifras estan medidas y publicadas...
STOP escritura: .../reports/p81_cierre_huecos.parcial.json ya existe y solo se admiten temporales
               hasta la publicacion atomica
```

**Dos defectos que se multiplican.** `publicar_parcial` publica el parcial **sobre su nombre
definitivo** en la linea 3889, luego la primera escritura, al terminar el bloque 1, deja el fichero en
disco y la segunda, tras A8, choca con el **guard de antitrunco de P80** de la linea 517, que prohibe
sobrescribir un destino existente. Y ese `SystemExit` no degrada a aviso porque la linea 3895 tiene
`except SystemExit: raise`, que lo relevanta **antes de que lo vea el `except Exception` de la 3897**,
que era el que iba a registrar `aviso: no se pudo escribir el JSON parcial`. **La red de seguridad del
propio escritor no existe para este fallo.**

**La ironia, sin adornos:** la caracteristica que se anadio para que un fallo final no tirase las
mediciones anteriores es la que tira la corrida, y se come la primera medicion del bloque 2, la que
pretendia salvar. **Las cuatro cifras de A8 mueren por tercera vez consecutiva, y esta por el
mecanismo que deberia impedirlas.**

**Bloque 1, por fin en disco, con numeros exactos.**

```text
umbral de zona betonal 0.99
por BRAZO, criterio declarado y el del primario
  MEDIA  excluye 0.35 y 0.50   mejor no betonal 0.7388888888888889 en lr 0.20
  SUMA   excluye 0.20, 0.35 y 0.50   mejor no betonal 0.03888888888888889 en lr 0.10
  diferencia_por_brazo = -0.7000000000000001   GANA LA MEDIA por 0.700000   primario en lr 0.1
por PUNTO, alternativa publicada, no usada
  sobran 0.02, 0.05 y 0.10   MEDIA 0.005555555555555556   SUMA 0.03888888888888889
  diferencia_por_punto = +0.03333333333333333   GANA LA SUMA por 0.0333
las_dos_lecturas_coinciden_en_signo = False
```

El signo se da la vuelta enteramente segun el criterio de exclusion, y por eso el criterio esta
escrito y la alternativa se publica.

**Recuento de victorias: la SUMA gana 2 de 6 y la MEDIA 4 de 6, reparto "2 a 4".** Curva no monotona,
**2 cambios de signo**, `recuento_en_banda_muerta: true`.

**Titular literal, y coincide con lo esperado en las cuatro cosas:**

```text
etiqueta: EL_TEST_NO_APOYA_A_LA_SUMA   apoya_a_la_suma: False   direccion: favor_a_la_media
```

```text
la diferencia entre los dos mejores de la rejilla es -0.700000, de magnitud suficiente para
discriminar pero con el SIGNO en contra de la SUMA. La MEDIA gana por 0.700000. El test discrimina
por magnitud, pero su direccion es contraria a la hipotesis de A6, y eso se publica: el bloque 1
NO apoya al objetivo de la referencia
```

```text
ZONA DE NO CONCLUSION: el reparto de victorias por learning rate queda en 2 a 4 y eso es ruido con
tres semillas. El primario NO debe leerse como confirmacion
```

**A5 CONFIRMADO por magnitud, A6 REFUTADO por direccion, EL_TEST_NO_APOYA_A_LA_SUMA con la MEDIA
ganando por 0.700000, y recuento 2 a 4 en zona de no conclusion.** Se publica tal cual. No se movio la
tolerancia de 0.02, ni la banda muerta de 0.005, ni la rejilla, ni el criterio de zona betonal, ni el
signo de la direccion.

**Bloque 2: A8 corrio y CONFIRMO, y sus cifras mueren otra vez.** El parcial tiene siete
afirmaciones, A1 a A7. **Las cuatro cifras de la proporcion de OOV no son recuperables por tercera
vez**, ni el reparto entre canal de relleno y canal de OOV **con magnitud**. No queda A9, A10, A11 ni
A12: el veredicto cualitativo de que el canal de OOV es al menos tan grande como el de relleno tampoco
esta en disco. El 0.155020 del prototipo y las tres cifras anteriores siguen sin reproducirse desde un
fichero publicado.

**Bloque 5: A18 y A19 NUNCA se ejecutaron.** El recorrido se corta en la primera escritura del bloque
2. **No tienen veredicto: ni REFUTADO ni ERROR**, no se ejecutaron. **La contingencia de que A19 salga
REFUTADO por un A18 en ERROR no se materializa**, luego no hay ningun REFUTado que descartar por fallo
aguas arriba. Los dos REFUTADO reales, A6 y A7, son mediciones del bloque 1 y no dependen de A18. El
`AttributeError` de `RE_TOKEN` de la segunda corrida **sigue sin corregir**: esta version no llega a A18.

**ASCII y verificador.** Parcial: 24196 bytes, **0 bytes por encima de 127, ASCII puro**. El `.json` y
el `.md` completos no se pueden verificar porque no existen. Verificador oficial
`tools/verificar_caracteres.py` sobre el proyecto entero: **salida 0, 0 caracteres no permitidos**.

**NO_DISPONIBLE en el `.md`: no aplica, el `.md` no se escribio.** En el parcial, que es el unico
informe en disco, son **0**. **El bug que llenaba el informe de huecos en silencio no se ha reproducido
en tres reventones seguidos**: la defensa de escritura de P80 ha aguantado las tres.

**Avisos de contrato.** Al arrancar salen **34 lineas** `aviso de contrato` sobre A0, A4 a A13, A15, A16,
A17 y A19: falta el origen declarado `medidas.<parametro>` para casi todos los parametros. **Son avisos,
no fallos**, y la corrida sigue. Pero el comprobador de contratos **no tiene declarada la procedencia de
casi todo lo que le pasa**: esa tabla esta a medio construir.

**Cuatro fallos seguidos, los cuatro de contrato asumido y no verificado en la frontera**, y el ultimo
es el guard antitrunco de P80 el que se enciende: el `KeyError` de clave literal, corregido; el modulo
equivocado de A18, sin corregir; la tabla de origenes, a medio construir; y el escritor incremental
publicando sobre su nombre final. **Ninguno lo cubren los 120 checks de humo**, porque el humo verifica
recuentos y no contratos entre capas. Hace falta un caso que recorra `main` de punta a punta, o una
asercion de contrato en cada frontera.

**CONCLUSION, con el titular ya en disco.** La pata **EJECUTABLE** del anclaje de H1 **no apoya a la
suma**: A5 discrimina por magnitud, A6 mide la direccion y la encuentra contraria, la MEDIA gana por
0.700000, y el reparto 2 a 4 con dos cambios de signo deja el primario en zona de no conclusion con
tres semillas. Luego **la base que sostiene que el objetivo no era SGNS es la pata DOCUMENTAL, el
codigo C de word2vec, y queda como base unica.** La reimplementacion propia no aporta apoyo ejecutable.

**Limite permanente.** El anclaje a una implementacion externa de SGNS queda **SIN HACER** porque gensim
no esta instalado y con Python 3.14.7 no compila: autoriza afirmar sobre el **OBJETIVO**, no sobre la
calidad relativa del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor que V03.**

**Fronteras respetadas y comprobadas.**

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura
```

```text
V03 remains the best verified leaderboard result.
```

<!-- P81-SEGUNDA-CORRIDA 2026-10-01 -->
## P81 segunda corrida: 17 de 20 medidas, 15 CONFIRMADO y 2 REFUTADO, se para en A18

```text
A1..A5 CONFIRMADO   A6 REFUTADO   A7 REFUTADO   A8..A17 CONFIRMADO
A0, A18 y A19 sin ejecutar
15 CONFIRMADO   2 REFUTADO   0 NO APLICABLE   0 STOP   3 sin ejecutar
567 s, codigo de salida 1, parada en A18 y NO en A6
bloques 2, 3 y 4 completos; el bloque 5 no llego a empezar
```

```text
experiments/p81_01_cierre_huecos.py
  00756ef3432ca2ee29fd2c99987a84f19b3c6f801dc1ea70a34cdd4bc78c966c
  4930 lineas, 83 de 83 en --humo
reports/p81_cierre_huecos.json  NO EXISTE
reports/p81_cierre_huecos.md    NO EXISTE
ningun temporal .tmp_p81, nada publicado, nada vaciado
```

**El arreglo de A6 funciona y los candados hacen su trabajo.** La funcion que deriva toda clave de
brazo a partir del nombre del objetivo elimina el `KeyError`, y el candado
`ERROR_DE_COHERENCIA_NO_PUBLICABLE` ya no tiene nada que devolver porque la corrida cae del lado
correcto.

**Fallo literal, sin arreglar.**

```text
File ".../experiments/p81_01_cierre_huecos.py", line 3331, in afirmacion_a18
    documentos = leer_primeras_n_lineas(CORPUS_TWITTER, RECRETE["lineas"], modulo_p72)
File ".../experiments/p81_01_cierre_huecos.py", line 3407, in leer_primeras_n_lineas
    tokens = modulo_p71.RE_TOKEN.findall(linea)
AttributeError: module 'p72_01' has no attribute 'RE_TOKEN'
```

**Cableado de modulos.** El parametro se llama `modulo_p71` y de ahi usa `RE_TOKEN`, pero quien llama
le pasa `modulo_p72`, que es `p72_01` y no lo tiene. `RE_TOKEN` esta en `p71_04`, que el script ya
importa y que ya esta en `contexto["modulos"]["p71_04"]`, y `afirmacion_a18` no lo recibe: su unico
argumento es `p72_01`. Necesita los dos modulos, uno para las constantes del curriculum y otro para el
tokenizador, y le dieron uno.

**Bloque 1: las dos lecturas de la zona betonal, y no coinciden en signo.**

```text
lr 0.02  MEDIA 0.0000  SUMA 0.0000   lr 0.20  MEDIA 0.7389  SUMA 1.0000
lr 0.05  MEDIA 0.0000  SUMA 0.0000   lr 0.35  MEDIA 1.0000  SUMA 1.0000
lr 0.10  MEDIA 0.0056  SUMA 0.0389   lr 0.50  MEDIA 1.0000  SUMA 1.0000
```

```text
por BRAZO, el criterio declarado y el del primario, linea 178
  MEDIA mejor no betonal 0.7389 en lr 0.20   SUMA mejor no betonal 0.0389 en lr 0.10
  diferencia = -0.700000   GANA LA MEDIA por 0.700000   primario en lr 0.1
por PUNTO, el alternativo que se calcula y se publica pero no se usa
  MEDIA 0.0056   SUMA 0.0389   diferencia = +0.0333   GANA LA SUMA por 0.0333
```

**El signo se da la vuelta enteramente segun el criterio de exclusion**, y por eso el criterio queda
escrito y la alternativa se publica: decidir solo con una seria ocultar la otra.

**Recuento de victorias: la SUMA gana 2 de 6 puntos. La curva NO es monotona.** El prototipo no se
reproduce: en 0.20 daba MEDIA 0.3611 contra SUMA 0.0000 y aqui la SUMA gana 1.0000 contra 0.7389; en
0.35 daba MEDIA 0.4667 contra SUMA 0.8278 y aqui empatan en 1.0000.

**El titular, literal, y sale como se esperaba.**

```text
A5: CONFIRMADO [EL_TEST_DISCRIMINA_POR_MAGNITUD]
A6: REFUTADO   la direccion es CONTRARIA a la hipotesis. La MEDIA gana por 0.700000 en el
               primario, y el margen exigido es de 0.02 a favor de la SUMA
A7: REFUTADO   el efecto no se sostiene: ['el conteo de victorias por semilla', 'el control
               negativo tambien recupera estructura']
```

**El test NO apoya a la suma, y es un resultado legitimo.** A5 etiqueta solo magnitud. A6 mide la
direccion y la da contraria. A7 dice que el efecto no aguanta ni en el conteo de victorias por
semilla ni en el control negativo, que tambien recupera estructura. Con diferencia `-0.700000`,
`lectura_del_bloque_1` entra por la rama de la linea 1339 y devuelve **`EL_TEST_NO_APOYA_A_LA_SUMA`**
con `apoya_a_la_suma` falso. **Ese titular no se publico en ningun fichero**, porque el fallo cae
antes de `construir_json` y `escribir_markdown`: existio solo en memoria.

**Sin retoque.** No se movio la tolerancia de 0.02, ni la banda muerta de 0.005, ni la rejilla, ni el
criterio de zona betonal, ni el signo de la direccion.

**Bloque 2: ejecutado completo, con las cifras perdidas.** A8 **CONFIRMADO**, las cuatro cifras
medidas y el 11.2726 por ciento de la discusion de auditoria **no** reproducido con ninguna de las
tres. A9 **CONFIRMADO**, celdas separadas de verdad. A10 **CONFIRMADO**, las seis magnitudes con al
menos una cifra distinta de cero por condicion y por clase. A11 **CONFIRMADO**: **el canal de OOV es
al menos tan grande como el de relleno en las dos clases.** A12 **CONFIRMADO**, variacion
exactamente cero entre dos pasadas con la misma semilla.

**Las cuatro cifras de la proporcion de OOV NO son recuperables.** Se midieron y se verificaron, y
murieron en memoria porque el proceso no llego a escribir. Del reparto relleno contra OOV sobrevive
solo el veredicto cualitativo de A11, sin magnitud. **El 0.155020 del prototipo y las tres cifras
anteriores siguen sin reproducirse de forma verificable desde un fichero publicado.**

**Lo que no se pudo verificar.** El ASCII byte a byte de los dos informes y el verificador oficial no
se pueden ejecutar, porque los ficheros no existen. El recuento de `NO_DISPONIBLE` en el `.md` no
aplica, porque no hay `.md`. **El bug que llenaba el informe de huecos en silencio no se ha
reproducido**: el fallo es de cableado y ocurre antes de la escritura, y la defensa de escritura de
P80 aguanta dos reventones seguidos.

**Tres fallos seguidos, los tres de contrato asumido y no verificado en la frontera.** El primero, el
`KeyError` de clave literal, corregido. Este, el modulo equivocado, sin corregir. Los 83 checks de
humo pasan enteros en las dos versiones y no cubren ninguno, porque el humo verifica **recuentos** y
no **contratos de frontera entre modulos**. No se arregla con mas checks de humo: hace falta un caso
que recorra el camino real de `main`, o una asercion de contrato en la frontera.

**CONCLUSION NUEVA.** La pata EJECUTABLE del anclaje **no apoya a la suma**: A5 discrimina por
magnitud, A6 mide la direccion y es contraria, A7 no sostiene el efecto. Luego **la pata que sostiene
que el objetivo no era SGNS es la DOCUMENTAL, la del codigo C de word2vec, y queda como unica
base.** La reimplementacion propia no aporta apoyo ejecutable, y eso se dice sin rodeos.

**Limite permanente.** El anclaje a una implementacion externa de SGNS queda **SIN HACER** porque
gensim no esta instalado y con Python 3.14.7 no compila: autoriza afirmar sobre el **OBJETIVO**, no
sobre la calidad relativa del artefacto frente a una libreria. **No se afirma que Word2Vec sea mejor
que V03.**

**Fronteras respetadas y comprobadas.**

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura
```

```text
V03 remains the best verified leaderboard result.
```

<!-- P81-INCONCLUSO 2026-09-30 -->
## P81: INCONCLUSO. 5 de 20 afirmaciones medidas, todas CONFIRMADO, se para en A6 con un KeyError

```text
A1..A5 CONFIRMADO   A0 y A6..A19 sin ejecutar
5 CONFIRMADO   0 REFUTADO   0 NO APLICABLE   0 STOP   15 sin ejecutar
codigo de salida 1   62.3 s con cronometro de subprocess, 77 s en la primera corrida
mismo punto de fallo en las dos, bit a bit
```

```text
experiments/p81_01_cierre_huecos.py  212aa786998f682a4ccefca2d91aa77f0cbf510968a0e8d21f1c687083f8e913
4517 lineas, 67 de 67 en --humo, APROBADO por ml-reviewer
reports/p81_cierre_huecos.json  NO EXISTE
reports/p81_cierre_huecos.md    NO EXISTE
ningun temporal .tmp_p81, ninguna salida publicada, ninguna vaciada
```

**Fallo literal, sin arreglar.**

```text
File ".../experiments/p81_01_cierre_huecos.py", line 1846, in afirmacion_a6
    m1_suma = primario["primarios"]["SUMA"]["m1_media"]
KeyError: 'm1_media'
```

`elegir_primario` genera la clave del brazo con `f"m1_{objetivo.lower()}"` en la linea 1265 y la
escribe en la 1277, luego SUMA guarda `m1_suma`. `afirmacion_a6` busca para SUMA el literal
`m1_media`, que es la clave del brazo MEDIA. `afirmacion_a5`, linea 1711, lo hace bien. Mismo patron
de clave literal, bien a una linea y mal a la otra.

**El limite de la suite de humo, aqui y en serio.** El revisor dejo sin cobertura de humo directa el
fallo de la tupla pasada como clave literal. Este fallo es **de esa misma familia** y **tampoco lo
cubre el humo**: 67 de 67 pasan y el recorrido real revienta en la sexta afirmacion. Smoke Test
Driven Development con recuento de comprobaciones no fija el contrato de claves que el resto del
recorrido da por hecho.

**Bloque 1, que es lo que si se midio.** Curva completa de M1 por learning rate, los dos brazos, seis
puntos de la enmienda 1 y tres semillas, sobre corpus plantado de 23004 pares con M3 del azar
0.000042 en 2000 permutaciones:

```text
lr 0.02  MEDIA 0.0000  SUMA 0.0000   lr 0.20  MEDIA 0.7389  SUMA 1.0000
lr 0.05  MEDIA 0.0000  SUMA 0.0000   lr 0.35  MEDIA 1.0000  SUMA 1.0000
lr 0.10  MEDIA 0.0056  SUMA 0.0389   lr 0.50  MEDIA 1.0000  SUMA 1.0000
```

**La SUMA gana 2 de 6 puntos y la curva NO es monotona.** El prototipo no se reproduce: daba MEDIA
0.3611 contra SUMA 0.0000 en 0.20, aqui la SUMA gana 1.0000 contra 0.7389; daba MEDIA 0.4667 contra
SUMA 0.8278 en 0.35, aqui empatan en el techo de 1.0000. Primario declarado en lr 0.1, con la zona
betonal de la enmienda 1 dejando fuera los 1.0000.

**El titular, con el signo delante.** `A5: CONFIRMADO [EL_TEST_DISCRIMINA]`, diferencia
`-0.700000`. La comprobacion de la linea 1726 es `abs(diferencia) >= TOL["discriminacion"]`:
**discrimina por magnitud, no por direccion.** Los dos mejores son 0.0389 de la SUMA en 0.10 y
0.7389 de la MEDIA en 0.20, y la SUMA **pierde por 0.70**. La afirmacion que iba a medir la
direccion es A6, y es la que revienta, luego **la direccion preregistrada queda sin medir**.

**Bloque 2: no se ha ejecutado.** El fallo cae antes de la linea 4360. **A8 no corrio.** La
proporcion de OOV con el tokenizador real de `v9_04` **no se ha obtenido**: ni el 0.155020 del
prototipo, ni las tres cifras anteriores, ni el reparto entre canal de relleno y canal de OOV. Los
bloques 3 y 4 tampoco.

**Lo que no se pudo verificar.** El ASCII byte a byte de `reports/p81_cierre_huecos.json` y `.md` y
el verificador oficial **no se pueden ejecutar**, porque los ficheros no existen. El recuento de
`NO_DISPONIBLE` en el markdown **no aplica**, porque no hay markdown. El bug que llenaba el `.md` de
`NO_DISPONIBLE` en silencio no se ha reproducido porque el script no llego a escribir.

**Fronteras respetadas y comprobadas.**

```text
test = 0   holdout = 0 identificadores deserializados   leaderboard = 0   submission = 0
P71 a P80 sin tocar   V03 sin tocar   umbrales 0.44 / 0.51 / 0.46 sin tocar   modelos/ sin escrituras
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a ANTES
congelado 302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a DESPUES
29478868 bytes, abierto en solo lectura, misma huella tras las dos corridas
```

**Limite permanente.** El anclaje a una implementacion externa de SGNS queda **SIN HACER** porque
gensim no esta instalado y con Python 3.14.7 no compila: autoriza afirmar sobre el **OBJETIVO**, no
sobre la calidad relativa del artefacto frente a una libreria. Dos salvedades mas: el texto ancla de
`docs/p81_enmienda_1.md` se commiteo en `5f4faf7` con dos tildes y se corrigio en `c3dee26` sin
reescribir historia, y el sha256 vinculante es el de `c3dee26`; y el fallo de la tupla pasada como
clave literal sigue sin caso de humo que lo fije.

```text
V03 remains the best verified leaderboard result.
```

<!-- P80-CERRADO 2026-09-30 -->
## P80: cerrado y publicado. 19 de 19, 7 de 7 CONFIRMADO, informe en JSON y markdown

```text
A1..A12 CONFIRMADO   A13 NO APLICABLE POR CONSTRUCCION   A14..A19 CONFIRMADO
18 CONFIRMADO   0 REFUTADO   1 NO APLICABLE   0 STOP   0 sin ejecutar
H1 H2 H3 H4 H5 H6 H7 = CONFIRMADO
corrida de 2 s, codigo de salida 0, sin fallos y sin traceback
```

```text
reports/p80_auditoria_w2v.json  174656 B  27efa9a37a755464b121581110a333831bfdf5f3acf63f092f8086503f6ad8e6
reports/p80_auditoria_w2v.md     11657 B  f8bab0dd93dd6eb299a9b04d89083ef380c882d1d382bfd396d7e47c66ce4e8f
figuras/p80/  3 PNG, firmas verificadas
publicados por renombrado atomico, asercion ASCII byte a byte sobre script, json y md
```

**El `.md` no sale vacio.** 19 filas de tabla de afirmaciones y 7 de hallazgos, todas pobladas. El
bug anterior hacia tablas vacias en silencio, asi que se comprobo el contenido, no la existencia. Un
unico `no disponible`, en la linea de A12, y es legitimo: `claves_json_recorridas` ya no existe en el
esquema de A12 tras el arreglo.

**Reproducible.** Las diecinueve medidas son **bit a bit identicas** a las de la segunda corrida, y los
veredictos por hallazgo tambien. Dos versiones distintas del script, mismas medidas: los arreglos
tocaron instrumentacion y no calculo.

**H2 CONFIRMADO, y el veredicto de P79 era erroneo.** `p72_01:92-95` planifica con
`range(m.EPOCAS)` y `SEMILLA + ep`, o sea 20261016 a 20261020; `p72_01:99-103` entrena con
`enumerate(planes, start=1)` y `SEMILLA + ep`, o sea 20261017 a 20261021. Solo en el plan 20261016,
solo en el entrenamiento 20261021. 1395 lotes planados frente a 1398 recorridos.

**H4 CONFIRMADO.** El congelado solo guarda `W`, la tabla de entrada. A12 da 0 candidatos sobre 19
`.npz` por cabecera y 320 `.json`. A13 NO APLICABLE POR CONSTRUCCION: la tabla de salida no existe,
**no se sustituye por `W`** y **ninguna cifra absoluta sobre el artefacto es evidencia del estado
final**.

**H5 CONFIRMADO**: la metrica que decide cuando restaurar pesos es la del pliegue externo que despues
se puntua. **H6 CONFIRMADO**: GRU antes de la mascara, arrastre y no explosion, 0.08399328135684547.
**H7 CONFIRMADO**: DL-03 con 1731 filas y 1483 + 248 = 1731, luego el holdout v2 estaba en el pool.

**Tres salvedades de alcance, que el informe del script no lleva.** La regla 3 de A19, el desempate
por nombre, puede clasificar como validacion un argumento que llega como parametro sin vinculacion
local, por ejemplo `x_val`; es latente, no se activa en los tres ficheros reales y ahi manda el vinculo
local, pero es una apuesta por nombre. `asignaciones_del_llamante` usa `setdefault`, luego si un
nombre se asigna dos veces gana la primera sin marcar ambiguedad; falla hacia el lado conservador, da
REFUTADO y no sobre-detecta, pero la lectura queda parcial sin aviso. Y la sonda de A16 mide una red
sintetica de 51 por 200, **no los BiGRU reales**, luego su magnitud **no se extrapola**.

**El `.tmp_p80` viejo ya no existe** y no lo borre yo: el script reutiliza esa ruta, la vacio con el
JSON de esta corrida y la renombro sobre el `.json`. Sigue en la historia, en `c317794`, sha256
`890bbb1037b3034a893e88634b5db261a2d9f6acad278722d32c6a00007c834f`.

```text
P80 = CERRADO   V03 = FROZEN   Word2Vec Twitter 100d = FROZEN
congelado 302c994d... intacto antes y despues      pre-registro 440ab13c... sin cambios en las tres corridas
Sin escrituras fuera de reports/ y figuras/p80/. Test, holdout, leaderboard y submission en cero.
P71 a P79, V03, los umbrales 0.44 / 0.51 / 0.46 y modelos/ sin tocar.
```

```text
V03 remains the best verified leaderboard result.
```
<!-- /P80-CERRADO 2026-09-30 -->

<!-- P80-SEGUNDA-CORRIDA 2026-09-30 -->
## P80: las 19 afirmaciones pasan. Los 7 hallazgos CONFIRMADO. El informe no se publico

```text
A1..A12 CONFIRMADO   A13 NO APLICABLE POR CONSTRUCCION   A14..A19 CONFIRMADO
18 CONFIRMADO   0 REFUTADO   1 NO APLICABLE   0 STOP   0 sin ejecutar
corrida de 2 s, codigo de salida 1, sin entrenamiento
```

**Misme en el markdown.** Revienta en `escribir_markdown:4704` con
`KeyError: 'claves_json_recorridas'`: la correccion de A12 quito esa clave de `medida.cobertura` y el
redactor la sigue leyendo. Las otras tres claves que lee de A12 si existen, luego es **una sola
clave**. No se ha tocado el script.

```text
reports/p80_auditoria_w2v.json            1936 B    el JSON de PARADA anterior, sin tocar
reports/p80_auditoria_w2v.json.tmp_p80  174103 B    el informe COMPLETO, huerfano
reports/p80_auditoria_w2v.md             NO EXISTE
figuras/p80/  3 PNG, firmados en el temporal
```

El renombrado atomico final no llego a ocurrir, luego no se publico un informe incompleto con
apariencia de validado. El temporal se commitea como evidencia.

**H2 CONFIRMADO, y el veredicto de P79 era erroneo.** `p72_01:92-95` planifica con
`range(m.EPOCAS)` y `SEMILLA + ep`, o sea 20261016 a 20261020; `p72_01:99-103` entrena con
`enumerate(planes, start=1)` y `SEMILLA + ep`, o sea 20261017 a 20261021. Los dos bucles se llaman
`ep` y P79 los clasifico juntos. Solo en el plan: 20261016. Solo en el entrenamiento: 20261021.
1395 lotes planados frente a 1398 recorridos.

**H4 CONFIRMADO.** El congelado solo tiene `W`, la tabla de entrada, y una sola matriz. A12 da **0
candidatos** sobre 19 `.npz` por cabecera y 320 `.json`, con criterio de forma exacta `(77657, 100)`,
que es el defecto que P80_arranco a corregir. A13 queda NO APLICABLE POR CONSTRUCCION: la tabla de
salida no existe, no se sustituye por `W` y **P80 no publica ninguna cifra sobre el artefacto**.
Las cifras absolutas de P79 describen un par `(E, S)` con `E = S = W` que nunca existio y **no son
evidencia del estado final**.

**H5 CONFIRMADO** en `v9_04` y `v10_11`: la metrica que decide cuando restaurar pesos es la del
pliegue externo que despues se puntua, sin particion interna. **H6 CONFIRMADO** con sonda
ejecutada: la GRU corre antes de enmascarar (`v9_04:104-105`, `v10_11:170-171`), el efecto es de
arrastre y no de explosion, y pesa 0.08399328135684547 en distancia relativa L2.
**H7 CONFIRMADO**: DL-03 entreno sobre 1731 filas etiquetadas completas, y 1483 + 248 = 1731, luego
**el holdout v2 estaba dentro del pool** de la validacion cruzada.

**Dos salvedades que el informe del script no lleva.** La regla 3 de A19, el desempate por nombre,
puede clasificar como validacion un argumento que llega como parametro sin vinculacion local, por
ejemplo `x_val`; es latente, no se activa en los tres ficheros reales y ahi manda el vinculo local,
pero es una apuesta por nombre y no una medicion. Y `asignaciones_del_llamante` usa `setdefault`,
luego si un nombre se asigna dos veces gana la primera sin marcar ambiguedad; se comprobo que falla
hacia el lado conservador, da REFUTADO y no sobre-detecta.

```text
H1 H2 H3 H4 H5 H6 H7 = CONFIRMADO
P80 = MEDIDO Y NO PUBLICADO   (falta una clave, vuelve a ml-coder)
congelado 302c994d... intacto antes y despues      pre-registro 440ab13c... sin cambios
```

**Sin escrituras fuera de** `reports/p80_auditoria_w2v.json.tmp_p80` y `figuras/p80/`. P71 a P79,
V03, los umbrales 0.44 / 0.51 / 0.46 y `modelos/` intactos. Test, holdout, leaderboard y submission
en cero.

```text
V03 remains the best verified leaderboard result.
```
<!-- /P80-SEGUNDA-CORRIDA 2026-09-30 -->

<!-- P80-PARADA-A12 2026-09-30 -->
## P80: la auditoria Word2Vec se para en A12. H1 H2 H3 confirmados, H4 a H7 abiertos

```text
A1  CONFIRMADO   A2  CONFIRMADO   A3  CONFIRMADO   A4  CONFIRMADO   A5  CONFIRMADO
A6  CONFIRMADO   A7  CONFIRMADO   A8  CONFIRMADO   A9  CONFIRMADO   A10 CONFIRMADO
A11 CONFIRMADO   A12 PARADA       A13 a A19 sin ejecutar

11 CONFIRMADO   0 REFUTADO   0 NO APLICABLE   1 STOP   7 sin ejecutar
corrida de menos de 2 segundos, codigo de salida 1, sin entrenamiento y sin GPU
```

**La parada** la manda el pre-registro, seccion 7, clausula 5. A12 busco una tabla de salida en todos
los `.npz` de `modelos/` y `features/` y en todas las claves de los `.json` de `reports/`, y salio
con siete candidatos. Ninguno es una tabla de salida: cuatro son el `W` de **entrada** de otras
corridas (77657 por 200 o 300) y tres son claves de forma, listas de dos enteros que describen una
forma y no pesos. El detector empareja por primera dimension `77657` sin mirar la dimension, y
cuenta cualquier clave `shape` o `forma`. No se corrigio nada. Vuelve a ml-coder.

**H2 queda CONFIRMADO y el veredicto de P79 era erroneo.** El scheduler de `p72_01` planifica con
`range(5)` y `SEMILLA + ep`, o sea 20261016 a 20261020, y entrena con `enumerate(planes, start=1)` y
`SEMILLA + ep`, o sea 20261017 a 20261021. Los dos bucles se llaman `ep` y P79 los clasifico juntos.
El desfase es de una epoca: 20261016 se planifica y no se entrena, 20261021 se entrena y no se
planifica, y el curriculum de `lr` sale de un total de lotes que no es el real. Lineas `p72_01:92-95`
contra `p72_01:99-103`, con `SEMILLA` y `EPOCAS` en `p71_04:62-64`.

**H3 queda CONFIRMADO.** La identidad con el signo correcto se cumple con residuo 1.11e-16 y el
control discriminante da 2.77, luego la asercion de P79 comparaba `pos + neg` contra `-(pos + neg)`
y fallaba siempre sin informar de nada.

```text
H1  CONFIRMADO     A1 A2 A3 A4 A5
H2  CONFIRMADO     A6 A7 A8
H3  CONFIRMADO     A9 A10
H4  SIN VEREDICTO  A11 CONFIRMADO, A12 PARADA, A13 sin ejecutar
H5  SIN VEREDICTO  A14 sin ejecutar
H6  SIN VEREDICTO  A15 A16 sin ejecutar
H7  SIN VEREDICTO  A17 A18 A19 sin ejecutar

P80 = INCONCLUSA  (sin informe completo, sin markdown, sin figuras)
congelado 302c994d... intacto antes y despues      pre-registro 440ab13c... verificado
```

**Sin escrituras fuera de `reports/p80_auditoria_w2v.json`.** P71 a P79, V03, los umbrales 0.44 /
0.51 / 0.46 y `modelos/` intactos. Test, holdout, leaderboard y submission siguen en cero.

```text
V03 remains the best verified leaderboard result.
```
<!-- /P80-PARADA-A12 2026-09-30 -->


<!-- P78-CONTROL-ANIDADO 2026-09-30 -->
## P78: el embedding aporta, pero el tokenizer era el cuello de botella

```text
P78 = RAMA 1 del criterio pre-registado

A Word2Vec  CRUDA        0.592444   sd 0.016930
B aleatoria  CRUDA        0.569137   sd 0.006724
C Word2Vec  NORMALIZADA  0.650999   sd 0.017442
D aleatoria  NORMALIZADA  0.626106   sd 0.005729

P1 = C - D  +0.024893  sd 0.021588  5/5 folds  IC95 boot (0.007902, 0.040645)
P2 = A - B  +0.023307  sd 0.023590  4/5 folds  IC95 boot (0.006199, 0.039429)
P3 = C - A  +0.058555  sd 0.015017  5/5 folds  IC95 boot (0.040710, 0.076529)
P4 = D - B  +0.056969  sd 0.001272  5/5 folds  IC95 boot (0.040363, 0.073252)

el gate pasa, pero es FRAGIL: el bootstrap propaga cero variabilidad de reentrenamiento
y el delta por semilla de P1 va de +0.0428 a +0.0009. El IC t con n=3 contiene el cero

el efecto de la normalizacion es MAS GRANDE que el del embedding: 65 por ciento del total
cobertura 68.64 -> 88.73 por ciento, OOV 31.36 -> 11.27, filas totalmente muertas 70 -> 1
el OOV era 68.3 por ciento ruido de caja, no falta de corpus

control pareado verificado por fuera: normas de fila identicas, y coseno intra-brazo
+0.1598 en Word2Vec frente a -0.0002 en la aleatoria. P1 mide informacion de similitud

P79: el objetivo de p71_04 NO es SGNS, el factor 1/K es exacto. El scheduler SI esta bien
```

**Reetiquetado**: P73 y P74 pasan de "evidencia reproducible" a **exploratorios, protocolo
contaminado**: elegian el checkpoint con el pliegue externo de validacion. Sus numeros se conservan.

**Cerrado**: la seccion de corpus. El techo de cobertura es de 88.73 a cerca de 91.3 por ciento y
la procedencia no lo permite. Lo que procede es `MIN_COUNT` de 3 a 2, que no necesita corpus nuevo.

```text
V03 = FROZEN      Word2Vec Twitter 100d = FROZEN
P72 = CLOSED      P73 = EXPLORATORIO      P74 = EXPLORATORIO
P75 = SIN TRANSFERENCIA      P76 = SIN TRANSFERENCIA      P77 = CASO 1
P78 = RAMA 1, gate fragil      P79 = OBJETIVO CONFIRMADO
TEXTCNN = ACTIVA SOLO BAJO EL ALCANCE DEL OPERADOR SELECTIVO
test = 0   holdout = 0   leaderboard = 0   submission = 0
```
<!-- /P78-CONTROL-ANIDADO 2026-09-30 -->

<!-- P77-SELECTIVO 2026-09-30 -->
## P77: el selector funciona y no alcanza

```text
P77 = CASO 1  NINGUN_OPERADOR_TRANSFIERE

A V03                    0.7772114525   switches    0
B CNN sola               0.5838755350   switches 1081
C determinista nivel 1   0.7579728990   switches  154   delta -0.0192385535   0/5 folds
D determinista nivel 2   0.7259394272   switches  393   delta -0.0512720253   0/5 folds
E gate lineal anidado    0.7215155047   switches  543   delta -0.0556959478   0/5 folds

monotono en el numero de conmutaciones. Bootstrap incondicional, los tres IC excluyen cero

el gate SI discrimina: AUC outer 0.61 a 0.74, interno medio 0.7016
precision del selector 0.3517 frente a una base de 0.2479
tipo de cambio 0.3296 -> 0.5426, y el macro-F1 necesita mas de 1

la senal del gate es sobre todo la confianza de V03: m_v cambia -0.61, m_c solo +0.16
TR dana mas que HS y AG, al contrario de lo que sugeria el AUC 0.6045 de P75
```

**Cerrado**: el operador selectivo de dos niveles con corte por cuantil, y el gate lineal anidado
de tres features, construidos sobre las salidas de V03 y del CNN. **Sin finalista. La segunda
particion no se ejecuta.**

**Abierto**: un operador que use informacion distinta del score del CNN, porque el score del CNN es
justo lo que no tiene poder discriminante suficiente.

```text
V03 = FROZEN      Word2Vec Twitter 100d = FROZEN
P72 = CLOSED      P73 = COMPLEMENTARY      P74 = EVIDENCIA REPRODUCIBLE
P75 = SIN TRANSFERENCIA      P76 = SIN TRANSFERENCIA      P77 = CASO 1
TEXTCNN = ACTIVA SOLO BAJO EL ALCANCE DEL OPERADOR SELECTIVO
test = 0   holdout = 0   leaderboard = 0   submission = 0
```
<!-- /P77-SELECTIVO 2026-09-30 -->
