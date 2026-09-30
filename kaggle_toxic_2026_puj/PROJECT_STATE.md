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
