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
