
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
