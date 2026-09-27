# V21: cierre de la fase. Triaje, dictamen y decision

Sin envio de submissions. Ningun modelo entrenado en la parte que decide.

## 1. La tabla que se pidio

| candidato | DeltaOOF | rescates | danos | **neto** | novedad | coste | submission |
|---|---:|---:|---:|---:|---|---|---|
| **FUSION-WS-3** | +0.005747 | 74 | 63 | **+11** | ya entrenada (ws=3) | **cero** | **no** |
| FUSION-VS-100 | +0.005046 | 77 | 67 | +10 | ya entrenada (dim=100) | **cero** | **no** |
| L1 C=4.0 | +0.006431 | 208 | 205 | +3 | **nueva** (P1-03) | bajo | **no** |
| All-but-the-top (D=1..3) | no medida | -- | -- | -- | **nueva** | bajo | **no** |
| SSWE (objetivo dirigido) | no medida | -- | -- | -- | **nueva** | alto | **no** |
| CBOW / GloVe / word2vec | no medida | -- | -- | -- | nueva parcial | **alto** | **no** |

Referencia: V03 = 0.7772114525165721 OOF, **0.78306 LB**. Submissions restantes: **4**.

Orden pre-registrado por si hubiera restriccion externa: **WS-3 > VS-100 > L1 C=4.0**.

## 2. Decision

# NO CANDIDATE JUSTIFIES SUBMISSION

Las cuatro submissions se reservan. Ninguna familia abierta por V21 justifica gastar una.

## 3. Por que, con los tres argumentos que resisten

### 3.1 El cambio de WS-3 no es estadisticamente distinguible de ruido

74 rescates contra 63 danos sobre 137 pares discordantes. Test exacto de McNemar
bilateral: **p = 0.3930**. La diferencia de proporciones entre pares discordantes es
+0.0803, y su desviacion tipica es 0.0857: el efecto esta a menos de un sigma.

Lo que ese perfil implica sobre el leaderboard, con la escala del proyecto (107 cambios
netos de celda ~ 0.003 de LB) y la tasa de acierto observada: **+0.00024**, es decir cerca
del **2 % de un error estandar** del LB (0.01219).

### 3.2 WS-3 y VS-100 no son dos oportunidades

Solapan el **80.3 %** de sus cambios. Son casi la misma perturbacion. En el mejor caso WS-3
la domina; en el peor son la misma. Contarlas como dos candidatas seria contarlas dos veces.

### 3.3 L1 tiene el mayor DeltaOOF y el peor perfil, y repite el mecanismo de V20

| candidata | celdas | neto | DeltaF1/celda | mayor desplazamiento de tasa |
|---|---:|---:|---:|---|
| WS-3 | 137 | +11 | 4.20e-05 | +0.0121 (HS) |
| VS-100 | 144 | +10 | 3.50e-05 | +0.0081 (HS) |
| **L1 C=4.0** | **413** | **+3** | 1.56e-05 | **+0.0357 (TR)** |

L1 es 2,7 veces menos eficiente por celda que WS-3, y su desplazamiento de tasa en TR es
ocho veces mayor que el de WS-3 en cualquier etiqueta, **con saldo negativo en esa misma
etiqueta** (-3 de 103). V20 movio 51 celdas de una sola etiqueta, todas en una sola
direccion, tasa +0.0442, saldo -6, y el leaderboard bajo **0.00643**. Es la misma mecanica.

## 4. Auditoria del dictamen del especialista: dos correcciones

El dictamen se sostiene, pero dos de sus cifras no.

### 4.1 McNemar: p = 0.3930, no 0.347

Recalculado con el test exacto bilateral sobre 137 pares discordantes. El especialista
aproximo. **La conclusion no cambia**: 0.393 tampoco es significativo.

### 4.2 La regresion dLB sobre dOOF no es una base solida, y se descarta

El especialista ajusto `dLB = -0.00851 + 0.252 * dOOF` con `n = 7`. Recalculada con las
sondas que tienen `local_f1` numerico, son **cuatro**, no siete, y el ajuste es otro:

```
dLB = -0.00744 + 0.508 * dOOF    r = +0.865    R2 = 0.748    n = 4
```

Un `r` de 0.865 con cuatro puntos no es una ley, es una recta por cuatro observaciones, y
esta dominada por V20, que es un punto anomalo (dOOF +0.0008 con dLB -0.0064). Ademas
`local_f1` **no significa lo mismo en todas las fases**: V16 optimizo la tasa predicha, y
sus `local_f1` miden otra magnitud que el `corrected_cv` de V03. Restar 0.777211 a eso no
produce un dOOF.

**La regresion no se usa para decidir.** La decision se sostiene en el McNemar y en el
perfil de celdas, que no dependen de esa construccion.

### 4.3 Correccion heredada de P0

La afirmacion de que "las cinco sondas de V16 se enviaron con DeltaOOF positivo" era
incorrecta y ya esta corregida en `reports/v21_p0_00_inventario_v5_v5_1.md`. Con los datos
disponibles, tres de cuatro tienen dOOF **negativo** y solo V20 lo tiene positivo. V16 no
fue "senal positiva que no transfirio", fue "senal ya nula o negativa que tampoco
transfirio". La conclusion de la fase no cambia, pero el diagnostico era mas erroneo de lo
que se creia.

## 5. El techo, y por que ninguna de las tres familias lo alcanza

Del informe de estado del arte (`reports/v21_p1_02_toxicidad_nlp_clasico.md`, 1.056 lineas,
referencias verificadas una a una):

- El techo medido de **todo** el programa de embeddings es **+0.005748**. Y casi iguala al
  DeltaOOF de la sonda v05 Optuna (+0.0050), que perdio 0.00704 en el leaderboard. Es decir,
  el maximo de la familia mas trabajada del proyecto es del orden de lo que ya se sabe que
  no transfiere.
- La **regresion** (con las salvedades de seccion 4.2) situaba el punto de equilibrio en un DeltaOOF
  de +0.033: seis veces el techo.
- La matriz de evidencia publicada por idioma es dura: en persa con 6.000 filas gana la linea
  clasica (+0.03 a +0.21 macro-F1), pero **en espanol toda la evidencia publicada favorece al
  modelo con embeddings**, porque no se pueden usar y porque el ensemble con word y char
  n-grams es mas de lo que aporta un vector de 200 numeros.
- La **unica** palanca con evidencia solida y reproducible en espanol es el
  **preprocesamiento del texto de la tarea** (elongaciones, emoticones, usuarios, hashtags
  como marcas). V5 lo aplico al *corpus* y jamas al *texto de la tarea*, que es
  precisamente la separacion que el encargo de V21 pedia explicitar y que nadie habia
  recorrido.

Ese ultimo punto es el unico que no queda cubierto por este cierre, y queda registrado como
tal: **no se ha medido**, no se ha descartado, y no se ha incluido en los candidatos porque
ningun dato de esta fase lo respalda todavia.

## 6. Lo que esta fase dejo, mas alla de la decision

Tres hallazgos que corrigen el registro y que no existian antes:

1. **El NB-SVM del proyecto estaba mal implementado**, en ninguno de los cinco puntos de la
   receta canonica (desviacion del ratio 4.32 frente a 0.79 del canonico, con 30-46 % de
   valores fuera de [-3, 3]). Y la version **correcta** es **peor** (0.7638), porque
   binariza y por tanto descarta el term frequency, que en tweets toxicos es senal.
2. **Las metricas de la linea de embeddings son sensibles a detalles no publicados.** Tres
   diferencias entre una reimplementacion razonable y el pipeline de V5 cambiaron el
   resultado, y una de ellas hacia que FUSION-B0-200 diera 0.7728 en vez de 0.7818. La
   causa: V5.1 consulta al binario de fastText sobre el `.bin`, donde los 2.282 tipos fuera
   de vocabulario reciben vector por subpalabra, mientras que el `.vec` solo exporta los
   100.098 tipos del modelo. Toda comparacion futura debe declarar de donde salen los
   vectores de tipo.
3. **L1 recorta la brecha train-OOF de 0.1980 a 0.0665** a C=2.0. El mecanismo es real y
   esta medido, pero no se convierte en celdas acertadas.

## 7. Lo que no se hizo y por que

- **No se gasto ninguna submission.** La instruccion era no gastar sin evidencia
  competitiva, y ninguna candidata la tiene.
- **No se abrio V22.** Tres familias quedan sin medir por decision informada, no por agotamiento de ideas.
- **No se toco V03.** Su sha256 sigue siendo
  `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`.
- **No se formaron familias para rellenar el presupuesto.** El encargo permitia 8
  experimentos; se gastaron 0 de entrenamiento y 2 de reevaluacion.
