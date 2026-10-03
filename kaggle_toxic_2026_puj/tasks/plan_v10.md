# Plan pre-registrado de la fase v10: calibracion, distribucion y gobierno

Fecha de pre-registro: **2026-09-26**. Este fichero se escribe **antes** de ejecutar
ninguna linea de la fase. Los gates que siguen son los unicos que se aplican, y su criterio
de cierre esta fijado por adelantado. Si un resultado no los cumple, la sub-via se cierra con
evidencia y no se reintenta con mas hiperparametros.

Worktree: `/home/carlos/Git/PLN-v10/kaggle_toxic_2026_puj`.
Rama: `proyecto-kaggle-toxic-2026-v10`, desde `23aac93`.

## 1. Estado de partida, verificado en P0

| comprobacion | valor | veredicto |
|---|---|---|
| `corrected_cv` reconstruido del OOF | 0.7772114525165721 | coincide exacto |
| `sha256` de la submission v03 | 7d7478f2...4520e | identico |
| filas de la submission v03 | 1154 | correcto |
| filas de `experiments.csv` | 207, con 207 claves unicas | correcto |
| sondas en el registro del leaderboard | 15 | correcto |
| prevalencias de entrenamiento | HS 0.4367 / TR 0.2998 / AG 0.3824 | correcto |
| grupos fusionados | 1645 sobre 1731 filas | correcto |
| artefactos en `features/` y `models/` | 22 y 9, copiados | correcto |

Referencia canonica: **0.777211** en el desarrollo de 1483 filas, y **0.78306** en el
externo, con un error estandar aproximado de 0.013 a 1154 filas.

## 2. Reglas que no se negocian

1. Sin transformers, sin modelos preentrenados de terceros, sin recursos externos
   etiquetados, sin pseudo-etiquetado y sin etiquetas de prueba.
2. El diagnostico de shift de covariables es **solo diagnostico** y, como maximo,
   repondera la validacion local. Nunca se entrena con la prueba.
3. **Nunca se optimizan umbrales sobre el mismo OOF que se reporta.** La auditoria midio que
   hacerlo infla +0.0036. Toda optimizacion va con validacion anidada.
4. El leaderboard no se consulta para ajustar nada durante el desarrollo. Las sondas externas
   son decisiones pre-registradas al final, **maximo una por fase**.
5. El holdout v2 esta agotado, 5 de 5. No se vuelve a usar.
6. Semillas fijas y reproducibilidad con tolerancia 1e-6. FastText con varios hilos no es
   determinista, asi que se compara por vocabulario y vecinos, no por huella.
7. Artefactos en ASCII puro. No se borra ni modifica historia; las correcciones se registran
   como original y corregido.
8. Sin push, sin merge, sin rebase.
9. Evidencia o `UNKNOWN`: nada se afirma sin artefacto que lo respalde.
10. **Filtro de detectabilidad de V8-02**, aplicado a cada linea de la via A como criterio de
    priorizacion: una candidata que cambie `C` celdas solo es resoluble por el externo si la
    tasa de acierto entre esas celdas supera `0,5 + 19,2 / C`. Por debajo del minimo, la
    sonda no se gasta. El externo tiene 1154 filas y solo resuelve unos 19,2 aciertos netos,
    por lo que un cambio pequeno es indistinguible del azar por construccion.

## 3. Advertencia que queda escrita antes de ver resultados

**El holdout2 no es un conjunto virginal.** Se construye con un pliegue del desarrollo actual,
y esas 1483 filas ya participaron en las decisiones de las fases v1 a v9. Su valor real es
**detectar que un candidato esta sobreajustado a su gate**, que es un uso legitimo y
necesario, y **no** es probar generalizacion sobre datos nunca vistos.

Decir lo contrario despues de ver los numeros seria una reinterpretacion, y por eso esta
advertencia se escribe antes de ejecutar nada. La consecuencia practica es que una mejora
aprobada en holdout2 debe leerse como "sobrevive a un conjunto que ya se uso para decidir
otras cosas", y no como "generaliza".

Por esa razon holdout2 se crea **antes** de la via A, para que el juez de las adopciones exista
cuando se empiece a buscar candidatas. Si se creara despues, las candidatas de la via A se
elegirian sobre las mismas filas que luego se usarian para juzgarlas.

## 4. Gates de cada sub-via

### P0b, holdout2 congelado

- Composicion: `StratifiedGroupKFold(n_splits=6, shuffle=True, seed=20260926)` sobre el
  desarrollo actual de 1483 filas con los grupos fusionados. Un pliegue de unas 247 filas es
  holdout2, y los otros cinco de unas 1236 son el desarrollo nuevo, **dev2**.
- Se congelan `config/holdout2.json` y `data/processed/holdout2_ids.csv` con `sha256`.
- **Gate**: cero grupos compartidos entre holdout2 y dev2, y la referencia de V03 recalculada
  en dev2 con los mismos umbrales congelados.
- **Cierre**: si el recalculo se aparta de 0.7772 en mas de 0.005, se documenta como coste de
  reducir el conjunto de ajuste y **no** se usa dev2 para ninguna decision de adopcion.

### P1, via A de la capa de decision

Objetivo: convertir el shift de prevalencia medido en una mejora **anidada** de al menos
+0.002, o cerrar la via con evidencia.

- **V10-01, calibracion fuerte.** Platt frente a isotonica recortada en [0.01, 0.99] frente
  a calibracion beta de tres parametros, todas por etiqueta y fold-local, evaluadas en OOF
  anidado. Se reportan ECE, MCE, Brier y NLL.
  **Gate**: la mejor calibracion reduce el ECE al menos un **20 %** frente a Platt **y** no
  empeora la F1 anidada.
  **Cierre**: si ninguna cumple, la sub-via de calibracion se cierra documentado.
- **V10-02, regla de decision F1 optima.** En vez de tres umbrales por celda, se usa la
  iteracion de punto fijo que maximiza la F1 esperada a partir del marginal calibrado y de
  la prevalencia estimada. Se evalua anidado, con prior natural y con prior simulado de 0.94.
  **Gate**: ganancia anidada de al menos **+0.002** bajo prior simulado 0.94, **con TR y AG no
  negativos**.
  **Cierre**: si no cumple, no se genera sonda.
- **V10-03, umbrales estables.** Tres parametrizaciones: tres umbrales libres, un desplazamiento
  global con escala por etiqueta, y un encogimiento hacia la mediana agrupada. Se reporta la
  ganancia anidada y la **dispersion** entre particiones.
  **Gate**: igualdad o mejora de ganancia **con menor dispersion**.
  **Cierre**: si la dispersion no mejora, se documenta que la holgura de la capa de decision
  es irreducible con estos datos.

Ademas, y con independencia de los gates de cada linea, **el filtro de detectabilidad de V8-02
debe superarse** para que la sonda v19 se genere. Sin eso, la linea se mide, se registra y no
se envia.

Expectativa declarada de antemano: **baja**. El antecedente mas directo es V9-02, que midio
+0.0006 de ganancia anidada para el mecanismo de shift de prevalencia, y V8-09, que encontro
que la relacion entre la tasa de positivos predicha y el externo es una curva con maximo
exacto en la tasa de V03. La via A es una apuesta disciplinada, no un tramite.

### P2, diagnostico de shift de features

- Discriminador entre entrenamiento y prueba sobre la union TF-IDF, con validacion cruzada
  por grupos, AUC fuera de muestra y las 20 caracteristicas mas desplazadas.
- **Gate de decision, pre-registrado**: AUC **menor o igual que 0.55** cierra la via de shift de
  caracteristicas y deja solo la correccion de prevalencia de P1. Entre 0.55 y 0.65 se usa el
  cociente de densidades **solo** para reponderar la validacion local y se reporta su efecto.
  Mayor que 0.65 se reporta y se propone un plan, sin entrenar nunca con la prueba.
- **Cierre**: la via se cierra con veredicto explicito, CERRADA o ABIERTA.

### P3, confirmacion de referencia en dev2

- Se verifica que los grupos no se solapan, se recalcula la referencia de V03 en dev2 y se
  comprueba que los artefactos de holdout2 no han cambiado.
- **Cierre**: sin candidatos que confirmar, esta fase se cierra con la verificacion.

### P4, via C de cierre cientifico de H1

- V10-06 repite V9-04 con embeddings **no congelados**, inicializados con el `.vec` de FastText,
  learning rate diferencial y early stopping estricto.
- Se compara contra 0.6943 (congelado), 0.6636 (media mas regresion lineal) y 0.7772 (V03).
- **No hay gate de externo**: es cierre cientifico. Si sigue por debajo de V03, H1 queda
  cerrada con dos variantes, congelada y afinada, documentado.

### P5, cierre de fase

Informe de fase, decisiones, `RUNLOG` y `PROJECT_STATE` al dia, notebook con seccion 20
reejecutada con cero errores.

## 5. Regla de adopcion de la submission final

V03 sigue siendo la submission recomendada salvo que un candidato cumpla **las cinco**
condiciones a la vez:

1. mejora anidada de al menos **+0.002**;
2. simulacion de shift coherente;
3. holdout2 aprobado en evaluacion unica;
4. reproducibilidad exacta con tolerancia 1e-6;
5. revision adversarial sin contraindicios.

Si un candidato las cumple, genera **una sola sonda** y se decide con su externo. Si nada las
cumple, **V03 se mantiene** y se documenta el cierre: ninguna familia se reabre sin un
mecanismo nuevo.

## 6. Regla de envio

Reenviar una submission ya puntuada no aporta nada. La unica sonda nueva con sentido al cierre
es la v19 de P1, y solo si su gate la respalda. La v18 es identica a V03 y no se envia. La v17
solo se considera si P1 la confirma.

## 7. Addendum de protocolo, fechado el 2026-09-26 DESPUES de P0b y V10-02

Este addendum se escribe **antes** de ejecutar ninguna linea de la via A. Se incorpora porque
V10-02 midio dos cosas que obligan a cambiar el protocolo, no porque ninguna linea de la via
A haya dado mal resultado: **la via A todavia no se ha ejecutado.**

### 7.1 Lo que midio V10-02

- El control de equivalencia da diferencia **exactamente 0.0**: el codigo reproduce la
  referencia canonica al ultimo digito. La desviacion de P0b no es un fallo de codigo.
- Con **ocho semillas de particion y sin cambiar nada mas**, la metrica corregida recorre de
  0.772021 a 0.789205: recorrido **0.017184**, desviacion tipica **0.005302**.
- Con **ocho subconjuntos de 1236 filas**, recorre de 0.762169 a 0.782417: recorrido
  **0.020248**, desviacion tipica **0.007134**. El 0.763436 de dev2 cae dentro de esa banda.
- Es decir: **la metrica corregida tiene una banda de ruido de mas de 0.017 entre particiones
  equivalentes**, y por tanto un delta de +0.002 medido en un solo calculo esta muy por debajo
  del suelo de medicion del propio instrumento.

Esto confirma por una via independiente, y sin gastar una sola sonda, el ruido de 0.01524 que
V8-02 medico en el externo. Dos instrumentos, misma magnitud.

### 7.2 Lo que cambia

El gate de la via A estaba escrito como "mejora anidada de al menos +0.002 en un calculo".
Eso es **inoperante** con la banda medida: un resultado de +0.002 es indistinguible de elegir
una particion afortunada, y es exactamente el mecanismo por el que D-067 concluyo que la
mejora local de las lineas anteriores era el promediado y no la tecnica.

Por eso, a partir de aqui, **toda candidata de la via A se evalua con las tres cifras
siguientes, y las tres se reportan siempre, gane o pierda:**

1. **Delta medio** sobre `K = 5` particiones de grupo con semillas distintas.
2. **Intervalo al 95 por ciento pareado** del delta, con re-muestreo de filas.
3. **Recuento de particiones que mejoran**, de 0 a `K`. Este es el criterio que en D-067
   resulto ser el mas informativo: una mejora en 1 de 5 es ruido de particion, y una mejora
   en 5 de 5 no lo es.

### 7.3 Criterio de adopcion corregido

Una candidata de la via A se considera **adoptable** solo si cumple las tres cosas:

- delta medio de al menos **+0.002**;
- **al menos 4 de 5** particiones mejoran, la misma regla de la fase v8;
- el intervalo pareado al 95 por ciento **no incluye el cero**.

Los gates por sub-linea del punto 4 del plan original **siguen vigentes** tal como estaban
escritos: reduccion de ECE del 20 por ciento para la calibracion, y TR y AG no negativos para
la regla de decision. Este addendum no los relaja, solo anade el instrumento con el que se
miden, que antes estaba dado por supuesto en un solo calculo.

### 7.4 Que pasa con holdout2, que fallo su gate

El gate de P0b **no se supero** y eso no se reinterpreta. Lo que cambia es el uso, no el
veredicto:

- **dev2 no se usa para ninguna decision de adopcion**, tal como dice la regla pre-registrada.
- **holdout2 si puede seguir funcionando como juez**, pero solo por **comparacion pareada
  dentro de sus propias 247 filas**: V03 contra la candidata sobre las mismas filas, con
  intervalo. No por comparacion de numeros absolutos contra 0.7772, porque la banda de ruido
  medida hace que esa comparacion no signifique nada.
- Y con un limite que hay que declarar: la banda de 247 filas no resuelve los efectos que las
  candidatas reales producen. Contraste medido: una perturbacion **pura de degradacion** del
  5 por ciento de las celdas da -0.0287 con intervalo que no incluye el cero, pero esa es una
  perturbacion en una sola direccion, y una candidata real mueve celdas en direcciones mixtas
  con un efecto neto mucho menor. **El numero de particiones que mejoran es mas informativo que
  el intervalo de holdout2**, y por eso es criterio primario.

### 7.5 Lo que este addendum no cambia

No se abre ninguna familia cerrada. No se relaja ninguna restriccion academica. No se permite
usar el externo para ajustar nada. El holdout v2 sigue agotado. La regla de envio de una sola
sonda por fase sigue igual, y ademas la sonda solo se genera si la candidata supera el filtro
de detectabilidad de V8-02.
