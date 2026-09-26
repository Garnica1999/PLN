# V12: auditoria de brecha (P0)

Generado por `experiments/v12_00_auditoria_brecha.py` (V12-00 (P0, auditoria de brecha)).
Commit: `039eae4`. Modulos analizados: **127**.

Referencia: **V03 = 0.7772114525165721**, leaderboard 0.78306. Huella de la submission
identica al congelado: **True**.

## 1. Por que esta auditoria se sostiene sobre una sonda estructural

V11-03 cometio el error de buscar palabras prohibidas en el texto de los ficheros y
produjo doce falsos positivos, porque las coincidencias eran las citas de la prohibicion y
su propia lista de patrones. **Un escaner textual es inservible en un proyecto que
documenta sus propias prohibiciones**: marcar sus propias listas ensena a ignorar sus
alarmas.

Aqui se analiza el **arbol sintactico**: se recoge cada `import`, cada `class` y cada
llamada de todos los modulos del proyecto, y se busca ahi. Un nombre en un `import` es un
uso; un nombre en un comentario no aparece. Eso permite ademas lo que la busqueda textual
no permite, que es **probar la ausencia**: afirmar que un mecanismo no se ha probado
exige un universo cerrado, y el universo es el codigo del proyecto.

## 2. Reparto del inventario

| estado | familias | significado |
| --- | --- | --- |
| CLOSED_VALID | 16 | probada con protocolo correcto y cerrada con evidencia, no por abandono |
| PARTIALLY_TESTED | 5 | el mecanismo se probo en una forma, pero la forma que se propone es distinta |
| GENUINELY_UNTESTED | 3 | no existe ninguna ocurrencia en el codigo: la sonda lo confirma, no lo supone |
| INVALIDLY_TESTED | 0 | ninguna: no se ha encontrado ninguna prueba invalida |

Se distinguiron **9 ejes**: model family, representation, loss, training objective, decision layer, ensemble strategy, sampling strategy, label structure, error correction.

## 3. Las tres familias genuinamente no probadas

Estas son las unicas tres, y la afirmacion es fuerte: **la sonda AST no encuentra ni una
sola ocurrencia** en todo el codigo del proyecto.

### Funcion de perdida propia: objetivos de scoring proper, perdida asimetrica, perdida tolerante al ruido

- **Eje**: loss
- **Hipotesis**: La entropia cruzada no es la unica perdida razonable para un F1 macro, y una perdida que tolere mejor el ruido de anotacion extraeria senal que la actual no ve
- **Experimentos**: ninguno
- **Mejor resultado**: nunca medido
- **Lo que si es nuevo**: **No existe ni una sola clase de perdida propia en todo el proyecto.** La sonda AST sobre el arbol sintactico completo no encuentra ninguna definicion de clase que derive de una perdida, ni ninguna funcion de perdida propia. Todo el ajuste se ha hecho con instance weights, que es otra cosa: pesa ejemplos, no cambia la forma de la funcion
- **Coste**: medio: hay que implementar y validar la derivada | **Riesgo de leakage**: bajo si los parametros se eligen dentro del train

### Entrenamiento contrastivo o por pares sobre TF-IDF disperso

- **Eje**: training objective
- **Hipotesis**: Separar explicitamente ejemplos de etiquetas distintas y acercar los de la misma, en el espacio disperso existente, expone estructura que el discriminador no codifica
- **Experimentos**: ninguno
- **Mejor resultado**: nunca medido
- **Lo que si es nuevo**: **Cero ocurrencias de contrastive, pairwise, margin_loss o hinge como objetivo propio en todo el codigo.** La sonda AST no encuentra ninguna clase ni ninguna llamada. Y conviene decirlo con precision: la perdida hinge **si** se probo, pero como clasificador completo, LinearSVC, no como termino auxiliar sobre el modelo logistico. Lo que no existe es el objetivo de pares
- **Coste**: medio-alto: pares implican una reformulacion del bucle de entrenamiento | **Riesgo de leakage**: bajo, pero la separabilidad debe medirse en validacion, no en el train

### Regularizacion distinta por etiqueta Y por grupo de caracteristicas

- **Eje**: representation
- **Hipotesis**: La misma caracteristica puede tener utilidad diferente para HS, TR y AG, y compartir una sola estructura de penalizacion puede limitar una o mas etiquetas
- **Experimentos**: ninguno equivalente
- **Mejor resultado**: nunca medido en esta forma
- **Lo que si es nuevo**: V8-06 varia el C **de la etiqueta** sobre el espacio entero. Aqui lo que se varia es el C **del grupo de caracteristicas dentro de la etiqueta**, sobre una union de word y char_wb que el codigo de V03 construye ya separada. Es un mecanismo distinto, y no aumenta la dimension
- **Coste**: bajo: es el mismo pipeline con dos constantes en vez de una | **Riesgo de leakage**: medio: dos constantes por etiqueta Multiplican el riesgo de ajustar en el exterior, asi que exige anidacion estricta

## 3bis. El escaner se valida a si mismo, con un positivo conocido

Una afirmacion de "no se ha probado" solo vale si el instrumento que la sostiene esta
validado. Y aqui hay un positivo conocido a mano: **este propio fichero contiene las
palabras "contrastive", "pairwise" y "margin_loss"**, porque las nombra al
describir la familia que no se ha probado.

- El escaner **textual** encuentra el patron en **1 fichero(s)**: v12_00_auditoria_brecha.py.
- El indice **estructural** devuelve **0**.

Es decir: **el unico fichero del proyecto donde aparece "contrastive" es el que dice
que no aparece**. Un escaner textual concluiria que el mecanismo **si** se ha probado, y
lo concluiria con la maxima confianza, porque habria encontrado una coincidencia
literal. El indice estructural, que solo mira `import`, `class` y llamadas, devuelve
cero, que es lo correcto.

Esto no es un argumento teorico: la auditoria **reprodujo el fallo de v11 en su propio
codigo** en el momento de ejecutarse. Por eso la distincion no es un criterio de estilo
sino una condicion para que las tres afirmaciones de la seccion 3 sean defendibles.

## 4. Las parcialmente probadas, y por que la forma propuesta si es distinta

Aqui la distincion importa mas, porque es facil declarar algo "ya probado" sin
comprobar que la forma coincide. En los cuatro casos **la forma nueva es genuinamente
distinta** de la que seCerro, y conviene decir cual es en cada uno.

### Modelos generativos: ComplementNB, y verosimilitud condicional de clase

- **Experimentos que la cierran**: MOD-02_BEST, MOD-02_W12, 33_class_conditional_lm_v4
- **Mejor resultado**: ComplementNB 0.6888 sobre la union y 0.7362 sobre solo palabra, frente a 0.777211 de V03
- **Por que la forma nueva no es la misma**: El generativo sobre **representacion de conteos** sigue sin probarse: ComplementNB solo se ejecuto sobre TF-IDF
- **Prioridad**: baja, la evidencia previa apunta en contra

### Constante y peso de clase por etiqueta

- **Experimentos que la cierran**: V8-06, V8-07, R1-C1, R2-MINDF5
- **Mejor resultado**: V8-06 se reporta sin recomendacion de envio; V8-07 es negativa con la tasa de acierto por debajo del filtro
- **Por que la forma nueva no es la misma**: **Un C distinto por etiqueta Y por grupo de caracteristicas** sigue sin probarse: V8-06 varia el C de la etiqueta, no el del grupo dentro de la etiqueta
- **Prioridad**: media, con control obligatorio

### Sospecha de instance-label y contradiccion

- **Experimentos que la cierran**: V4-A02, V4-B05, 10_sondas_robustez
- **Mejor resultado**: El techo de la via de downweight por contradiccion esta por debajo del 3 % de los ejemplos
- **Por que la forma nueva no es la misma**: una **funcion de perdida** robusta en lugar de una instance weight sigue sin probarse
- **Prioridad**: media

### Regimenes de error y rebanadas con modelo o umbral propio

- **Experimentos que la cierran**: E2-16-LOCAL-SIN-SHRINKAGE, E2-16-SHRINK-M25, E2-16-SHRINK-M50, E2-16-SHRINK-M100, E2-16-MEJOR, E2-20-ESPECIALISTAS
- **Mejor resultado**: E2-16-MEJOR: CERRADA sin mejora. E2-20-ESPECIALISTAS: CERRADA sin mejora
- **Por que la forma nueva no es la misma**: una perdida **consciente del regimen** sobre un modelo compartido sigue sin probarse, que es distinta de un modelo por rebanada
- **Prioridad**: baja-media

### Corrector sobre margenes y desacuerdo entre pliegues

- **Experimentos que la cierran**: V8-08
- **Mejor resultado**: 192 de 3462 celdas cambian, en 161 filas; por etiqueta HS 114, TR 19, AG 59
- **Por que la forma nueva no es la misma**: interacciones **multiplicativas** entre margenes, que el corrector lineal no puede representar. Distinto de esto
- **Prioridad**: media

## 5. Hallazgo de trazabilidad del propio registro

El registro tiene **228 filas** y **V03 no tiene fila propia**:
presencia en el registro: **False**.

V03 es la referencia contra la que se mide todo, y 228 filas la citan en sus campos de
texto, pero no existe una fila que la declare. Su metrica vive en
`features/oof_v03_final.npz` y en los informes, no en la fuente de verdad de trazabilidad.

No es un error de calculo: es un hueco de **trazabilidad**, y el enunciado obliga a
registrar los huecos en vez de dejarlos. La cifra 0.7772114525165721 se ha verificado hoy
por tres vias independientes, la huella sha256 entre ellas, asi que la referencia es
solida; lo que falta es su fila.

## 6. Las diez cosas que la auditoria debe entregar

1. **Inventario completo**: 24 familias en `reports/v12_gap_audit.csv`, con los nueve
   ejes distinguidos y los identificadores de experimento de cada una.
2. **Familias ya cerradas**: 16 en CLOSED_VALID.
3. **Parcialmente probadas**: 5, cada una con la distincion razonada de
   por que la forma propuesta no es la que se cerro.
4. **Genuinamente no probadas**: 3, con sonda AST que confirma la ausencia.
5. **Bugs metodologicos o huecos**: el de la fila ausente de V03, y el hecho de que las
   cuatro variantes NB de E2-15 usaban la constante 1.0 mientras V03 usa 2.0, lo que
   hace que su cierre no sea directamente comparable. V8-04 ya lo Documento y anadio el
   control que faltaba.
6. **Artefactos reutilizables**: `features/oof_v03_final.npz` con el OOF canonico,
   `experiments/v10_03_arnes_evaluacion.py` como arnes de evaluacion validado en los dos
   sentidos, `src/splits.py` con la particion agrupada, y el corpus de 24.646 dimensiones.
7. **Coste por hipotesis**: registrado en la columna `coste` de la tabla.
8. **Riesgo de leakage**: registrado en la columna `riesgo_leakage`. Los tres huecos
   genuinos tienen riesgo bajo o medio-bajo, y ninguno exige el test.
9. **Hipotesis seleccionadas**: tres, en la seccion 8.
10. **Orden de ejecucion**: en la seccion 9.

## 7. Lo que este inventario dice sobre el estado del proyecto

De 24 familias, **16 estan cerradas validamente**. El proyecto
ha sido exhaustivo en un eje concreto: el **espacio de la representacion y del
clasificador**, con 16 de 24 familias cerradas alli.

Los tres huecos que quedan tienen una forma comun: **todos son objetivos de
entrenamiento, no representaciones**. No se ha tocado nunca la funcion que se minimiza.
Todo lo que se ha variesdo ha sido el espacio de entrada, el clasificador, la capa de
decision o el ensemble. Esa asimetria es el hallazgo de esta auditoria, y es lo que
justifica abrir V12 en lugar de seguir barriendo hiperparametros.

Y hay una segunda asimetria, en la situacion del proyecto: el techo de la
complementariedad esta medido en **12 filas de interseccion de errores**, y la mayor
ganancia anidada de toda la historia es de **+0.000148** frente a un gate de +0.002. Un
hallazgo que no mueva mas que eso no se distinguira del ruido, cuya banda entre
particiones equivalentes es de **0.017184**.

## 8. Las tres hipotesis seleccionadas, y por que estas

El enunciado pide entre 3 y 6, elegidas por ser genuinamente no probadas, por novedad de
mecanismo, por la evidencia historica, por complemento esperado, por coste y por riesgo de
leakage. Se eligen **tres**, y la justificacion va con cada una.

### H1. Funcion de perdida propia: objetivos de scoring proper, perdida asimetrica, perdida tolerante al ruido

- **Por que es genuinamente nueva**: **No existe ni una sola clase de perdida propia en todo el proyecto.** La sonda AST sobre el arbol sintactico completo no encuentra ninguna definicion de clase que derive de una perdida, ni ninguna funcion de perdida propia. Todo el ajuste se ha hecho con instance weights, que es otra cosa: pesa ejemplos, no cambia la forma de la funcion
- **Evidencia historica a favor**: el hueco de tres objetivos sin tocar, medido en la
  seccion 7.
- **Coste**: medio: hay que implementar y validar la derivada. **Riesgo de leakage**: bajo si los parametros se eligen dentro del train.
- **Por que no se eligen mas de tres**: las otras dos familias no probadas del
  inventario son la parte parcial de este mismo bloque, y meterlas seria
  desdoblar una sola idea en tres experimentos.

### H2. Entrenamiento contrastivo o por pares sobre TF-IDF disperso

- **Por que es genuinamente nueva**: **Cero ocurrencias de contrastive, pairwise, margin_loss o hinge como objetivo propio en todo el codigo.** La sonda AST no encuentra ninguna clase ni ninguna llamada. Y conviene decirlo con precision: la perdida hinge **si** se probo, pero como clasificador completo, LinearSVC, no como termino auxiliar sobre el modelo logistico. Lo que no existe es el objetivo de pares
- **Evidencia historica a favor**: el hueco de tres objetivos sin tocar, medido en la
  seccion 7.
- **Coste**: medio-alto: pares implican una reformulacion del bucle de entrenamiento. **Riesgo de leakage**: bajo, pero la separabilidad debe medirse en validacion, no en el train.
- **Por que no se eligen mas de tres**: las otras dos familias no probadas del
  inventario son la parte parcial de este mismo bloque, y meterlas seria
  desdoblar una sola idea en tres experimentos.

### H3. Regularizacion distinta por etiqueta Y por grupo de caracteristicas

- **Por que es genuinamente nueva**: V8-06 varia el C **de la etiqueta** sobre el espacio entero. Aqui lo que se varia es el C **del grupo de caracteristicas dentro de la etiqueta**, sobre una union de word y char_wb que el codigo de V03 construye ya separada. Es un mecanismo distinto, y no aumenta la dimension
- **Evidencia historica a favor**: el hueco de tres objetivos sin tocar, medido en la
  seccion 7.
- **Coste**: bajo: es el mismo pipeline con dos constantes en vez de una. **Riesgo de leakage**: medio: dos constantes por etiqueta Multiplican el riesgo de ajustar en el exterior, asi que exige anidacion estricta.
- **Por que no se eligen mas de tres**: las otras dos familias no probadas del
  inventario son la parte parcial de este mismo bloque, y meterlas seria
  desdoblar una sola idea en tres experimentos.

## 9. Orden de ejecucion

El orden lo fija el **coste y el riesgo**, no la novedad. Empezar por la mas barata con
el control mas limpio deja el resultado mas interpretable como linea base de las otras.

| orden | hipotesis | coste | riesgo de leakage | por que aqui |
| --- | --- | --- | --- | --- |
| 1 | Regularizacion por etiqueta y por grupo | bajo | medio | el pipeline es identico con dos constantes en vez de una, y su control es el mas nitido del proyecto |
| 2 | Funcion de perdida propia | medio | bajo | ataque al mecanismo central, y su control es un solo linea de codigo adicional |
| 3 | Entrenamiento por pares sobre TF-IDF | medio-alto | bajo | el mas costoso y el que mas puede fallar, pero con la condicion de medir separabilidad antes de la metrica |

**Ninguna de las tres se ejecutara sin su control equivalente**, y el control se escribe
antes que la variante, no despues.

## 10. Lo que esta auditoria NO ha hecho

- **No ha entrenado nada.** Ni un solo modelo. El enunciado lo prohibe antes de cerrar
  P0, y no hay ninguna cifra nueva en este informe.
- **No ha ejecutado P11, la revision de literatura.** El enunciado la coloca despues
  del audit y de los resultados historicos.
- **No ha evaluado holdout2**, ni ha consultado el leaderboard, ni ha usado el texto de
  la prueba para nada.
- **No ha reabierto ninguna familia cerrada.** Donde el veredicto es CLOSED_VALID, la
  lectura correcta es que no se ejecuta, no que se ejecuta con mas cuidado.


Fecha: 2026-09-26T22:27:50+00:00. Duracion: 0.7 s.
