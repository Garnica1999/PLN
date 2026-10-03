# Post-P67 - Gap analysis

Generado por `experiments/p67_04_gap_analysis.py`. Fecha 2026-09-29T17:15:50+00:00.

## La pregunta central despues de P67

```text
Cuando V03 falla, que estructura observable EXCLUSIVAMENTE DESDE EL TRAIN
permite que otro modelo acierte, y que mecanismo DIFERENTE de una
transformacion de scores puede explotar esa informacion?
```

No es `que otra formula podemos aplicar al score`. Esa pregunta tiene respuesta desde
P67: ninguna que no sea la misma familia.

## Por que el analisis empieza por los errores

P67 dejo dos hechos que juntos obligan a cambiar de sitio la busqueda:

1. La senal de MI_arbol **existe**: AUC `0,5796` fuera de la banda de margen, y con
   `beta` fijo es positiva en los seis valores mientras el ruido de igual varianza es
   negativa en los seis.
2. La via de integracion por score **esta agotada**: el mecanismo estrella de P66 es
   algebraicamente un blend, con diferencia `8,882e-16`.

Luego la unica pregunta que queda abierta es donde esta la informacion y quien ya la
alcanza. Eso es lo que mide este script.

## La interseccion de errores

```text
A   V03 falla y TODAS las alternativas fallan    informacion que nadie alcanza
B   V03 falla y AL MENOS UNA acierta             informacion que SI se alcanza
```

- Celdas donde V03 acierta: **3692**
- A: **542** celdas
- B: **215** celdas  <-- la poblacion prioritaria
- De todo lo que V03 falla, alguien alcanza el **28.4 por ciento**

### Por etiqueta

| etiqueta | V03 falla | A | B | fraccion de B |
|---|---:|---:|---:|---:|
| HS | 316 | 218 | 98 | 31.0% |
| TR | 164 | 125 | 39 | 23.8% |
| AG | 277 | 199 | 78 | 28.2% |

### Por rebanada de longitud, con los cortes heredados de P63

| rebanada | filas | V03 falla | A | B | fraccion de B |
|---|---:|---:|---:|---:|---:|
| CORTA | 167 | 33 | 27 | 6 | 18.2% |
| MEDIA | 602 | 305 | 224 | 81 | 26.6% |
| LARGA | 547 | 306 | 210 | 96 | 31.4% |
| MUY_LARGA | 167 | 113 | 81 | 32 | 28.3% |

## La clasificacion de las familias

Cada familia con las doce cosas que el encargo pide, y las tres dimensiones separadas:
**senal de la representacion**, **mecanismo de integracion** y **mecanismo de
decision**. No se declara agotada una representacion porque concatenarla fallara, ni se
reabre con otra operacion equivalente.

### Representacion: segmentacion de evidencia (MI_seg)

- **Dimension**: representacion
- **Clasificacion**: `CLOSED`
- **Evidencia historica**: P64-05, delta -0,0549336945, el peor de los tres MI
- **Por que**: el concepto esta cerrado con cobertura corregida. La segmentacion densa destruye senal en vez de crearla
- **Experimentos**: P64-05
- **Fallo conocido**: la segmentacion pierde la composicion global del documento
- **Reformulacion posible**: ninguna que no sea otra concatenacion equivalente. La representacion no es el problema
- **Coste**: alto | **Riesgo de leakage**: bajo | **Prioridad**: descartada

### Representacion: MI_arbol (multiple instance con gradient boosting)

- **Dimension**: representacion
- **Clasificacion**: `PARTIALLY_OPEN`
- **Evidencia historica**: P66 y P67: AUC del residuo 0,5796 fuera de banda, 0,5654 contra el error, positiva en los SEIS betas de la correccion residual
- **Por que**: la senal EXISTE y esta medida. Lo que no existe es un mecanismo de integracion que la convierta en mejora robusta
- **Experimentos**: P64-05, P66-02, P66-03, P67-01
- **Mecanismo**: arbol de decision sobre evidencia segmental densa, no lineal
- **Integracion probada**: blend de probabilidad, blend de logits, residualizacion, fusion LR. Los cuatro son la misma familia por algebra
- **Fallo conocido**: la senal no cabe en un espacio de 20000 dimensiones por concatenacion, y las transformaciones de score dan 4 celdas de 1483
- **Reformulacion posible**: usa MI_arbol como un EXPERTO que decide sobre un SUBCONJUNTO de filas, en vez de como una columna o un score que se mezcla. Eso no es ni concatenar ni mezclar: es descomponer la decision
- **Coste**: medio | **Riesgo de leakage**: medio | **Prioridad**: ALTA

### Representacion: n-gramas de orden superior (word 1,3 y skip-grams)

- **Dimension**: representacion
- **Clasificacion**: `CLOSED`
- **Evidencia historica**: P65: word(1,3) -0,0039 con brecha de capacidad +0,0597; skip-grams -0,0001 en V6-05 y los huecos 1-2 reproducen el -0,0001
- **Por que**: la reproduccion del -0,0001 de V6-05 con una implementacion nueva es un control de implementacion que confirma el cierre
- **Experimentos**: V6-05, R2-MINDF5, P65-02
- **Fallo conocido**: el presupuesto L2 del espacio de features, no la disponibilidad
- **Reformulacion posible**: ninguna. Es otra bolsa de palabras mas ancha
- **Coste**: alto | **Riesgo de leakage**: bajo | **Prioridad**: descartada

### Representacion: cohesion documental, huecos y contraste

- **Dimension**: representacion
- **Clasificacion**: `CLOSED`
- **Evidencia historica**: P65: cohesion -0,0027, huecos 3-5 y 6-10 y contraste tampoco aportan. Las brechas de capacidad son amplias y el cv baja
- **Por que**: tres medidas de estructura de largo alcance, las tres neutras o peores
- **Experimentos**: P65-03
- **Fallo conocido**: es el indicador multi-segmento disfrazado, ventaja -0,025999
- **Reformulacion posible**: ninguna por concatenacion
- **Coste**: alto | **Riesgo de leakage**: bajo | **Prioridad**: descartada

### Representacion: negacion

- **Dimension**: representacion
- **Clasificacion**: `COMPLEMENTARY`
- **Evidencia historica**: P64-06: COMPLEMENTARY, brecha de capacidad +0,0201566146
- **Por que**: es complementaria y esta medida, pero su residuo tiene AUC 0,472985, por DEBAJO del azar: su parte unica es antisimetrica con el error
- **Experimentos**: P64-06, P67-01
- **Fallo conocido**: el residuo no sobrevive en ninguna banda
- **Reformulacion posible**: ninguna que cambie el mecanismo
- **Coste**: medio | **Riesgo de leakage**: bajo | **Prioridad**: baja

### Representacion: especialista por longitud

- **Dimension**: representacion
- **Clasificacion**: `INVALIDATED`
- **Evidencia historica**: P65: especialista y reentrenado dan +0,0000000000 entre si, BIT A BIT IDENTICOS
- **Por que**: el criterio de longitud no cambia NADA del modelo. La longitud no es el mecanismo, y quedo demostrado por identidad, no por parecido
- **Experimentos**: P65-05
- **Fallo conocido**: gana +0,0212 en MUY_LARGA y pierde -0,0257 en LARGA
- **Reformulacion posible**: ninguna: el resultado dice que no hay mecanismo que reformular
- **Coste**: bajo | **Riesgo de leakage**: bajo | **Prioridad**: descartada

### Integracion: concatenacion de features

- **Dimension**: integracion
- **Clasificacion**: `CLOSED`
- **Evidencia historica**: P64 y P65: SIETE representaciones baten a sus dos controles y NINGUNA mejora el corrected_cv al concatenarse
- **Por que**: la ley de P65 lo cierra: el cuello es el presupuesto L2 de un espacio de veinte mil dimensiones, no la disponibilidad de la senal
- **Experimentos**: P64 completo, P65 completo
- **Mecanismo**: anadir columnas
- **Fallo conocido**: el espacio no tiene presupuesto para la columna nueva
- **Reformulacion posible**: solo si la senal entra por un camino que NO sea una columna: por ejemplo decidiendo sobre un subconjunto
- **Coste**: alto | **Riesgo de leakage**: bajo | **Prioridad**: descartada

### Integracion: blend, stacking y residualizacion de scores

- **Dimension**: integracion
- **Clasificacion**: `DUPLICATE`
- **Evidencia historica**: P66 y P67. Y el hallazgo algebraico: z + beta*R == (1-beta*b0)*z_V03 + beta*z_C - beta*a, con diferencia 8,882e-16
- **Por que**: los cinco mecanismos de P66 son MENOS de los que parecian. `residual` pertenece algebraicamente a la familia de `blend_logit`
- **Experimentos**: V6-05, P59-09, P66-02, P66-03, P67-01
- **Mecanismo**: transformar el score
- **Fallo conocido**: 4 celdas de 1483, IC95 incluye el cero, 4 de 7 candidatos responden
- **Reformulacion posible**: ninguna dentro de la familia. Cualquier otra variante es la misma transformacion con otro nombre
- **Coste**: bajo | **Riesgo de leakage**: medio | **Prioridad**: descartada

### Decision: correccion de umbral y calibracion

- **Dimension**: decision
- **Clasificacion**: `CLOSED`
- **Evidencia historica**: P67-01: un escalar por etiqueta reentrenado de forma anidada da -0,0029731513, PEOR que el umbral congelado
- **Por que**: el modelo de calibracion mas barato posible empeora el resultado
- **Experimentos**: P67-01
- **Mecanismo**: mover la frontera
- **Fallo conocido**: el umbral congelado ya estaba bien puesto en esta particion
- **Reformulacion posible**: ninguna
- **Coste**: bajo | **Riesgo de leakage**: alto | **Prioridad**: descartada

### Decision: ensemble con pesos anidados

- **Dimension**: decision
- **Clasificacion**: `CLOSED`
- **Evidencia historica**: P61: CLOSED. El control negativo decide
- **Por que**: los pesos anidados no baten al mejor miembro
- **Experimentos**: P61
- **Fallo conocido**: la diversidad superficial no se traduce en ganancia
- **Reformulacion posible**: solo con diversidad REAL, no con correlaciones de 0,93 a 0,99
- **Coste**: medio | **Riesgo de leakage**: medio | **Prioridad**: descartada

## El criterio estricto de `GENUINELY_NEW`

Una hipotesis solo es nueva si cambia **materialmente el mecanismo** por el que intenta
producir senal predictiva. **NO cuentan** como nuevas:

```text
otro C                     otra semilla                otro umbral
otro numero de features    otra concatenacion          otro blend
otro stacking              otra residualizacion        otra correccion de umbral
```

Y hay una razon **demostrada**, no estetica, para descartar la ultima fila: `residual`
es algebraicamente `blend_logit`. No es que se parezcan: son el mismo objeto
matematico con otro nombre.

## La hipotesis que sobrevive

### `P68-H1` - descomposicion de la decision por enrutado sobre estructura del EJEMPLO

**Enunciado**: delegar un subconjunto de filas a un experto entrenado para ese subconjunto, con el enrutador aprendido ANIDADO y basado en features observables en train, nunca en el score ni en la etiqueta

- **Evidencia previa**: el analisis de errores de este script: la poblacion B existe, esta medida y tiene tamano
- **Novedad**: no concatena columnas (P64/P65, cerrado), no mezcla scores (P66/P67, DUPLICATE por algebra), no mueve el umbral (P67-01, cerrado), no es un ensemble de pesos (P61, cerrado). Cambia QUIEN decide cada fila
- **Control**: enrutador con las MISMAS features pero etiqueta del enrutador BARAJADA, mas un enrutador aleatorio de igual anchura
- **Modo de fallo esperado**: que el enrutador aprenda a mirar la longitud o el margen de V03 y que el especialista no aporte nada: se vera en la importancia de features del enrutador
- **Criterio de exito**: delta >= +0,0015 y >= 4 de 5 pliegues y control pasado
- **Presupuesto**: 1 principal + 1 control
- **Riesgo de leakage**: MEDIO. El enrutador es un modelo supervisado: su etiqueta tiene que construirse SOLO con informacion de train del pliegue, y su entrenamiento tiene que ser anidado. Si el enrutador ve el OOF completo, hay leakage

### Por que esa y no otra

```text
error residual real          B tiene tamano medido, no supuesto
informacion observable       el enrutador solo ve features del ejemplo y de train
mecanismo genuinamente nuevo cambia QUIEN decide, no COMO se combina un score
protocolo aislable           enrutador anidado, control con etiqueta barajada
riesgo de leakage controlable el enrutador es el unico punto de fuga, y es anidable
coste razonable              1 principal + 1 control
```

### Por que NO se elige la de mayor delta historico

El mayor delta historico de una familia viva es el `+0,0026743167` del residual, y su
familia esta clasificada **DUPLICATE** por algebra. Elegirlo seria gastar el presupuesto
en repetir lo ya medido. La regla del encargo es explicita: una hipotesis con delta
historico mayor pero mecanismo duplicado se clasifica `DUPLICATE` y **no se ejecuta**.

## El presupuesto

| concepto | valor |
|---|---:|
| experimentos historicos totales | 373 |
| de P67 | 4 |
| de P66 | 4 |
| de P65 | 5 |
| de P64 | 5 |
| P67 tracks principales previstos | 12 |
| P67 tracks principales ejecutados | 1 |
| P67 tracks reserva previstos | 4 |
| P67 tracks reserva ejecutados | 0 |
| P67 familias cerradas | 6 |
| P67 familias invalidadas | 1 |
| P67 familias duplicadas | 1 |
| P67 familias marginales | 0 |
| P67 familias complementarias | 1 |
| P67 familias parcialmente abiertas | 1 |
| restante principal de P67 | 11 |
| restante reserva de P67 | 4 |

el presupuesto sin gastar NO se gasta por gastarlo. Un presupuesto sin usar es preferible a experimentos de baja informacion.

## Lo que este gap analysis NO hace

```text
no entrena nada con la etiqueta del test
no usa el holdout
no usa el leaderboard
no promedia predicciones de candidatos
no crea un ensemble por diversidad superficial
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p67_04_gap_analysis.py`. Sin push.*