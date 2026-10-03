# P68-H2 - Rescue Re-weighting: auditoria, y por que NO se ejecuta

Generado por `experiments/p68_02_h2_auditoria.py`. Fecha 2026-09-29T17:26:55+00:00.

## La clasificacion

```text
DUPLICATE

coste en experimentos: 0
```

## La propuesta

Reponderar los pares etiqueta-instancia donde V03 falla y un modelo alternativo acierta,
y reentrenar el mismo clasificador con esos pesos, para ver si el mapa de rescates
**transfiere al entrenamiento** en vez de corregir el score despues.

```text
R_i = (pred_V03_i != y_i) AND (pred_OTRO_i == y_i)
w   = 2 si R_i, 1 en el resto
```

## La auditoria que lo decide

La pregunta no es si el nombre es nuevo. Es de donde sale el peso. Cada ponderado del
registro se clasifica por la **fuente de su indicador**:

| experimento | fuente | indicador | resultado | estado |
|---|---|---|---|---|
| `V4-B03` | propio | dureza OOF de V03: margen y violacion del umbral | upweighting de la dureza EMPEORA: medium_up -0,0096. Solo mejora | `CLOSED` |
| `V4-B04` | propio | dureza de la propia etiqueta | 0,7686 frente a 0,7772: empeora | `CLOSED` |
| `V4-B05` | propio | contradiccion textual del ejemplo | downweight +0,0005 y remove -0,0004: nada | `CLOSED` |
| `V4-C01..C05` | propio | violacion del umbral combinada con el peso de clase | EL UNICO CASO QUE FUNCIONA: 0,785416, +0,008205, 5 de 5 pliegues | `NO ADOPTADO` |
| `V6-03` | propio | repeticiones del ejemplo | la ponderacion por tamano de grupo no corrige el problema | `CLOSED` |
| `V6-07 / V30-00` | propio | inconsistencia del par instancia-etiqueta detectada  | el entrenamiento robusto OOF no reduce el impacto de forma utili | `CLOSED` |
| `P60-07 (P60_I)` | propio | eps_i = <g_i, G>/<G, G>, alineacion del gradiente de | -0,3011 frente a V03, con pesos casi degenerados | `CLOSED` |
| `P60-08` | propio | region de error, modulada por la perdida y NO por un | reformula el reparto del gradiente entre regiones de error | `CLOSED` |
| `V8-07` | propio | peso de clase elegido con validacion anidada | no mejora sobre balanced | `CLOSED` |
| `V51-01 / V51-02` | ajeno | P(canal acierta) estimado por canal y celda: es el u | -0,0101586431, con 121 rescates y 183 danos, neto -62 | `INVALIDATED` |
| `P68-H2 (propuesta)` | ajeno | R_i = (pred_V03 != y) AND (pred_OTRO == y) | no ejecutado | `DUPLICATE` |

**9 con indicador propio** (funcion de `p_V03` e `y`) y **2 con indicador ajeno**.

## El argumento, en una linea

El indicador que P68-H2 propone es el de V4-B03 **intersecado** con la correccion de otro
modelo:

```text
V4-B03 ya pondero:        pred_V03 != y          (hard example mining)
P68-H2 propone:           pred_V03 != y  AND  pred_OTRO == y

el segundo termino no anade informacion al indicador: le QUITA filas
```

Y la medicion que cierra el caso:

```text
celdas donde V03 falla                        757
de esas, alguna alternativa acierta           215 (28.4%)
celdas de error que el mapa EXCLUYE            542
```

El mapa de rescates cubre el `28.4 por ciento` de los errores de V03, luego
el indicador propuesto es un **subconjunto estricto** de uno que ya se midio. Un
subconjunto de un indicador no puede ser un mecanismo nuevo.

## Y ademas, en la direccion que salio mal

V4-B03 probo las dos direcciones sobre la dureza:

```text
mild_up   0,7751   -0,0021   subir el peso a lo dificil EMPEORA
medium_up 0,7676   -0,0096   subirlo mas, EMPEORA mas
down_violacion 0,7784  +0,0036   BAJAR el peso a los que violan: mejora
```

P68-H2 propone **subir** el peso (`w = 2`), y sobre una poblacion que es la parte mas
dificil del corpus. La razon por la que subir no funciona esta escrita desde V4 y no ha
cambiado:

> en este corpus los ejemplos que el modelo falla son, desde el punto de vista del
> modelo, frontera irreductible con la representacion disponible: subir su peso empuja
> la frontera y degrada el resto

## El unico ponderado que funciono, y por que no es un mapa de rescates

`A_cost_sensitive_downweight` alcanzo `0,785416` (`+0,008205`) con **5 de 5 pliegues**.
Es el mejor ponderado del proyecto y no se adopto porque **fallo el criterio 6, el de
artefacto**, del gate de V4-X01, y no tiene veredicto externo.

Su indicador es la **violacion del umbral del propio V03 combinada con el peso de
clase**. No mira a ningun otro modelo. Es la direccion que funciona: bajar, no subir.

## Y el antecedente que ya construyo el mapa de rescates

P50 y P51 no solo construyeron el mapa de rescates de los canales, sino que midieron su
techo y fallaron al explotarlo:

```text
P50   techo del oracle                         +0,2098
P51   gating por utilidad esperada             -0,0102   (121 rescates, 183 danos)
```

Es la misma leccion de P68-H1 con otro nombre: el techo existe y no es alcanzable.

## Las secciones que el encargo pedia, y su estado

| seccion | estado |
|---|---|
| novedad | `DUPLICATE`: unico cambio: la fuente del indicador. El indicador es un subconjunto estric |
| hipotesis | `NO EJECUTADA`: R_i = (pred_V03 != y) AND (pred_OTRO == y), con w = 2 |
| protocolo nested | `NO EJECUTADO`: no se gasto presupuesto en un mecanismo duplicado |
| construccion del rescue map | `YA EXISTE`: P50 y P51 lo construyeron: techo del oracle +0,2098 y -0,0102 real |
| pesos | `NO EJECUTADOS`: w = 2 propuesto |
| real | `NO EJECUTADO`: n/a |
| shuffled | `NO EJECUTADO`: n/a |
| random | `NO EJECUTADO`: n/a |
| resultados | `NO EJECUTADOS`: el coste de esta hipotesis es 0 experimentos |
| bootstrap | `NO APLICA`: solo se calcula para un candidato que pase el gate inicial |
| complementarity | `MEDIDA`: 122 rescates exclusivos, 122 en el gap analysis previo |
| decision | `DUPLICATE`: no se ejecuta; se cierra la fase de investigacion porque no quedan ejes |

## Lo que NO se hace

```text
no se entrena con sample weights derivados del mapa de rescates
no se lanza una grid de pesos
no se reabre P68-H1
no se toca el test, ni el holdout, ni el leaderboard
no se consume presupuesto para repetir una familia cerrada
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p68_02_h2_auditoria.py`. Sin push.*